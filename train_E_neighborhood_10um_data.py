#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-cleaning training/inference pipeline with streaming CSV and Z-slice plots.
- Train/Test: header=None, drop first row, keep rows where column 20 (1-based) == 1
- Inference:  header=None, drop first row, keep rows where column 24 (1-based) == 1
- Model input: [B, 1, 17]
- GPU AMP enabled when CUDA is available.

Outputs:
- best_model.pth
- label_mapping.json
- inference_lite.csv
- plots: Z-slice scatter images for combined TRUE (train+test) and predicted inference.

"""

from __future__ import annotations

import argparse
import json
import os
import time
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Your models
from network17 import MAFNet

# ---------------------------------------------------------------------
# Baseline fallback model (kept for parity/testing; not used by default)
class Simple1DCNN(nn.Module):
    def __init__(self, num_classes: int, seq_len: int = 17):
        super().__init__()
        if seq_len < 4:
            raise ValueError("seq_len must be >= 4 for two MaxPool1d(2) layers.")
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
# ---------------------------------------------------------------------


# ---------------------------- Utilities -----------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # allow autotune but deterministic kernels when possible


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_tuple3(values: Sequence[float | int]) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("Expected exactly 3 values.")
    a, b, c = float(values[0]), float(values[1]), float(values[2])
    return a, b, c


# ------------------------- Color map helpers ------------------------
def build_color_map(label_index: int) -> Dict[int, Tuple[float, float, float]]:
    """Return RGB colors in [0,1] per original label id."""
    def norm(rgb): r, g, b = rgb; return (r/255.0, g/255.0, b/255.0)
    cmap17_raw = {
        1:(250,0,135),2:(97,226,164),3:(208,0,0),4:(22,242,242),5:(27,67,50),
        6:(204,255,51),7:(249,84,238),8:(112,224,0),9:(177,153,255),10:(50,131,254),
        11:(69,0,153),12:(255,102,0),13:(144,224,239),14:(170,13,254),15:(242,130,102),
        16:(1,214,105),17:(250,163,7),18:(13,71,161),19:(0,114,0),20:(158,240,26),
        21:(56,176,0),22:(236,64,103),23:(107,92,165),24:(240,160,255),25:(8,99,117),
        26:(114,25,90),27:(0,150,199),28:(255,251,70),29:(157,2,8)
    }
    cmap18_raw = {1:(77,88,255),2:(255,38,120),3:(158,60,204),4:(252,129,15),5:(16,135,65),6:(68,226,179),7:(166,144,111)}
    raw = cmap18_raw if label_index == 18 else cmap17_raw
    return {int(k): norm(v) for k, v in raw.items()}


def colors_for_labels(labels: np.ndarray, color_map: dict, default=(0.2, 0.2, 0.2)):
    return [color_map.get(int(l), default) for l in labels]


# ------------------------------ I/O --------------------------------
def _read_csv_no_header(path: str, drop_first_row: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, low_memory=False)
    if drop_first_row and len(df) > 0:
        df = df.iloc[1:, :].reset_index(drop=True)
    return df


def load_train_or_test(
    csv_paths: Iterable[str],
    *,
    label_index: int,
    keep_col20_equal_1: bool = True,
    drop_first_row: bool = True,
    feature_cols: Sequence[int] = tuple(range(17)),
    scale_cols: Sequence[int] = (14, 15, 16),
    max_vec: Sequence[float] | None = None,
    clip_to_unit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and stack multiple train/test CSVs.
    - Filter rows where col20 (1-based) == 1 if requested.
    - Return:
        X:   float32 [N, F]
        y:   int     [N]
        zxy: float32 [N, 3] (cols 14..16: z,x,y raw)
    """
    X_list, y_list, zxy_list = [], [], []
    for p in csv_paths:
        df = _read_csv_no_header(p, drop_first_row=drop_first_row)
        if keep_col20_equal_1:
            before = len(df)
            df = df.loc[df.iloc[:, 19].astype(float) == 1.0].reset_index(drop=True)
            print(f"[FILTER col20==1] {os.path.basename(p)}: kept {len(df)}/{before}")

        zxy_raw = df.iloc[:, [14, 15, 16]].astype(float).values
        X = df.iloc[:, list(feature_cols)].astype(np.float32).values
        if max_vec is not None and len(scale_cols) > 0:
            mv = np.asarray(max_vec, dtype=np.float32)
            X[:, list(scale_cols)] = X[:, list(scale_cols)] / (mv + 1e-8)
            if clip_to_unit:
                X[:, list(scale_cols)] = np.clip(X[:, list(scale_cols)], 0.0, 1.0)

        y = df.iloc[:, label_index].astype(int).values
        X_list.append(X); y_list.append(y); zxy_list.append(zxy_raw)

    X = np.vstack(X_list).astype(np.float32, copy=False)
    y = np.concatenate(y_list).astype(np.int32, copy=False)
    zxy_raw = np.vstack(zxy_list).astype(np.float32, copy=False)
    return X, y, zxy_raw


def reindex_labels_to_contiguous(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], np.ndarray]:
    """
    Map arbitrary integer labels (e.g., {4,7,9}) to contiguous {0..K-1}.
    Returns: y_new, forward_map, inverse_map, original_sorted_labels
    """
    uniq = np.unique(y).astype(int)
    ordered = [int(v) for v in uniq]
    mapping = {orig: i for i, orig in enumerate(ordered)}
    inv = {i: orig for i, orig in enumerate(ordered)}
    y_new = np.vectorize(lambda t: mapping[int(t)])(y)
    return y_new.astype(np.int64), mapping, inv, np.array(ordered, dtype=int)


def read_predfinal_filter_col24_eq1(csv_path: str, drop_first_row: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    For inference files:
    - header=None
    - drop first row
    - keep rows where column 24 (1-based index 23) == 1
    Returns:
        feats: float32 [N, 17]
        zxy:   float32 [N, 3]  (cols 14..16: z, x, y)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"file not found: {csv_path}")
    df = _read_csv_no_header(csv_path, drop_first_row=drop_first_row)
    before = len(df)
    df = df.loc[df.iloc[:, 23].astype(float) == 1.0].reset_index(drop=True)
    print(f"[FILTER col24==1] {os.path.basename(csv_path)}: kept {len(df)}/{before}")

    zxy_raw = df.iloc[:, [14, 15, 16]].astype(float).values
    feats = df.iloc[:, :17].astype(np.float32).values
    return feats, zxy_raw


# ----------------------------- Plots -------------------------------
def plot_by_z(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels_orig: np.ndarray,
    color_map: Dict[int, Tuple[float, float, float]],
    out_dir: str,
    prefix: str,
    s: int = 2,
    alpha: float = 0.8,
) -> None:
    """
    Create a scatter per unique integer Z slice. Points colored by original labels.
    """
    ensure_dir(out_dir)
    z_int = z.astype(int)
    uniq_z = np.unique(z_int)
    for zi in tqdm(uniq_z, desc=f"Plot {prefix} by Z"):
        mask = (z_int == int(zi))
        if not np.any(mask):
            continue
        cols = colors_for_labels(labels_orig[mask], color_map)
        plt.figure(figsize=(7, 6))
        plt.scatter(x[mask], y[mask], c=cols, s=s, alpha=alpha, edgecolors='none')
        plt.xlabel("x"); plt.ylabel("y"); plt.title(f"{prefix} | Z={int(zi)}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_Z{int(zi):04d}.png")
        plt.savefig(out_path, dpi=240)
        plt.close()


# ------------------------ Training / Eval --------------------------
def train_and_save_best(
    model: nn.Module,
    *,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    device: torch.device,
    out_path: str | Path,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
) -> nn.Module:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    te_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=False,
                           pin_memory=(device.type == "cuda"), num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                           pin_memory=(device.type == "cuda"), num_workers=0)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_amp))
    best_acc, best_f1 = -1.0, -1.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train(); running = 0.0
        pbar = tqdm(tr_loader, desc=f"Train {epoch:03d}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.unsqueeze(1).to(device, non_blocking=True)  # [B,1,17]
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and use_amp)):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{running / max(1, pbar.n):.4f}")
        train_loss = running / max(1, len(tr_loader))

        # Eval
        model.eval(); y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in te_loader:
                xb = xb.unsqueeze(1).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and use_amp)):
                    logits = model(xb)
                y_true.extend(yb.numpy().tolist())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        scheduler.step()

        print(f"[Epoch {epoch:03d}/{epochs}] train_loss={train_loss:.4f} | acc={acc:.4f} | f1w={f1w:.4f}")

        improved = (acc > best_acc) or (acc == best_acc and f1w > best_f1)
        if improved:
            best_acc, best_f1 = acc, f1w
            torch.save(model.state_dict(), out_path)
            print(f"  -> Best saved to {out_path} (acc={best_acc:.4f}, f1w={best_f1:.4f})")

    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    return model


# ---------------------- Streaming Inference ------------------------
def stream_infer_and_save_csv(
    model: nn.Module,
    *,
    inv_map: Dict[int, int],
    inf_paths: Sequence[str],
    save_dir: str,
    inf_coord_max: Sequence[float],
    batch_size: int = 8192,
    csv_name: str = "inference_lite.csv",
) -> str:
    """
    For each file:
      - read & filter (col24==1)
      - normalize z/x/y (feature cols 14..16) by inf_coord_max and clip to [0,1]
      - forward pass with AMP
      - append rows to a streaming CSV with fsync after each batch
    """
    out_csv = os.path.join(save_dir, csv_name)
    ensure_dir(save_dir)
    device = next(model.parameters()).device

    f = open(out_csv, "w", buffering=64 * 1024 * 1024)
    f.write("_source_file,_row_in_source,z,x,y,pred_label_mapped,pred_label_original\n")
    f.flush(); os.fsync(f.fileno())

    total_written = 0
    total_bar = tqdm(total=0, desc="TOTAL rows", unit="row", position=0, leave=True)

    with torch.no_grad():
        for p in inf_paths:
            base = os.path.basename(p)
            print(f"[Infer] {base}")
            feats, zxy_raw = read_predfinal_filter_col24_eq1(p, drop_first_row=True)

            feats = feats.copy()
            inf_cm = np.asarray(inf_coord_max, dtype=np.float32)
            feats[:, [14, 15, 16]] = feats[:, [14, 15, 16]] / (inf_cm + 1e-8)
            feats[:, [14, 15, 16]] = np.clip(feats[:, [14, 15, 16]], 0.0, 1.0)

            file_rows = feats.shape[0]
            total_bar.total += file_rows
            total_bar.refresh()

            ds_tensor = torch.from_numpy(feats)
            loader = DataLoader(ds_tensor, batch_size=batch_size, shuffle=False, drop_last=False,
                                pin_memory=(device.type == "cuda"), num_workers=0)

            file_bar = tqdm(total=file_rows, desc=f"{base}", unit="row", position=1, leave=False)
            row_start = 0
            rows_done_prev, t_prev = 0, time.perf_counter()

            for xb in loader:
                bsz = xb.shape[0]
                xb = xb.unsqueeze(1).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(xb)
                pred_mapped = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32)
                pred_orig = np.array([inv_map[int(v)] for v in pred_mapped], dtype=np.int32)

                idx = slice(row_start, row_start + bsz)
                zr = zxy_raw[idx, 0]; xr = zxy_raw[idx, 1]; yr = zxy_raw[idx, 2]

                lines = [
                    f"{base},{row_start+i},{zr[i]},{xr[i]},{yr[i]},{int(pred_mapped[i])},{int(pred_orig[i])}\n"
                    for i in range(bsz)
                ]
                f.writelines(lines)
                f.flush(); os.fsync(f.fileno())

                row_start += bsz
                total_written += bsz
                total_bar.update(bsz)
                file_bar.update(bsz)

                now = time.perf_counter()
                dt = max(1e-6, now - t_prev)
                rate = (row_start - rows_done_prev) / dt
                file_bar.set_postfix(done=f"{row_start}/{file_rows}", rps=f"{rate:,.0f}/s")
                t_prev, rows_done_prev = now, row_start

            file_bar.close()
            print(f"[CSV] {base}: wrote {row_start} rows")

    f.close()
    total_bar.close()
    print(f"[CSV] Done -> {out_csv}  (total rows: {total_written:,})")
    return out_csv


# ------------------------------- CLI -------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-cleaning train/infer + streaming CSV + Z-slice plots (AMP).")
    # data (repeatable)
    p.add_argument("--train_csv", action="append", required=True,
                   help="Path(s) to training CSV. Repeat flag to add multiple.")
    p.add_argument("--test_csv",  action="append", required=True,
                   help="Path(s) to test CSV. Repeat flag to add multiple.")
    p.add_argument("--infer_csv", action="append", required=True,
                   help="Path(s) to inference CSV. Repeat flag to add multiple.")

    # columns & normalization
    p.add_argument("--label_index", type=int, default=18,
                   help="0-based label column index for train/test (default: 18).")
    p.add_argument("--feature_cols", type=str, default="0:17",
                   help="Feature columns (0-based). Format 'start:end' (end exclusive). Default '0:17'.")
    p.add_argument("--scale_cols", type=str, default="14,15,16",
                   help="Columns to normalize with *_coord_max. Comma list. Default '14,15,16'.")
    p.add_argument("--train_coord_max", type=float, nargs=3, default=[76.0, 11000.0, 11000.0],
                   help="Normalization max for z/x/y in training CSVs (3 numbers).")
    p.add_argument("--infer_coord_max", type=float, nargs=3, default=[1320.0, 1100.0, 1100.0],
                   help="Normalization max for z/x/y in inference CSVs (3 numbers).")

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seq_len", type=int, default=17)
    p.add_argument("--model", type=str, choices=["mafnet", "simple"], default="mafnet")

    # output
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--plots_dir_true", type=str, default=None)
    p.add_argument("--plots_dir_infer", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true", help="Disable AMP even on CUDA.")
    return p.parse_args()


def parse_cols(spec: str) -> List[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in spec.split(",") if s.strip() != ""]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve dirs
    save_dir = ensure_dir(args.save_dir)
    plots_dir_true = ensure_dir(args.plots_dir_true or (save_dir / "plots_by_z_combined_true"))
    plots_dir_infer = ensure_dir(args.plots_dir_infer or (save_dir / "plots_by_z_infer_pred"))
    model_path = save_dir / "best_model.pth"

    # Column definitions
    FEATURE_COLS = parse_cols(args.feature_cols)  # default 0..16
    SCALE_COLS = parse_cols(args.scale_cols)      # default 14,15,16
    TRAIN_COORD_MAX = np.array(as_tuple3(args.train_coord_max), dtype=np.float32)
    INF_COORD_MAX   = np.array(as_tuple3(args.infer_coord_max), dtype=np.float32)

    # ---------------- Load train/test ----------------
    print("===> Loading train/test (no cleaning, col20==1) ...")
    X_tr, y_tr, zxy_tr = load_train_or_test(
        args.train_csv,
        label_index=args.label_index,
        keep_col20_equal_1=True,
        drop_first_row=True,
        feature_cols=FEATURE_COLS,
        scale_cols=SCALE_COLS,
        max_vec=TRAIN_COORD_MAX,
        clip_to_unit=True,
    )
    X_te, y_te, zxy_te = load_train_or_test(
        args.test_csv,
        label_index=args.label_index,
        keep_col20_equal_1=True,
        drop_first_row=True,
        feature_cols=FEATURE_COLS,
        scale_cols=SCALE_COLS,
        max_vec=TRAIN_COORD_MAX,
        clip_to_unit=True,
    )

    # Re-index labels to contiguous 0..K-1
    y_tr_mapped, forward_map, inv_map, orig_labels = reindex_labels_to_contiguous(y_tr)
    y_te_mapped = np.vectorize(lambda t: forward_map[int(t)])(y_te)

    with open(save_dir / "label_mapping.json", "w") as f:
        json.dump({
            "orig_labels": orig_labels.tolist(),
            "map":   {str(k): int(v) for k, v in forward_map.items()},
            "inv_map": {str(k): int(v) for k, v in inv_map.items()},
        }, f, indent=2)
    print("Saved label mapping:", save_dir / "label_mapping.json")

    # Prepare combined TRUE arrays for plotting
    zxy_comb = np.vstack([zxy_tr, zxy_te]).astype(np.float32, copy=False)
    y_comb_mapped = np.concatenate([y_tr_mapped, y_te_mapped])
    y_comb_orig   = np.array([inv_map[int(v)] for v in y_comb_mapped], dtype=int)

    # ---------------- Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    num_classes = int(np.unique(y_tr_mapped).size)

    if args.model == "mafnet":
        model: nn.Module = MAFNet(num_classes=num_classes, seq_len=args.seq_len).to(device)
    else:
        model = Simple1DCNN(num_classes=num_classes, seq_len=args.seq_len).to(device)

    # ---------------- Train (if needed) ----------------
    if not model_path.exists():
        print(f"[Info] Training and saving best to {model_path} ...")
        model = train_and_save_best(
            model,
            X_tr=X_tr, y_tr=y_tr_mapped,
            X_te=X_te, y_te=y_te_mapped,
            device=device, out_path=str(model_path),
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            use_amp=(not args.no_amp),
        )
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"[Info] Loaded model: {model_path}")

    # ---------------- Streaming inference ----------------
    print("===> Inference (col24==1, streaming CSV) ...")
    out_csv = stream_infer_and_save_csv(
        model=model,
        inv_map=inv_map,
        inf_paths=args.infer_csv,
        save_dir=str(save_dir),
        inf_coord_max=INF_COORD_MAX,
        batch_size=8192,
        csv_name="inference_lite.csv",
    )

    # ---------------- Plots by Z ----------------
    print("===> Plotting by Z slices ...")
    color_map = build_color_map(args.label_index)

    # TRUE (train+test)
    plot_by_z(
        x=zxy_comb[:, 1], y=zxy_comb[:, 2], z=zxy_comb[:, 0],
        labels_orig=y_comb_orig, color_map=color_map,
        out_dir=str(plots_dir_true), prefix="Combined_TRUE"
    )

    # PRED (inference)
    z_list, x_list, y_list, pred_orig_all = [], [], [], []
    with torch.no_grad():
        for p in args.infer_csv:
            feats, zxy_raw = read_predfinal_filter_col24_eq1(p, drop_first_row=True)
            feats = feats.astype(np.float32)
            feats[:, [14, 15, 16]] = feats[:, [14, 15, 16]] / (INF_COORD_MAX + 1e-8)
            feats[:, [14, 15, 16]] = np.clip(feats[:, [14, 15, 16]], 0.0, 1.0)
            ds_tensor = torch.from_numpy(feats)
            loader = DataLoader(ds_tensor, batch_size=8192, shuffle=False, drop_last=False,
                                pin_memory=(device.type == "cuda"), num_workers=0)
            preds_file: List[np.ndarray] = []
            for xb in tqdm(loader, leave=False, desc=f"Plot infer {os.path.basename(p)}"):
                xb = xb.unsqueeze(1).to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                    logits = model(xb)
                pm = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32)
                preds_file.append(pm)
            pm_all = np.concatenate(preds_file) if len(preds_file) > 0 else np.empty((0,), dtype=np.int32)
            pred_orig = np.array([inv_map[int(v)] for v in pm_all], dtype=np.int32)
            pred_orig_all.append(pred_orig)
            z_list.append(zxy_raw[:, 0]); x_list.append(zxy_raw[:, 1]); y_list.append(zxy_raw[:, 2])

    if len(z_list) > 0:
        z_inf = np.concatenate(z_list).astype(np.float32)
        x_inf = np.concatenate(x_list).astype(np.float32)
        y_inf = np.concatenate(y_list).astype(np.float32)
        preds_orig = np.concatenate(pred_orig_all).astype(np.int32) if len(pred_orig_all) > 0 else np.empty((0,), dtype=np.int32)

        plot_by_z(
            x=x_inf, y=y_inf, z=z_inf,
            labels_orig=preds_orig, color_map=color_map,
            out_dir=str(plots_dir_infer), prefix="Infer_PRED"
        )

    print("\nAll done. Inference (col24==1) + streaming CSV + Z-slice plots finished.")
    print("Output CSV:", out_csv)


if __name__ == "__main__":
    main()
