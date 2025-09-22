#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python train_infer_mafnet.py \
  --train_csv "/project/AIRC/NWang_lab/shared/rui/Cell_type/separate_E_I/data_group/dMRIcelltype_N54943_10timesres_trainingdata_MRIonly_allslices_addEI_nooverlapping_cleaned_E_group1_training.csv" \
  --test_csv  "/project/AIRC/NWang_lab/shared/rui/Cell_type/separate_E_I/data_group/dMRIcelltype_N54943_10timesres_trainingdata_MRIonly_allslices_addEI_nooverlapping_cleaned_E_group1_testing.csv" \
  --new_csv   "/project/AIRC/NWang_lab/shared/rui/Cell_type/10um/dMRIcelltype_N58171_CCF10umdata_NeuNcellmasked_cleaned_E_group1.csv" \
  --save_dir  "/project/AIRC/NWang_lab/shared/rui/Cell_type/10um/result/results_E1" \
  --epochs 200 --batch_size 64 --lr 1e-4 --weight_decay 1e-5 --step_size 10 --gamma 0.9 \
  --early_stop_patience 20 --save_prob
"""

from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from network17 import MAFNet  

TRAIN_COORD_MAX = np.array([76.0, 11000.0, 11000.0], dtype=float)
INF_COORD_MAX   = np.array([1320.0, 800.0, 1140.0], dtype=float)


FEATURE_SLICE = slice(0, 17)  
COORD_SLICE   = slice(14, 17)  
LABEL_COL     = 19             


# ---------------------- Utils ----------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights; tolerate missing classes."""
    unique, counts = np.unique(labels, return_counts=True)
    max_label = int(unique.max()) if unique.size > 0 else 1
    n_classes = max(max_label + 1, 2)  # 至少二分类
    total = counts.sum() if counts.size > 0 else 1
    # 初始化为 1.0
    weights = np.ones(n_classes, dtype=np.float32)
    for cls, cnt in zip(unique, counts):
        weights[int(cls)] = float(total / max(cnt, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def check_columns(df: pd.DataFrame) -> None:
    """Basic sanity checks on expected column indices."""
    needed_max = max(FEATURE_SLICE.stop - 1, COORD_SLICE.stop - 1, LABEL_COL)
    if df.shape[1] <= needed_max:
        raise ValueError(
            f"CSV has only {df.shape[1]} columns, but indices up to {needed_max} are required. "
            f"Expected: features 0-16, coords 14-16, label 19."
        )


# ---------------------- Datasets ----------------------
class DMRI_CellType_BinaryDataset(Dataset):
    """
    Training/validation dataset normalized by TRAIN_COORD_MAX.
    CSV expectation:
      - features in cols 0–16,
      - coords in cols 14–16,
      - label  in col 19 (only 0/1 are used).
    """
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        check_columns(df)
        features = df.iloc[:, FEATURE_SLICE].to_numpy(dtype=float)
        labels   = df.iloc[:, LABEL_COL].to_numpy(dtype=int)
        coords   = df.iloc[:, COORD_SLICE].to_numpy(dtype=float) / TRAIN_COORD_MAX

        mask = np.isin(labels, [0, 1])  
        self.features = features[mask]
        self.labels   = labels[mask]
        self.coords   = coords[mask]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x     = torch.tensor(self.features[idx], dtype=torch.float32)
        y     = torch.tensor(self.labels[idx],   dtype=torch.long)
        coord = torch.tensor(self.coords[idx],   dtype=torch.float32)
        return x, y, coord


class DMRI_InferenceDataset(Dataset):
    """
    Batch inference dataset normalized by INF_COORD_MAX.
    CSV expectation:
      - features in cols 0–16,
      - coords in cols 14–16.
    """
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        check_columns(df)
        self.features = df.iloc[:, FEATURE_SLICE].to_numpy(dtype=float)
        coords_raw    = df.iloc[:, COORD_SLICE].to_numpy(dtype=float)
        self.coords   = coords_raw / INF_COORD_MAX

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x     = torch.tensor(self.features[idx], dtype=torch.float32)
        coord = torch.tensor(self.coords[idx],   dtype=torch.float32)
        return x, coord


# ---------------------- Config ----------------------
@dataclass
class Config:
    # data
    train_csv: str
    test_csv: str
    new_csv: str
    save_dir: str = "./outputs"

    # training
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5
    step_size: int = 10
    gamma: float = 0.9
    early_stop_patience: int = 20  # epochs without val acc improvement

    # runtime
    num_workers: int = 4
    pin_memory: bool = True
    log_every_n: int = 50
    save_prob: bool = False  # save class probability for inference

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------- Training / Eval ----------------------
def create_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    train_ds = DMRI_CellType_BinaryDataset(cfg.train_csv)
    test_ds  = DMRI_CellType_BinaryDataset(cfg.test_csv)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device,
             criterion: Optional[torch.nn.Module] = None) -> Dict[str, float | List[int]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    running_loss = 0.0
    n_batches = 0

    for x, y, _ in loader:
        x = x.unsqueeze(1).to(device)  # [B, C=1, L=17]
        y = y.to(device)
        logits = model(x)
        if criterion is not None:
            running_loss += float(criterion(logits, y).item())
            n_batches += 1
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())

    metrics: Dict[str, float | List[int]] = {}
    metrics["val_loss"] = running_loss / max(n_batches, 1)
    metrics["val_acc"]  = float(accuracy_score(y_true, y_pred))
    metrics["val_f1_w"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["y_true"]   = y_true
    metrics["y_pred"]   = y_pred
    return metrics


def train_and_validate(cfg: Config) -> Tuple[torch.nn.Module, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    set_seed(cfg.seed, deterministic=True)
    train_loader, test_loader = create_loaders(cfg)

    # class weights from training labels
    train_labels = train_loader.dataset.labels  # type: ignore[attr-defined]
    class_weights = safe_class_weights(train_labels, device=device)
    torch.save(class_weights.detach().cpu(), Path(cfg.save_dir) / "class_weights.pt")

    num_classes = int(class_weights.numel())
    model = MAFNet(num_classes=num_classes, seq_len=17).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # save initial state
    torch.save(model.state_dict(), Path(cfg.save_dir) / "initial_weights.pth")

    best_acc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.num_epochs + 1):
        # ---------------- Train ----------------
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} Train", leave=False)
        for i, (x, y, _) in enumerate(pbar, start=1):
            x = x.unsqueeze(1).to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            if i % cfg.log_every_n == 0:
                pbar.set_postfix(loss=f"{running_loss/i:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = running_loss / max(len(train_loader), 1)

        # ---------------- Validate ----------------
        val_metrics = evaluate(model, test_loader, device, criterion)
        val_loss = float(val_metrics["val_loss"])
        val_acc  = float(val_metrics["val_acc"])
        val_f1   = float(val_metrics["val_f1_w"])

        print(f"Epoch {epoch}/{cfg.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1(w): {val_f1:.4f}")

        # Text report (also print at each epoch for traceability)
        report = classification_report(
            val_metrics["y_true"], val_metrics["y_pred"],
            target_names=['others', 'E'], zero_division=0
        )
        cm = confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"])
        print("Validation Report:\n", report)
        print("Validation Confusion Matrix:\n", cm)

        # save per-epoch snapshots (optional, only metrics)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_w": val_f1,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        # scheduler step
        scheduler.step()

        # best checkpoint on Val Acc
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), Path(cfg.save_dir) / "best_model.pth")
            print(f"[Info] ✔ New best model saved at epoch {epoch} (val_acc={best_acc:.4f})")
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= cfg.early_stop_patience:
            print(f"[Info] Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (acc={best_acc:.4f})")
            break

    # persist training history
    pd.DataFrame(history).to_csv(Path(cfg.save_dir) / "training_history.csv", index=False)

    # final evaluation on test set using best checkpoint
    model.load_state_dict(torch.load(Path(cfg.save_dir) / "best_model.pth", map_location=device))
    final = evaluate(model, test_loader, device, criterion=None)
    final_cm = confusion_matrix(final["y_true"], final["y_pred"])
    final_report = classification_report(final["y_true"], final["y_pred"],
                                         target_names=['others','E'], zero_division=0)

    # save results
    pd.DataFrame(final_cm, index=['others','E'], columns=['pred_others','pred_E'])\
      .to_csv(Path(cfg.save_dir) / "confusion_matrix.csv")
    with open(Path(cfg.save_dir) / "classification_report.txt", "w") as f:
        f.write(final_report)
    with open(Path(cfg.save_dir) / "metrics.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "final_test_acc": float(accuracy_score(final["y_true"], final["y_pred"])),
            "final_test_f1_w": float(f1_score(final["y_true"], final["y_pred"], average="weighted", zero_division=0)),
        }, f, indent=2)

    print("\n[Final Test Set Evaluation]")
    print(final_report)
    print("Confusion Matrix:\n", final_cm)

    return model, {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
    }


@torch.no_grad()
def run_inference(cfg: Config, model: torch.nn.Module) -> Path:
    device = next(model.parameters()).device
    print(f"\n[Info] Running inference on new data: {cfg.new_csv}")

    # keep full df for appending predictions
    df_full = pd.read_csv(cfg.new_csv)
    check_columns(df_full)

    dataset = DMRI_InferenceDataset(cfg.new_csv)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    preds: List[int] = []
    probs: List[float] = []  # probability of class 1 (E)
    coords: List[List[float]] = []

    model.eval()
    for x, coord in tqdm(loader, desc="NewDataInference", leave=False):
        x = x.unsqueeze(1).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        preds.extend(pred)
        if cfg.save_prob:
            p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
            probs.extend(p)
        coords.extend(coord.detach().cpu().numpy().tolist())

    # append predictions
    df_full["pred_label"] = preds
    if cfg.save_prob:
        # 如果只有二分类，这里存储类1(E)的概率
        df_full["prob_E"] = probs

    out_path = Path(cfg.save_dir) / "prediction_new_appended.csv"
    df_full.to_csv(out_path, index=False)

    # distribution
    dist_counts = df_full["pred_label"].value_counts().sort_index()
    dist_percent = df_full["pred_label"].value_counts(normalize=True).sort_index() * 100.0
    print("\n[Prediction Distribution]")
    print(dist_counts.to_string())
    print("\n[Prediction Proportions (%)]")
    print(dist_percent.round(3).to_string())

    dist_counts.to_csv(Path(cfg.save_dir) / "prediction_class_counts.csv")
    dist_percent.to_csv(Path(cfg.save_dir) / "prediction_class_proportions.csv")

    return out_path


# ---------------------- Main ----------------------
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train & infer MAFNet (binary).")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv",  type=str, required=True)
    parser.add_argument("--new_csv",   type=str, required=True)
    parser.add_argument("--save_dir",  type=str, default="./outputs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, dest="num_epochs", default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--save_prob", action="store_true", help="also save class-1 probability in inference")

    args = parser.parse_args()

    cfg = Config(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        new_csv=args.new_csv,
        save_dir=args.save_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        save_prob=bool(args.save_prob),
    )
    return cfg


def main():
    cfg = parse_args()
    save_dir = Path(cfg.save_dir)
    ensure_dir(save_dir)
    cfg.to_json(save_dir / "config.json")

    # log device early
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    print(f"[Info] Saving to: {save_dir.resolve()}")

    model, best = train_and_validate(cfg)
    # reload best and run inference
    model.load_state_dict(torch.load(Path(cfg.save_dir) / "best_model.pth", map_location=device))
    model.to(device)
    out_path = run_inference(cfg, model)
    print(f"\n[Done] Saved inference output to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
