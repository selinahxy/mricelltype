#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from network17 import MFNet


# ---------------------------- Dataset ----------------------------
class DMRI_E4_Dataset(Dataset):
    """
    Uses first 4 columns as features (ABA_template, z, x, y).
    Uses column index 6 as the *original* label.
    Does not assume labels are contiguous; mapping is handled outside.
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.features = df.iloc[:, 0:4].to_numpy(np.float32)
        self.labels_orig = df.iloc[:, 6].to_numpy(int)  # may be non-contiguous (e.g., {3,5,7})

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            int(self.labels_orig[idx])
        )


# ----------------------------- Utils -----------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_label_mapping(train_labels_orig: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int], List[int]]:
    """Map arbitrary integer labels to contiguous indices 0..K-1."""
    uniq = sorted(int(v) for v in np.unique(train_labels_orig))
    fwd = {lab: i for i, lab in enumerate(uniq)}   # orig -> idx
    inv = {i: lab for lab, i in fwd.items()}       # idx  -> orig
    return fwd, inv, uniq


def remap_labels(labels_orig: np.ndarray, fwd: Dict[int, int]) -> np.ndarray:
    return np.array([fwd[int(v)] for v in labels_orig], dtype=np.int64)


def compute_weights_balanced(y_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    classes = np.arange(num_classes)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_idx)
    return torch.tensor(cw, dtype=torch.float32)


def save_curves(save_dir: Path, hist: List[dict]) -> None:
    df = pd.DataFrame(hist)
    df.to_csv(save_dir / "train_log.csv", index=False)

    plt.figure()
    plt.plot(df["train_loss"], label="train")
    plt.plot(df["test_loss"], label="test")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png"); plt.close()

    plt.figure()
    plt.plot(df["test_acc"], label="acc")
    plt.plot(df["test_f1"], label="f1")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / "test_metric_curve.png"); plt.close()


def save_confusion(save_dir: Path, y_true_idx: np.ndarray, y_pred_idx: np.ndarray,
                   inv: Dict[int, int]) -> None:
    labels_idx = sorted(inv.keys())
    names = [f"Class {inv[i]}" for i in labels_idx]
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels_idx)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Final Confusion Matrix (Best Model)")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png"); plt.close()

    report = classification_report(y_true_idx, y_pred_idx,
                                   labels=labels_idx, target_names=stringsafe(names),
                                   zero_division=0)
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(report)


def stringsafe(names: List[str]) -> List[str]:
    return [str(n) for n in names]


# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    # ---- Hyperparams & setup ----
    seed = 42
    batch_size = 64
    epochs = 200
    lr = 1e-5
    weight_decay = 1e-5
    seq_len = 4
    amp = True
    early_stop_patience = 25  # stop if no F1 improvement for N epochs

    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Paths ----
    train_csv = "/project/AIRC/NWang_lab/shared/rui/Cell_type/ABA_only_data/dMRIcelltype_N54943_10timesres_trainingdata_ABAonly_allslices_addEI_nooverlapping_cleaned_E_training.csv"
    test_csv  = "/project/AIRC/NWang_lab/shared/rui/Cell_type/ABA_only_data/dMRIcelltype_N54943_10timesres_trainingdata_ABAonly_allslices_addEI_nooverlapping_cleaned_E_testing.csv"
    save_dir = Path("/project/AIRC/NWang_lab/shared/rui/Cell_type/ABA_only_data/results_E")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_ds = DMRI_E4_Dataset(train_csv)
    test_ds  = DMRI_E4_Dataset(test_csv)

    # Build contiguous mapping from **train** labels
    fwd, inv, orig_sorted = build_label_mapping(train_ds.labels_orig)
    with open(save_dir / "label_map.json", "w") as f:
        json.dump({"forward": {str(k): int(v) for k,v in fwd.items()},
                   "inverse": {str(k): int(v) for k,v in inv.items()},
                   "orig_labels_sorted": [int(x) for x in orig_sorted]}, f, indent=2)

    # Remap labels to 0..K-1
    y_tr_idx = remap_labels(train_ds.labels_orig, fwd)
    y_te_idx = remap_labels(test_ds.labels_orig,  fwd)  # may fail if test has unseen -> KeyError

    # Guard: at least two classes
    if np.unique(y_tr_idx).size < 2:
        raise RuntimeError("Training set has <2 classes after mapping. Check label column/filters.")

    # Loaders
    x_tr = torch.from_numpy(train_ds.features)  # [N,4]
    x_te = torch.from_numpy(test_ds.features)
    tr_loader = DataLoader(TensorDataset(x_tr, torch.from_numpy(y_tr_idx)),
                           batch_size=batch_size, shuffle=True, drop_last=True)
    te_loader = DataLoader(TensorDataset(x_te, torch.from_numpy(y_te_idx)),
                           batch_size=batch_size, shuffle=False)

    # ---- Model / Loss / Opt ----
    num_classes = int(np.unique(y_tr_idx).size)
    class_weights = compute_weights_balanced(y_tr_idx, num_classes).to(device)
    print("Balanced class weights:", class_weights.tolist())

    model = MFNet(num_classes=num_classes, seq_len=seq_len).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp))

    # ---- Train/eval loop ----
    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[dict] = []

    log_txt = save_dir / "epoch_log.txt"
    if log_txt.exists():
        log_txt.unlink()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        run_loss = 0.0
        pbar = tqdm(tr_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.unsqueeze(1).to(device, non_blocking=True)  # [B,1,4]
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp)):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += float(loss.item())
            pbar.set_postfix(loss=f"{run_loss/max(1,pbar.n):.4f}")
        train_loss = run_loss / max(1, len(tr_loader))

        # Eval
        model.eval()
        y_true, y_pred = [], []
        run_loss = 0.0
        with torch.no_grad():
            for xb, yb in te_loader:
                xb = xb.unsqueeze(1).to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp)):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                run_loss += float(loss.item())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                y_true.extend(yb.cpu().numpy().tolist())
        test_loss = run_loss / max(1, len(te_loader))
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
                        "test_acc": acc, "test_f1": f1w})
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {acc:.4f} | F1: {f1w:.4f}")

        # Log details
        labels_idx = sorted(inv.keys())
        target_names = [f"Class {inv[i]}" for i in labels_idx]
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
        report = classification_report(y_true, y_pred, labels=labels_idx,
                                       target_names=stringsafe(target_names), zero_division=0)
        with open(log_txt, "a") as f:
            f.write(f"===== Epoch {epoch} =====\n")
            f.write(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                    f"Acc: {acc:.4f} | F1: {f1w:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
            f.write("\nClassification Report:\n")
            f.write(report)
            f.write("\n\n")

        # Save curves each epoch
        save_curves(save_dir, history)

        # Checkpoint on best F1 (tie-breaker: higher acc)
        improved = (f1w > best_f1) or (f1w == best_f1 and acc > history[best_epoch-1]["test_acc"] if best_epoch > 0 else True)
        if improved:
            best_f1 = f1w
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("Best model saved (by weighted F1).")

            # Save per-sample CSV for best snapshot
            best_true, best_pred = [], []
            with torch.no_grad():
                for xb, yb in tqdm(te_loader, desc="BestModelTestSet", leave=False):
                    xb = xb.unsqueeze(1).to(device, non_blocking=True)
                    preds = model(xb).argmax(dim=1).cpu().numpy().tolist()
                    best_pred.extend(preds)
                    best_true.extend(yb.cpu().numpy().tolist())
            pd.DataFrame({
                "true_idx": best_true,
                "pred_idx": best_pred,
                "true_orig": [inv[i] for i in best_true],
                "pred_orig": [inv[i] for i in best_pred],
            }).to_csv(save_dir / "prediction_result_best.csv", index=False)

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step()
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no F1 improvement for {early_stop_patience} epochs).")
            break

    # ---------------- Final evaluation with best ----------------
    print("\n===> Final test using the best saved model...")
    best_path = save_dir / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError("best_model.pth not found. No checkpoint was saved during training.")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in tqdm(te_loader, desc="Final Test"):
            xb = xb.unsqueeze(1).to(device, non_blocking=True)
            preds = model(xb).argmax(dim=1).cpu().numpy().tolist()
            all_pred.extend(preds)
            all_true.extend(yb.cpu().numpy().tolist())

    # Save CM + report + final predictions (mapped & original)
    save_confusion(save_dir, np.array(all_true), np.array(all_pred), inv)
    pd.DataFrame({
        "true_idx": all_true,
        "pred_idx": all_pred,
        "true_orig": [inv[i] for i in all_true],
        "pred_orig": [inv[i] for i in all_pred],
    }).to_csv(save_dir / "prediction_result_final.csv", index=False)

    print("Saved: train_log.csv, curves, best_model.pth, prediction_result_best.csv,")
    print("       confusion_matrix.png, classification_report.txt, prediction_result_final.csv")
