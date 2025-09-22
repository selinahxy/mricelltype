#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E-only training/evaluation for ABA-only data:
- Input CSV: keep rows where col[6] == 1 (0-based).
- Features: first 4 columns [ABA_template, z, x, y].
- Coordinates (z,x,y) are max-normalized by SCALE_MAX = (76, 11000, 11000).
- Label column: col[5] (original subtype id). Train set defines the mapping to {0..K-1}.
- Model: Simple1DCNN expecting input [B, 1, 4].
- Class weighting: balanced by default using sklearn's compute_class_weight.
- Saves: best checkpoint by weighted F1, logs, curves, confusion matrix, per-sample CSVs.
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("warning")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# If you want to switch to your advanced model later, just import and instantiate it:
# from network17 import MAFNet

# -------------------------- Model (baseline) --------------------------
class Simple1DCNN(nn.Module):
    """
    A small 1D-CNN baseline for seq_len=4 input.
    """
    def __init__(self, num_classes: int, seq_len: int = 4) -> None:
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
            nn.Flatten(),          # [B,128,1] -> [B,128]
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# -------------------------- Dataset --------------------------
@dataclass(frozen=True)
class ENormConfig:
    """Max values for (z, x, y) normalization."""
    z: float = 76.0
    x: float = 11000.0
    y: float = 11000.0

    def as_array(self) -> np.ndarray:
        return np.array([self.z, self.x, self.y], dtype=np.float32)


class DMRI_CellType_EOnlyDataset(Dataset):
    """
    E-only dataset (ABA-only variant).
    - Keep rows where df.iloc[:,6] == 1.
    - Features: df.iloc[:, 0:4] -> [ABA_template, z, x, y] with (z,x,y) max-normalized.
    - Label: df.iloc[:, 5] (original subtype id). Train set builds the mapping {orig -> idx}.
    """
    def __init__(self, csv_path: str | Path, label_map: Dict[int, int] | None = None,
                 scale_max: Tuple[float, float, float] = (76, 11000, 11000)) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        if df.shape[1] < 7:
            raise ValueError(f"CSV has {df.shape[1]} columns; expected at least 7.")

        # Filter E-only (col[6] == 1)
        df = df[df.iloc[:, 6] == 1].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("Empty dataset after filtering col[6] == 1.")
        self.df = df  # keep a handle for result exports

        feats = df.iloc[:, 0:4].values.astype(np.float32)  # [N,4]
        coords = feats[:, 1:4].astype(np.float32, copy=True)  # (z,x,y)
        scale = np.asarray(scale_max, dtype=np.float32)       # (76,11000,11000)
        if scale.shape != (3,):
            raise ValueError("scale_max must be a length-3 tuple/array.")
        feats[:, 1:4] = coords / (scale + 1e-8)
        self.features = feats

        orig_np = df.iloc[:, 5].values
        orig = np.array([int(v) for v in orig_np], dtype=int)

        if label_map is None:
            unique = sorted(int(u) for u in np.unique(orig).tolist())
            self.label_to_int: Dict[int, int] = {lab: i for i, lab in enumerate(unique)}
            print("Built subtype mapping:", self.label_to_int)
        else:
            self.label_to_int = {int(k): int(v) for k, v in label_map.items()}
            unknown = set(int(u) for u in np.unique(orig).tolist()) - set(self.label_to_int.keys())
            if unknown:
                raise ValueError(f"Test set contains unseen labels: {sorted(unknown)}")

        self.int_to_label: Dict[int, int] = {i: lab for lab, i in self.label_to_int.items()}
        self.labels = np.array([self.label_to_int[int(lab)] for lab in orig], dtype=int)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)  # [4]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, idx


# -------------------------- Main script --------------------------
if __name__ == "__main__":
    # ----- Hyperparameters & setup -----
    random_seed = 42
    batch_size   = 64
    num_epochs   = 100
    lr           = 1e-4
    weight_decay = 1e-5
    seq_len      = 4

    SCALE_MAX = ENormConfig(76, 11000, 11000).as_array()
    USE_EQUAL_WEIGHTS = False  # False => use balanced class weights

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Paths -----
    base_dir  = "/project/AIRC/NWang_lab/shared/rui/Cell_type/ABA_only_data"
    train_csv = os.path.join(base_dir, "dMRIcelltype_N54943_10timesres_trainingdata_ABAonly_allslices_addEI_nooverlapping_cleaned_E_training.csv")
    test_csv  = os.path.join(base_dir, "dMRIcelltype_N54943_10timesres_trainingdata_ABAonly_allslices_addEI_nooverlapping_cleaned_E_testing.csv")

    save_dir   = os.path.join(base_dir, "results_E7_maxnorm")
    os.makedirs(save_dir, exist_ok=True)
    txt_log    = os.path.join(save_dir, "epoch_log.txt")
    model_path = os.path.join(save_dir, "best_model.pth")
    map_json   = os.path.join(save_dir, "label_map.json")
    norm_json  = os.path.join(save_dir, "coord_maxnorm.json")

    # ----- Datasets -----
    train_ds = DMRI_CellType_EOnlyDataset(train_csv, scale_max=SCALE_MAX)
    test_ds  = DMRI_CellType_EOnlyDataset(test_csv, label_map=train_ds.label_to_int,
                                          scale_max=SCALE_MAX)

    with open(map_json, "w") as f:
        json.dump({
            "label_to_int": {str(k): int(v) for k, v in train_ds.label_to_int.items()},
            "int_to_label": {str(k): int(v) for k, v in train_ds.int_to_label.items()}
        }, f, indent=2)
    with open(norm_json, "w") as f:
        z, x, y = SCALE_MAX.tolist()
        json.dump({"type": "maxnorm", "scale_max": [int(z), int(x), int(y)]}, f, indent=2)

    # ----- DataLoaders -----
    num_workers = 4
    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    # ----- Classes & names -----
    num_classes  = len(train_ds.label_to_int)
    labels_all   = list(range(num_classes))
    target_names = [f"Class {train_ds.int_to_label[i]}" for i in range(num_classes)]
    print(f"Number of classes: {num_classes}, targets: {target_names}")

    # ----- Model, Loss, Optimizer, Scheduler -----
    model = Simple1DCNN(num_classes, seq_len).to(device)
    # To switch to MAFNet later:
    # model = MAFNet(num_classes=num_classes, seq_len=seq_len).to(device)

    if USE_EQUAL_WEIGHTS:
        criterion = nn.CrossEntropyLoss()
        print("Using equal class weights (no weighting).")
    else:
        classes = np.arange(num_classes)
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=train_ds.labels)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
        print("Using balanced class weights:", class_weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # ----- Clear previous logs -----
    if os.path.exists(txt_log):
        os.remove(txt_log)

    best_f1   = -1.0
    train_log: List[Dict[str, float]] = []

    # =========================
    # Training / Evaluation
    # =========================
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # ---- Train ----
        model.train()
        loss_sum = 0.0
        for x, y, _ in tqdm(train_loader, desc="Train", leave=False):
            x = x.unsqueeze(1).to(device)  # [B,1,4]
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        train_loss = loss_sum / max(1, len(train_loader))

        # ---- Eval ----
        model.eval()
        loss_sum, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for x, y, _ in tqdm(test_loader, desc="Eval", leave=False):
                x = x.unsqueeze(1).to(device)
                y = y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                loss_sum += float(loss.item())
                y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
                y_true.extend(y.cpu().numpy().tolist())

        test_loss = loss_sum / max(1, len(test_loader))
        acc       = accuracy_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred, labels=labels_all,
                             average="weighted", zero_division=0)

        train_log.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "test_acc": float(acc),
            "test_f1": float(f1),
        })
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        # ---- Append epoch report to txt ----
        cm     = confusion_matrix(y_true, y_pred, labels=labels_all)
        report = classification_report(y_true, y_pred, labels=labels_all,
                                       target_names=target_names, zero_division=0)
        with open(txt_log, "a") as f:
            f.write(f"=== Epoch {epoch} ===\n")
            f.write(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                    f"Acc: {acc:.4f} | F1: {f1:.4f}\n")
            f.write("CM:\n" + np.array2string(cm) + "\n")
            f.write("Report:\n" + report + "\n\n")

        # ---- Curves & CSV ----
        df_log = pd.DataFrame(train_log)
        plt.figure(); plt.plot(df_log["train_loss"], label="train"); plt.plot(df_log["test_loss"], label="test")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_curve.png")); plt.close()

        plt.figure(); plt.plot(df_log["test_acc"], label="acc"); plt.plot(df_log["test_f1"], label="f1")
        plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "test_metric_curve.png")); plt.close()

        df_log.to_csv(os.path.join(save_dir, "train_log.csv"), index=False)

        # ---- Save best by weighted F1 ----
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
            print("Best model saved (by weighted F1).")

            # per-sample predictions (best snapshot)
            best_records = []
            with torch.no_grad():
                for x, y, idxs in tqdm(test_loader, desc="BestPred", leave=False):
                    x = x.unsqueeze(1).to(device)
                    preds = model(x).argmax(1).cpu().numpy()
                    y_np  = y.numpy()
                    for i, idx in enumerate(idxs.numpy()):
                        row = {
                            "ABA_template": test_ds.df.iat[idx, 0],
                            "z":            float(test_ds.df.iat[idx, 1]),
                            "x":            float(test_ds.df.iat[idx, 2]),
                            "y":            float(test_ds.df.iat[idx, 3]),
                            "true_mapped":  int(y_np[i]),
                            "pred_mapped":  int(preds[i]),
                            "true_orig":    test_ds.int_to_label[int(y_np[i])],
                            "pred_orig":    test_ds.int_to_label[int(preds[i])],
                        }
                        best_records.append(row)
            pd.DataFrame(best_records).to_csv(
                os.path.join(save_dir, "prediction_result_best.csv"), index=False
            )

        scheduler.step()

    # =========================
    # Final evaluation
    # =========================
    print("\n=== Final evaluation with best model ===")
    if not os.path.exists(model_path):
        raise FileNotFoundError("best_model.pth not found. No checkpoint was saved during training.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    final_true, final_pred = [], []
    with torch.no_grad():
        for x, y, _ in tqdm(test_loader, desc="FinalTest"):
            x = x.unsqueeze(1).to(device)
            logits = model(x)
            final_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            final_true.extend(y.numpy().tolist())

    cm_final = confusion_matrix(final_true, final_pred, labels=labels_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_final, annot=True, fmt="d",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title("Final CM")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png")); plt.close()

    final_report = classification_report(final_true, final_pred, labels=labels_all,
                                         target_names=target_names, zero_division=0)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(final_report)
    print(final_report)

    # Final per-sample CSV
    final_records = []
    with torch.no_grad():
        for x, y, idxs in tqdm(test_loader, desc="FinalPred"):
            x = x.unsqueeze(1).to(device)
            preds = model(x).argmax(1).cpu().numpy()
            y_np  = y.numpy()
            for i, idx in enumerate(idxs.numpy()):
                row = {
                    "ABA_template": test_ds.df.iat[idx, 0],
                    "z":            float(test_ds.df.iat[idx, 1]),
                    "x":            float(test_ds.df.iat[idx, 2]),
                    "y":            float(test_ds.df.iat[idx, 3]),
                    "true_mapped":  int(y_np[i]),
                    "pred_mapped":  int(preds[i]),
                    "true_orig":    test_ds.int_to_label[int(y_np[i])],
                    "pred_orig":    test_ds.int_to_label[int(preds[i])],
                }
                final_records.append(row)
    pd.DataFrame(final_records).to_csv(
        os.path.join(save_dir, "prediction_result_final.csv"), index=False
    )

    print("All done. Results saved in", save_dir)
