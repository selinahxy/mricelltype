# datasets.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["DMRI_CellType_Dataset_v2", "DMRI_CellType_BinaryDataset", "DatasetConfig"]


# ----------------------------- Config -----------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """
    Column configuration for the dMRI CSV format.

    Defaults match your original code:
      - features: columns [0..16]  (17 features total)
      - coords:   columns [14..16] (3 coordinates, overlapping the tail of features)
      - label:    column 19        (1-based classes for multiclass; 0/1 for binary)
    """
    feature_slice: slice = slice(0, 17)
    coord_slice: slice = slice(14, 17)
    label_col: int = 19


def _check_required_columns(df: pd.DataFrame, cfg: DatasetConfig) -> None:
    needed_max = max(cfg.feature_slice.stop - 1, cfg.coord_slice.stop - 1, cfg.label_col)
    if df.shape[1] <= needed_max:
        raise ValueError(
            f"CSV has {df.shape[1]} columns, but indices up to {needed_max} are required. "
            f"Expected: features {cfg.feature_slice}, coords {cfg.coord_slice}, label {cfg.label_col}."
        )


def _to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)


def _to_int64(x: np.ndarray) -> np.ndarray:
    return x.astype(np.int64, copy=False)


# --------------------- Multiclass Dataset (v2) --------------------
class DMRI_CellType_Dataset_v2(Dataset):
    """
    Multiclass dataset (4 classes).
    - Features: columns 0..16 (17 dims)
    - Label:    column 19, assumed to be in {1,2,3,4}; we map to {0,1,2,3} by (label-1).
    - Coordinates are NOT returned (kept compatible with your original class).
    """
    def __init__(self,
                 csv_path: str | Path,
                 cfg: DatasetConfig = DatasetConfig(),
                 enforce_label_range: bool = True) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        _check_required_columns(df, cfg)

        feats = df.iloc[:, cfg.feature_slice].to_numpy()
        labels_raw = df.iloc[:, cfg.label_col].to_numpy()

        # Keep rows with labels in {1,2,3,4} (before shifting), then shift to {0,1,2,3}
        mask = (labels_raw >= 1) & (labels_raw <= 4)
        if enforce_label_range and not np.all(mask):
            # If there are unexpected labels, drop them but warn via print
            dropped = int((~mask).sum())
            print(f"[DMRI_CellType_Dataset_v2] Dropping {dropped} rows with labels outside 1..4.")

        feats = feats[mask]
        labels = labels_raw[mask] - 1

        self.features = _to_float32(feats)
        self.labels = _to_int64(labels)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[idx])  # float32
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ----------------------- Binary Dataset (E/I) ---------------------
class DMRI_CellType_BinaryDataset(Dataset):
    """
    Binary dataset (0/1 labels).
    - Features: columns 0..16 (17 dims)
    - Coords:   columns 14..16 (3 dims), optionally normalized by coord_max
    - Label:    column 19, we keep only rows where label in {0,1}
    Returns: (feature: FloatTensor[17], label: LongTensor[], coord: FloatTensor[3])
    """
    def __init__(self,
                 csv_path: str | Path,
                 cfg: DatasetConfig = DatasetConfig(),
                 coord_max: Optional[np.ndarray] = None) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        _check_required_columns(df, cfg)

        labels = df.iloc[:, cfg.label_col].to_numpy()
        feats = df.iloc[:, cfg.feature_slice].to_numpy()
        coords = df.iloc[:, cfg.coord_slice].to_numpy()

        # Keep only binary classes {0,1}
        mask = (labels == 0) | (labels == 1)
        if not np.all(mask):
            dropped = int((~mask).sum())
            print(f"[DMRI_CellType_BinaryDataset] Dropping {dropped} rows with labels not in {{0,1}}.")

        feats = feats[mask]
        labels = labels[mask]
        coords = coords[mask]

        # Optional normalization for coordinates
        if coord_max is not None:
            coord_max = np.asarray(coord_max, dtype=np.float32).reshape(1, -1)
            if coord_max.shape[1] != coords.shape[1]:
                raise ValueError(
                    f"coord_max length ({coord_max.shape[1]}) must match coord dim ({coords.shape[1]})."
                )
            # Avoid divide-by-zero
            coords = coords / np.clip(coord_max, 1e-8, None)

        self.features = _to_float32(feats)
        self.labels = _to_int64(labels)
        self.coords = _to_float32(coords)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[idx])          # float32
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        c = torch.from_numpy(self.coords[idx])            # float32
        return x, y, c
