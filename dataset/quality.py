"""Dataset utilities for hand-image quality assessment."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.age import AgeDataset


QUALITY_TARGET_COLUMNS = [
    "abs_error",
]


class QualityDataset(Dataset):
    """Image dataset backed by quality labels generated from out-of-fold age predictions."""

    def __init__(
        self,
        quality_targets: pd.DataFrame,
        transform=None,
        use_masks: bool = False,
        target_columns: list[str] | None = None,
    ):
        self.records = quality_targets.copy().reset_index(drop=True)
        self.target_columns = target_columns or QUALITY_TARGET_COLUMNS
        missing = {"image_path", "age", "user_id", *self.target_columns} - set(self.records.columns)
        if missing:
            raise ValueError(f"Quality targets missing columns: {sorted(missing)}")
        self.records["image_path"] = self.records["image_path"].map(Path)
        self.base = AgeDataset(
            self.records,
            transform=transform,
            use_masks=use_masks,
            return_image_path=True,
        )
        self.records = self.base.records

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        image, _age, user_id, image_path = self.base[idx]
        row = self.records.iloc[idx]
        targets = torch.tensor(
            [float(row[col]) for col in self.target_columns],
            dtype=torch.float32,
        )
        return image, targets, str(user_id), str(image_path)
