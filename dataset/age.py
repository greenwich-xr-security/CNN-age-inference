"""Dataset utilities for age regression."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from displayUtils import DisplayUtils


class AgeDataset(Dataset):
    """Simple age regression dataset that supports optional bbox cropping."""

    def __init__(self, records: pd.DataFrame, transform=None):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        image_path: Path = row["image_path"]
        age = float(row["age"])
        image = Image.open(image_path).convert("RGB")

        # Optional: crop to square bbox with padding if available
        bbox: Optional[list | tuple] = row.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox]
                if xmax > xmin and ymax > ymin:
                    w, h = image.size
                    sq_xmin, sq_ymin, sq_xmax, sq_ymax = DisplayUtils.make_square_bbox(
                        (xmin, ymin, xmax, ymax)
                    )

                    # Compute required padding to keep crop inside image bounds
                    pad_left = max(0, -sq_xmin)
                    pad_top = max(0, -sq_ymin)
                    pad_right = max(0, sq_xmax - w)
                    pad_bottom = max(0, sq_ymax - h)

                    if pad_left or pad_top or pad_right or pad_bottom:
                        image = ImageOps.expand(
                            image,
                            border=(pad_left, pad_top, pad_right, pad_bottom),
                            fill=(0, 0, 0),
                        )
                        # Shift square bbox into padded image coords
                        sq_xmin += pad_left
                        sq_xmax += pad_left
                        sq_ymin += pad_top
                        sq_ymax += pad_top

                    # Final safety clamp then crop
                    sq_xmin = max(0, sq_xmin)
                    sq_ymin = max(0, sq_ymin)
                    sq_xmax = max(sq_xmin + 1, min(image.size[0], sq_xmax))
                    sq_ymax = max(sq_ymin + 1, min(image.size[1], sq_ymax))
                    image = image.crop((sq_xmin, sq_ymin, sq_xmax, sq_ymax))
            except Exception:
                # If anything goes wrong with bbox handling, fall back to full image
                pass
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)
