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
    """Age regression dataset with optional bbox cropping and normal map loading.

    When ``normals_transform`` is provided and the DataFrame contains a
    ``normals_path`` column, ``__getitem__`` returns a 4-tuple
    ``(image, age, user_id, normals_tensor_or_None)``; otherwise a 3-tuple.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        transform=None,
        use_masks: bool = False,
        normals_transform=None,
        return_image_path: bool = False,
    ):
        self.records = records.copy()
        if "aspect" in self.records.columns:
            dorsal_mask = self.records["aspect"].astype(str).str.contains(
                "dorsal", case=False, na=False
            )
            dropped = int((~dorsal_mask).sum())
            if dropped:
                print(f"[AgeDataset] Dropped {dropped} non-dorsal samples before loading.")
            self.records = self.records[dorsal_mask]
        self.records = self.records.reset_index(drop=True)
        self.transform = transform
        self.use_masks = bool(use_masks)
        self.normals_transform = normals_transform
        self.return_image_path = bool(return_image_path)
        self._has_normals_col = "normals_path" in self.records.columns

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _resolve_mask_path(image_path: Path) -> Optional[Path]:
        """Infer a mask path from the image path based on known dataset layouts."""
        parent = image_path.parent
        name = image_path.name

        # If explicit mask_path column exists in records, caller should pass it; this is heuristic fallback.
        candidates = []
        # handRGBD: rgb / rgb_jpg -> rgb_mask
        if parent.name in ("rgb", "rgb_jpg"):
            candidates.append(parent.with_name(f"{parent.name}_mask") / name)
        # 11kHands: Hands -> Masks
        if parent.name.lower() == "hands":
            candidates.append(parent.with_name("Masks") / name)
        # archive: Photos -> Masks
        if parent.name.lower() == "photos":
            candidates.append(parent.with_name("Masks") / name)

        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        image_path: Path = row["image_path"]
        age = float(row["age"])
        image = Image.open(image_path).convert("RGB")

        # Optional: apply binary mask to zero-out background
        mask_img = None
        if self.use_masks:
            # Skip masking for handRGBD (already masked in source)
            if any("handrgbd" in str(p).lower() for p in image_path.parents):
                mask_img = None
            else:
                explicit = row.get("mask_path") if isinstance(row, pd.Series) else None
                if explicit is not None and isinstance(explicit, (str, Path)) and str(explicit):
                    cand = Path(explicit)
                    if cand.is_file():
                        mask_img = cand
                if mask_img is None:
                    mask_path = self._resolve_mask_path(image_path)
                    if mask_path is not None:
                        mask_img = mask_path
            if mask_img is not None:
                try:
                    mask = Image.open(mask_img).convert("L")
                    # binarize
                    mask = mask.point(lambda p: 255 if p >= 128 else 0)
                    # apply
                    mask_rgb = Image.merge("RGB", (mask, mask, mask))
                    image = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask_rgb)
                except Exception:
                    # if masking fails, fall back to original image
                    pass

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

        if self.normals_transform is None or not self._has_normals_col:
            base = (image, torch.tensor(age, dtype=torch.float32), row["user_id"])
            if self.return_image_path:
                return (*base, str(image_path))
            return base

        # Load paired normal map if available.
        normals_tensor = None
        normals_raw = row.get("normals_path")
        if normals_raw is not None and not (isinstance(normals_raw, float) and pd.isna(normals_raw)):
            normals_path = Path(normals_raw) if not isinstance(normals_raw, Path) else normals_raw
            try:
                normals_img = Image.open(normals_path).convert("RGB")
                normals_tensor = self.normals_transform(normals_img)
            except Exception:
                normals_tensor = None

        base = (image, torch.tensor(age, dtype=torch.float32), row["user_id"], normals_tensor)
        if self.return_image_path:
            return (*base, str(image_path))
        return base
