"""SSL dataset for paired hand images."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, get_worker_info

from displayUtils import DisplayUtils


def _build_pair_map(user_ids: Sequence[object], aspects: Sequence[object]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = defaultdict(list)
    for idx, (uid, aspect) in enumerate(zip(user_ids, aspects)):
        key = f"{uid}|{aspect}"
        mapping[key].append(idx)
    return dict(mapping)


class HandSSLPairDataset(Dataset):
    """Return multi-crop views for paired hand images."""

    def __init__(
        self,
        records: pd.DataFrame,
        *,
        transform=None,
        pair_same_hand_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.transform = transform
        self.pair_same_hand_prob = float(pair_same_hand_prob)
        self.seed = int(seed)
        self.epoch = 0
        self._pair_keys = [
            f"{uid}|{aspect}"
            for uid, aspect in zip(self.records["user_id"], self.records["aspect"])
        ]
        self._pair_map = _build_pair_map(self.records["user_id"], self.records["aspect"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _get_rng(self) -> random.Random:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        return random.Random(self.seed + self.epoch * 997 + worker_id * 101)

    def _load_image(self, row: pd.Series) -> Image.Image:
        image_path: Path = row["image_path"]
        image = Image.open(image_path).convert("RGB")

        bbox: Optional[list | tuple] = row.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                xmin, ymin, xmax, ymax = [int(v) for v in bbox]
                if xmax > xmin and ymax > ymin:
                    w, h = image.size
                    sq_xmin, sq_ymin, sq_xmax, sq_ymax = DisplayUtils.make_square_bbox(
                        (xmin, ymin, xmax, ymax)
                    )
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
                        sq_xmin += pad_left
                        sq_xmax += pad_left
                        sq_ymin += pad_top
                        sq_ymax += pad_top
                    sq_xmin = max(0, sq_xmin)
                    sq_ymin = max(0, sq_ymin)
                    sq_xmax = max(sq_xmin + 1, min(image.size[0], sq_xmax))
                    sq_ymax = max(sq_ymin + 1, min(image.size[1], sq_ymax))
                    image = image.crop((sq_xmin, sq_ymin, sq_xmax, sq_ymax))
            except Exception:
                pass
        return image

    def __getitem__(self, idx: int):
        rng = self._get_rng()
        row = self.records.iloc[idx]
        img1 = self._load_image(row)
        img2 = None

        if rng.random() < self.pair_same_hand_prob:
            pair_key = self._pair_keys[idx]
            candidates = self._pair_map.get(pair_key, [idx])
            if len(candidates) > 1:
                other = idx
                while other == idx:
                    other = rng.choice(candidates)
                img2 = self._load_image(self.records.iloc[other])
        if img2 is None:
            img2 = img1.copy()

        if self.transform is None:
            return [img1, img2]
        return self.transform(img1, img2)
