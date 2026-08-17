#!/usr/bin/env python3
"""Create a contact sheet showing archive masking and square-bbox crop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.age import AgeDataset
from dataset.hand_metadata import load_combined_metadata
from dataset.utils import filter_metadata
from displayUtils import DisplayUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual QA for archive preprocessing.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets",
    )
    parser.add_argument("--split", type=str, default="splits/archive_users_all.json")
    parser.add_argument(
        "--out",
        type=str,
        default="runs/archive_preprocessing_visual_qa/contact_sheet.png",
    )
    parser.add_argument("--num-with-mask", type=int, default=6)
    parser.add_argument("--num-without-mask", type=int, default=2)
    parser.add_argument("--thumb-size", type=int, default=220)
    return parser.parse_args()


def fit_thumb(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def apply_mask(img: Image.Image, mask_path: Path | None) -> Image.Image:
    if mask_path is None or not mask_path.is_file():
        return img.copy()
    mask = Image.open(mask_path).convert("L").point(lambda p: 255 if p >= 128 else 0)
    return Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)


def square_crop(img: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    sx1, sy1, sx2, sy2 = DisplayUtils.make_square_bbox(bbox)
    pad_left = max(0, -sx1)
    pad_top = max(0, -sy1)
    pad_right = max(0, sx2 - img.width)
    pad_bottom = max(0, sy2 - img.height)
    if pad_left or pad_top or pad_right or pad_bottom:
        img = Image.new("RGB", img.size, (0, 0, 0)) if img.mode != "RGB" else img
        img = Image.fromarray(np.asarray(img))
        from PIL import ImageOps

        img = ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0),
        )
        sx1 += pad_left
        sx2 += pad_left
        sy1 += pad_top
        sy2 += pad_top
    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(img.width, sx2)
    sy2 = min(img.height, sy2)
    return img.crop((sx1, sy1, sx2, sy2))


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    with Path(args.split).open("r", encoding="utf-8") as fp:
        test_ids = {str(uid) for uid in json.load(fp)["test_user_ids"]}

    meta = filter_metadata(
        load_combined_metadata(
            root=root,
            include_handrgbd=False,
            include_hagrid=False,
            include_synthetic_dorsal=False,
            include_synthetic_dorsal2=False,
            include_prolific=False,
            include_primary=False,
            include_archive=True,
        ),
        max_samples_per_user=None,
        max_samples_per_age_bin=None,
    )
    meta = meta[meta["user_id"].astype(str).isin(test_ids)].reset_index(drop=True)

    with_mask = meta[meta["mask_path"].notna()].head(args.num_with_mask)
    without_mask = meta[meta["mask_path"].isna()].head(args.num_without_mask)
    selected = list(with_mask.index) + list(without_mask.index)
    if not selected:
        raise RuntimeError("No archive rows selected for visualisation.")

    headers = ["original + bbox", "mask", "masked rgb", "final crop", "final 384"]
    cell = args.thumb_size
    label_h = 28
    cols = len(headers)
    rows = len(selected) + 1
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for ci, header in enumerate(headers):
        draw.text((ci * cell + 8, 8), header, fill="black")

    for ri, idx in enumerate(selected, start=1):
        row = meta.loc[idx]
        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"]) if row.get("mask_path") is not None and str(row.get("mask_path")) else None
        bbox = tuple(int(v) for v in row["bbox"])
        img = Image.open(img_path).convert("RGB")

        original = img.copy()
        original_draw = ImageDraw.Draw(original)
        original_draw.rectangle(bbox, outline="yellow", width=4)
        square = DisplayUtils.make_square_bbox(bbox)
        original_draw.rectangle(square, outline="magenta", width=3)

        if mask_path is not None and mask_path.is_file():
            mask_img = Image.open(mask_path).convert("L")
            mask_vis = Image.merge("RGB", (mask_img, mask_img, mask_img))
        else:
            mask_vis = Image.new("RGB", img.size, (245, 245, 245))
            ImageDraw.Draw(mask_vis).text((10, 10), "no mask", fill="black")

        masked = apply_mask(img, mask_path)
        crop = square_crop(masked, bbox)
        final = crop.resize((384, 384), Image.Resampling.BILINEAR)

        panels = [original, mask_vis, masked, crop, final]
        for ci, panel in enumerate(panels):
            thumb = fit_thumb(panel, cell)
            x = ci * cell
            y = ri * (cell + label_h)
            sheet.paste(thumb, (x, y + label_h))
            caption = f"{row['user_id']} age {int(row['age'])}"
            if ci == 0:
                caption += " masked" if mask_path is not None else " no-mask"
            draw.text((x + 8, y + 6), caption[:28], fill="black")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
