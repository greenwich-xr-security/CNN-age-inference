#!/usr/bin/env python3
"""Export age-model test-time augmentation consistency features for quality labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.age import AgeDataset
from models import resolve_model_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic TTA age predictions for consistency scoring.")
    parser.add_argument("--samples-csv", required=True, help="CSV containing image_path, age, and user_id columns.")
    parser.add_argument("--checkpoint", required=True, help="Age model checkpoint path.")
    parser.add_argument("--output-file", required=True, help="CSV path for TTA consistency features.")
    parser.add_argument("--model", default="v2_m", help="Age model backbone name.")
    parser.add_argument("--img-size", type=int, default=None, help="Override model image size.")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension used by the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-masks", action="store_true")
    parser.add_argument("--age-threshold", type=float, default=18.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_tta_transforms(img_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def compose(extra):
        ops = [transforms.Resize((img_size, img_size))]
        if extra is not None:
            ops.append(extra)
        ops.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(ops)

    return [
        ("identity", compose(None)),
        ("hflip", compose(transforms.Lambda(ImageOps.mirror))),
        ("vflip", compose(transforms.Lambda(ImageOps.flip))),
        ("rot_pos10", compose(transforms.Lambda(lambda img: img.rotate(10, resample=2)))),
        ("rot_neg10", compose(transforms.Lambda(lambda img: img.rotate(-10, resample=2)))),
    ]


def load_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_path", "age", "user_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    records = df[["image_path", "age", "user_id"]].copy()
    records["image_path"] = records["image_path"].map(Path)
    return records


def run_inference(model, records: pd.DataFrame, transform, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    ds = AgeDataset(records, transform=transform, use_masks=args.use_masks, return_image_path=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    preds: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, _ages, _user_ids, _image_paths = batch
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                mean = outputs[0]
            else:
                mean = outputs
            preds.extend(mean.detach().cpu().numpy().astype(float).tolist())
    return np.asarray(preds, dtype=float)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    records = load_samples(Path(args.samples_csv))
    model_builder, default_size, _desc, _key = resolve_model_builder(args.model, embed_dim=args.embed_dim)
    img_size = args.img_size if args.img_size is not None else default_size
    model = model_builder()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    pred_columns = {}
    for name, transform in build_tta_transforms(img_size):
        pred_columns[f"tta_{name}"] = run_inference(model, records, transform, args, device)

    pred_matrix = np.column_stack([pred_columns[key] for key in pred_columns])
    base = pred_matrix[:, 0]
    adult_rate = np.mean(pred_matrix >= float(args.age_threshold), axis=1)
    out = pd.DataFrame(
        {
            "image_path": records["image_path"].astype(str).to_numpy(),
            "tta_pred_mean": np.mean(pred_matrix, axis=1),
            "tta_pred_std": np.std(pred_matrix, axis=1),
            "tta_max_drift": np.max(np.abs(pred_matrix - base[:, None]), axis=1),
            "tta_boundary_flip_rate": 2.0 * np.minimum(adult_rate, 1.0 - adult_rate),
        }
    )
    for key, values in pred_columns.items():
        out[key] = values
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"Saved TTA consistency features for {len(out)} samples to {output_file}")


if __name__ == "__main__":
    main()
