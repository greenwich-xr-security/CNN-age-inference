#!/usr/bin/env python3
"""Run a trained quality assessor on samples from a CSV or age-prediction NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.quality import QUALITY_TARGET_COLUMNS, QualityDataset
from dataset.transforms import build_transforms
from models import resolve_quality_model_builder
from train_quality_distributed import decode_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict quality scores for hand images.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        nargs="+",
        help="One or more quality-assessor checkpoints. Multiple checkpoints are averaged as an ensemble.",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--per-checkpoint-output-dir",
        default=None,
        help="Optional directory where one CSV per checkpoint is saved before ensemble averaging.",
    )
    parser.add_argument("--model", default="b0")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--samples-csv", default=None, help="CSV containing image_path, age, user_id.")
    parser.add_argument("--age-predictions", default=None, help="Raw age prediction NPZ containing image_path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-masks", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> pd.DataFrame:
    if bool(args.samples_csv) == bool(args.age_predictions):
        raise ValueError("Provide exactly one of --samples-csv or --age-predictions.")
    if args.samples_csv:
        df = pd.read_csv(args.samples_csv)
        required = {"image_path", "age", "user_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"samples CSV missing columns: {sorted(missing)}")
        return df[["image_path", "age", "user_id"]].copy()
    data = np.load(args.age_predictions, allow_pickle=False)
    required = {"image_path", "targets", "user_ids"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"age predictions missing keys: {sorted(missing)}")
    return pd.DataFrame(
        {
            "image_path": data["image_path"].astype(str),
            "age": data["targets"].astype(float),
            "user_id": data["user_ids"].astype(str),
        }
    )


def predict_checkpoint(
    checkpoint: str,
    model_builder,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model = model_builder()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for images, _targets, user_ids, image_paths in loader:
            outputs = model(images.to(device, non_blocking=True))
            decoded = decode_outputs(outputs)
            for i, image_path in enumerate(image_paths):
                row = {"image_path": str(image_path), "user_id": str(user_ids[i])}
                for key, values in decoded.items():
                    row[key] = float(values[i])
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["quality_checkpoint"] = str(checkpoint)
    return frame


def ensemble_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No prediction frames to ensemble.")
    combined = pd.concat(frames, ignore_index=True)
    pred_cols = [col for col in combined.columns if col.startswith("pred_")]
    grouped = combined.groupby(["image_path", "user_id"], as_index=False, sort=False)
    mean_df = grouped[pred_cols].mean()
    std_df = grouped[pred_cols].std(ddof=0).fillna(0.0)
    std_df = std_df.rename(columns={col: f"{col}_std" for col in pred_cols})
    counts = grouped.size().rename(columns={"size": "quality_ensemble_size"})
    return mean_df.merge(std_df, on=["image_path", "user_id"]).merge(counts, on=["image_path", "user_id"])


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    for col in QUALITY_TARGET_COLUMNS:
        if col not in samples.columns:
            samples[col] = 0.0
    device = torch.device(args.device)
    model_builder, default_size, _desc, _key = resolve_quality_model_builder(args.model)
    img_size = args.img_size if args.img_size is not None else default_size
    _train_tf, eval_tf = build_transforms(img_size)
    ds = QualityDataset(samples, transform=eval_tf, use_masks=args.use_masks)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    frames = [predict_checkpoint(checkpoint, model_builder, loader, device) for checkpoint in args.checkpoint]

    if args.per_checkpoint_output_dir:
        per_checkpoint_dir = Path(args.per_checkpoint_output_dir)
        per_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for idx, (checkpoint, frame) in enumerate(zip(args.checkpoint, frames)):
            fold_name = Path(checkpoint).parent.name or f"checkpoint_{idx}"
            frame.to_csv(per_checkpoint_dir / f"quality_predictions_test_{fold_name}.csv", index=False)

    out = ensemble_predictions(frames)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(
        f"Saved ensembled quality predictions for {len(out)} samples "
        f"from {len(frames)} checkpoint(s) to {output_file}"
    )


if __name__ == "__main__":
    main()
