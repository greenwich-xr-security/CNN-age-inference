#!/usr/bin/env python3
"""Generate out-of-fold quality labels from age-regression prediction artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quality-assessment targets from k-fold age-CNN validation predictions."
    )
    parser.add_argument("--kfold-root", required=True, help="Directory containing fold_* age run folders.")
    parser.add_argument("--output-file", required=True, help="CSV path for generated quality targets.")
    parser.add_argument(
        "--prediction-name",
        default="val_predictions_raw_ddp.npz",
        help="Per-fold validation prediction file name (default: val_predictions_raw_ddp.npz).",
    )
    parser.add_argument(
        "--consistency-file",
        default=None,
        help="Optional CSV from generate_age_tta_predictions.py to merge by image_path.",
    )
    parser.add_argument("--age-threshold", type=float, default=18.0, help="Adult/minor boundary age.")
    parser.add_argument(
        "--boundary-window",
        type=float,
        default=2.0,
        help="Samples within this many years of the threshold are marked near-boundary.",
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=5.0,
        help="Error scale in years used when mapping error to quality score.",
    )
    parser.add_argument(
        "--uncertainty-scale",
        type=float,
        default=5.0,
        help="Predicted std scale in years used when mapping uncertainty to quality score.",
    )
    parser.add_argument(
        "--consistency-scale",
        type=float,
        default=2.0,
        help="TTA std scale in years used when mapping consistency to quality score.",
    )
    return parser.parse_args()


def discover_folds(root: Path) -> list[Path]:
    folds = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    folds.sort(key=lambda p: int(p.name.split("_")[-1]))
    if not folds:
        raise FileNotFoundError(f"No fold_* directories found under {root}")
    return folds


def _required(data: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"{path} is missing required key '{key}'. Re-run age export with sample identifiers.")
    return data[key]


def load_fold_predictions(path: Path, fold_index: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    data = np.load(path, allow_pickle=False)
    targets = _required(data, "targets", path).astype(float)
    pred_mean = _required(data, "pred_mean", path).astype(float)
    pred_log_var = _required(data, "pred_log_var", path).astype(float)
    user_ids = _required(data, "user_ids", path).astype(str)
    image_paths = _required(data, "image_path", path).astype(str)
    if not (len(targets) == len(pred_mean) == len(pred_log_var) == len(user_ids) == len(image_paths)):
        raise ValueError(f"Array lengths do not match in {path}")
    return pd.DataFrame(
        {
            "fold_index": fold_index,
            "image_path": image_paths,
            "user_id": user_ids,
            "age": targets,
            "age_pred_mean": pred_mean,
            "age_pred_log_var": pred_log_var,
        }
    )


def add_quality_columns(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    log_var = np.clip(out["age_pred_log_var"].to_numpy(dtype=float), LOG_VAR_MIN, LOG_VAR_MAX)
    pred_std = np.exp(0.5 * log_var)
    abs_error = np.abs(out["age_pred_mean"].to_numpy(dtype=float) - out["age"].to_numpy(dtype=float))
    sq_error = abs_error ** 2
    true_adult = out["age"].to_numpy(dtype=float) >= args.age_threshold
    pred_adult = out["age_pred_mean"].to_numpy(dtype=float) >= args.age_threshold
    consistency = out.get("tta_pred_std", pd.Series(np.zeros(len(out)), index=out.index)).fillna(0.0).to_numpy(float)

    out["abs_error"] = abs_error
    out["squared_error"] = sq_error
    out["pred_std"] = pred_std
    out["uncertainty_score"] = pred_std
    out["boundary_error"] = (true_adult != pred_adult).astype(int)
    out["boundary_margin"] = np.abs(out["age_pred_mean"].to_numpy(dtype=float) - args.age_threshold)
    out["true_boundary_margin"] = np.abs(out["age"].to_numpy(dtype=float) - args.age_threshold)
    out["near_boundary"] = (out["true_boundary_margin"] <= args.boundary_window).astype(int)
    out["consistency_score"] = consistency
    penalty = (
        (abs_error / max(args.error_scale, 1e-6))
        + (pred_std / max(args.uncertainty_scale, 1e-6))
        + (consistency / max(args.consistency_scale, 1e-6))
    )
    out["quality_score"] = 1.0 / (1.0 + penalty)
    out["usefulness_score"] = out["quality_score"]
    return out


def main() -> None:
    args = parse_args()
    kfold_root = Path(args.kfold_root)
    rows: list[pd.DataFrame] = []
    for fold_dir in discover_folds(kfold_root):
        fold_index = int(fold_dir.name.split("_")[-1])
        rows.append(load_fold_predictions(fold_dir / args.prediction_name, fold_index))
    df = pd.concat(rows, ignore_index=True)
    df["image_path"] = df["image_path"].astype(str)
    if df["image_path"].eq("").any():
        raise ValueError("Some predictions have empty image_path identifiers.")
    duplicates = df["image_path"].duplicated(keep=False)
    if duplicates.any():
        dupes = df.loc[duplicates, "image_path"].head(10).tolist()
        raise ValueError(f"Duplicate image_path entries found in OOF predictions, examples: {dupes}")

    if args.consistency_file:
        consistency_df = pd.read_csv(args.consistency_file)
        required = {"image_path", "tta_pred_std", "tta_max_drift", "tta_boundary_flip_rate"}
        missing = required - set(consistency_df.columns)
        if missing:
            raise ValueError(f"Consistency file missing columns: {sorted(missing)}")
        df = df.merge(consistency_df[list(required)], on="image_path", how="left")

    out = add_quality_columns(df, args)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"Saved {len(out)} quality targets to {output_file}")


if __name__ == "__main__":
    main()
