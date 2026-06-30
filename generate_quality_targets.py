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
        "--age-bin-width",
        type=float,
        default=1.0,
        help="Age-bin width in years for normalising absolute error by OOF age-bin MAE.",
    )
    parser.add_argument(
        "--min-age-bin-mae",
        type=float,
        default=0.25,
        help="Lower bound for age-bin MAE when computing normalised error.",
    )
    parser.add_argument(
        "--min-age-bin-uncertainty",
        type=float,
        default=0.25,
        help="Lower bound for age-bin mean predicted std when computing normalised uncertainty.",
    )
    parser.add_argument(
        "--min-age-bin-consistency",
        type=float,
        default=0.05,
        help="Lower bound for age-bin mean TTA std when computing normalised consistency.",
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=1.0,
        help="Scale for normalised error when mapping error to quality score.",
    )
    parser.add_argument(
        "--uncertainty-scale",
        type=float,
        default=1.0,
        help="Scale for normalised predicted std when mapping uncertainty to quality score.",
    )
    parser.add_argument(
        "--consistency-scale",
        type=float,
        default=1.0,
        help="Scale for normalised TTA std when mapping consistency to quality score.",
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


def collapse_duplicate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    duplicates = df["image_path"].duplicated(keep=False)
    if not duplicates.any():
        return df

    dup_df = df.loc[duplicates].copy()
    cross_fold = dup_df.groupby("image_path")["fold_index"].nunique()
    cross_fold = cross_fold[cross_fold > 1]
    if not cross_fold.empty:
        examples = cross_fold.head(10).index.tolist()
        raise ValueError(
            "Duplicate image_path entries span multiple folds, which violates OOF isolation. "
            f"Examples: {examples}"
        )

    for column in ("user_id", "age"):
        inconsistent = dup_df.groupby("image_path")[column].nunique(dropna=False)
        inconsistent = inconsistent[inconsistent > 1]
        if not inconsistent.empty:
            examples = inconsistent.head(10).index.tolist()
            raise ValueError(f"Duplicate image_path entries disagree on {column}. Examples: {examples}")

    before = len(df)
    grouped = (
        df.groupby("image_path", as_index=False)
        .agg(
            fold_index=("fold_index", "first"),
            user_id=("user_id", "first"),
            age=("age", "first"),
            age_pred_mean=("age_pred_mean", "mean"),
            age_pred_log_var=("age_pred_log_var", "mean"),
        )
        .sort_values(["fold_index", "user_id", "image_path"], kind="stable")
        .reset_index(drop=True)
    )
    removed = before - len(grouped)
    print(f"Collapsed {removed} duplicate OOF prediction rows by image_path within folds.")
    return grouped


def add_quality_columns(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    log_var = np.clip(out["age_pred_log_var"].to_numpy(dtype=float), LOG_VAR_MIN, LOG_VAR_MAX)
    pred_std_raw = np.exp(0.5 * log_var)
    ages = out["age"].to_numpy(dtype=float)
    abs_error_raw = np.abs(out["age_pred_mean"].to_numpy(dtype=float) - ages)
    bin_width = max(float(args.age_bin_width), 1e-6)
    age_bin = np.floor(ages / bin_width) * bin_width
    out["age_bin"] = age_bin
    out["raw_abs_error"] = abs_error_raw
    out["raw_pred_std"] = pred_std_raw
    age_bin_mae = out.groupby("age_bin")["raw_abs_error"].transform("mean").to_numpy(dtype=float)
    age_bin_mae = np.maximum(age_bin_mae, float(args.min_age_bin_mae))
    normalised_abs_error = abs_error_raw / age_bin_mae
    sq_error = normalised_abs_error ** 2
    true_adult = out["age"].to_numpy(dtype=float) >= args.age_threshold
    pred_adult = out["age_pred_mean"].to_numpy(dtype=float) >= args.age_threshold
    consistency_raw = out.get("tta_pred_std", pd.Series(np.zeros(len(out)), index=out.index)).fillna(0.0).to_numpy(float)
    out["raw_consistency_score"] = consistency_raw

    age_bin_uncertainty = out.groupby("age_bin")["raw_pred_std"].transform("mean").to_numpy(dtype=float)
    age_bin_uncertainty = np.maximum(age_bin_uncertainty, float(args.min_age_bin_uncertainty))
    normalised_uncertainty = pred_std_raw / age_bin_uncertainty

    age_bin_consistency = out.groupby("age_bin")["raw_consistency_score"].transform("mean").to_numpy(dtype=float)
    age_bin_consistency = np.maximum(age_bin_consistency, float(args.min_age_bin_consistency))
    normalised_consistency = consistency_raw / age_bin_consistency

    out["age_bin_mae"] = age_bin_mae
    out["normalized_abs_error"] = normalised_abs_error
    out["age_bin_uncertainty"] = age_bin_uncertainty
    out["normalized_uncertainty"] = normalised_uncertainty
    out["age_bin_consistency"] = age_bin_consistency
    out["normalized_consistency"] = normalised_consistency
    out["abs_error"] = normalised_abs_error
    out["squared_error"] = sq_error
    out["pred_std"] = normalised_uncertainty
    out["uncertainty_score"] = normalised_uncertainty
    out["boundary_error"] = (true_adult != pred_adult).astype(int)
    out["boundary_margin"] = np.abs(out["age_pred_mean"].to_numpy(dtype=float) - args.age_threshold)
    out["true_boundary_margin"] = np.abs(out["age"].to_numpy(dtype=float) - args.age_threshold)
    out["near_boundary"] = (out["true_boundary_margin"] <= args.boundary_window).astype(int)
    out["consistency_score"] = normalised_consistency
    penalty = (
        (normalised_abs_error / max(args.error_scale, 1e-6))
        + (normalised_uncertainty / max(args.uncertainty_scale, 1e-6))
        + (normalised_consistency / max(args.consistency_scale, 1e-6))
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
    df = collapse_duplicate_predictions(df)

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
