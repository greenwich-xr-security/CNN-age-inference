#!/usr/bin/env python3
"""Evaluate quality-assessor predictions against generated quality targets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise quality-assessor validation predictions.")
    parser.add_argument("--predictions-file", default=None, help="Single quality_predictions_val.csv file.")
    parser.add_argument("--quality-root", default=None, help="Root containing fold_*/quality_predictions_val.csv.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    if args.predictions_file:
        frames.append(pd.read_csv(args.predictions_file))
    if args.quality_root:
        root = Path(args.quality_root)
        for path in sorted(root.glob("fold_*/quality_predictions_val.csv")):
            frame = pd.read_csv(path)
            frame["fold"] = path.parent.name
            frames.append(frame)
    if not frames:
        raise ValueError("Provide --predictions-file or --quality-root.")
    return pd.concat(frames, ignore_index=True)


def drop_ddp_padding_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if "image_path" not in df.columns:
        return df
    duplicates = df["image_path"].duplicated(keep=False)
    if not duplicates.any():
        return df

    if "fold" in df.columns:
        cross_fold = df.loc[duplicates].groupby("image_path")["fold"].nunique()
        cross_fold = cross_fold[cross_fold > 1]
        if not cross_fold.empty:
            examples = cross_fold.head(10).index.tolist()
            raise ValueError(f"Duplicate image_path entries span multiple quality folds: {examples}")

    before = len(df)
    deduped = df.drop_duplicates(subset=["image_path"], keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(deduped)} duplicate prediction rows by image_path before evaluation.")
    return deduped


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return float("nan")
    return float(a.corr(b))


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    if y_true.nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    args = parse_args()
    df = drop_ddp_padding_duplicates(load_predictions(args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "samples": len(df),
        "corr_quality": safe_corr(df["target_quality_score"], df["pred_quality_score"]),
        "corr_abs_error": safe_corr(df["target_abs_error"], df["pred_expected_abs_error"]),
        "mae_abs_error": float(mean_absolute_error(df["target_abs_error"], df["pred_expected_abs_error"])),
        "rmse_abs_error": float(
            np.sqrt(np.mean((df["target_abs_error"] - df["pred_expected_abs_error"]) ** 2))
        ),
        "auc_boundary_error": safe_auc(df["target_boundary_error"], df["pred_boundary_error_prob"]),
    }
    high_error = df["target_abs_error"] >= df["target_abs_error"].quantile(0.75)
    summary["auc_high_error_top_quartile"] = safe_auc(high_error.astype(int), df["pred_expected_abs_error"])
    pd.DataFrame([summary]).to_csv(output_dir / "quality_summary.csv", index=False)

    bins = pd.qcut(df["pred_quality_score"], q=min(10, max(2, len(df) // 10)), duplicates="drop")
    calibration = (
        df.assign(quality_bin=bins)
        .groupby("quality_bin", observed=True)
        .agg(
            samples=("image_path", "count"),
            pred_quality_mean=("pred_quality_score", "mean"),
            target_quality_mean=("target_quality_score", "mean"),
            target_abs_error_mean=("target_abs_error", "mean"),
            boundary_error_rate=("target_boundary_error", "mean"),
        )
        .reset_index()
    )
    calibration.to_csv(output_dir / "quality_calibration.csv", index=False)
    df.to_csv(output_dir / "quality_predictions_all.csv", index=False)
    print(f"Saved quality evaluation to {output_dir}")


if __name__ == "__main__":
    main()
