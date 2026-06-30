#!/usr/bin/env python3
"""Reassess adult/minor age gate performance with quality-aware filtering."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate age assurance with quality scores.")
    parser.add_argument("--age-predictions", required=True, help="Raw age prediction .npz with image_path.")
    parser.add_argument("--quality-predictions", required=True, help="CSV with pred_quality_score by image_path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--age-threshold", type=float, default=18.0)
    parser.add_argument("--decision-threshold", type=float, default=18.0)
    parser.add_argument(
        "--quality-quantiles",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Lowest quality fractions to reject before computing gate metrics.",
    )
    parser.add_argument("--near-boundary-window", type=float, default=2.0)
    return parser.parse_args()


def load_age_predictions(path: Path) -> pd.DataFrame:
    data = np.load(path, allow_pickle=False)
    required = {"targets", "pred_mean", "pred_log_var", "user_ids", "image_path"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} missing required keys: {sorted(missing)}")
    return pd.DataFrame(
        {
            "image_path": data["image_path"].astype(str),
            "user_id": data["user_ids"].astype(str),
            "age": data["targets"].astype(float),
            "age_pred_mean": data["pred_mean"].astype(float),
            "age_pred_log_var": data["pred_log_var"].astype(float),
        }
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true.astype(int), y_score.astype(float)))


def gate_metrics(df: pd.DataFrame, age_threshold: float, decision_threshold: float) -> dict[str, float]:
    true_adult = df["age"].to_numpy(float) >= age_threshold
    pred_adult = df["age_pred_mean"].to_numpy(float) >= decision_threshold
    minors = ~true_adult
    adults = true_adult
    fp = float(np.sum(pred_adult & minors))
    tn = float(np.sum((~pred_adult) & minors))
    fn = float(np.sum((~pred_adult) & adults))
    tp = float(np.sum(pred_adult & adults))
    return {
        "samples": float(len(df)),
        "minor_count": float(np.sum(minors)),
        "adult_count": float(np.sum(adults)),
        "fpr": fp / max(1.0, fp + tn),
        "fnr": fn / max(1.0, fn + tp),
        "tpr": tp / max(1.0, fn + tp),
        "tnr": tn / max(1.0, fp + tn),
        "auc_adult_gate": safe_auc(true_adult, df["age_pred_mean"].to_numpy(float)),
        "mae": float(np.mean(np.abs(df["age_pred_mean"].to_numpy(float) - df["age"].to_numpy(float)))),
    }


def plot_quality_histogram(merged: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(merged["pred_quality_score"].to_numpy(float), bins=30, color="#2f6f8f", edgecolor="white")
    ax.set_xlabel("Predicted quality score")
    ax.set_ylabel("Frequency")
    ax.set_title("Held-out Test Quality Score Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_score_histogram.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    age_df = load_age_predictions(Path(args.age_predictions))
    quality_df = pd.read_csv(args.quality_predictions)
    if "image_path" not in quality_df.columns or "pred_quality_score" not in quality_df.columns:
        raise ValueError("quality predictions must include image_path and pred_quality_score columns.")
    merged = age_df.merge(
        quality_df[["image_path", "pred_quality_score", "pred_boundary_error_prob", "pred_expected_abs_error"]],
        on="image_path",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping image_path values between age and quality predictions.")

    rows = []
    for reject_fraction in sorted(set(float(q) for q in args.quality_quantiles)):
        reject_fraction = min(max(reject_fraction, 0.0), 0.95)
        cutoff = float(merged["pred_quality_score"].quantile(reject_fraction))
        admitted = merged[merged["pred_quality_score"] >= cutoff].copy()
        metrics = gate_metrics(admitted, args.age_threshold, args.decision_threshold)
        metrics.update(
            {
                "rule": "quality_score_filter",
                "reject_fraction": reject_fraction,
                "quality_cutoff": cutoff,
                "coverage": len(admitted) / max(1, len(merged)),
            }
        )
        rows.append(metrics)

    near = merged[np.abs(merged["age"] - args.age_threshold) <= args.near_boundary_window].copy()
    near_rows = []
    if not near.empty:
        for reject_fraction in sorted(set(float(q) for q in args.quality_quantiles)):
            cutoff = float(near["pred_quality_score"].quantile(min(max(reject_fraction, 0.0), 0.95)))
            admitted = near[near["pred_quality_score"] >= cutoff]
            metrics = gate_metrics(admitted, args.age_threshold, args.decision_threshold)
            metrics.update(
                {
                    "rule": "near_boundary_quality_score_filter",
                    "reject_fraction": reject_fraction,
                    "quality_cutoff": cutoff,
                    "coverage": len(admitted) / max(1, len(near)),
                }
            )
            near_rows.append(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_quality_histogram(merged, output_dir)
    pd.DataFrame(rows).to_csv(output_dir / "quality_age_gate_reassessment.csv", index=False)
    pd.DataFrame(near_rows).to_csv(output_dir / "quality_age_gate_near_boundary.csv", index=False)
    merged.to_csv(output_dir / "quality_age_gate_joined_predictions.csv", index=False)
    print(f"Saved quality-aware age-gate reassessment to {output_dir}")


if __name__ == "__main__":
    main()
