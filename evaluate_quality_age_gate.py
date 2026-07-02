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
        "--mae-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap resamples for MAE 95%% confidence intervals (0 disables).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
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


def mae_interval(
    df: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    abs_error = np.abs(df["age_pred_mean"].to_numpy(float) - df["age"].to_numpy(float))
    if abs_error.size == 0:
        return {"mae_se": float("nan"), "mae_ci_low": float("nan"), "mae_ci_high": float("nan")}

    if abs_error.size == 1 or bootstrap_samples <= 0:
        se = float(np.std(abs_error, ddof=1) / np.sqrt(abs_error.size)) if abs_error.size > 1 else 0.0
        mae = float(np.mean(abs_error))
        return {"mae_se": se, "mae_ci_low": mae - 1.96 * se, "mae_ci_high": mae + 1.96 * se}

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, abs_error.size, size=(int(bootstrap_samples), abs_error.size))
    boot_mae = abs_error[sample_idx].mean(axis=1)
    return {
        "mae_se": float(np.std(boot_mae, ddof=1)),
        "mae_ci_low": float(np.quantile(boot_mae, 0.025)),
        "mae_ci_high": float(np.quantile(boot_mae, 0.975)),
    }


def bootstrap_mean_interval(values: np.ndarray, *, bootstrap_samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean_value = float(np.mean(values))
    if values.size == 1 or bootstrap_samples <= 0:
        return mean_value, mean_value

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(int(bootstrap_samples), values.size))
    boot_means = values[sample_idx].mean(axis=1)
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def plot_quality_histogram(
    merged: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    quality = merged["pred_quality_score"].to_numpy(float)
    abs_error = np.abs(merged["age_pred_mean"].to_numpy(float) - merged["age"].to_numpy(float))
    bins = np.linspace(float(np.min(quality)), float(np.max(quality)), 16)
    if np.unique(bins).size < 2:
        bins = np.linspace(0.0, 1.0, 16)

    bin_ids = np.digitize(quality, bins[1:-1], right=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    mean_errors = np.full(len(bin_centers), np.nan, dtype=float)
    error_low = np.full(len(bin_centers), np.nan, dtype=float)
    error_high = np.full(len(bin_centers), np.nan, dtype=float)
    counts = np.zeros(len(bin_centers), dtype=int)
    for idx in range(len(bin_centers)):
        mask = bin_ids == idx
        counts[idx] = int(np.sum(mask))
        if counts[idx] > 0:
            bin_errors = abs_error[mask]
            mean_errors[idx] = float(np.mean(bin_errors))
            error_low[idx], error_high[idx] = bootstrap_mean_interval(
                bin_errors,
                bootstrap_samples=bootstrap_samples,
                seed=seed + idx,
            )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=150, sharex=True)
    axes[0].hist(quality, bins=30, color="#2f6f8f", edgecolor="white")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Held-out Test Quality Score Distribution")
    axes[0].grid(axis="y", alpha=0.25)

    lower = np.where(np.isfinite(error_low), np.maximum(0.0, mean_errors - error_low), 0.0)
    upper = np.where(np.isfinite(error_high), np.maximum(0.0, error_high - mean_errors), 0.0)
    axes[1].errorbar(
        bin_centers,
        mean_errors,
        yerr=np.vstack([lower, upper]),
        marker="o",
        color="#a23b3b",
        ecolor="#5f5f5f",
        elinewidth=1.2,
        capsize=3,
        linewidth=2,
    )
    axes[1].set_xlabel("Predicted quality score")
    axes[1].set_ylabel("Mean absolute age error (95% CI)")
    axes[1].grid(alpha=0.25)
    for x, y, count in zip(bin_centers, mean_errors, counts):
        if np.isfinite(y):
            axes[1].annotate(str(count), (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "quality_score_histogram.png")
    plt.close(fig)


def plot_quality_gate_metrics(rows: list[dict[str, float]], output_path: Path, title: str) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values("reject_fraction")
    if frame.empty:
        return

    x = frame["reject_fraction"].to_numpy(float)
    mae = frame["mae"].to_numpy(float)
    ci_low = frame.get("mae_ci_low", pd.Series(np.full(len(frame), np.nan))).to_numpy(float)
    ci_high = frame.get("mae_ci_high", pd.Series(np.full(len(frame), np.nan))).to_numpy(float)
    lower = np.where(np.isfinite(ci_low), np.maximum(0.0, mae - ci_low), 0.0)
    upper = np.where(np.isfinite(ci_high), np.maximum(0.0, ci_high - mae), 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), dpi=150, sharex=True)
    axes[0].errorbar(
        x,
        mae,
        yerr=np.vstack([lower, upper]),
        marker="o",
        capsize=4,
        linewidth=2,
        color="#2f6f8f",
        label="MAE (95% bootstrap CI)",
    )
    axes[0].set_ylabel("MAE (years)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for metric, label, color in [
        ("auc_adult_gate", "AUC", "#2f6f8f"),
        ("fpr", "FPR", "#a23b3b"),
        ("fnr", "FNR", "#7a5aa6"),
    ]:
        if metric in frame.columns:
            axes[1].plot(x, frame[metric].to_numpy(float), marker="o", linewidth=2, label=label, color=color)
    axes[1].set_xlabel("Rejected lowest-quality fraction")
    axes[1].set_ylabel("Age-gate metric")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", ncol=3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    age_df = load_age_predictions(Path(args.age_predictions))
    quality_df = pd.read_csv(args.quality_predictions)
    if "image_path" not in quality_df.columns or "pred_quality_score" not in quality_df.columns:
        raise ValueError("quality predictions must include image_path and pred_quality_score columns.")
    merged = age_df.merge(
        quality_df[["image_path", "pred_quality_score"]],
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
            mae_interval(
                admitted,
                bootstrap_samples=args.mae_bootstrap_samples,
                seed=args.bootstrap_seed + int(round(reject_fraction * 1000)),
            )
        )
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
                mae_interval(
                    admitted,
                    bootstrap_samples=args.mae_bootstrap_samples,
                    seed=args.bootstrap_seed + 10000 + int(round(reject_fraction * 1000)),
                )
            )
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
    plot_quality_histogram(
        merged,
        output_dir,
        bootstrap_samples=args.mae_bootstrap_samples,
        seed=args.bootstrap_seed + 20000,
    )
    plot_quality_gate_metrics(rows, output_dir / "quality_age_gate_metrics.png", "Quality Filtering Age-Gate Metrics")
    plot_quality_gate_metrics(
        near_rows,
        output_dir / "quality_age_gate_near_boundary_metrics.png",
        "Near-Boundary Quality Filtering Age-Gate Metrics",
    )
    pd.DataFrame(rows).to_csv(output_dir / "quality_age_gate_reassessment.csv", index=False)
    pd.DataFrame(near_rows).to_csv(output_dir / "quality_age_gate_near_boundary.csv", index=False)
    merged.to_csv(output_dir / "quality_age_gate_joined_predictions.csv", index=False)
    print(f"Saved quality-aware age-gate reassessment to {output_dir}")


if __name__ == "__main__":
    main()
