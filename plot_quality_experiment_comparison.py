#!/usr/bin/env python3
"""Compare quality-filter age-gate aggregates across quality experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LABELS = {
    "v2_small_quality_full_wall3_b32_quality": "V2-S baseline",
    "v2_small_quality_full_wall3_b32_quality_age_norm": "V2-S age-error norm",
    "v2_small_quality_full_wall3_b32_quality_age_norm_components": "V2-S component norm",
    "v2_small_quality_full_wall3_b32_quality_age_norm_components_b0": "B0 component norm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quality experiment aggregate comparisons.")
    parser.add_argument("--quality-root", required=True, help="Directory containing quality experiment folders.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <quality-root>/comparison).")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=list(DEFAULT_LABELS),
        help="Experiment folder names to compare. Defaults to known quality runs.",
    )
    return parser.parse_args()


def load_experiment(root: Path, name: str) -> pd.DataFrame | None:
    aggregate_dir = root / name / "aggregate"
    sidecar_path = aggregate_dir / "quality_age_gate_reassessment_with_mae_ci.csv"
    path = sidecar_path if sidecar_path.exists() else aggregate_dir / "quality_age_gate_reassessment.csv"
    if not path.exists():
        print(f"[warn] missing aggregate: {path}")
        return None
    frame = pd.read_csv(path)
    frame["experiment"] = name
    frame["label"] = DEFAULT_LABELS.get(name, name)
    return frame


def plot_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=150, sharex=True)
    colors = plt.get_cmap("tab10")
    labels = list(dict.fromkeys(frame["label"].tolist()))

    for idx, label in enumerate(labels):
        part = frame[frame["label"] == label].sort_values("reject_fraction")
        x = part["reject_fraction"].to_numpy(float)
        mae = part["mae"].to_numpy(float)
        ci_low = part.get("mae_ci_low", pd.Series(np.full(len(part), np.nan))).to_numpy(float)
        ci_high = part.get("mae_ci_high", pd.Series(np.full(len(part), np.nan))).to_numpy(float)
        lower = np.where(np.isfinite(ci_low), np.maximum(0.0, mae - ci_low), 0.0)
        upper = np.where(np.isfinite(ci_high), np.maximum(0.0, ci_high - mae), 0.0)
        color = colors(idx % 10)
        axes[0].errorbar(x, mae, yerr=np.vstack([lower, upper]), marker="o", capsize=3, linewidth=1.8, label=label, color=color)
        axes[1].plot(x, part["auc_adult_gate"].to_numpy(float), marker="o", linewidth=1.8, label=label, color=color)

    axes[0].set_ylabel("MAE (years)")
    axes[0].set_title("Quality Filtering Comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_xlabel("Rejected lowest-quality fraction")
    axes[1].set_ylabel("Adult-gate AUC")
    axes[1].set_ylim(0.88, 0.98)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    quality_root = Path(args.quality_root)
    output_dir = Path(args.output_dir) if args.output_dir else quality_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_experiment(quality_root, name) for name in args.experiments]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError("No quality aggregate CSVs found.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "quality_experiment_reassessment_compare.csv", index=False)
    plot_comparison(combined, output_dir / "quality_experiment_mae_auc_compare.png")
    print(f"Saved quality experiment comparison to {output_dir}")


if __name__ == "__main__":
    main()
