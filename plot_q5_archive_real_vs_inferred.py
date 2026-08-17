#!/usr/bin/env python3
"""Plot real versus inferred age for Q5 archive masked-only results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("R0", "Pooled")
SEEDS = tuple(range(42, 49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real vs inferred age for archive Q5 outputs.")
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813",
        help="Masked-only archive result root.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813/archive_real_vs_inferred_scatter.png",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813/archive_real_vs_inferred_ensemble.csv",
    )
    return parser.parse_args()


def load_arm_matrix(run_root: Path, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets_ref: np.ndarray | None = None
    user_ids_ref: np.ndarray | None = None
    pred_cols = []
    for seed in SEEDS:
        for fold_idx in range(5):
            path = run_root / f"{arm}_seed{seed}" / f"fold_{fold_idx}" / "test_predictions_raw_ddp.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                targets = data["targets"].astype(float)
                user_ids = data["user_ids"].astype(str)
                preds = data["pred_mean"].astype(float)
            if targets_ref is None:
                targets_ref = targets
                user_ids_ref = user_ids
            elif not np.array_equal(targets_ref, targets) or not np.array_equal(user_ids_ref, user_ids):
                raise ValueError(f"Prediction order differs for {path}")
            pred_cols.append(preds)
    return targets_ref, user_ids_ref, np.stack(pred_cols, axis=1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["arm", "sample_index", "user_id", "real_age", "inferred_age_mean", "inferred_age_std"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)

    arm_data = {}
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        targets, user_ids, pred_matrix = load_arm_matrix(run_root, arm)
        pred_mean = pred_matrix.mean(axis=1)
        pred_std = pred_matrix.std(axis=1)
        arm_data[arm] = (targets, pred_mean, pred_std)
        for idx, (uid, target, pred, std) in enumerate(zip(user_ids, targets, pred_mean, pred_std)):
            rows.append(
                {
                    "arm": arm,
                    "sample_index": idx,
                    "user_id": uid,
                    "real_age": float(target),
                    "inferred_age_mean": float(pred),
                    "inferred_age_std": float(std),
                }
            )
    write_csv(out_csv, rows)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=True, sharey=True)
    colors = {"R0": "#1f77b4", "Pooled": "#d62728"}
    for ax, arm in zip(axes, ARMS):
        targets, pred_mean, _ = arm_data[arm]
        mae = float(np.mean(np.abs(pred_mean - targets)))
        rmse = float(np.sqrt(np.mean((pred_mean - targets) ** 2)))
        ax.scatter(targets, pred_mean, s=14, alpha=0.28, color=colors[arm], edgecolors="none")
        min_age = float(min(targets.min(), pred_mean.min()))
        max_age = float(max(targets.max(), pred_mean.max()))
        pad = 2.0
        lo = max(0.0, min_age - pad)
        hi = max_age + pad
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, label="ideal")
        ax.axvline(18.0, color="gray", linestyle=":", linewidth=1.0)
        ax.axhline(18.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{arm} ensemble mean\nMAE={mae:.2f}, RMSE={rmse:.2f}")
        ax.set_xlabel("Real age")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel("Inferred age")
    fig.suptitle("Q5 Archive Masked-Only: Real vs Inferred Age", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
