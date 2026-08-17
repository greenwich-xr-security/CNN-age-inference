#!/usr/bin/env python3
"""Plot masked-only Q5 archive MAE by target age."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("R0", "Pooled")
SEEDS = tuple(range(42, 49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Q5 archive MAE per age.")
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813",
        help="Masked-only result root.",
    )
    parser.add_argument("--window", type=int, default=5, help="Centered rolling age-window size.")
    parser.add_argument(
        "--out-png",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813/archive_mae_by_age_smoothed.png",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813/archive_mae_by_age.csv",
    )
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.empty_like(values, dtype=float)
    half = window // 2
    for idx in range(values.size):
        start = max(0, idx - half)
        end = min(values.size, idx + half + 1)
        out[idx] = float(np.nanmean(values[start:end]))
    return out


def collect_arm_predictions(run_root: Path, arm: str) -> tuple[np.ndarray, np.ndarray]:
    targets_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    for seed in SEEDS:
        for fold_idx in range(5):
            path = run_root / f"{arm}_seed{seed}" / f"fold_{fold_idx}" / "test_predictions_raw_ddp.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                targets_all.append(data["targets"].astype(float))
                preds_all.append(data["pred_mean"].astype(float))
    return np.concatenate(targets_all), np.concatenate(preds_all)


def per_age_rows(run_root: Path, window: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for arm in ARMS:
        targets, preds = collect_arm_predictions(run_root, arm)
        ages = np.asarray(np.round(targets), dtype=int)
        unique_ages = np.asarray(sorted(set(ages.tolist())), dtype=int)
        mae = []
        rmse = []
        counts = []
        for age in unique_ages:
            mask = ages == age
            errors = preds[mask] - targets[mask]
            mae.append(float(np.mean(np.abs(errors))))
            rmse.append(float(np.sqrt(np.mean(errors**2))))
            counts.append(int(mask.sum()))
        mae_arr = np.asarray(mae, dtype=float)
        smoothed = rolling_mean(mae_arr, window)
        for age, count, mae_value, rmse_value, smooth_value in zip(unique_ages, counts, mae, rmse, smoothed):
            rows.append(
                {
                    "arm": arm,
                    "age": int(age),
                    "predictions": int(count),
                    "mae": float(mae_value),
                    "rmse": float(rmse_value),
                    "mae_smoothed": float(smooth_value),
                    "smoothing_window": int(window),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["arm", "age", "predictions", "mae", "rmse", "mae_smoothed", "smoothing_window"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path: Path, rows: list[dict[str, float | int | str]], window: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {"R0": "#1f77b4", "Pooled": "#d62728"}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        ages = np.asarray([row["age"] for row in arm_rows], dtype=float)
        mae = np.asarray([row["mae"] for row in arm_rows], dtype=float)
        smooth = np.asarray([row["mae_smoothed"] for row in arm_rows], dtype=float)
        ax.plot(ages, mae, marker="o", linewidth=1.0, alpha=0.35, color=colors[arm], label=f"{arm} per-age MAE")
        ax.plot(ages, smooth, linewidth=2.4, color=colors[arm], label=f"{arm} smoothed")
    ax.set_title(f"Q5 Archive Masked-Only MAE by Age (rolling window={window})")
    ax.set_xlabel("Target age")
    ax.set_ylabel("MAE")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    rows = per_age_rows(run_root, args.window)
    write_csv(Path(args.out_csv), rows)
    plot_rows(Path(args.out_png), rows, args.window)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
