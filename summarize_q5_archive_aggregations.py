#!/usr/bin/env python3
"""Summarise Q5 archive metrics across aggregation sizes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


ARMS = ("R0", "Pooled")
SEEDS = tuple(range(42, 49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise archive aggregation sizes.")
    parser.add_argument("--run-root", type=str, default="runs/q5_archive_eval_local_masked_only_20260813")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--out-csv",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813/archive_aggregation_metrics.csv",
    )
    return parser.parse_args()


def choose_operating_point(path: Path, target_fpr: float = 0.05) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = [
            {
                "tau": float(row["tau"]),
                "fpr": float(row["fpr"]),
                "fnr": float(row["fnr"]),
            }
            for row in csv.DictReader(fp)
        ]
    eligible = [row for row in rows if row["fpr"] <= target_fpr]
    if eligible:
        chosen = min(eligible, key=lambda row: (row["fnr"], -row["fpr"], row["tau"]))
        attained = 1
    else:
        chosen = min(rows, key=lambda row: (row["fpr"], row["fnr"], row["tau"]))
        attained = 0
    return {
        "fpr": chosen["fpr"],
        "fnr": chosen["fnr"],
        "tau": chosen["tau"],
        "attained": attained,
    }


def read_fold_rows(run_root: Path, arm: str, seed: int, group_size: int) -> list[dict[str, float]]:
    summary_path = run_root / f"{arm}_seed{seed}" / f"kfold_test_summary_n{group_size}.csv"
    with summary_path.open(newline="", encoding="utf-8") as fp:
        summary_rows = list(csv.DictReader(fp))
    rows = []
    for fold_idx, row in enumerate(summary_rows):
        op = choose_operating_point(
            run_root / f"{arm}_seed{seed}" / f"fold_{fold_idx}" / f"test_age_gate_metrics_n{group_size}_ddp.csv"
        )
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "fold": fold_idx,
                "group_size": group_size,
                "samples": float(row["samples"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "auc_adult_gate": float(row["auc_adult_gate"]),
                "fpr": op["fpr"],
                "adult_fnr": op["fnr"],
                "tau": op["tau"],
                "attained": op["attained"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    fold_rows = []
    for group_size in args.group_sizes:
        for seed in SEEDS:
            for arm in ARMS:
                fold_rows.extend(read_fold_rows(run_root, arm, seed, group_size))

    summary_rows = []
    for group_size in args.group_sizes:
        for arm in ARMS:
            rows = [row for row in fold_rows if row["group_size"] == group_size and row["arm"] == arm]
            summary_rows.append(
                {
                    "group_size": group_size,
                    "arm": arm,
                    "folds": len(rows),
                    "samples_per_fold": mean(row["samples"] for row in rows),
                    "mae": mean(row["mae"] for row in rows),
                    "rmse": mean(row["rmse"] for row in rows),
                    "auc_adult_gate": mean(row["auc_adult_gate"] for row in rows),
                    "mean_fpr": mean(row["fpr"] for row in rows),
                    "adult_fnr": mean(row["adult_fnr"] for row in rows),
                    "mean_tau": mean(row["tau"] for row in rows),
                    "folds_attained_target_fpr": sum(row["attained"] for row in rows),
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_csv}")
    for row in summary_rows:
        print(
            f"n={row['group_size']} {row['arm']}: samples/fold={row['samples_per_fold']:.1f} "
            f"MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} AUC={row['auc_adult_gate']:.4f} "
            f"FPR={100*row['mean_fpr']:.2f}% FNR={100*row['adult_fnr']:.2f}%"
        )


if __name__ == "__main__":
    main()
