#!/usr/bin/env python3
"""Add per-user aggregated prediction outputs for masked-only Q5 archive results."""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from metrics import aggregate_predictions_by_user, compute_age_gate_curves_direct_threshold


ARMS = ("R0", "Pooled")
SEEDS = tuple(range(42, 49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute aggregation sizes for masked-only Q5 archive outputs.")
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813",
        help="Masked-only result root.",
    )
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--aggregation-seed", type=int, default=42)
    parser.add_argument("--age-gate-threshold-min", type=float, default=10.0)
    parser.add_argument("--age-gate-threshold-max", type=float, default=30.0)
    parser.add_argument("--num-thresholds", type=int, default=201)
    return parser.parse_args()


def write_age_gate_metrics(path: Path, gate: dict) -> None:
    adult_gate = gate["adult_gate"]
    with path.open("w", encoding="utf-8") as fp:
        fp.write("gate,tau,fpr,fnr,tpr,tnr\n")
        for tau, fpr, fnr, tpr_val, tnr in zip(
            adult_gate["thresholds"],
            adult_gate["fpr"],
            adult_gate["fnr"],
            adult_gate["tpr"],
            adult_gate["tnr"],
        ):
            fp.write(f"adult_gate,{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n")


def add_fold_outputs(
    fold_dir: Path,
    group_sizes: list[int],
    *,
    aggregation_seed: int,
    age_min: float,
    age_max: float,
    num_thresholds: int,
) -> None:
    raw_path = fold_dir / "test_predictions_raw_ddp.npz"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    with np.load(raw_path, allow_pickle=False) as data:
        targets = data["targets"].astype(float)
        pred_mean = data["pred_mean"].astype(float)
        pred_log_var = data["pred_log_var"].astype(float)
        user_ids = data["user_ids"].astype(str)
        skin_color = data["skin_color"].astype(str) if "skin_color" in data.files else None

    summary_rows = []
    for group_size in sorted(set(group_sizes)):
        aggregated = aggregate_predictions_by_user(
            user_ids,
            targets,
            pred_mean,
            pred_log_var,
            group_size=group_size,
            rng=random.Random(aggregation_seed + group_size),
        )
        agg_targets = aggregated["targets"]
        agg_preds = aggregated["pred_mean"]
        agg_log_vars = aggregated["pred_log_var"]
        agg_user_ids = aggregated["user_ids"]
        gate = compute_age_gate_curves_direct_threshold(
            agg_targets,
            agg_preds,
            age_min=age_min,
            age_max=age_max,
            num_thresholds=num_thresholds,
        )
        adult_gate = gate["adult_gate"]
        suffix = f"n{group_size}_ddp"
        mae = float(np.mean(np.abs(agg_preds - agg_targets)))
        rmse = float(np.sqrt(np.mean((agg_preds - agg_targets) ** 2)))
        auc = float(adult_gate["auc"])
        np.savez(
            fold_dir / f"test_predictions_{suffix}.npz",
            targets=agg_targets,
            pred_mean=agg_preds,
            pred_log_var=agg_log_vars,
            adult_prob=gate["adult_prob"],
            user_ids=agg_user_ids,
            skin_color=skin_color[: agg_user_ids.size] if skin_color is not None and group_size == 1 else np.asarray(["unknown"] * agg_user_ids.size),
            group_size=group_size,
        )
        write_age_gate_metrics(fold_dir / f"test_age_gate_metrics_{suffix}.csv", gate)
        summary_rows.append(
            {
                "group_size": group_size,
                "mae": mae,
                "rmse": rmse,
                "auc_adult_gate": auc,
                "samples": int(agg_targets.size),
            }
        )

    with (fold_dir / "test_summary_ddp.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["group_size", "mae", "rmse", "auc_adult_gate", "samples"])
        writer.writeheader()
        writer.writerows(summary_rows)


def run_kfold_aggregate(repo_root: Path, run_dir: Path, group_sizes: list[int], args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "aggregate_kfold.py"),
        "--kfold-root",
        str(run_dir),
        "--output-dir",
        str(run_dir),
        "--group-sizes",
        *[str(n) for n in group_sizes],
        "--eval-age-gate-mode",
        "age_threshold",
        "--age-gate-threshold-min",
        str(args.age_gate_threshold_min),
        "--age-gate-threshold-max",
        str(args.age_gate_threshold_max),
        "--skip-val",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root)
    group_sizes = sorted({int(n) for n in args.group_sizes if int(n) > 0})
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = run_root / f"{arm}_seed{seed}"
            for fold_idx in range(5):
                add_fold_outputs(
                    run_dir / f"fold_{fold_idx}",
                    group_sizes,
                    aggregation_seed=args.aggregation_seed,
                    age_min=args.age_gate_threshold_min,
                    age_max=args.age_gate_threshold_max,
                    num_thresholds=args.num_thresholds,
                )
            run_kfold_aggregate(repo_root, run_dir, group_sizes, args)
            print(f"Aggregated {run_dir} for n={','.join(str(n) for n in group_sizes)}")


if __name__ == "__main__":
    main()
