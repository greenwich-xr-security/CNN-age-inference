#!/usr/bin/env python3
"""Recalculate adult-gate AUC and 5%-FPR operating metrics from prediction dumps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import compute_age_gate_curves_direct_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--age-min", type=float, default=0.0)
    parser.add_argument("--age-max", type=float, default=100.0)
    parser.add_argument("--num-thresholds", type=int, default=1001)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    return parser.parse_args()


def select_operating_point(gate: dict, target_fpr: float) -> int:
    fpr = np.asarray(gate["fpr"], dtype=float)
    fnr = np.asarray(gate["fnr"], dtype=float)
    eligible = np.flatnonzero(fpr <= target_fpr)
    if eligible.size:
        return int(min(eligible, key=lambda idx: (fnr[idx], -fpr[idx])))
    return int(np.argmin(fpr))


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for run_dir in args.run_dir:
        fold_paths = sorted(run_dir.glob("fold_*/test_predictions_n1_ddp.npz"))
        if not fold_paths:
            raise FileNotFoundError(f"No n=1 held-out prediction dumps under {run_dir}")

        fold_rows: list[dict[str, object]] = []
        for path in fold_paths:
            data = np.load(path)
            gate = compute_age_gate_curves_direct_threshold(
                data["targets"],
                data["pred_mean"],
                age_min=args.age_min,
                age_max=args.age_max,
                num_thresholds=args.num_thresholds,
            )["adult_gate"]
            idx = select_operating_point(gate, args.target_fpr)
            fold_rows.append(
                {
                    "run_dir": run_dir.name,
                    "fold": path.parent.name,
                    "auc_adult_gate": float(gate["auc"]),
                    "fpr": float(gate["fpr"][idx]),
                    "adult_fnr": float(gate["fnr"][idx]),
                    "threshold": float(gate["thresholds"][idx]),
                    "within_target_fpr": bool(gate["fpr"][idx] <= args.target_fpr),
                }
            )

        rows.extend(fold_rows)
        rows.append(
            {
                "run_dir": run_dir.name,
                "fold": "mean",
                "auc_adult_gate": float(np.mean([row["auc_adult_gate"] for row in fold_rows])),
                "fpr": float(np.mean([row["fpr"] for row in fold_rows])),
                "adult_fnr": float(np.mean([row["adult_fnr"] for row in fold_rows])),
                "threshold": float(np.mean([row["threshold"] for row in fold_rows])),
                "within_target_fpr": all(row["within_target_fpr"] for row in fold_rows),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
