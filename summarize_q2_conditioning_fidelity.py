#!/usr/bin/env python3
"""Summarize age-conditioning fidelity from Q2 RS/TRTS prediction files.

The five RS folds are five real-trained checkpoints evaluated on one shared
held-out SyntheticDorsalHands2 set.  Consequently, this script computes one
Spearman correlation per fold and decade, then summarizes those five values;
it deliberately does not pool duplicate images across folds.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/q2_fullreal_rs_v2s_seed42"),
        help="RS/TRTS run directory containing fold_* directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <run-dir>/conditioning_fidelity_by_decade.csv).",
    )
    return parser.parse_args()


def load_fold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("targets", "pred_mean", "user_ids")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        targets = np.asarray(data["targets"], dtype=float)
        predictions = np.asarray(data["pred_mean"], dtype=float)
        user_ids = np.asarray(data["user_ids"], dtype=str)
    if not (len(targets) == len(predictions) == len(user_ids)):
        raise ValueError(f"{path} has inconsistent prediction-array lengths")
    if not (np.isfinite(targets).all() and np.isfinite(predictions).all()):
        raise ValueError(f"{path} contains non-finite targets or predictions")
    return targets, predictions, user_ids


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "conditioning_fidelity_by_decade.csv"
    fold_paths = sorted(run_dir.glob("fold_*/test_predictions_n1_ddp.npz"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold predictions found under {run_dir}")

    reference_targets: np.ndarray | None = None
    reference_user_ids: np.ndarray | None = None
    fold_predictions: list[np.ndarray] = []
    fold_names: list[str] = []
    for path in fold_paths:
        targets, predictions, user_ids = load_fold(path)
        if reference_targets is None:
            reference_targets, reference_user_ids = targets, user_ids
        elif not (np.array_equal(targets, reference_targets) and np.array_equal(user_ids, reference_user_ids)):
            raise ValueError(
                "RS folds do not evaluate the same ordered synthetic examples; "
                "refusing to summarize them as repeated model evaluations."
            )
        fold_predictions.append(predictions)
        fold_names.append(path.parent.name)

    assert reference_targets is not None
    rows: list[dict[str, str | int | float]] = []
    lower_decade = int(np.floor(reference_targets.min() / 10.0) * 10)
    upper_decade = int(np.floor(reference_targets.max() / 10.0) * 10)
    for lower in range(lower_decade, upper_decade + 1, 10):
        upper = lower + 9
        mask = (reference_targets >= lower) & (reference_targets < lower + 10)
        bin_targets = reference_targets[mask]
        if bin_targets.size < 2 or np.unique(bin_targets).size < 2:
            raise ValueError(f"Age {lower}-{upper} has insufficient distinct conditioned ages for Spearman rho")
        rhos = [float(spearmanr(bin_targets, predictions[mask]).statistic) for predictions in fold_predictions]
        if not np.isfinite(rhos).all():
            raise ValueError(f"Age {lower}-{upper} produced a non-finite Spearman rho")
        row: dict[str, str | int | float] = {
            "age_decade": f"{lower}-{upper}",
            "n_images": int(mask.sum()),
            "n_distinct_conditioned_ages": int(np.unique(bin_targets).size),
            "rho_mean": float(np.mean(rhos)),
            "rho_sd": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
            "rho_min": float(np.min(rhos)),
            "rho_max": float(np.max(rhos)),
        }
        row.update({f"rho_{fold_name}": rho for fold_name, rho in zip(fold_names, rhos)})
        rows.append(row)

    fieldnames = list(rows[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    for row in rows:
        print(
            f"{row['age_decade']}: n={row['n_images']}, "
            f"rho={row['rho_mean']:.3f} +/- {row['rho_sd']:.3f} "
            f"(range {row['rho_min']:.3f}-{row['rho_max']:.3f})"
        )


if __name__ == "__main__":
    main()
