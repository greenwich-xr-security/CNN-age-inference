#!/usr/bin/env python3
"""Summarize age-conditioning monotonicity and calibration from Q2 RS/TRTS predictions.

Supersedes the per-decade Spearman diagnostic in
``summarize_q2_conditioning_fidelity.py``, which was rejected: slicing into
decades leaves only ~10 years of conditioned-age spread against an RS error of
~12-13 years, so the within-bin correlation is attenuated to a ceiling of about
0.23 regardless of how faithful the generator is.

This script instead asks the question at the scale it is posed:

1. Global Spearman/Pearson across the full conditioned-age range, per fold.
   Here the conditioned-age spread is ~18 years, lifting the attenuation
   ceiling to ~0.84 and giving the statistic real discriminating power.
2. The OLS slope of predicted on conditioned age, which measures compression of
   the age axis directly (slope < 1 means the generator's rendered range is
   narrower than its labels claim).
3. Decade-level mean predicted age with standard errors -- a calibration curve.
   Averaging within a bin cuts the noise by sqrt(n), so bin means remain
   informative where the within-bin correlation does not.

The approximate attenuation ceiling is reported alongside every correlation so
the numbers are not read without it.

As in the superseded script, the five RS folds are five real-trained
checkpoints evaluated on one shared held-out synthetic set, so statistics are
computed per fold and then summarized; images are never pooled across folds.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/q2_fullreal_rs_v2s_seed42"),
        help="RS/TRTS run directory containing fold_* directories.",
    )
    parser.add_argument(
        "--global-output",
        type=Path,
        default=None,
        help="Global summary CSV (default: <run-dir>/age_monotonicity_global.csv).",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=None,
        help="Decade calibration CSV (default: <run-dir>/age_calibration_by_decade.csv).",
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


def load_run(run_dir: Path) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Load every fold, asserting they evaluate the same ordered examples."""
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
    return reference_targets, fold_predictions, fold_names


def attenuation_ceiling(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Approximate correlation ceiling for an additive-noise reader.

    For ``predicted = true + noise`` the attainable correlation is
    ``sigma_true / sqrt(sigma_true^2 + sigma_resid^2)``. Residual spread is
    taken as std(predicted - true), which folds any slope compression into the
    noise term, so this is an approximation and a lower bound on the ceiling.
    """
    sigma_true = float(np.std(targets, ddof=1))
    sigma_resid = float(np.std(predictions - targets, ddof=1))
    if sigma_true <= 0.0:
        raise ValueError("Conditioned ages have zero spread; correlation is undefined")
    return sigma_true / float(np.hypot(sigma_true, sigma_resid))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite statistic encountered across folds")
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def global_rows(
    targets: np.ndarray, fold_predictions: list[np.ndarray], fold_names: list[str]
) -> list[dict[str, object]]:
    """Per-fold global correlation, OLS slope, and attenuation ceiling."""
    if np.unique(targets).size < 2:
        raise ValueError("Global summary needs at least two distinct conditioned ages")

    per_fold: dict[str, list[float]] = {"spearman": [], "pearson": [], "slope": [], "intercept": [], "ceiling": []}
    rows: list[dict[str, object]] = []
    for name, predictions in zip(fold_names, fold_predictions):
        slope, intercept = (float(value) for value in np.polyfit(targets, predictions, 1))
        stats = {
            "spearman": float(spearmanr(targets, predictions).statistic),
            "pearson": float(pearsonr(targets, predictions).statistic),
            "slope": slope,
            "intercept": intercept,
            "ceiling": attenuation_ceiling(targets, predictions),
        }
        if not np.isfinite(list(stats.values())).all():
            raise ValueError(f"{name} produced a non-finite global statistic")
        for key, value in stats.items():
            per_fold[key].append(value)
        rows.append({"scope": name, "n_images": int(targets.size), **stats})

    for key, values in per_fold.items():
        summary = summarize(values)
        rows.append(
            {
                "scope": f"all_folds_{key}",
                "n_images": int(targets.size),
                "spearman": "",
                "pearson": "",
                "slope": "",
                "intercept": "",
                "ceiling": "",
                **{f"fold_{stat}": value for stat, value in summary.items()},
            }
        )
    fieldnames = list(rows[-1])
    return [{name: row.get(name, "") for name in fieldnames} for row in rows]


def calibration_rows(
    targets: np.ndarray, fold_predictions: list[np.ndarray], fold_names: list[str]
) -> list[dict[str, object]]:
    """Decade-level mean predicted age, with image-level SE and between-fold SD."""
    lower_decade = int(np.floor(targets.min() / 10.0) * 10)
    upper_decade = int(np.floor(targets.max() / 10.0) * 10)

    rows: list[dict[str, object]] = []
    for lower in range(lower_decade, upper_decade + 1, 10):
        mask = (targets >= lower) & (targets < lower + 10)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_means = [float(predictions[mask].mean()) for predictions in fold_predictions]
        # Image-level scatter within the bin, averaged over folds, as a standard error.
        image_ses = [
            float(np.std(predictions[mask], ddof=1) / np.sqrt(count)) if count > 1 else float("nan")
            for predictions in fold_predictions
        ]
        mean_predicted = float(np.mean(bin_means))
        rows.append(
            {
                "age_decade": f"{lower}-{lower + 9}",
                "n_images": count,
                "mean_conditioned_age": float(targets[mask].mean()),
                "mean_predicted_age": mean_predicted,
                "predicted_minus_conditioned": mean_predicted - float(targets[mask].mean()),
                "image_level_se": float(np.mean(image_ses)) if count > 1 else "",
                "between_fold_sd": float(np.std(bin_means, ddof=1)) if len(bin_means) > 1 else 0.0,
                **{f"mean_pred_{name}": value for name, value in zip(fold_names, bin_means)},
            }
        )
    if not rows:
        raise ValueError("No populated age decades found")
    return rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    global_output = (args.global_output or run_dir / "age_monotonicity_global.csv").resolve()
    calibration_output = (args.calibration_output or run_dir / "age_calibration_by_decade.csv").resolve()

    targets, fold_predictions, fold_names = load_run(run_dir)
    globals_ = global_rows(targets, fold_predictions, fold_names)
    calibration = calibration_rows(targets, fold_predictions, fold_names)

    write_csv(global_output, globals_)
    write_csv(calibration_output, calibration)
    print(f"Wrote {global_output}")
    print(f"Wrote {calibration_output}")

    summaries = {row["scope"]: row for row in globals_ if str(row["scope"]).startswith("all_folds_")}
    spearman = summaries["all_folds_spearman"]
    ceiling = summaries["all_folds_ceiling"]
    slope = summaries["all_folds_slope"]
    print(
        f"\nGlobal (n={targets.size}, conditioned-age SD {np.std(targets, ddof=1):.1f} yr): "
        f"Spearman rho = {spearman['fold_mean']:.3f} +/- {spearman['fold_sd']:.3f}, "
        f"attenuation ceiling ~{ceiling['fold_mean']:.3f}"
    )
    print(
        f"Interpretation: rho near the ceiling means the conditioned-age axis is monotone; "
        f"rho far below it means it is not."
    )
    print(f"OLS slope = {slope['fold_mean']:.3f} +/- {slope['fold_sd']:.3f} (1.0 = uncompressed age axis)\n")

    print("Decade calibration (mean predicted age; monotone rise = conditioning holds):")
    for row in calibration:
        se = row["image_level_se"]
        se_text = f" +/- {se:.2f}" if isinstance(se, float) else ""
        print(
            f"  {row['age_decade']}: n={row['n_images']}, "
            f"conditioned {row['mean_conditioned_age']:.1f} -> predicted {row['mean_predicted_age']:.1f}{se_text} "
            f"(delta {row['predicted_minus_conditioned']:+.1f} yr)"
        )


if __name__ == "__main__":
    main()
