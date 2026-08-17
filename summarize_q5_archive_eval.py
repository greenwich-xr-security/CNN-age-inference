#!/usr/bin/env python3
"""Summarise Q5 archive-only evaluation outputs."""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np

try:
    from scipy import stats
except Exception:  # noqa: BLE001
    stats = None


ARMS = ("R0", "Pooled")
SEEDS = tuple(range(42, 49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise Q5 archive evaluation runs.")
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/q5_archive_eval_20260813",
        help="Root containing R0_seed*/Pooled_seed* evaluation directories.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="Q5_ARCHIVE_EVAL_RESULTS.md",
        help="Markdown results file to write.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target maximum FPR operating point for Adult FNR reporting.",
    )
    return parser.parse_args()


def read_kfold_summary(run_dir: Path) -> dict[str, float]:
    path = run_dir / "kfold_test_summary_n1.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregate summary: {path}")
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"Empty aggregate summary: {path}")
    return {
        "mae": mean(float(row["mae"]) for row in rows),
        "rmse": mean(float(row["rmse"]) for row in rows),
        "auc_adult_gate": mean(float(row["auc_adult_gate"]) for row in rows),
        "samples_per_fold": mean(float(row["samples"]) for row in rows),
        "folds": len(rows),
    }


def read_operating_points(run_dir: Path, target_fpr: float) -> dict[str, float]:
    fprs = []
    fnrs = []
    taus = []
    attainable = 0
    for fold_idx in range(5):
        path = run_dir / f"fold_{fold_idx}" / "test_age_gate_metrics_n1_ddp.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing age-gate metrics: {path}")
        with path.open(newline="", encoding="utf-8") as fp:
            rows = [
                {
                    "tau": float(row["tau"]),
                    "fpr": float(row["fpr"]),
                    "fnr": float(row["fnr"]),
                    "tpr": float(row["tpr"]),
                }
                for row in csv.DictReader(fp)
            ]
        eligible = [row for row in rows if row["fpr"] <= target_fpr]
        if eligible:
            chosen = min(eligible, key=lambda row: (row["fnr"], -row["fpr"], row["tau"]))
            attainable += 1
        else:
            chosen = min(rows, key=lambda row: (row["fpr"], row["fnr"], row["tau"]))
        fprs.append(chosen["fpr"])
        fnrs.append(chosen["fnr"])
        taus.append(chosen["tau"])
    return {
        "mean_fpr": mean(fprs),
        "adult_fnr": mean(fnrs),
        "mean_tau": mean(taus),
        "folds_attained_target_fpr": attainable,
    }


def paired_tests(values_a: list[float], values_b: list[float], *, higher_is_better: bool) -> dict[str, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = b - a if higher_is_better else a - b
    out = {
        "mean_advantage": float(np.mean(diff)),
        "wins": int(np.sum(diff > 0)),
        "n": int(diff.size),
        "paired_t_p_two_sided": float("nan"),
        "wilcoxon_p_two_sided": float("nan"),
    }
    if stats is not None and diff.size > 1:
        out["paired_t_p_two_sided"] = float(stats.ttest_rel(b, a).pvalue)
        try:
            out["wilcoxon_p_two_sided"] = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
        except ValueError:
            out["wilcoxon_p_two_sided"] = 1.0
    return out


def fmt_num(value: float, digits: int = 3) -> str:
    if np.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_pct(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_md = Path(args.out_md)
    out_csv = run_root / "archive_seed_metrics.csv"
    paired_csv = run_root / "archive_paired_comparison.csv"

    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = run_root / f"{arm}_seed{seed}"
            summary = read_kfold_summary(run_dir)
            op = read_operating_points(run_dir, args.target_fpr)
            rows.append({"seed": seed, "arm": arm, **summary, **op})

    run_root.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_arm_seed = {(row["arm"], row["seed"]): row for row in rows}
    metric_specs = [
        ("mae", False),
        ("rmse", False),
        ("auc_adult_gate", True),
        ("adult_fnr", False),
    ]
    paired_rows = []
    for metric, higher_is_better in metric_specs:
        r0_vals = [by_arm_seed[("R0", seed)][metric] for seed in SEEDS]
        pooled_vals = [by_arm_seed[("Pooled", seed)][metric] for seed in SEEDS]
        test = paired_tests(r0_vals, pooled_vals, higher_is_better=higher_is_better)
        paired_rows.append({"metric": metric, **test})

    with paired_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    means = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        means[arm] = {
            "mae": mean(row["mae"] for row in arm_rows),
            "rmse": mean(row["rmse"] for row in arm_rows),
            "auc_adult_gate": mean(row["auc_adult_gate"] for row in arm_rows),
            "mean_fpr": mean(row["mean_fpr"] for row in arm_rows),
            "adult_fnr": mean(row["adult_fnr"] for row in arm_rows),
        }

    lines = [
        "# Q5 Archive Evaluation Results",
        "",
        f"Last updated: {datetime.now().astimezone().replace(microsecond=0).isoformat()}",
        "",
        "Archive-only inference for the Q5 paired fine-tuned model set. Each row is a five-fold unweighted `n=1` aggregate over the filtered archive dataset.",
        "",
        f"Run root: `{run_root}`",
        f"Execution context: `{('Slurm job ' + os.environ['SLURM_JOB_ID']) if os.environ.get('SLURM_JOB_ID') else 'local'}`",
        f"Operating point: best fold-level threshold with FPR <= {100 * args.target_fpr:.1f}%; if unavailable, lowest-FPR threshold in the 10-30 age-threshold sweep.",
        "",
        "## Seed-Level Results",
        "",
        "| Seed | Arm | Folds | Samples/fold | MAE | RMSE | Adult-gate AUC | Mean FPR | Adult FNR | Mean tau | Folds <= target FPR |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | `{row['arm']}` | {int(row['folds'])} | "
            f"{row['samples_per_fold']:.0f} | {fmt_num(row['mae'])} | "
            f"{fmt_num(row['rmse'])} | {fmt_num(row['auc_adult_gate'], 4)} | "
            f"{fmt_pct(row['mean_fpr'])} | {fmt_pct(row['adult_fnr'])} | "
            f"{fmt_num(row['mean_tau'], 2)} | {int(row['folds_attained_target_fpr'])}/5 |"
        )

    lines.extend([
        "",
        "## Arm Means",
        "",
        "| Arm | Mean MAE | Mean RMSE | Mean Adult-gate AUC | Mean FPR | Adult FNR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm in ARMS:
        row = means[arm]
        lines.append(
            f"| `{arm}` | {fmt_num(row['mae'])} | {fmt_num(row['rmse'])} | "
            f"{fmt_num(row['auc_adult_gate'], 4)} | {fmt_pct(row['mean_fpr'])} | "
            f"{fmt_pct(row['adult_fnr'])} |"
        )

    lines.extend([
        "",
        "## Paired Comparisons",
        "",
        "Positive mean advantage means `Pooled` is better than `R0`; for MAE, RMSE, and Adult FNR this is a reduction, while for AUC it is an increase.",
        "",
        "| Metric | Mean Pooled advantage | Wins | Paired t p two-sided | Wilcoxon p two-sided |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in paired_rows:
        metric = row["metric"]
        suffix = " pp" if metric == "adult_fnr" else ""
        value = row["mean_advantage"] * 100 if metric == "adult_fnr" else row["mean_advantage"]
        lines.append(
            f"| `{metric}` | {fmt_num(value, 4)}{suffix} | {row['wins']}/{row['n']} | "
            f"{fmt_num(row['paired_t_p_two_sided'], 4)} | {fmt_num(row['wilcoxon_p_two_sided'], 4)} |"
        )

    lines.extend([
        "",
        "## Output Files",
        "",
        f"- Seed metrics CSV: `{out_csv}`",
        f"- Paired comparison CSV: `{paired_csv}`",
    ])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {paired_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
