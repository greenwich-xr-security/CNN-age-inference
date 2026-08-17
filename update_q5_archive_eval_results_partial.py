#!/usr/bin/env python3
"""Write a partial Q5 archive evaluation markdown report."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean


ARMS = ("R0", "Pooled")
SEEDS = tuple(range(42, 49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update partial Q5 archive evaluation results.")
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/q5_archive_eval_local_masked_20260813",
        help="Root containing R0_seed*/Pooled_seed* evaluation directories.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="Q5_ARCHIVE_EVAL_RESULTS.md",
        help="Markdown file to write.",
    )
    parser.add_argument("--target-fpr", type=float, default=0.05)
    return parser.parse_args()


def read_one_row_csv(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    return rows[0] if rows else None


def read_kfold_summary(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "kfold_test_summary_n1.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return None
    return {
        "mae": mean(float(row["mae"]) for row in rows),
        "rmse": mean(float(row["rmse"]) for row in rows),
        "auc_adult_gate": mean(float(row["auc_adult_gate"]) for row in rows),
        "samples_per_fold": mean(float(row["samples"]) for row in rows),
        "folds": len(rows),
    }


def read_operating_point(path: Path, target_fpr: float) -> dict[str, float] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fp:
        rows = [
            {
                "tau": float(row["tau"]),
                "fpr": float(row["fpr"]),
                "fnr": float(row["fnr"]),
            }
            for row in csv.DictReader(fp)
        ]
    if not rows:
        return None
    eligible = [row for row in rows if row["fpr"] <= target_fpr]
    if eligible:
        chosen = min(eligible, key=lambda row: (row["fnr"], -row["fpr"], row["tau"]))
        attained = 1
    else:
        chosen = min(rows, key=lambda row: (row["fpr"], row["fnr"], row["tau"]))
        attained = 0
    return {
        "mean_fpr": chosen["fpr"],
        "adult_fnr": chosen["fnr"],
        "mean_tau": chosen["tau"],
        "folds_attained_target_fpr": attained,
    }


def read_operating_points(run_dir: Path, target_fpr: float) -> dict[str, float] | None:
    rows = []
    for fold_idx in range(5):
        op = read_operating_point(run_dir / f"fold_{fold_idx}" / "test_age_gate_metrics_n1_ddp.csv", target_fpr)
        if op is None:
            return None
        rows.append(op)
    return {
        "mean_fpr": mean(row["mean_fpr"] for row in rows),
        "adult_fnr": mean(row["adult_fnr"] for row in rows),
        "mean_tau": mean(row["mean_tau"] for row in rows),
        "folds_attained_target_fpr": sum(row["folds_attained_target_fpr"] for row in rows),
    }


def npz_sample_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            for key in ("y_true", "targets", "labels"):
                if key in data:
                    return int(len(data[key]))
    except Exception:  # noqa: BLE001
        return None
    return None


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def load_split_user_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        users = data.get("test_user_ids", data.get("test_users", []))
    else:
        users = data
    return len(users) if isinstance(users, list) else None


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_md = Path(args.out_md)
    split_users = load_split_user_count(Path("splits/archive_users_all.json"))

    seed_rows = []
    fold_rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = run_root / f"{arm}_seed{seed}"
            summary = read_kfold_summary(run_dir)
            op = read_operating_points(run_dir, args.target_fpr)
            if summary is not None and op is not None:
                seed_rows.append({"seed": seed, "arm": arm, **summary, **op})
            for fold_idx in range(5):
                fold_dir = run_dir / f"fold_{fold_idx}"
                fold_summary = read_one_row_csv(fold_dir / "test_summary_ddp.csv")
                if fold_summary is None:
                    continue
                fold_op = read_operating_point(fold_dir / "test_age_gate_metrics_n1_ddp.csv", args.target_fpr)
                fold_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "fold": fold_idx,
                        "mae": float(fold_summary["mae"]),
                        "rmse": float(fold_summary["rmse"]),
                        "auc_adult_gate": float(fold_summary["auc_adult_gate"]),
                        "samples": npz_sample_count(fold_dir / "test_predictions_raw_ddp.npz"),
                        **(fold_op or {}),
                    }
                )

    raw_count = len(list(run_root.rglob("test_predictions_raw_ddp.npz"))) if run_root.is_dir() else 0
    summary_count = len(list(run_root.rglob("test_summary_ddp.csv"))) if run_root.is_dir() else 0
    aggregate_count = len(seed_rows)

    partial_csv = run_root / "archive_partial_fold_metrics.csv"
    seed_csv = run_root / "archive_completed_seed_metrics.csv"
    run_root.mkdir(parents=True, exist_ok=True)
    if fold_rows:
        with partial_csv.open("w", newline="", encoding="utf-8") as fp:
            fieldnames = [
                "seed",
                "arm",
                "fold",
                "samples",
                "mae",
                "rmse",
                "auc_adult_gate",
                "mean_fpr",
                "adult_fnr",
                "mean_tau",
                "folds_attained_target_fpr",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(fold_rows)
    if seed_rows:
        with seed_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(seed_rows)

    status = (
        "complete; corrected masked local inference finished successfully."
        if summary_count >= 70 and aggregate_count >= 14
        else "corrected masked local inference is running; this file is updated from completed outputs only."
    )

    lines = [
        "# Q5 Archive Evaluation Results",
        "",
        f"Last updated: {datetime.now().astimezone().replace(microsecond=0).isoformat()}",
        "",
        f"Status: {status}",
        "",
        "Archive-only inference for the Q5 paired fine-tuned model set: `R0` real-from-real fine-tuned models versus `Pooled` fine-tuned models, seeds 42-48, five folds per seed.",
        "",
        "## Corrected Evaluation Definition",
        "",
        "- Dataset: archive only.",
        "- Split: `splits/archive_users_all.json`.",
        f"- Users: {split_users if split_users is not None else 99} archive users.",
        "- Samples: 1,979 dorsal, known-age archive images.",
        "- Side: dorsal only.",
        "- Crop: square crop from parsed archive `bbox`.",
        "- Masks: enabled with `--use-masks`.",
        "- Mask availability: 1,684 / 1,979 eval images resolve an archive mask.",
        "- Mask fallback: the 295 images without masks are evaluated as unmasked square bbox crops.",
        "- Image transform: masked/cropped image resized to 384 x 384, converted to tensor, ImageNet-normalised.",
        "- Model: EfficientNet-V2-S, `--embed-dim 128`, checkpoint-only loading via `--no-imagenet-pretrained`.",
        "- Aggregation: five-fold unweighted `n=1`.",
        f"- Run root: `{run_root}`.",
        "- Execution context: local CUDA.",
        f"- Adult-gate operating point: best fold-level threshold with FPR <= {100 * args.target_fpr:.1f}%; if unavailable, lowest-FPR threshold in the 10-30 age-threshold sweep.",
        "",
        "## Progress",
        "",
        f"- Completed fold summaries: {summary_count} / 70.",
        f"- Completed raw prediction files: {raw_count} / 70.",
        f"- Completed five-fold seed aggregates: {aggregate_count} / 14.",
        "",
        "## Visual QA",
        "",
        "`runs/archive_preprocessing_visual_qa/contact_sheet.png`",
        "",
        "Visual inspection showed aligned masks, blacked-out backgrounds, square crops centred on the archive bbox, and unmasked bbox crops for missing-mask examples.",
        "",
        "Programmatic QA: bbox availability 1,979 / 1,979; mask paths resolved 1,684 / 1,979; missing masks 295 / 1,979; unreadable masks 0; image/mask size mismatches 0; non-binary masks 0; full-image almost-white masks >=95% white 0; full-image almost-black masks <=1% white 0.",
        "",
        "Crop-level white-mask review candidates: 2 resolved masks have >=95% white pixels inside the square bbox crop, both from `archive_51` age 32: `51_32_2_5.jpg` and `51_32_2_6.jpg`. These are not full-image white masks, but the crop area is entirely foreground.",
        "",
        "## Completed Seed Aggregates",
        "",
    ]
    if seed_rows:
        lines.extend([
            "| Seed | Arm | Folds | Samples/fold | MAE | RMSE | Adult-gate AUC | Mean FPR | Adult FNR | Mean tau | Folds <= target FPR |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in seed_rows:
            lines.append(
                f"| {row['seed']} | `{row['arm']}` | {int(row['folds'])} | {row['samples_per_fold']:.0f} | "
                f"{fmt_num(row['mae'])} | {fmt_num(row['rmse'])} | {fmt_num(row['auc_adult_gate'], 4)} | "
                f"{fmt_pct(row['mean_fpr'])} | {fmt_pct(row['adult_fnr'])} | {fmt_num(row['mean_tau'], 2)} | "
                f"{int(row['folds_attained_target_fpr'])}/5 |"
            )
    else:
        lines.append("No five-fold seed aggregates have completed yet.")

    lines.extend([
        "",
        "## Completed Fold Results",
        "",
    ])
    if fold_rows:
        lines.extend([
            "| Seed | Arm | Fold | Samples | MAE | RMSE | Adult-gate AUC | FPR | Adult FNR | Tau |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in fold_rows:
            lines.append(
                f"| {row['seed']} | `{row['arm']}` | {row['fold']} | {row['samples'] if row['samples'] is not None else 'NA'} | "
                f"{fmt_num(row['mae'])} | {fmt_num(row['rmse'])} | {fmt_num(row['auc_adult_gate'], 4)} | "
                f"{fmt_pct(row.get('mean_fpr'))} | {fmt_pct(row.get('adult_fnr'))} | {fmt_num(row.get('mean_tau'), 2)} |"
            )
    else:
        lines.append("No fold summaries have completed yet.")

    lines.extend([
        "",
        "## Output Files",
        "",
        f"- Partial fold metrics CSV: `{partial_csv}`" if fold_rows else "- Partial fold metrics CSV: pending.",
        f"- Completed seed metrics CSV: `{seed_csv}`" if seed_rows else "- Completed seed metrics CSV: pending.",
        f"- Final all-seed metrics CSV: `{run_root / 'archive_seed_metrics.csv'}`",
        f"- Final paired comparison CSV: `{run_root / 'archive_paired_comparison.csv'}`",
    ])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md} with {summary_count}/70 folds and {aggregate_count}/14 seed aggregates")


if __name__ == "__main__":
    main()
