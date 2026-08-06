"""Summarize Q3 fold-level MAE/AUC variability from evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Q3Cell:
    arm: str
    fraction: str
    job: str
    run_dir: str


TAGS = {
    "5%": "f05",
    "10%": "f10",
    "25%": "f25",
    "50%": "f50",
    "100%": "f100",
}


def build_cells() -> list[Q3Cell]:
    cells: list[Q3Cell] = []
    specs = [
        (
            "R0",
            ["1050948", "1050949", "1050950", "1050951", "1050952"],
            "q3_r0_realfrac_{tag}_seed42_v2s_384",
        ),
        (
            "R1",
            ["1050953", "1050954", "1050955", "1050956", "1050957"],
            "q3_r1_realfrac_{tag}_seed42_v2s_384",
        ),
        (
            "S-age LR-control",
            ["1050979", "1050980", "1050981", "1050982", "1050975"],
            "q3_sage_realfrac_{tag}_lr2e5_seed42_v2s_384",
        ),
        (
            "S-ssl LR-control",
            ["1050983", "1050984", "1050985", "1050986", "1050987"],
            "q3_sssl_realfrac_{tag}_lr2e5_seed42_v2s_384_embed128_2gpu_standard",
        ),
        (
            "S-shuffle LR-control",
            ["1051003", "1051004", "1051005", "1051006", "1051007"],
            "q3_sshuffle_realfrac_{tag}_lr2e5_seed42_v2s_384",
        ),
        (
            "U-ssl LR-control",
            ["1051107", "1051108", "1051109", "1051110", "1051111"],
            "q3_ussl8_realfrac_{tag}_lr2e5_seed42_v2s_384",
        ),
    ]
    fractions = list(TAGS)
    for arm, jobs, template in specs:
        for fraction, job in zip(fractions, jobs):
            tag = TAGS[fraction]
            cells.append(Q3Cell(arm, fraction, job, template.format(tag=tag)))
    return cells


def read_fold_summary(path: Path) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    for row in rows:
        if row.get("group_size") == "1":
            return float(row["mae"]), float(row["auc_adult_gate"])
    raise ValueError(f"No group_size=1 row found in {path}")


def summarize_cell(root: Path, cell: Q3Cell) -> dict[str, object]:
    fold_paths = sorted((root / cell.run_dir).glob("fold_*/test_summary_ddp.csv"))
    if len(fold_paths) != 5:
        raise FileNotFoundError(
            f"Expected 5 fold summaries for {cell.arm} {cell.fraction} at "
            f"{root / cell.run_dir}, found {len(fold_paths)}"
        )

    maes: list[float] = []
    aucs: list[float] = []
    for path in fold_paths:
        mae, auc = read_fold_summary(path)
        maes.append(mae)
        aucs.append(auc)

    return {
        "arm": cell.arm,
        "fraction": cell.fraction,
        "job": cell.job,
        "folds": len(fold_paths),
        "mae_mean": statistics.mean(maes),
        "mae_std": statistics.stdev(maes),
        "auc_mean": statistics.mean(aucs),
        "auc_std": statistics.stdev(aucs),
    }


def fmt_mean_std(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def emit_markdown(rows: list[dict[str, object]]) -> None:
    print("| Arm | Real-label fraction | Job | Folds | MAE mean ± std | Adult-gate AUC mean ± std |")
    print("| --- | ---: | --- | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {arm} | {fraction} | `{job}` | {folds} | {mae} | {auc} |".format(
                arm=row["arm"],
                fraction=row["fraction"],
                job=row["job"],
                folds=row["folds"],
                mae=fmt_mean_std(float(row["mae_mean"]), float(row["mae_std"]), 3),
                auc=fmt_mean_std(float(row["auc_mean"]), float(row["auc_std"]), 4),
            )
        )


def emit_csv(rows: list[dict[str, object]]) -> None:
    import sys

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "arm",
            "fraction",
            "job",
            "folds",
            "mae_mean",
            "mae_std",
            "auc_mean",
            "auc_std",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Q3 fold_*/test_summary_ddp.csv files and compute mean/std "
            "for image-level MAE and adult-gate AUC."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs"),
        help="Root containing copied HPC run directories (default: runs).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize_cell(args.root, cell) for cell in build_cells()]
    if args.format == "csv":
        emit_csv(rows)
    else:
        emit_markdown(rows)


if __name__ == "__main__":
    main()
