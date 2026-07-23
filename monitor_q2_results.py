"""Write a live per-fold MAE/AUC table for the Question 2 main matrix."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("rr", "ss", "sr", "rs")


def _read_test_metrics(path: Path) -> tuple[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    row = next((item for item in rows if item.get("group_size") == "1"), rows[0])
    return row.get("mae", ""), row.get("auc_adult_gate", "")


def _fold_status(fold_dir: Path) -> str:
    if (fold_dir / "test_summary_ddp.csv").is_file():
        return "completed"
    if (fold_dir / "history_distributed.log").is_file() or (fold_dir / "config.txt").is_file():
        return "running"
    return "pending"


def build_rows(runs_root: Path, folds: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for condition in CONDITIONS:
        candidates = sorted(runs_root.glob(f"q2_main_{condition}_*"))
        run_dir = candidates[-1] if candidates else None
        for fold_index in range(folds):
            fold_dir = run_dir / f"fold_{fold_index}" if run_dir else Path()
            metrics = _read_test_metrics(fold_dir / "test_summary_ddp.csv") if run_dir else None
            rows.append(
                {
                    "condition": condition.upper(),
                    "fold": fold_index,
                    "status": "completed" if metrics else (_fold_status(fold_dir) if run_dir else "pending"),
                    "mae": metrics[0] if metrics else "",
                    "auc_adult_gate": metrics[1] if metrics else "",
                    "run_directory": str(run_dir) if run_dir else "",
                }
            )
    return rows


def write_tables(rows: list[dict[str, str | int]], csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "fold", "status", "mae", "auc_adult_gate", "run_directory"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    updated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "# Question 2 main-matrix monitoring",
        "",
        f"Last updated (UTC): `{updated}`",
        "",
        "| Condition | Fold | Status | Test MAE | Adult-gate AUC |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        mae = f"{float(row['mae']):.4f}" if row["mae"] else "—"
        auc = f"{float(row['auc_adult_gate']):.4f}" if row["auc_adult_gate"] else "—"
        lines.append(f"| {row['condition']} | {row['fold']} | {row['status']} | {mae} | {auc} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=Path("runs/q2_main_results.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("runs/q2_main_results.md"))
    args = parser.parse_args()
    write_tables(build_rows(args.runs_root, args.folds), args.csv, args.markdown)


if __name__ == "__main__":
    main()
