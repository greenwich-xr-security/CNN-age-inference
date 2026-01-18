#!/usr/bin/env python3
"""Inspect Optuna study results from a local storage backend."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import optuna
from optuna.trial import TrialState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore Optuna study results.")
    parser.add_argument("--storage", required=True, help="Optuna storage URL (e.g. sqlite:///path.db).")
    parser.add_argument("--study-name", required=True, help="Optuna study name.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top trials to show (default: 10).")
    parser.add_argument(
        "--sort-by",
        choices=["auc", "mae"],
        default="auc",
        help="Sort criterion for the top list (default: auc).",
    )
    parser.add_argument("--out-csv", type=str, default=None, help="Optional CSV path to export all trials.")
    parser.add_argument("--show-pareto", action="store_true", help="Print Pareto front trials.")
    return parser.parse_args()


def format_params(params: dict) -> str:
    if not params:
        return ""
    pairs = [f"{k}={v}" for k, v in params.items()]
    return " ".join(pairs)


def collect_rows(trials: list[optuna.trial.FrozenTrial]) -> list[dict]:
    rows = []
    for trial in trials:
        values = list(trial.values) if trial.values is not None else []
        auc = float(values[0]) if len(values) > 0 else float("nan")
        mae = float(values[1]) if len(values) > 1 else float("nan")
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "auc": auc,
                "mae": mae,
                "params": format_params(trial.params),
                "output_dir": trial.user_attrs.get("output_dir", ""),
                "intra_user_std": trial.user_attrs.get("avg_intra_user_std", ""),
            }
        )
    return rows


def print_table(title: str, rows: list[dict], top_n: int) -> None:
    print(title)
    if not rows:
        print("  (no trials)")
        return
    rows = rows[:top_n]
    header = ["trial", "auc", "mae", "params", "output_dir"]
    widths = {
        "trial": max(len(header[0]), max(len(str(r["number"])) for r in rows)),
        "auc": max(len(header[1]), 8),
        "mae": max(len(header[2]), 8),
        "params": max(len(header[3]), max(len(r["params"]) for r in rows)),
        "output_dir": max(len(header[4]), max(len(r["output_dir"]) for r in rows)),
    }
    line = (
        f"{header[0]:<{widths['trial']}}  "
        f"{header[1]:<{widths['auc']}}  "
        f"{header[2]:<{widths['mae']}}  "
        f"{header[3]:<{widths['params']}}  "
        f"{header[4]:<{widths['output_dir']}}"
    )
    print(line)
    print("-" * len(line))
    for row in rows:
        print(
            f"{row['number']:<{widths['trial']}}  "
            f"{row['auc']:<{widths['auc']}.6f}  "
            f"{row['mae']:<{widths['mae']}.6f}  "
            f"{row['params']:<{widths['params']}}  "
            f"{row['output_dir']:<{widths['output_dir']}}"
        )


def export_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "number",
                "state",
                "auc",
                "mae",
                "params",
                "output_dir",
                "intra_user_std",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    rows = collect_rows(completed)

    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {len(study.trials)} total | {len(completed)} complete\n")

    if completed:
        best_by_auc = max(rows, key=lambda r: r["auc"])
        best_by_mae = min(rows, key=lambda r: r["mae"])
        print(f"Best AUC: trial {best_by_auc['number']} | auc={best_by_auc['auc']:.6f} | mae={best_by_auc['mae']:.6f}")
        print(f"Best MAE: trial {best_by_mae['number']} | auc={best_by_mae['auc']:.6f} | mae={best_by_mae['mae']:.6f}\n")

    if args.sort_by == "auc":
        rows_sorted = sorted(rows, key=lambda r: r["auc"], reverse=True)
        print_table(f"Top {args.top_n} by AUC:", rows_sorted, args.top_n)
    else:
        rows_sorted = sorted(rows, key=lambda r: r["mae"])
        print_table(f"Top {args.top_n} by MAE:", rows_sorted, args.top_n)

    if args.show_pareto:
        pareto = collect_rows(study.best_trials)
        pareto_sorted = sorted(pareto, key=lambda r: r["auc"], reverse=True)
        print("")
        print_table("Pareto front (sorted by AUC):", pareto_sorted, len(pareto_sorted))

    if args.out_csv:
        export_csv(Path(args.out_csv), rows)
        print(f"\nSaved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
