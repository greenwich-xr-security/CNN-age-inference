#!/usr/bin/env python3
"""Periodically refresh the partial Q5 archive markdown report."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a Q5 archive eval run and refresh partial results.")
    parser.add_argument("--run-root", type=str, default="runs/q5_archive_eval_local_masked_20260813")
    parser.add_argument("--out-md", type=str, default="Q5_ARCHIVE_EVAL_RESULTS.md")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--expected-folds", type=int, default=70)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument(
        "--log-file",
        type=str,
        default="runs/q5_archive_eval_local_masked_20260813/partial_report_watcher.log",
    )
    return parser.parse_args()


def count_outputs(run_root: Path) -> tuple[int, int]:
    if not run_root.is_dir():
        return 0, 0
    summaries = len(list(run_root.rglob("test_summary_ddp.csv")))
    aggregates = len(list(run_root.rglob("kfold_test_summary_n1.csv")))
    return summaries, aggregates


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = repo_root / run_root
    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = repo_root / out_md
    log_file = Path(args.log_file)
    if not log_file.is_absolute():
        log_file = repo_root / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    updater = repo_root / "update_q5_archive_eval_results_partial.py"
    deadline = time.time() + args.max_hours * 3600.0
    while True:
        cmd = [
            sys.executable,
            str(updater),
            "--run-root",
            str(run_root),
            "--out-md",
            str(out_md),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        summaries, aggregates = count_outputs(run_root)
        stamp = datetime.now().astimezone().replace(microsecond=0).isoformat()
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(f"[{stamp}] summaries={summaries}/{args.expected_folds} aggregates={aggregates}/14 rc={completed.returncode}\n")
            if completed.stdout:
                fp.write(completed.stdout)
            if completed.stderr:
                fp.write(completed.stderr)
        if summaries >= args.expected_folds or time.time() >= deadline:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
