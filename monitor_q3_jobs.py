#!/usr/bin/env python3
"""Monitor Q3 Slurm jobs and write a compact status report."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Q3 Slurm jobs from a JSON manifest.")
    parser.add_argument("--jobs-file", required=True, help="JSON file with a 'jobs' list.")
    parser.add_argument("--out-dir", required=True, help="Directory for status JSON/Markdown outputs.")
    return parser.parse_args()


def run_sacct(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    cmd = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,JobName,State,Elapsed,Start,End,ExitCode",
        "-P",
        "-n",
        "-X",
    ]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rows: dict[str, dict[str, str]] = {}
    if proc.returncode != 0:
        return rows
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 7:
            continue
        job_id, name, state, elapsed, start, end, exit_code = parts
        rows[job_id] = {
            "job_name": name,
            "state": state,
            "elapsed": elapsed,
            "start": start,
            "end": end,
            "exit_code": exit_code,
        }
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_result(run_dir: Path) -> dict[str, float] | None:
    summary = run_dir / "kfold_test_summary_n1.csv"
    if not summary.is_file():
        return None
    rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
    if not rows:
        return None
    result = {
        "mae": _mean([float(row["mae"]) for row in rows if row.get("mae")]),
        "rmse": _mean([float(row["rmse"]) for row in rows if row.get("rmse")]),
        "auc_adult_gate": _mean(
            [float(row["auc_adult_gate"]) for row in rows if row.get("auc_adult_gate")]
        ),
    }
    fprs: list[float] = []
    fnrs: list[float] = []
    fallback_folds = 0
    for fold_idx in range(len(rows)):
        gate_path = run_dir / f"fold_{fold_idx}" / "test_age_gate_metrics_n1_ddp.csv"
        if not gate_path.is_file():
            continue
        gate_rows = list(csv.DictReader(gate_path.open(newline="", encoding="utf-8")))
        if not gate_rows:
            continue
        numeric = [
            {key: (float(value) if key != "gate" else value) for key, value in row.items()}
            for row in gate_rows
        ]
        under = [row for row in numeric if row["fpr"] <= 0.05]
        if under:
            best = min(under, key=lambda row: (row["fnr"], -row["fpr"]))
        else:
            best = min(numeric, key=lambda row: row["fpr"])
            fallback_folds += 1
        fprs.append(best["fpr"])
        fnrs.append(best["fnr"])
    if fprs:
        result["mean_fpr"] = _mean(fprs)
        result["adult_fnr"] = _mean(fnrs)
        result["fallback_folds"] = float(fallback_folds)
    return {key: value for key, value in result.items() if value is not None}


def fold_progress(run_dir: Path, k: int = 5) -> dict[str, object]:
    started: list[int] = []
    completed: list[int] = []
    for fold_idx in range(k):
        fold_dir = run_dir / f"fold_{fold_idx}"
        if fold_dir.is_dir():
            started.append(fold_idx)
        if (
            (fold_dir / "test_predictions_n1_ddp.npz").is_file()
            or (fold_dir / "test_group_summary_n1_ddp.csv").is_file()
        ):
            completed.append(fold_idx)
    return {
        "started_folds": started,
        "completed_folds": completed,
        "started_fold_count": len(started),
        "completed_fold_count": len(completed),
        "total_folds": k,
    }


def format_metric(value: float | None, *, percent: bool = False, digits: int = 3) -> str:
    if value is None:
        return ""
    if percent:
        return f"{value * 100:.2f}%"
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    jobs_path = Path(args.jobs_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    job_ids = [str(job["job_id"]) for job in jobs]
    sacct = run_sacct(job_ids)

    status_rows = []
    for job in jobs:
        job_id = str(job["job_id"])
        row = dict(job)
        row.update(sacct.get(job_id, {}))
        run_dir = Path(row["run_dir"])
        row["fold_progress"] = fold_progress(run_dir)
        result = load_result(run_dir)
        if result:
            row["result"] = result
        status_rows.append(row)

    status_payload = {
        "updated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "source_jobs_file": str(jobs_path),
        "jobs": status_rows,
    }
    (out_dir / "q3_status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    lines = [
        "# Q3 Monitor Status",
        "",
        f"Updated: {status_payload['updated_at']}",
        "",
        "| Job | Arm | Fraction | State | Elapsed | Folds Done | MAE | RMSE | AUC | Mean FPR | Adult FNR |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in status_rows:
        result = row.get("result", {})
        folds = row.get("fold_progress", {})
        fold_text = f"{folds.get('completed_fold_count', 0)}/{folds.get('total_folds', 5)}"
        lines.append(
            "| {job_id} | {arm} | {fraction} | {state} | {elapsed} | {folds} | {mae} | {rmse} | {auc} | {fpr} | {fnr} |".format(
                job_id=row.get("job_id", ""),
                arm=row.get("arm", ""),
                fraction=row.get("fraction", ""),
                state=row.get("state", "UNKNOWN"),
                elapsed=row.get("elapsed", ""),
                folds=fold_text,
                mae=format_metric(result.get("mae")),
                rmse=format_metric(result.get("rmse")),
                auc=format_metric(result.get("auc_adult_gate"), digits=4),
                fpr=format_metric(result.get("mean_fpr"), percent=True),
                fnr=format_metric(result.get("adult_fnr"), percent=True),
            )
        )
    (out_dir / "q3_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
