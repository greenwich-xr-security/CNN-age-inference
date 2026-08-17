#!/usr/bin/env python3
"""Run Q5 R0/Pooled archive inference locally with the repo Python environment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


Q5_RUNS: list[tuple[str, int, str]] = [
    ("R0", 42, "q3_r0_from1050753_real_finetune_lr2e5_seed42_v2s_384"),
    ("Pooled", 42, "v2s_hrgbd_real_finetuning_nll_lr2e5_uncapped_test20_k5_4gpu_16cpu_20260730_384"),
]
for seed in range(43, 49):
    Q5_RUNS.append(("R0", seed, f"minpair_seed{seed}_real_from_real_lr2e5_v2s_384"))
    Q5_RUNS.append(("Pooled", seed, f"minpair_seed{seed}_real_from_syn2_lr2e5_v2s_384"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q5 archive evaluation locally.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets",
        help="Local dataset root.",
    )
    parser.add_argument(
        "--test-split-file",
        type=str,
        default="splits/archive_users_all.json",
        help="Archive-only test split JSON.",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/q5_archive_eval_local_20260813",
        help="Output root for local archive evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Disable dataset masks. By default archive evaluation uses masks when available.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of arm/seed runs to execute, useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a fold when its raw prediction output already exists.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def update_partial_report(repo_root: Path, run_root: Path) -> None:
    run_command([
        sys.executable,
        str(repo_root / "update_q5_archive_eval_results_partial.py"),
        "--run-root",
        str(run_root),
        "--out-md",
        str(repo_root / "Q5_ARCHIVE_EVAL_RESULTS.md"),
    ])


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_root = repo_root / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    selected = Q5_RUNS[args.start_index:]
    if args.limit is not None:
        selected = selected[: args.limit]

    py = sys.executable
    for arm, seed, source_run in selected:
        dest_run = run_root / f"{arm}_seed{seed}"
        dest_run.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {arm} seed {seed} from {source_run} ===", flush=True)

        for fold_idx in range(5):
            checkpoint = repo_root / "runs" / source_run / f"fold_{fold_idx}" / "v2_s_age_regressor_ddp.pth"
            out_dir = dest_run / f"fold_{fold_idx}"
            expected = out_dir / "test_predictions_raw_ddp.npz"
            if args.skip_existing and expected.is_file():
                print(f"Skipping existing {expected}", flush=True)
                continue
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            cmd = [
                py,
                str(repo_root / "evaluate_test.py"),
                "--checkpoint",
                str(checkpoint),
                "--test-users-file",
                str(repo_root / args.test_split_file),
                "--model",
                "v2_s",
                "--img-size",
                "384",
                "--embed-dim",
                "128",
                "--no-imagenet-pretrained",
                "--data-root",
                args.data_root,
                "--max-samples-per-user",
                "0",
                "--max-samples-per-age-bin",
                "0",
                "--include-archive",
            ]
            if not args.no_masks:
                cmd.append("--use-masks")
            cmd.extend([
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--eval-aggregation-sizes",
                "1",
                "--eval-aggregation-seed",
                "42",
                "--eval-age-gate-mode",
                "age_threshold",
                "--age-gate-threshold-min",
                "10",
                "--age-gate-threshold-max",
                "30",
                "--ddp-output-names",
                "--output-dir",
                str(out_dir),
            ])
            run_command(cmd)
            update_partial_report(repo_root, run_root)

        run_command([
            py,
            str(repo_root / "aggregate_kfold.py"),
            "--kfold-root",
            str(dest_run),
            "--output-dir",
            str(dest_run),
            "--group-sizes",
            "1",
            "--eval-age-gate-mode",
            "age_threshold",
            "--age-gate-threshold-min",
            "10",
            "--age-gate-threshold-max",
            "30",
            "--skip-val",
        ])
        update_partial_report(repo_root, run_root)

    ran_full_set = args.start_index == 0 and args.limit is None
    if ran_full_set:
        run_command([
            py,
            str(repo_root / "summarize_q5_archive_eval.py"),
            "--run-root",
            str(run_root),
            "--out-md",
            str(repo_root / "Q5_ARCHIVE_EVAL_RESULTS.md"),
        ])
    else:
        print("\nPartial run complete; skipping final all-seed summary.", flush=True)


if __name__ == "__main__":
    main()
