#!/usr/bin/env python3
"""Run Optuna trials by launching submit_distributed.slurm."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna worker for k-fold HPO.")
    parser.add_argument("--study-name", required=True, help="Optuna study name.")
    parser.add_argument("--storage", required=True, help="Optuna storage URL (e.g. sqlite:///path.db).")
    parser.add_argument("--n-trials", type=int, default=1, help="Number of trials to run in this worker.")
    parser.add_argument("--base-output", type=str, default="runs/optuna", help="Base output directory for runs.")
    parser.add_argument("--submit-script", type=str, default="submit_distributed.slurm", help="Training script to run.")
    parser.add_argument("--model", type=str, default="b2", help="Model name for training runs.")
    parser.add_argument("--seed", type=int, default=32, help="Seed for training and folds.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--img-size", type=int, default=None, help="Override image size for training runs.")
    parser.add_argument("--tune-img-size", action="store_true", help="Tune image size.")
    parser.add_argument("--img-size-min", type=int, default=224, help="Min image size when tuning.")
    parser.add_argument("--img-size-max", type=int, default=600, help="Max image size when tuning.")
    parser.add_argument("--img-size-step", type=int, default=1, help="Step for image size when tuning.")
    parser.add_argument("--user-group-size", type=int, default=2, help="User group size for training.")
    parser.add_argument("--epochs", type=int, default=240, help="Training epochs.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of folds to run.")
    parser.add_argument("--agg-size", type=int, default=1, help="Aggregation group size to score.")
    parser.add_argument(
        "--auc-case",
        type=str,
        default="mean",
        choices=["case1", "case2", "mean"],
        help="Which ROC AUC to optimize (default: mean of case1/case2).",
    )
    parser.add_argument("--tune-lr", action="store_true", help="Tune learning rate.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Fixed learning rate when --tune-lr is off.")
    parser.add_argument("--lr-min", type=float, default=1e-4, help="Min LR when tuning.")
    parser.add_argument("--lr-max", type=float, default=1e-3, help="Max LR when tuning.")
    parser.add_argument("--tune-weight-decay", action="store_true", help="Tune weight decay.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Fixed weight decay when --tune-weight-decay is off.")
    parser.add_argument("--weight-decay-min", type=float, default=0.0, help="Min weight decay when tuning.")
    parser.add_argument("--weight-decay-max", type=float, default=0.1, help="Max weight decay when tuning.")
    parser.add_argument("--nll-min", type=float, default=0.1, help="Min NLL weight.")
    parser.add_argument("--nll-max", type=float, default=1.0, help="Max NLL weight.")
    parser.add_argument("--mse-min", type=float, default=0.0, help="Min MSE weight.")
    parser.add_argument("--mse-max", type=float, default=1.0, help="Max MSE weight.")
    parser.add_argument("--mae-min", type=float, default=0.0, help="Min MAE weight.")
    parser.add_argument("--mae-max", type=float, default=1.0, help="Max MAE weight.")
    parser.add_argument("--spread-min", type=float, default=0.0, help="Min spread weight.")
    parser.add_argument("--spread-max", type=float, default=1.0, help="Max spread weight.")
    return parser.parse_args()


def format_float(value: float) -> str:
    text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")


def folder_name(
    *,
    model: str,
    batch_size: int,
    seed: int,
    nll: str,
    mse: str,
    mae: str,
    spread: str,
    user_group_size: int,
    run_suffix: str | None,
) -> str:
    base = (
        f"{model}_bs{batch_size}_seed{seed}_nll{nll}_mse{mse}_mae{mae}"
        f"_spread{spread}_ug{user_group_size}"
    )
    if run_suffix:
        return f"{base}_{run_suffix}"
    return base


def read_kfold_metrics(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in metrics file: {path}")

    def mean(values: list[float]) -> float:
        return sum(values) / float(len(values))

    mae_vals = [float(row["mae"]) for row in rows]
    auc1_vals = [float(row["auc_case1"]) for row in rows]
    auc2_vals = [float(row["auc_case2"]) for row in rows]
    intra_vals = [
        float(row.get("intra_user_std_mean", "nan"))
        for row in rows
        if row.get("intra_user_std_mean") not in (None, "")
    ]
    avg_mae = mean(mae_vals)
    avg_auc1 = mean(auc1_vals)
    avg_auc2 = mean(auc2_vals)
    avg_intra = mean(intra_vals) if intra_vals else float("nan")
    return {
        "avg_mae": avg_mae,
        "avg_auc_case1": avg_auc1,
        "avg_auc_case2": avg_auc2,
        "avg_intra_user_std": avg_intra,
    }


def main() -> None:
    args = parse_args()
    base_output = Path(args.base_output).resolve()
    base_output.mkdir(parents=True, exist_ok=True)
    submit_script = Path(args.submit_script).resolve()
    repo_root = Path(__file__).resolve().parent

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial):
        nll = format_float(trial.suggest_float("loss_weight_nll", args.nll_min, args.nll_max))
        mse = format_float(trial.suggest_float("loss_weight_mse", args.mse_min, args.mse_max))
        mae = format_float(trial.suggest_float("loss_weight_mae", args.mae_min, args.mae_max))
        spread = format_float(trial.suggest_float("loss_weight_spread", args.spread_min, args.spread_max))
        if args.tune_lr:
            lr = format_float(trial.suggest_float("lr", args.lr_min, args.lr_max, log=True))
        else:
            lr = format_float(args.lr)
        if args.tune_weight_decay:
            weight_decay = format_float(
                trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max)
            )
        else:
            weight_decay = format_float(args.weight_decay)

        run_suffix = f"optuna_t{trial.number}"
        folder = folder_name(
            model=args.model,
            batch_size=args.batch_size,
            seed=args.seed,
            nll=nll,
            mse=mse,
            mae=mae,
            spread=spread,
            user_group_size=args.user_group_size,
            run_suffix=run_suffix,
        )
        output_root = base_output
        config_root = output_root / folder
        metrics_path = config_root / f"kfold_summary_n{args.agg_size}.csv"

        if args.tune_img_size:
            if args.img_size_step < 1:
                raise ValueError("img-size-step must be >= 1.")
            img_size = int(
                trial.suggest_int(
                    "img_size",
                    args.img_size_min,
                    args.img_size_max,
                    step=args.img_size_step,
                )
            )
        else:
            img_size = args.img_size

        env = os.environ.copy()
        env.update(
            {
                "PROJECT_ROOT": str(repo_root),
                "OUTPUT_ROOT": str(output_root),
                "MODELS": args.model,
                "SEEDS": str(args.seed),
                "SEED": str(args.seed),
                "BATCH_SIZE": str(args.batch_size),
                **({"IMG_SIZE": str(img_size)} if img_size is not None else {}),
                "USER_GROUP_SIZES": str(args.user_group_size),
                "LOSS_WEIGHT_NLL": nll,
                "LOSS_WEIGHT_MSE": mse,
                "LOSS_WEIGHT_MAE": mae,
                "LOSS_WEIGHT_SPREAD": spread,
                "LR": lr,
                "WEIGHT_DECAY": weight_decay,
                "EPOCHS": str(args.epochs),
                "NUM_WORKERS": str(args.num_workers),
                "KFOLDS": str(args.kfolds),
                "RUN_KFOLD_AGG": "1",
                "RUN_SUFFIX": run_suffix,
                "AGG_SIZES": str(args.agg_size),
            }
        )

        subprocess.run(["bash", str(submit_script)], cwd=repo_root, env=env, check=True)

        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")

        metrics = read_kfold_metrics(metrics_path)
        if args.auc_case == "case1":
            avg_auc = metrics["avg_auc_case1"]
        elif args.auc_case == "case2":
            avg_auc = metrics["avg_auc_case2"]
        else:
            avg_auc = 0.5 * (metrics["avg_auc_case1"] + metrics["avg_auc_case2"])

        trial.set_user_attr("output_dir", str(config_root))
        trial.set_user_attr("avg_intra_user_std", metrics["avg_intra_user_std"])
        return avg_auc, metrics["avg_mae"]

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)


if __name__ == "__main__":
    main()
