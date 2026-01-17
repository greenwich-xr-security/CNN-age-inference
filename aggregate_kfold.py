#!/usr/bin/env python3
"""Aggregate k-fold evaluation artifacts into summary plots and tables."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from metrics import compute_age_gate_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate k-fold evaluation outputs.")
    parser.add_argument(
        "--kfold-root",
        type=str,
        required=True,
        help="Root directory containing fold_* subfolders.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subfolder to read within each fold directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write aggregated outputs (default: <kfold-root>/aggregate[/<run-name>]).",
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Aggregation group sizes to include (default: derived from files).",
    )
    parser.add_argument(
        "--age-threshold",
        type=float,
        default=18.0,
        help="Age threshold for adult/minor ROC calculations (default: 18).",
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=201,
        help="Number of thresholds for ROC curves (default: 201).",
    )
    return parser.parse_args()


def discover_folds(kfold_root: Path) -> list[Path]:
    fold_dirs = []
    for entry in kfold_root.iterdir():
        if entry.is_dir() and re.match(r"fold_\d+$", entry.name):
            fold_dirs.append(entry)
    fold_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
    return fold_dirs


def discover_group_sizes(run_dir: Path) -> list[int]:
    sizes = set()
    for path in run_dir.glob("val_predictions_n*_ddp.npz"):
        match = re.search(r"val_predictions_n(\d+)_ddp\.npz", path.name)
        if match:
            sizes.add(int(match.group(1)))
    return sorted(sizes)


def load_predictions(run_dir: Path, group_size: int) -> dict:
    path = run_dir / f"val_predictions_n{group_size}_ddp.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    data = np.load(path)
    return {
        "targets": data["targets"].astype(float),
        "pred_mean": data["pred_mean"].astype(float),
        "pred_log_var": data["pred_log_var"].astype(float),
    }


def load_raw_predictions(run_dir: Path) -> dict | None:
    path = run_dir / "val_predictions_raw_ddp.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "targets": data["targets"].astype(float),
        "pred_mean": data["pred_mean"].astype(float),
        "user_ids": data["user_ids"].astype(str),
    }


def compute_intra_user_variability(user_ids: np.ndarray, preds: np.ndarray) -> tuple[float, float, int]:
    buckets: dict[str, list[float]] = {}
    for uid, pred in zip(user_ids, preds):
        buckets.setdefault(str(uid), []).append(float(pred))

    stds = [float(np.std(vals)) for vals in buckets.values() if len(vals) > 1]
    if not stds:
        return float("nan"), float("nan"), 0
    return float(np.mean(stds)), float(np.median(stds)), len(stds)


def compute_age_errors(targets: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ages_int = targets.astype(int)
    unique_ages = np.unique(ages_int)
    unique_ages.sort()
    mae_vals = []
    rmse_vals = []
    for age in unique_ages:
        mask = ages_int == age
        if not np.any(mask):
            mae_vals.append(np.nan)
            rmse_vals.append(np.nan)
            continue
        abs_err = np.abs(preds[mask] - targets[mask])
        sq_err = (preds[mask] - targets[mask]) ** 2
        mae_vals.append(float(abs_err.mean()))
        rmse_vals.append(float(np.sqrt(sq_err.mean())))
    return unique_ages, np.asarray(mae_vals), np.asarray(rmse_vals)


def plot_roc_curves(curves: list[dict], mean_curve: dict, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for entry in curves:
        ax.plot(entry["fpr"], entry["tpr"], alpha=0.6, linewidth=1.2, label=entry["label"])
    ax.plot(
        mean_curve["fpr"],
        mean_curve["tpr"],
        color="black",
        linewidth=2.0,
        label=mean_curve["label"],
    )
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_age_error(
    ages: np.ndarray,
    mae_mean: np.ndarray,
    mae_std: np.ndarray,
    rmse_mean: np.ndarray,
    rmse_std: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values.copy()
        arr = np.asarray(values, dtype=float)
        n = arr.size
        half = window // 2
        out = np.empty_like(arr)
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window_vals = arr[start:end]
            if np.all(np.isnan(window_vals)):
                out[i] = np.nan
            else:
                out[i] = float(np.nanmean(window_vals))
        return out

    smooth_window = 10
    mae_smooth = rolling_mean(mae_mean, smooth_window)
    rmse_smooth = rolling_mean(rmse_mean, smooth_window)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ages, mae_mean, label="MAE", marker="o", linewidth=1.5)
    ax.plot(ages, rmse_mean, label="RMSE", marker="s", linewidth=1.5)
    ax.plot(ages, mae_smooth, label="MAE (smooth)", linestyle="--", linewidth=1.5)
    ax.plot(ages, rmse_smooth, label="RMSE (smooth)", linestyle="--", linewidth=1.5)
    ax.fill_between(ages, mae_mean - mae_std, mae_mean + mae_std, alpha=0.15)
    ax.fill_between(ages, rmse_mean - rmse_std, rmse_mean + rmse_std, alpha=0.15)
    ax.set_xlabel("Target Age")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    kfold_root = Path(args.kfold_root)
    fold_dirs = discover_folds(kfold_root)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {kfold_root}")

    run_dirs = []
    for fold_dir in fold_dirs:
        run_dir = fold_dir / args.run_name if args.run_name else fold_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")
        run_dirs.append(run_dir)

    if args.group_sizes:
        group_sizes = sorted({int(n) for n in args.group_sizes})
    else:
        group_sizes = discover_group_sizes(run_dirs[0])
    if not group_sizes:
        raise ValueError("No aggregation sizes found. Use --group-sizes to specify them.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = kfold_root / "aggregate"
        if args.run_name:
            output_dir = output_dir / args.run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    intra_user_stats = []
    for fold_dir, run_dir in zip(fold_dirs, run_dirs):
        raw_preds = load_raw_predictions(run_dir)
        if raw_preds is None:
            intra_user_stats.append(
                {
                    "fold": fold_dir.name,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "count": 0,
                }
            )
            continue
        mean_std, median_std, count = compute_intra_user_variability(
            raw_preds["user_ids"], raw_preds["pred_mean"]
        )
        intra_user_stats.append(
            {
                "fold": fold_dir.name,
                "mean": mean_std,
                "median": median_std,
                "count": count,
            }
        )

    intra_path = output_dir / "kfold_intra_user_variability.csv"
    with intra_path.open("w", encoding="utf-8") as fp:
        fp.write("fold,users_with_variability,intra_user_std_mean,intra_user_std_median\n")
        for row in intra_user_stats:
            fp.write(
                f"{row['fold']},{row['count']},{row['mean']:.6f},{row['median']:.6f}\n"
            )

    for group_size in group_sizes:
        fold_rows = []
        roc_case1 = []
        roc_case2 = []
        age_tables = []

        for fold_dir, run_dir in zip(fold_dirs, run_dirs):
            preds = load_predictions(run_dir, group_size)
            targets = preds["targets"]
            pred_mean = preds["pred_mean"]
            pred_log_var = preds["pred_log_var"]
            mae = float(np.mean(np.abs(pred_mean - targets)))
            rmse = float(np.sqrt(np.mean((pred_mean - targets) ** 2)))

            gate = compute_age_gate_curves(
                targets,
                pred_mean,
                pred_log_var,
                age_threshold=args.age_threshold,
                num_thresholds=args.num_thresholds,
            )
            fold_label = fold_dir.name
            roc_case1.append(
                {
                    "label": fold_label,
                    "fpr": gate["case1"]["fpr"],
                    "tpr": gate["case1"]["tpr"],
                    "auc": gate["case1"]["auc"],
                }
            )
            roc_case2.append(
                {
                    "label": fold_label,
                    "fpr": gate["case2"]["fpr"],
                    "tpr": gate["case2"]["tpr"],
                    "auc": gate["case2"]["auc"],
                }
            )

            ages, mae_vals, rmse_vals = compute_age_errors(targets, pred_mean)
            age_tables.append({"ages": ages, "mae": mae_vals, "rmse": rmse_vals})

            stats = intra_user_stats[len(fold_rows)] if intra_user_stats else None
            fold_rows.append(
                {
                    "fold": fold_label,
                    "group_size": group_size,
                    "mae": mae,
                    "rmse": rmse,
                    "auc_case1": gate["case1"]["auc"],
                    "auc_case2": gate["case2"]["auc"],
                    "samples": int(targets.size),
                    "intra_user_std_mean": (stats["mean"] if stats else float("nan")),
                    "intra_user_std_median": (stats["median"] if stats else float("nan")),
                    "users_with_variability": (stats["count"] if stats else 0),
                }
            )

        fold_rows_path = output_dir / f"kfold_summary_n{group_size}.csv"
        with fold_rows_path.open("w", encoding="utf-8") as fp:
            fp.write(
                "fold,group_size,mae,rmse,auc_case1,auc_case2,samples,"
                "intra_user_std_mean,intra_user_std_median,users_with_variability\n"
            )
            for row in fold_rows:
                fp.write(
                    f"{row['fold']},{row['group_size']},{row['mae']:.6f},{row['rmse']:.6f},"
                    f"{row['auc_case1']:.6f},{row['auc_case2']:.6f},{row['samples']},"
                    f"{row['intra_user_std_mean']:.6f},{row['intra_user_std_median']:.6f},"
                    f"{row['users_with_variability']}\n"
                )

        fpr_grid = np.linspace(0.0, 1.0, 101)
        mean_case1 = []
        mean_case2 = []
        for entry in roc_case1:
            order = np.argsort(entry["fpr"])
            mean_case1.append(np.interp(fpr_grid, entry["fpr"][order], entry["tpr"][order]))
        for entry in roc_case2:
            order = np.argsort(entry["fpr"])
            mean_case2.append(np.interp(fpr_grid, entry["fpr"][order], entry["tpr"][order]))

        mean_case1_tpr = np.mean(mean_case1, axis=0)
        mean_case2_tpr = np.mean(mean_case2, axis=0)
        mean_auc_case1 = float(np.mean([entry["auc"] for entry in roc_case1]))
        mean_auc_case2 = float(np.mean([entry["auc"] for entry in roc_case2]))

        plot_roc_curves(
            roc_case1,
            {
                "fpr": fpr_grid,
                "tpr": mean_case1_tpr,
                "label": f"mean (AUC={mean_auc_case1:.3f})",
            },
            title=f"ROC - Adult Content Gate (k-fold, n={group_size})",
            output_path=output_dir / f"roc_case1_kfold_n{group_size}.png",
        )
        plot_roc_curves(
            roc_case2,
            {
                "fpr": fpr_grid,
                "tpr": mean_case2_tpr,
                "label": f"mean (AUC={mean_auc_case2:.3f})",
            },
            title=f"ROC - Child Platform Gate (k-fold, n={group_size})",
            output_path=output_dir / f"roc_case2_kfold_n{group_size}.png",
        )

        all_ages = sorted({int(age) for entry in age_tables for age in entry["ages"]})
        mae_matrix = []
        rmse_matrix = []
        for entry in age_tables:
            age_to_mae = {int(a): v for a, v in zip(entry["ages"], entry["mae"])}
            age_to_rmse = {int(a): v for a, v in zip(entry["ages"], entry["rmse"])}
            mae_matrix.append([age_to_mae.get(age, np.nan) for age in all_ages])
            rmse_matrix.append([age_to_rmse.get(age, np.nan) for age in all_ages])

        mae_arr = np.asarray(mae_matrix, dtype=float)
        rmse_arr = np.asarray(rmse_matrix, dtype=float)
        mae_mean = np.nanmean(mae_arr, axis=0)
        rmse_mean = np.nanmean(rmse_arr, axis=0)
        mae_std = np.nanstd(mae_arr, axis=0)
        rmse_std = np.nanstd(rmse_arr, axis=0)

        age_csv = output_dir / f"age_error_kfold_n{group_size}.csv"
        with age_csv.open("w", encoding="utf-8") as fp:
            fp.write("age,mae_mean,mae_std,rmse_mean,rmse_std\n")
            for age, m_mae, s_mae, m_rmse, s_rmse in zip(
                all_ages, mae_mean, mae_std, rmse_mean, rmse_std
            ):
                fp.write(
                    f"{age},{m_mae:.6f},{s_mae:.6f},{m_rmse:.6f},{s_rmse:.6f}\n"
                )

        plot_age_error(
            np.asarray(all_ages),
            mae_mean,
            mae_std,
            rmse_mean,
            rmse_std,
            title=f"MAE/RMSE per Age (k-fold mean, n={group_size})",
            output_path=output_dir / f"age_error_kfold_n{group_size}.png",
        )

    print(f"Saved k-fold aggregates to: {output_dir}")


if __name__ == "__main__":
    main()
