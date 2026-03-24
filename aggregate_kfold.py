#!/usr/bin/env python3
"""Aggregate k-fold validation and held-out-test artifacts into summary plots and tables."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from metrics import compute_age_gate_curves, compute_group_summary_rows, save_group_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate k-fold validation and test outputs.")
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
    for pattern in ("val_predictions_n*_ddp.npz", "test_predictions_n*_ddp.npz"):
        for path in run_dir.glob(pattern):
            match = re.search(r"(?:val|test)_predictions_n(\d+)_ddp\.npz", path.name)
            if match:
                sizes.add(int(match.group(1)))
    return sorted(sizes)


def load_predictions(run_dir: Path, group_size: int, *, split: str = "val") -> dict:
    path = run_dir / f"{split}_predictions_n{group_size}_ddp.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    data = np.load(path)
    return {
        "targets": data["targets"].astype(float),
        "pred_mean": data["pred_mean"].astype(float),
        "pred_log_var": data["pred_log_var"].astype(float),
        "user_ids": data["user_ids"].astype(str) if "user_ids" in data.files else None,
        "skin_color": data["skin_color"].astype(str) if "skin_color" in data.files else None,
    }


def load_raw_predictions(run_dir: Path, *, split: str = "val") -> dict | None:
    path = run_dir / f"{split}_predictions_raw_ddp.npz"
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


def collect_per_user_std_by_age(user_ids: np.ndarray, preds: np.ndarray, targets: np.ndarray) -> list[tuple[int, float]]:
    buckets: dict[str, list[float]] = {}
    age_buckets: dict[str, list[float]] = {}
    for uid, pred, tgt in zip(user_ids, preds, targets):
        key = str(uid)
        buckets.setdefault(key, []).append(float(pred))
        age_buckets.setdefault(key, []).append(float(tgt))

    rows: list[tuple[int, float]] = []
    for key in buckets:
        vals = buckets[key]
        if len(vals) < 2:
            continue
        age_vals = age_buckets.get(key, [])
        if not age_vals:
            continue
        user_age = int(round(float(np.mean(age_vals))))
        user_std = float(np.std(vals))
        rows.append((user_age, user_std))
    return rows


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


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float).copy()
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


def plot_scatter(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    if targets.size == 0:
        return
    min_val = float(np.min([targets.min(), preds.min()]))
    max_val = float(np.max([targets.max(), preds.max()]))
    pad = max(1.0, 0.05 * (max_val - min_val))
    axis_min = min_val - pad
    axis_max = max_val + pad
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, s=18, alpha=0.45, edgecolors="none", color="#1f77b4")
    ax.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1)
    ax.axvline(18.0, color="black", linestyle=":", linewidth=1)
    ax.axhline(18.0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("True Age")
    ax.set_ylabel("Predicted Age")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_intra_user_per_age(
    ages: np.ndarray,
    stds: np.ndarray,
    per_age_stats: dict[int, dict[str, float]],
    *,
    title: str,
    output_path: Path,
    smooth_window: int = 5,
) -> None:
    if ages.size == 0:
        return
    age_sorted = sorted(per_age_stats.keys())
    mean_vals = np.asarray([per_age_stats[a]["mean"] for a in age_sorted], dtype=float)
    median_vals = np.asarray([per_age_stats[a]["median"] for a in age_sorted], dtype=float)
    iqr_low = np.asarray([per_age_stats[a]["iqr_low"] for a in age_sorted], dtype=float)
    iqr_high = np.asarray([per_age_stats[a]["iqr_high"] for a in age_sorted], dtype=float)
    mean_smooth = rolling_mean(mean_vals, smooth_window)
    median_smooth = rolling_mean(median_vals, smooth_window)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ages, stds, s=10, alpha=0.15, color="#1f77b4", label="users")
    ax.plot(age_sorted, mean_vals, color="#ff7f0e", linewidth=1.6, label="mean")
    ax.plot(age_sorted, median_vals, color="#2ca02c", linewidth=1.6, label="median")
    ax.plot(age_sorted, mean_smooth, color="#d62728", linestyle="--", linewidth=1.8, label=f"mean (smooth {smooth_window})")
    ax.plot(age_sorted, median_smooth, color="#9467bd", linestyle="--", linewidth=1.8, label=f"median (smooth {smooth_window})")
    ax.fill_between(age_sorted, iqr_low, iqr_high, color="#ff7f0e", alpha=0.12, label="IQR (25–75)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Std of predicted mean per user")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def aggregate_intra_user_outputs(
    fold_dirs: list[Path],
    run_dirs: list[Path],
    *,
    split: str,
    output_dir: Path,
    intra_csv_name: str,
    per_age_csv_name: str,
    per_age_plot_name: str,
    plot_title: str,
) -> list[dict]:
    intra_user_stats = []
    per_user_rows: list[tuple[int, float]] = []

    for fold_dir, run_dir in zip(fold_dirs, run_dirs):
        raw_preds = load_raw_predictions(run_dir, split=split)
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
        per_user_rows.extend(
            collect_per_user_std_by_age(
                raw_preds["user_ids"], raw_preds["pred_mean"], raw_preds["targets"]
            )
        )
        intra_user_stats.append(
            {
                "fold": fold_dir.name,
                "mean": mean_std,
                "median": median_std,
                "count": count,
            }
        )

    intra_path = output_dir / intra_csv_name
    with intra_path.open("w", encoding="utf-8") as fp:
        fp.write("fold,users_with_variability,intra_user_std_mean,intra_user_std_median\n")
        for row in intra_user_stats:
            fp.write(
                f"{row['fold']},{row['count']},{row['mean']:.6f},{row['median']:.6f}\n"
            )

    if per_user_rows:
        ages_all = np.asarray([r[0] for r in per_user_rows], dtype=int)
        stds_all = np.asarray([r[1] for r in per_user_rows], dtype=float)
        per_age_stats: dict[int, dict[str, float]] = {}
        unique_ages = sorted(set(ages_all.tolist()))
        for age in unique_ages:
            mask = ages_all == age
            vals = stds_all[mask]
            per_age_stats[age] = {
                "count": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "iqr_low": float(np.percentile(vals, 25)),
                "iqr_high": float(np.percentile(vals, 75)),
            }
        per_age_csv = output_dir / per_age_csv_name
        with per_age_csv.open("w", encoding="utf-8") as fp:
            fp.write("age,count,mean,median,std,iqr_low,iqr_high\n")
            for age in unique_ages:
                stats = per_age_stats[age]
                fp.write(
                    f"{age},{stats['count']},{stats['mean']:.6f},{stats['median']:.6f},"
                    f"{stats['std']:.6f},{stats['iqr_low']:.6f},{stats['iqr_high']:.6f}\n"
                )
        plot_intra_user_per_age(
            ages_all,
            stds_all,
            per_age_stats,
            title=plot_title,
            output_path=output_dir / per_age_plot_name,
            smooth_window=5,
        )
    else:
        print(f"Warning: No per-user variability data found for split='{split}'; skipping intra-user-per-age plot.")

    return intra_user_stats


def aggregate_prediction_outputs(
    fold_dirs: list[Path],
    run_dirs: list[Path],
    group_sizes: list[int],
    intra_user_stats: list[dict],
    *,
    split: str,
    output_dir: Path,
    age_threshold: float,
    num_thresholds: int,
    summary_name_template: str,
    roc_name_template: str,
    age_error_name_template: str,
    scatter_name_template: str,
    skin_summary_name_template: str,
    roc_title_template: str,
    age_error_title_template: str,
    scatter_title_template: str,
    required: bool,
) -> None:
    for group_size in group_sizes:
        fold_rows = []
        roc_adult_gate = []
        age_tables = []
        all_targets_concat: list[float] = []
        all_preds_concat: list[float] = []
        all_log_vars_concat: list[float] = []
        all_user_ids_concat: list[str] = []
        all_skin_colors_concat: list[str] = []

        for fold_idx, (fold_dir, run_dir) in enumerate(zip(fold_dirs, run_dirs)):
            try:
                preds = load_predictions(run_dir, group_size, split=split)
            except FileNotFoundError:
                if required:
                    raise
                print(
                    f"Warning: Missing {split} predictions for {fold_dir.name} "
                    f"(n={group_size}); skipping that fold in aggregate outputs."
                )
                continue

            targets = preds["targets"]
            pred_mean = preds["pred_mean"]
            pred_log_var = preds["pred_log_var"]
            all_targets_concat.extend(targets.tolist())
            all_preds_concat.extend(pred_mean.tolist())
            all_log_vars_concat.extend(pred_log_var.tolist())
            if preds.get("user_ids") is not None:
                all_user_ids_concat.extend(preds["user_ids"].tolist())
            if preds.get("skin_color") is not None:
                all_skin_colors_concat.extend(preds["skin_color"].tolist())
            mae = float(np.mean(np.abs(pred_mean - targets)))
            rmse = float(np.sqrt(np.mean((pred_mean - targets) ** 2)))

            gate = compute_age_gate_curves(
                targets,
                pred_mean,
                pred_log_var,
                age_threshold=age_threshold,
                num_thresholds=num_thresholds,
            )
            adult_gate = gate["adult_gate"]
            fold_label = fold_dir.name
            roc_adult_gate.append(
                {
                    "label": fold_label,
                    "fpr": adult_gate["fpr"],
                    "tpr": adult_gate["tpr"],
                    "auc": adult_gate["auc"],
                }
            )

            ages, mae_vals, rmse_vals = compute_age_errors(targets, pred_mean)
            age_tables.append({"ages": ages, "mae": mae_vals, "rmse": rmse_vals})

            stats = intra_user_stats[fold_idx] if fold_idx < len(intra_user_stats) else None
            fold_rows.append(
                {
                    "fold": fold_label,
                    "group_size": group_size,
                    "mae": mae,
                    "rmse": rmse,
                    "auc_adult_gate": adult_gate["auc"],
                    "samples": int(targets.size),
                    "intra_user_std_mean": (stats["mean"] if stats else float("nan")),
                    "intra_user_std_median": (stats["median"] if stats else float("nan")),
                    "users_with_variability": (stats["count"] if stats else 0),
                }
            )

        if not fold_rows:
            print(f"Warning: No {split} predictions found for n={group_size}; skipping aggregate outputs.")
            continue

        fold_rows_path = output_dir / summary_name_template.format(group_size=group_size)
        with fold_rows_path.open("w", encoding="utf-8") as fp:
            fp.write(
                "fold,group_size,mae,rmse,auc_adult_gate,samples,"
                "intra_user_std_mean,intra_user_std_median,users_with_variability\n"
            )
            for row in fold_rows:
                fp.write(
                    f"{row['fold']},{row['group_size']},{row['mae']:.6f},{row['rmse']:.6f},"
                    f"{row['auc_adult_gate']:.6f},{row['samples']},"
                    f"{row['intra_user_std_mean']:.6f},{row['intra_user_std_median']:.6f},"
                    f"{row['users_with_variability']}\n"
                )

        fpr_grid = np.linspace(0.0, 1.0, 101)
        mean_adult_gate = []
        for entry in roc_adult_gate:
            order = np.argsort(entry["fpr"])
            mean_adult_gate.append(np.interp(fpr_grid, entry["fpr"][order], entry["tpr"][order]))

        mean_adult_gate_tpr = np.mean(mean_adult_gate, axis=0)
        mean_auc_adult_gate = float(np.mean([entry["auc"] for entry in roc_adult_gate]))

        plot_roc_curves(
            roc_adult_gate,
            {
                "fpr": fpr_grid,
                "tpr": mean_adult_gate_tpr,
                "label": f"mean (AUC={mean_auc_adult_gate:.3f})",
            },
            title=roc_title_template.format(group_size=group_size),
            output_path=output_dir / roc_name_template.format(group_size=group_size),
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

        age_csv = output_dir / age_error_name_template.format(group_size=group_size, ext="csv")
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
            title=age_error_title_template.format(group_size=group_size),
            output_path=output_dir / age_error_name_template.format(group_size=group_size, ext="png"),
        )

        if all_targets_concat:
            plot_scatter(
                np.asarray(all_targets_concat, dtype=float),
                np.asarray(all_preds_concat, dtype=float),
                title=scatter_title_template.format(group_size=group_size),
                output_path=output_dir / scatter_name_template.format(group_size=group_size),
            )
        if all_skin_colors_concat and len(all_skin_colors_concat) == len(all_targets_concat):
            save_group_summary_csv(
                output_dir / skin_summary_name_template.format(group_size=group_size),
                compute_group_summary_rows(
                    np.asarray(all_skin_colors_concat, dtype=str),
                    np.asarray(all_targets_concat, dtype=float),
                    np.asarray(all_preds_concat, dtype=float),
                    np.asarray(all_log_vars_concat, dtype=float),
                    user_ids=np.asarray(all_user_ids_concat, dtype=str) if all_user_ids_concat else None,
                    age_threshold=age_threshold,
                    num_thresholds=num_thresholds,
                ),
                group_name="skin_color",
            )


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

    val_intra_user_stats = aggregate_intra_user_outputs(
        fold_dirs,
        run_dirs,
        split="val",
        output_dir=output_dir,
        intra_csv_name="kfold_intra_user_variability.csv",
        per_age_csv_name="intra_user_std_per_age.csv",
        per_age_plot_name="intra_user_std_per_age.png",
        plot_title=f"Intra-user variability by age (k-fold validation, folds={len(fold_dirs)})",
    )
    aggregate_prediction_outputs(
        fold_dirs,
        run_dirs,
        group_sizes,
        val_intra_user_stats,
        split="val",
        output_dir=output_dir,
        age_threshold=args.age_threshold,
        num_thresholds=args.num_thresholds,
        summary_name_template="kfold_summary_n{group_size}.csv",
        roc_name_template="roc_adult_gate_kfold_n{group_size}.png",
        age_error_name_template="age_error_kfold_n{group_size}.{ext}",
        scatter_name_template="age_val_scatter_kfold_n{group_size}.png",
        skin_summary_name_template="kfold_skin_color_summary_n{group_size}.csv",
        roc_title_template="ROC - Adult Gate (k-fold validation, n={group_size})",
        age_error_title_template="MAE/RMSE per Age (k-fold validation mean, n={group_size})",
        scatter_title_template="Validation scatter (all folds, n={group_size})",
        required=True,
    )

    test_intra_user_stats = aggregate_intra_user_outputs(
        fold_dirs,
        run_dirs,
        split="test",
        output_dir=output_dir,
        intra_csv_name="kfold_test_intra_user_variability.csv",
        per_age_csv_name="test_intra_user_std_per_age.csv",
        per_age_plot_name="test_intra_user_std_per_age.png",
        plot_title=f"Intra-user variability by age (k-fold held-out test, folds={len(fold_dirs)})",
    )
    aggregate_prediction_outputs(
        fold_dirs,
        run_dirs,
        group_sizes,
        test_intra_user_stats,
        split="test",
        output_dir=output_dir,
        age_threshold=args.age_threshold,
        num_thresholds=args.num_thresholds,
        summary_name_template="kfold_test_summary_n{group_size}.csv",
        roc_name_template="test_roc_adult_gate_kfold_n{group_size}.png",
        age_error_name_template="test_age_error_kfold_n{group_size}.{ext}",
        scatter_name_template="test_age_scatter_kfold_n{group_size}.png",
        skin_summary_name_template="kfold_test_skin_color_summary_n{group_size}.csv",
        roc_title_template="ROC - Adult Gate [held-out test] (k-fold checkpoints, n={group_size})",
        age_error_title_template="MAE/RMSE per Age [held-out test] (k-fold mean, n={group_size})",
        scatter_title_template="Held-out test scatter (all fold checkpoints, n={group_size})",
        required=False,
    )

    print(f"Saved k-fold aggregates to: {output_dir}")


if __name__ == "__main__":
    main()
