#!/usr/bin/env python3
"""Post-filter Q5 archive predictions to entries with resolved masks only."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np

from aggregate_kfold import compute_intra_user_variability
from dataset.hand_metadata import load_combined_metadata
from dataset.utils import build_user_skin_color_series, filter_metadata, map_user_series_to_array
from metrics import aggregate_predictions_by_user, compute_age_gate_curves_direct_threshold


ARMS = ("R0", "Pooled")
SEEDS = tuple(range(42, 49))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter completed Q5 archive eval outputs to masked samples only.")
    parser.add_argument(
        "--source-root",
        type=str,
        default="runs/q5_archive_eval_local_masked_20260813",
        help="Existing full archive run root.",
    )
    parser.add_argument(
        "--dest-root",
        type=str,
        default="runs/q5_archive_eval_local_masked_only_20260813",
        help="Destination root for masked-only recomputed outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets",
    )
    parser.add_argument("--split", type=str, default="splits/archive_users_all.json")
    parser.add_argument("--age-gate-threshold-min", type=float, default=10.0)
    parser.add_argument("--age-gate-threshold-max", type=float, default=30.0)
    parser.add_argument("--num-thresholds", type=int, default=201)
    parser.add_argument("--aggregation-seed", type=int, default=42)
    return parser.parse_args()


def archive_eval_metadata(data_root: Path, split_path: Path):
    with split_path.open("r", encoding="utf-8") as fp:
        split = json.load(fp)
    test_ids = {str(uid) for uid in split["test_user_ids"]}
    meta = filter_metadata(
        load_combined_metadata(
            root=data_root,
            include_handrgbd=False,
            include_hagrid=False,
            include_synthetic_dorsal=False,
            include_synthetic_dorsal2=False,
            include_prolific=False,
            include_primary=False,
            include_archive=True,
        ),
        max_samples_per_user=None,
        max_samples_per_age_bin=None,
    )
    return meta[meta["user_id"].astype(str).isin(test_ids)].reset_index(drop=True)


def write_age_gate_metrics(path: Path, gate: dict) -> None:
    adult_gate = gate["adult_gate"]
    with path.open("w", encoding="utf-8") as fp:
        fp.write("gate,tau,fpr,fnr,tpr,tnr\n")
        for tau, fpr, fnr, tpr_val, tnr in zip(
            adult_gate["thresholds"],
            adult_gate["fpr"],
            adult_gate["fnr"],
            adult_gate["tpr"],
            adult_gate["tnr"],
        ):
            fp.write(f"adult_gate,{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n")


def write_fold_outputs(
    src_fold: Path,
    dst_fold: Path,
    keep_mask: np.ndarray,
    skin_color_by_user,
    *,
    age_min: float,
    age_max: float,
    num_thresholds: int,
    aggregation_seed: int,
) -> dict[str, float]:
    raw_path = src_fold / "test_predictions_raw_ddp.npz"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    with np.load(raw_path, allow_pickle=False) as data:
        if len(data["targets"]) != keep_mask.size:
            raise ValueError(f"{raw_path} has {len(data['targets'])} predictions, expected {keep_mask.size}")
        targets = data["targets"].astype(float)[keep_mask]
        pred_mean = data["pred_mean"].astype(float)[keep_mask]
        pred_log_var = data["pred_log_var"].astype(float)[keep_mask]
        user_ids = data["user_ids"].astype(str)[keep_mask]
        skin_color = data["skin_color"].astype(str)[keep_mask] if "skin_color" in data.files else None

    dst_fold.mkdir(parents=True, exist_ok=True)
    if src_fold.joinpath("eval_config.txt").is_file():
        shutil.copy2(src_fold / "eval_config.txt", dst_fold / "eval_config_full_archive_source.txt")

    np.savez(
        dst_fold / "test_predictions_raw_ddp.npz",
        targets=targets,
        pred_mean=pred_mean,
        pred_log_var=pred_log_var,
        user_ids=user_ids,
        skin_color=skin_color if skin_color is not None else map_user_series_to_array(user_ids, skin_color_by_user),
    )

    aggregated = aggregate_predictions_by_user(
        user_ids,
        targets,
        pred_mean,
        pred_log_var,
        group_size=1,
        rng=random.Random(aggregation_seed + 1),
    )
    agg_targets = aggregated["targets"]
    agg_preds = aggregated["pred_mean"]
    agg_log_vars = aggregated["pred_log_var"]
    agg_user_ids = aggregated["user_ids"]
    gate = compute_age_gate_curves_direct_threshold(
        agg_targets,
        agg_preds,
        age_min=age_min,
        age_max=age_max,
        num_thresholds=num_thresholds,
    )
    adult_gate = gate["adult_gate"]
    mae = float(np.mean(np.abs(agg_preds - agg_targets)))
    rmse = float(np.sqrt(np.mean((agg_preds - agg_targets) ** 2)))
    auc = float(adult_gate["auc"])
    np.savez(
        dst_fold / "test_predictions_n1_ddp.npz",
        targets=agg_targets,
        pred_mean=agg_preds,
        pred_log_var=agg_log_vars,
        adult_prob=gate["adult_prob"],
        user_ids=agg_user_ids,
        skin_color=map_user_series_to_array(agg_user_ids, skin_color_by_user),
        group_size=1,
    )
    (dst_fold / "test_summary_ddp.csv").write_text(
        "group_size,mae,rmse,auc_adult_gate\n"
        f"1,{mae:.6f},{rmse:.6f},{auc:.6f}\n",
        encoding="utf-8",
    )
    write_age_gate_metrics(dst_fold / "test_age_gate_metrics_n1_ddp.csv", gate)
    std_mean, std_median, std_count = compute_intra_user_variability(user_ids, pred_mean)
    return {
        "mae": mae,
        "rmse": rmse,
        "auc_adult_gate": auc,
        "samples": int(agg_targets.size),
        "intra_user_std_mean": std_mean,
        "intra_user_std_median": std_median,
        "users_with_variability": int(std_count),
    }


def write_kfold_summary(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "fold",
                "group_size",
                "mae",
                "rmse",
                "auc_adult_gate",
                "samples",
                "intra_user_std_mean",
                "intra_user_std_median",
                "users_with_variability",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    meta = archive_eval_metadata(Path(args.data_root), Path(args.split))
    keep_mask = meta["mask_path"].notna().to_numpy()
    skin_color_by_user = build_user_skin_color_series(meta[keep_mask])
    print(
        f"Archive eval rows: {len(meta)} | with masks: {int(keep_mask.sum())} | "
        f"without masks removed: {int((~keep_mask).sum())}"
    )

    for seed in SEEDS:
        for arm in ARMS:
            src_run = source_root / f"{arm}_seed{seed}"
            dst_run = dest_root / f"{arm}_seed{seed}"
            fold_rows = []
            for fold_idx in range(5):
                stats = write_fold_outputs(
                    src_run / f"fold_{fold_idx}",
                    dst_run / f"fold_{fold_idx}",
                    keep_mask,
                    skin_color_by_user,
                    age_min=args.age_gate_threshold_min,
                    age_max=args.age_gate_threshold_max,
                    num_thresholds=args.num_thresholds,
                    aggregation_seed=args.aggregation_seed,
                )
                fold_rows.append({"fold": f"fold_{fold_idx}", "group_size": 1, **stats})
            write_kfold_summary(dst_run / "kfold_test_summary_n1.csv", fold_rows)
            print(f"Wrote {dst_run}")


if __name__ == "__main__":
    main()
