#!/usr/bin/env python3
"""Evaluate a trained model on the held-out test set.

Run this script ONCE after all model selection and hyperparameter tuning is
complete.  It loads the best checkpoint, runs inference on the test users that
were never seen during training, and writes predictions + metrics to disk.

Example
-------
python evaluate_test.py \
    --checkpoint runs/my_run/best_model.pth \
    --test-users-file splits/test_users.json \
    --model v2_m \
    --output-dir runs/my_run/test_eval
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.age import AgeDataset
from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.transforms import build_transforms
from dataset.utils import filter_metadata, load_test_split
from metrics import (
    CHALLENGE_BINS,
    CHALLENGE_FNR_BINS,
    CHALLENGE_PROB_TAU,
    aggregate_predictions_by_user,
    compute_age_gate_curves,
    compute_challenge_fnr_table_case1,
    compute_challenge_fpr_table,
)
from models import resolve_model_builder
from displayUtils import DisplayUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the held-out test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth) produced during training.",
    )
    parser.add_argument(
        "--test-users-file",
        type=str,
        required=True,
        help="Path to the held-out test split JSON (from make_test_split.py).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="v2_m",
        help="Backbone name — must match the checkpoint (default: v2_m).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override input resolution. By default uses the canonical size for the chosen model.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="Embedding head dimension used during training (0 = disabled; must match checkpoint).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=16,
        help="Maximum samples per user after dorsal filtering (default: 16; set 0 to disable).",
    )
    parser.add_argument(
        "--use-masks",
        action="store_true",
        help="Apply dataset masks (black out backgrounds) during loading.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--eval-aggregation-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Per-user aggregation group sizes for evaluation (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--eval-aggregation-seed",
        type=int,
        default=42,
        help="Seed for random per-user aggregation (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predictions and metrics are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_root:
        set_dataset_root(args.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load test user IDs ────────────────────────────────────────────────────
    test_data = load_test_split(args.test_users_file)
    test_ids = {str(uid) for uid in test_data["test_user_ids"]}
    print(
        f"Test split: {len(test_ids)} users "
        f"(seed={test_data.get('seed')}, "
        f"stratified={test_data.get('stratified')}, "
        f"fraction={test_data.get('test_size')})"
    )

    # ── Build test dataset ────────────────────────────────────────────────────
    metadata = filter_metadata(
        load_combined_metadata(root=get_dataset_root()),
        max_samples_per_user=args.max_samples_per_user,
    )
    test_meta = metadata[metadata["user_id"].astype(str).isin(test_ids)].copy()
    if test_meta.empty:
        raise RuntimeError(
            "No test samples found. Check --data-root and --test-users-file."
        )
    print(f"Test samples: {len(test_meta)} | Test users found: {test_meta['user_id'].nunique()}")

    model_builder, default_size, model_desc, _ = resolve_model_builder(
        args.model, embed_dim=args.embed_dim
    )
    img_size = args.img_size if args.img_size is not None else default_size
    _, test_transform = build_transforms(img_size)

    test_ds = AgeDataset(test_meta, transform=test_transform, use_masks=args.use_masks)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # ── Load model checkpoint ─────────────────────────────────────────────────
    model = model_builder().to(device)
    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Model: {model_desc} | Image size: {img_size}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_targets: list[float] = []
    all_preds: list[float] = []
    all_log_vars: list[float] = []
    all_user_ids: list[str] = []

    with torch.no_grad():
        for images, ages, batch_user_ids in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                pred_mean, pred_log_var = outputs[0], outputs[1]
            else:
                pred_mean, pred_log_var = outputs, torch.zeros_like(outputs)

            all_targets.extend(ages.tolist())
            all_preds.extend(pred_mean.detach().cpu().tolist())
            all_log_vars.extend(pred_log_var.detach().cpu().tolist())
            all_user_ids.extend(list(batch_user_ids))

    targets_arr = np.asarray(all_targets, dtype=float)
    preds_arr = np.asarray(all_preds, dtype=float)
    log_vars_arr = np.asarray(all_log_vars, dtype=float)

    sample_mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    sample_rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
    print(f"\nSample-level  MAE={sample_mae:.4f}  RMSE={sample_rmse:.4f}")

    # ── Save raw predictions ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "test_predictions_raw.npz"
    np.savez(
        raw_path,
        targets=targets_arr,
        pred_mean=preds_arr,
        pred_log_var=log_vars_arr,
        user_ids=np.asarray(all_user_ids, dtype=str),
    )
    print(f"Saved raw predictions: {raw_path}")

    # ── Per-aggregation-size evaluation ──────────────────────────────────────
    eval_group_sizes = sorted({int(n) for n in args.eval_aggregation_sizes if int(n) > 0})
    summary_rows: list[str] = [
        "group_size,mae,rmse,auc_case1,auc_case2"
    ]

    for group_size in eval_group_sizes:
        agg_rng = random.Random(args.eval_aggregation_seed + group_size)
        aggregated = aggregate_predictions_by_user(
            all_user_ids,
            targets_arr,
            preds_arr,
            log_vars_arr,
            group_size=group_size,
            rng=agg_rng,
        )
        agg_targets = aggregated["targets"]
        agg_preds = aggregated["pred_mean"]
        agg_log_vars = aggregated["pred_log_var"]
        suffix = f"n{group_size}"

        mae = float(np.mean(np.abs(agg_preds - agg_targets)))
        rmse = float(np.sqrt(np.mean((agg_preds - agg_targets) ** 2)))

        gate_results = compute_age_gate_curves(
            agg_targets,
            agg_preds,
            agg_log_vars,
            age_threshold=18.0,
            num_thresholds=201,
        )
        auc1 = gate_results["case1"]["auc"]
        auc2 = gate_results["case2"]["auc"]
        print(
            f"n={group_size:2d}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
            f"AUC(case1)={auc1:.4f}  AUC(case2)={auc2:.4f}"
        )
        summary_rows.append(f"{group_size},{mae:.6f},{rmse:.6f},{auc1:.6f},{auc2:.6f}")

        # Aggregated predictions
        np.savez(
            output_dir / f"test_predictions_{suffix}.npz",
            targets=agg_targets,
            pred_mean=agg_preds,
            pred_log_var=agg_log_vars,
            adult_prob=gate_results["adult_prob"],
            group_size=group_size,
        )

        # ROC curves
        def _best_tau(fprs, tprs, thrs):
            fprs_np = np.asarray(fprs, dtype=float)
            tprs_np = np.asarray(tprs, dtype=float)
            thrs_np = np.asarray(thrs, dtype=float)
            if fprs_np.size == 0:
                return None
            idx = int(np.argmin(fprs_np**2 + (1.0 - tprs_np)**2))
            return float(thrs_np[idx])

        best_tau_case1 = _best_tau(
            gate_results["case1"]["fpr"],
            gate_results["case1"]["tpr"],
            gate_results["case1"]["thresholds"],
        )
        best_tau_case2 = _best_tau(
            gate_results["case2"]["fpr"],
            gate_results["case2"]["tpr"],
            gate_results["case2"]["thresholds"],
        )
        selected_tau_case1 = best_tau_case1 if best_tau_case1 is not None else CHALLENGE_PROB_TAU

        DisplayUtils.plot_roc_curve(
            gate_results["case1"]["fpr"],
            gate_results["case1"]["tpr"],
            thresholds=gate_results["case1"]["thresholds"],
            save_path=output_dir / f"roc_case1_adult_gate_{suffix}.png",
            title=f"ROC - Adult Content Gate (TEST, n={group_size})",
            auc_value=auc1,
            highlight_tau=best_tau_case1,
            show=False,
        )
        DisplayUtils.plot_roc_curve(
            gate_results["case2"]["fpr"],
            gate_results["case2"]["tpr"],
            thresholds=gate_results["case2"]["thresholds"],
            save_path=output_dir / f"roc_case2_child_gate_{suffix}.png",
            title=f"ROC - Child Platform Gate (TEST, n={group_size})",
            auc_value=auc2,
            highlight_tau=best_tau_case2,
            show=False,
        )

        # Age-gate metrics CSV
        metrics_csv = output_dir / f"test_age_gate_metrics_{suffix}.csv"
        with metrics_csv.open("w", encoding="utf-8") as fp:
            fp.write("case,tau,fpr,fnr,tpr,tnr\n")
            for case_name, case_data in (
                ("adult_content_gate", gate_results["case1"]),
                ("child_platform_gate", gate_results["case2"]),
            ):
                for tau, fpr, fnr, tpr_val, tnr in zip(
                    case_data["thresholds"],
                    case_data["fpr"],
                    case_data["fnr"],
                    case_data["tpr"],
                    case_data["tnr"],
                ):
                    fp.write(f"{case_name},{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n")

        # Challenge FPR / FNR tables
        challenge_thresholds = np.arange(18, 31, 1, dtype=float)
        fpr_rows = compute_challenge_fpr_table(
            agg_targets,
            agg_preds,
            agg_log_vars,
            thresholds=challenge_thresholds,
            prob_threshold=selected_tau_case1,
            bins=CHALLENGE_BINS,
        )
        fpr_csv = output_dir / f"test_challenge_fpr_bins_{suffix}.csv"
        with fpr_csv.open("w", encoding="utf-8") as fp:
            header = ["threshold"] + [label for label, _, _ in CHALLENGE_BINS] + ["total"]
            fp.write(",".join(header) + "\n")
            for row in fpr_rows:
                values = [f"{row['threshold']:.1f}"] + [
                    f"{row[label]:.6f}" for label, _, _ in CHALLENGE_BINS
                ] + [f"{row['total']:.6f}"]
                fp.write(",".join(values) + "\n")

        fnr_rows = compute_challenge_fnr_table_case1(
            agg_targets,
            agg_preds,
            agg_log_vars,
            thresholds=challenge_thresholds,
            prob_threshold=selected_tau_case1,
        )
        fnr_csv = output_dir / f"test_challenge_fnr_bins_{suffix}.csv"
        with fnr_csv.open("w", encoding="utf-8") as fp:
            header = ["threshold"] + [label for label, _, _ in CHALLENGE_FNR_BINS] + ["total"]
            fp.write(",".join(header) + "\n")
            for row in fnr_rows:
                values = [f"{row['threshold']:.1f}"] + [
                    f"{row[label]:.6f}" for label, _, _ in CHALLENGE_FNR_BINS
                ] + [f"{row['total']:.6f}"]
                fp.write(",".join(values) + "\n")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = output_dir / "test_summary.csv"
    summary_path.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")

    # ── Config record ─────────────────────────────────────────────────────────
    config_path = output_dir / "eval_config.txt"
    with config_path.open("w", encoding="utf-8") as fp:
        fp.write(f"evaluated_at={datetime.now().astimezone().replace(microsecond=0).isoformat()}\n")
        for key, value in sorted(vars(args).items()):
            fp.write(f"{key}={value}\n")
        fp.write(f"test_users={len(test_ids)}\n")
        fp.write(f"test_samples={len(targets_arr)}\n")
        fp.write(f"sample_mae={sample_mae:.6f}\n")
        fp.write(f"sample_rmse={sample_rmse:.6f}\n")
    print(f"Config written to:   {config_path}")


if __name__ == "__main__":
    main()
