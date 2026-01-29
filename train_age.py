import argparse
import random
from pathlib import Path
from typing import Callable, Iterable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.age import AgeDataset
from dataset.hand_metadata import (
    get_dataset_root,
    load_combined_metadata,
    set_dataset_root,
)
from dataset.samplers import GroupedBatchSampler
from dataset.transforms import build_transforms
from dataset.utils import (
    compute_age_weight_map,
    dataset_composition_stats,
    filter_metadata,
    load_kfold_splits,
    oversample_by_age,
)
from displayUtils import DisplayUtils
from metrics import (
    CHALLENGE_BINS,
    CHALLENGE_FNR_BINS,
    CHALLENGE_PROB_TAU,
    LossWeights,
    aggregate_predictions_by_user,
    compute_age_gate_curves,
    compute_challenge_fnr_table_case1,
    compute_challenge_fpr_table,
    compute_challenge_fpr_table_weighted,
    intra_user_spread_loss,
    weighted_regression_loss,
)
from models import resolve_model_builder

# Example (Windows): python train_age.py --data-root "C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets" --output-dir runs\b4_efficientnet --model b4 --img-size 380 --batch-size 32 --epochs 40 --seed 42 --lr 0.0003
# --- Config -----------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_LR = 3e-4
DEFAULT_MODEL_VARIANT = "b0"
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_PATIENCE = 20


def _select_challenge_threshold(fpr_rows: list[dict], target_total: float = 0.001) -> float | None:
    """Return the first threshold whose total <= target_total, else the min-total threshold."""
    # Ensure ascending threshold order
    for row in sorted(fpr_rows, key=lambda r: float(r.get("threshold", 1e9))):
        total = float(row.get("total", 1.0))
        if total <= target_total:
            return float(row["threshold"])
    if fpr_rows:
        best = min(fpr_rows, key=lambda r: float(r.get("total", 1.0)))
        return float(best["threshold"])
    return None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _append_dataset_stats(config_path: Path, label: str, df: pd.DataFrame) -> None:
    stats = dataset_composition_stats(df)
    with config_path.open("a", encoding="utf-8") as fp:
        for key in sorted(stats):
            fp.write(f"{label}_{key}={stats[key]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet hand age regressor."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where checkpoints and plots will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        help=(
            "Backbone to use. EfficientNet: b0-b7. ConvNeXt: convnext_{tiny,small,base,large,xlarge} "
            "or aliases cnt,cns,cnb,cnl,cnx."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override the input resolution (discouraged). By default the canonical size for the chosen model is used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for training (default: 32).",
    )
    parser.add_argument(
        "--user-group-size",
        type=int,
        default=2,
        help="Samples per user in each training batch (default: 2).",
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=16,
        help="Maximum samples per user after dorsal filtering (default: 16; set 0 to disable).",
    )
    parser.add_argument(
        "--age-reweight-loss",
        action="store_true",
        help="Apply inverse-frequency age weights to the training loss.",
    )
    parser.add_argument(
        "--age-weight-eps",
        type=float,
        default=1.0,
        help="Smoothing term added to age counts when building loss weights (default: 1.0).",
    )
    parser.add_argument(
        "--age-weight-power",
        type=float,
        default=1.0,
        help="Exponent for inverse-frequency age weights (default: 1.0).",
    )
    parser.add_argument(
        "--age-weight-min",
        type=float,
        default=0.25,
        help="Lower clip for age loss weights (default: 0.25).",
    )
    parser.add_argument(
        "--age-weight-max",
        type=float,
        default=4.0,
        help="Upper clip for age loss weights (default: 4.0).",
    )
    parser.add_argument(
        "--age-oversample",
        action="store_true",
        help="Oversample under-represented integer ages in the training set.",
    )
    parser.add_argument(
        "--age-oversample-target",
        type=int,
        default=None,
        help="Target samples per age when oversampling (default: max age count).",
    )
    parser.add_argument(
        "--age-oversample-max-multiplier",
        type=float,
        default=3.0,
        help="Cap on per-age growth factor during oversampling (default: 3.0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for RNGs, ensuring reproducibility (default: 42).",
    )
    parser.add_argument(
        "--eval-aggregation-sizes",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Group sizes (n) for per-user random aggregation during evaluation outputs. "
            "Provide one or more values; default is 1 (no aggregation)."
        ),
    )
    parser.add_argument(
        "--eval-aggregation-seed",
        type=int,
        default=None,
        help="Seed used for random per-user aggregation; defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for AdamW optimizer (default: 3e-4).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01).",
    )
    parser.add_argument(
        "--fold-file",
        type=str,
        default=None,
        help="Path to a k-fold split JSON file. When set, --fold-index selects the test fold.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Fold index to use as test set (0-based). Required with --fold-file.",
    )
    parser.add_argument(
        "--loss-weight-nll",
        type=float,
        default=0.5,
        help="Weight for the Gaussian NLL component (default: 0.5; set to 1.0 for legacy behaviour).",
    )
    parser.add_argument(
        "--loss-weight-mse",
        type=float,
        default=0.25,
        help="Weight for the MSE component (default: 0.25).",
    )
    parser.add_argument(
        "--loss-weight-mae",
        type=float,
        default=0.25,
        help="Weight for the MAE component (default: 0.25).",
    )
    parser.add_argument(
        "--loss-weight-spread",
        type=float,
        default=0.0,
        help="Weight for intra-user prediction spread penalty (default: 0.0).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience in epochs (default: {DEFAULT_PATIENCE}).",
    )
    args = parser.parse_args()
    loss_weights = LossWeights(
        nll=args.loss_weight_nll,
        mse=args.loss_weight_mse,
        mae=args.loss_weight_mae,
    )
    loss_weights.validate()
    if args.loss_weight_spread < 0:
        raise ValueError("loss-weight-spread must be non-negative.")
    if args.user_group_size < 1:
        raise ValueError("user-group-size must be at least 1.")
    eval_group_sizes = sorted({int(n) for n in args.eval_aggregation_sizes if int(n) > 0})
    if not eval_group_sizes:
        raise ValueError("At least one positive --eval-aggregation-sizes value is required.")
    eval_agg_seed = (
        args.eval_aggregation_seed if args.eval_aggregation_seed is not None else args.seed
    )
    set_random_seed(args.seed)
    model_builder, default_size, model_desc, model_key = resolve_model_builder(
        args.model
    )
    img_size = args.img_size if args.img_size is not None else default_size
    if args.img_size is not None and args.img_size != default_size:
        print(
            f"[train] Using requested --img-size {args.img_size} "
            f"(default for {model_desc} is {default_size})."
        )
    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.txt"
    with config_path.open("w", encoding="utf-8") as fp:
        fp.write("# Training configuration\n")
        for key, value in sorted(vars(args).items()):
            fp.write(f"{key}={value}\n")
        fp.write(f"resolved_model={model_desc}\n")
        fp.write(f"resolved_img_size={img_size}\n")
        fp.write(f"resolved_batch_size={args.batch_size}\n")
        fp.write(f"resolved_lr={args.lr}\n")
        fp.write(f"resolved_weight_decay={args.weight_decay}\n")
        fp.write(f"resolved_patience={args.patience}\n")
        fp.write(f"resolved_user_group_size={args.user_group_size}\n")
        fp.write(f"resolved_loss_weights_nll={loss_weights.nll}\n")
        fp.write(f"resolved_loss_weights_mse={loss_weights.mse}\n")
        fp.write(f"resolved_loss_weights_mae={loss_weights.mae}\n")
    train_transform, test_transform = build_transforms(img_size)
    metadata = filter_metadata(
        load_combined_metadata(root=active_root),
        max_samples_per_user=args.max_samples_per_user,
    )
    _append_dataset_stats(config_path, "dataset", metadata)
    fold_info = None
    if args.fold_file:
        if args.fold_index is None:
            raise ValueError("--fold-index is required when --fold-file is set.")
        fold_data = load_kfold_splits(args.fold_file)
        folds = fold_data.get("folds", [])
        if not folds:
            raise ValueError("Fold file does not contain any folds.")
        if args.fold_index < 0 or args.fold_index >= len(folds):
            raise ValueError(f"fold-index must be in [0, {len(folds) - 1}].")
        test_ids = {str(uid) for uid in folds[args.fold_index]}
        train_ids = set()
        for idx, fold in enumerate(folds):
            if idx == args.fold_index:
                continue
            train_ids.update(str(uid) for uid in fold)
        available_ids = set(metadata["user_id"].astype(str).unique())
        test_ids = [uid for uid in test_ids if uid in available_ids]
        train_ids = [uid for uid in train_ids if uid in available_ids and uid not in test_ids]
        fold_info = {
            "k": len(folds),
            "index": args.fold_index,
        }
    else:
        user_ids = metadata["user_id"].unique()
        train_ids, test_ids = train_test_split(
            user_ids, test_size=0.2, random_state=args.seed
        )
    train_meta = metadata[metadata["user_id"].isin(train_ids)]
    test_meta = metadata[metadata["user_id"].isin(test_ids)]

    if args.age_oversample:
        before = len(train_meta)
        train_meta = oversample_by_age(
            train_meta,
            target_per_age=args.age_oversample_target,
            max_multiplier=args.age_oversample_max_multiplier,
            seed=args.seed,
        )
        print(f"[data] Oversampled train set from {before} to {len(train_meta)} samples.")

    age_weight_map: dict[int, float] | None = None
    if args.age_reweight_loss:
        age_weight_map = compute_age_weight_map(
            train_meta["age"],
            eps=args.age_weight_eps,
            power=args.age_weight_power,
            min_w=args.age_weight_min,
            max_w=args.age_weight_max,
            normalise=True,
        )
        if age_weight_map:
            values = list(age_weight_map.values())
            print(f"[loss] Age reweighting enabled (min={min(values):.3f}, max={max(values):.3f}).")
        else:
            print("[loss] Age reweighting requested but no weights were computed.")

    def _build_age_weight_tensor(batch_ages: torch.Tensor) -> torch.Tensor | None:
        if not age_weight_map:
            return None
        age_ints = torch.round(batch_ages).to(torch.int64).cpu().tolist()
        weights = [age_weight_map.get(int(a), 1.0) for a in age_ints]
        return batch_ages.new_tensor(weights)

    print(
        f"Using dataset root: {active_root}\n"
        f"Saving artifacts to: {output_dir}\n"
        f"Train users: {train_meta['user_id'].nunique()} | Train images: {len(train_meta)}\n"
        f"Test users:  {test_meta['user_id'].nunique()} | Test images:  {len(test_meta)}\n"
        f"Model: {model_desc} | Image size: {img_size} | Batch size: {args.batch_size} | "
        f"User group size: {args.user_group_size}\n"
        f"Epochs: {args.epochs} | Learning rate: {args.lr:.2e} | Seed: {args.seed}"
    )
    print(
        f"Loss weights -> NLL: {loss_weights.nll:.3f}, "
        f"MSE: {loss_weights.mse:.3f}, MAE: {loss_weights.mae:.3f}, "
        f"Spread: {args.loss_weight_spread:.3f}"
    )
    print(
        f"Eval aggregation group sizes: {eval_group_sizes} | "
        f"aggregation seed: {eval_agg_seed}"
    )
    if fold_info:
        print(
            "Split mode: k-fold "
            f"(fold {fold_info['index'] + 1}/{fold_info['k']})"
        )
    else:
        print("Split mode: Unstratified per-user split (random).")
    train_ds = AgeDataset(train_meta, transform=train_transform)
    test_ds = AgeDataset(test_meta, transform=test_transform)
    train_sampler = GroupedBatchSampler(
        train_ds.records["user_id"].tolist(),
        batch_size=args.batch_size,
        group_size=args.user_group_size,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = model_builder()
    if DEVICE.type == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Using {gpu_count} GPUs via DataParallel.")
            model = nn.DataParallel(model)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_val_loss = float("inf")
    best_model_path = output_dir / f"{model_key}_age_regressor.pth"
    history_log_path = output_dir / "history.log"
    min_delta = 0.001
    patience = max(1, int(args.patience))
    epochs_without_improvement = 0
    history_entries: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        running_std = 0.0
        running_spread = 0.0
        for images, ages, batch_user_ids in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            images, ages = images.to(DEVICE), ages.to(DEVICE)
            optimizer.zero_grad()
            pred_mean, pred_log_var = model(images)
            sample_weights = _build_age_weight_tensor(ages)
            base_loss = weighted_regression_loss(
                pred_mean,
                pred_log_var,
                ages,
                loss_weights,
                sample_weights=sample_weights,
            )
            if args.loss_weight_spread > 0:
                spread_loss = intra_user_spread_loss(pred_mean, batch_user_ids)
                loss = base_loss + args.loss_weight_spread * spread_loss
                running_spread += spread_loss.item()
            else:
                loss = base_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            mae = torch.mean(torch.abs(pred_mean - ages)).item()
            mse = torch.mean((pred_mean - ages) ** 2).item()
            avg_std = torch.mean(
                torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))
            ).item()
            running_mae += mae
            running_mse += mse
            running_std += avg_std
        denom = max(1, len(train_loader))
        train_loss = running_loss / denom
        train_mae = running_mae / denom
        train_mse = running_mse / denom
        train_std = running_std / denom
        train_spread = running_spread / denom if args.loss_weight_spread > 0 else 0.0
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        val_std = 0.0
        val_spread = 0.0
        val_targets = []
        val_predictions = []
        val_log_vars = []
        val_user_ids = []
        with torch.no_grad():
            for images, ages, batch_user_ids in test_loader:
                images, ages = images.to(DEVICE), ages.to(DEVICE)
                pred_mean, pred_log_var = model(images)
                batch_loss = weighted_regression_loss(
                    pred_mean, pred_log_var, ages, loss_weights
                ).item()
                val_loss += batch_loss
                val_mae += torch.mean(torch.abs(pred_mean - ages)).item()
                val_mse += torch.mean((pred_mean - ages) ** 2).item()
                val_std += torch.mean(
                    torch.exp(
                        0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0)
                    )
                ).item()
                if args.loss_weight_spread > 0:
                    val_spread += intra_user_spread_loss(pred_mean, batch_user_ids).item()
                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(pred_mean.detach().cpu().tolist())
                val_log_vars.extend(pred_log_var.detach().cpu().tolist())
                val_user_ids.extend(list(batch_user_ids))
        denom = max(1, len(test_loader))
        val_loss /= denom
        val_mae /= denom
        val_mse /= denom
        val_std /= denom
        val_spread = val_spread / denom if args.loss_weight_spread > 0 else 0.0
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, "
            f"train_mse={train_mse:.4f}, train_std={train_std:.4f}, train_spread={train_spread:.4f} | "
            f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, val_mse={val_mse:.4f}, "
            f"val_std={val_std:.4f}, val_spread={val_spread:.4f}"
        )
        with history_log_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(
                f"Epoch {epoch},train_loss={train_loss:.6f},train_mae={train_mae:.6f},train_mse={train_mse:.6f},"
                f"train_std={train_std:.6f},train_spread={train_spread:.6f},"
                f"val_loss={val_loss:.6f},val_mae={val_mae:.6f},val_mse={val_mse:.6f},"
                f"val_std={val_std:.6f},val_spread={val_spread:.6f}\n"
            )
        history_entries.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_rmse": float(np.sqrt(train_mse)),
                "val_rmse": float(np.sqrt(val_mse)),
            }
        )

        # Save the model and evaluation artifacts only if validation loss improves
        if val_loss < best_val_loss:
            improvement = (
                float("inf")
                if best_val_loss == float("inf")
                else best_val_loss - val_loss
            )
            best_val_loss = val_loss
            model_to_save = (
                model.module if isinstance(model, nn.DataParallel) else model
            )
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path} (val_loss={val_loss:.4f})")
            saved_artifacts = []
            for group_size in eval_group_sizes:
                agg_rng = random.Random(eval_agg_seed + group_size)
                aggregated = aggregate_predictions_by_user(
                    val_user_ids,
                    val_targets,
                    val_predictions,
                    val_log_vars,
                    group_size=group_size,
                    rng=agg_rng,
                )
                gate_results = compute_age_gate_curves(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    age_threshold=18.0,
                    num_thresholds=201,
                )
                # Select tau closest to top-left (0,1) for each case
                def _select_best_tau(fprs_arr, tprs_arr, thresholds_arr):
                    fprs_np = np.asarray(fprs_arr, dtype=float)
                    tprs_np = np.asarray(tprs_arr, dtype=float)
                    thr_np = np.asarray(thresholds_arr, dtype=float)
                    if fprs_np.size == 0:
                        return None
                    idx = int(np.argmin((fprs_np ** 2) + ((1.0 - tprs_np) ** 2)))
                    return float(thr_np[idx])

                best_tau_case1 = _select_best_tau(
                    gate_results["case1"]["fpr"], gate_results["case1"]["tpr"], gate_results["case1"]["thresholds"]
                )
                best_tau_case2 = _select_best_tau(
                    gate_results["case2"]["fpr"], gate_results["case2"]["tpr"], gate_results["case2"]["thresholds"]
                )
                selected_tau_case1 = best_tau_case1 if best_tau_case1 is not None else CHALLENGE_PROB_TAU
                suffix = f"n{group_size}"
                preds_dump_path = output_dir / f"val_predictions_{suffix}.npz"
                np.savez(
                    preds_dump_path,
                    targets=aggregated["targets"],
                    pred_mean=aggregated["pred_mean"],
                    pred_log_var=aggregated["pred_log_var"],
                    adult_prob=gate_results["adult_prob"],
                    epoch=epoch,
                    group_size=group_size,
                )
                challenge_thresholds = np.arange(18, 31, 1, dtype=float)  # 18..30
                fpr_rows = compute_challenge_fpr_table(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    thresholds=challenge_thresholds,
                    prob_threshold=selected_tau_case1,
                    bins=CHALLENGE_BINS,
                )
                challenge_csv = output_dir / f"challenge_fpr_bins_{suffix}.csv"
                with challenge_csv.open("w", encoding="utf-8") as fp:
                    header = ["threshold"] + [label for label, _, _ in CHALLENGE_BINS] + ["total"]
                    fp.write(",".join(header) + "\n")
                    for row in fpr_rows:
                        values = [f"{row['threshold']:.1f}"] + [
                            f"{row[label]:.6f}" for label, _, _ in CHALLENGE_BINS
                        ] + [f"{row['total']:.6f}"]
                        fp.write(",".join(values) + "\n")
                fnr_rows = compute_challenge_fnr_table_case1(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    thresholds=challenge_thresholds,
                    prob_threshold=selected_tau_case1,
                )
                fnr_csv = output_dir / f"challenge_fnr_bins_case1_{suffix}.csv"
                with fnr_csv.open("w", encoding="utf-8") as fp:
                    header = ["threshold"] + [label for label, _, _ in CHALLENGE_FNR_BINS] + ["total"]
                    fp.write(",".join(header) + "\n")
                    for row in fnr_rows:
                        values = [f"{row['threshold']:.1f}"] + [
                            f"{row[label]:.6f}" for label, _, _ in CHALLENGE_FNR_BINS
                        ] + [f"{row['total']:.6f}"]
                        fp.write(",".join(values) + "\n")
                roc_case1_path = output_dir / f"roc_case1_adult_gate_{suffix}.png"
                roc_case2_path = output_dir / f"roc_case2_child_gate_{suffix}.png"
                DisplayUtils.plot_roc_curve(
                    gate_results["case1"]["fpr"],
                    gate_results["case1"]["tpr"],
                    thresholds=gate_results["case1"]["thresholds"],
                    save_path=roc_case1_path,
                    title=f"ROC - Adult Content Gate (admit adults, n={group_size})",
                    auc_value=gate_results["case1"]["auc"],
                    highlight_tau=best_tau_case1,
                    show=False,
                )
                DisplayUtils.plot_roc_curve(
                    gate_results["case2"]["fpr"],
                    gate_results["case2"]["tpr"],
                    thresholds=gate_results["case2"]["thresholds"],
                    save_path=roc_case2_path,
                    title=f"ROC - Child Platform Gate (admit minors, n={group_size})",
                    auc_value=gate_results["case2"]["auc"],
                    highlight_tau=best_tau_case2,
                    show=False,
                )
                metrics_csv_path = output_dir / f"age_gate_metrics_{suffix}.csv"
                with metrics_csv_path.open("w", encoding="utf-8") as metrics_fp:
                    metrics_fp.write("case,tau,fpr,fnr,tpr,tnr\n")
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
                            metrics_fp.write(
                                f"{case_name},{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n"
                            )
                challenge_thresholds = np.arange(18, 31, 1, dtype=float)
                fpr_rows_weighted_best = compute_challenge_fpr_table_weighted(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    thresholds=challenge_thresholds,
                    prob_threshold=selected_tau_case1,
                    bins=CHALLENGE_BINS,
                )
                challenge_line = _select_challenge_threshold(fpr_rows_weighted_best, target_total=0.001)
                print(f"[debug] challenge_line (weighted total<=0.001) for n={group_size}: {challenge_line}")

                scatter_path = output_dir / f"age_val_scatter_{suffix}.png"
                if DisplayUtils.save_regression_scatter(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    save_path=scatter_path,
                    title=f"Validation Age Predictions (epoch {epoch}, n={group_size})",
                    axis_limits=(0.0, 70.0),
                    point_size=20,
                    alpha=0.6,
                ):
                    saved_artifacts.append(
                        f"n={group_size} -> {roc_case1_path.name}, {roc_case2_path.name}, {metrics_csv_path.name}, {preds_dump_path.name}, {challenge_csv.name}, {fnr_csv.name}, {scatter_path.name}"
                    )
                else:
                    saved_artifacts.append(
                        f"n={group_size} -> {roc_case1_path.name}, {roc_case2_path.name}, {metrics_csv_path.name}, {preds_dump_path.name}, {challenge_csv.name}, {fnr_csv.name}"
                    )
                error_plot_path = output_dir / f"age_error_by_target_{suffix}.png"
                if DisplayUtils.save_error_by_age(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    save_path=error_plot_path,
                    title=f"Error vs Target Age (epoch {epoch}, n={group_size})",
                ):
                    saved_artifacts.append(f"n={group_size} -> {error_plot_path.name}")
            print("Saved/updated eval artifacts for: " + "; ".join(saved_artifacts))
            if improvement == float("inf") or improvement >= min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            early_msg = (
                f"Early stopping at epoch {epoch}: validation loss did not improve by at least "
                f"{min_delta:.3f} for {patience} consecutive epochs."
            )
            print(early_msg)
            with history_log_path.open("a", encoding="utf-8") as log_fp:
                log_fp.write(f"# {early_msg}\n")
            history_plot_path = output_dir / "history_plot.png"
            DisplayUtils.plot_loss_history(
                history_entries,
                save_path=history_plot_path,
                show=False,
                title="Training History (MAE & RMSE)",
            )
            print(f"Saved training history plot to {history_plot_path}")
            break
    # Save per-user age distribution histograms for train/test splits
    train_user_ages = train_meta.groupby("user_id")["age"].mean().to_numpy()
    test_user_ages = test_meta.groupby("user_id")["age"].mean().to_numpy()
    hist_path = output_dir / "age_distribution_users.png"
    saved_hist = DisplayUtils.plot_age_histograms(
        train_user_ages,
        test_user_ages,
        save_path=hist_path,
        show=False,
        title="Per-user age distribution (train vs test)",
    )
    if saved_hist:
        print(f"Saved per-user age histograms to {saved_hist}")
    # Final challenge-threshold table (single evaluation pass using best model)
    if best_model_path.exists():
        print("Computing challenge-threshold FPR table on test set using best model...")
        eval_model = model_builder()
        state = torch.load(best_model_path, map_location=DEVICE)
        eval_model.load_state_dict(state)
        eval_model = eval_model.to(DEVICE)
        eval_model.eval()
        all_targets: list[float] = []
        all_means: list[float] = []
        all_log_vars: list[float] = []
        all_user_ids: list = []
        with torch.no_grad():
            for images, ages, batch_user_ids in test_loader:
                images = images.to(DEVICE)
                ages = ages.to(DEVICE)
                mean, log_var = eval_model(images)
                all_targets.extend(ages.cpu().tolist())
                all_means.extend(mean.cpu().tolist())
                all_log_vars.extend(log_var.cpu().tolist())
                all_user_ids.extend(list(batch_user_ids))
        challenge_thresholds = np.arange(18, 31, 1, dtype=float)  # 18..30
        for group_size in eval_group_sizes:
            agg_rng = random.Random(eval_agg_seed + group_size)
            aggregated = aggregate_predictions_by_user(
                all_user_ids,
                all_targets,
                all_means,
                all_log_vars,
                group_size=group_size,
                rng=agg_rng,
            )
            # Select tau closest to top-left (0,1) on test ROC
            gate_results = compute_age_gate_curves(
                aggregated["targets"],
                aggregated["pred_mean"],
                aggregated["pred_log_var"],
                age_threshold=18.0,
                num_thresholds=201,
            )
            def _select_best_tau(fprs_arr, tprs_arr, thresholds_arr):
                fprs_np = np.asarray(fprs_arr, dtype=float)
                tprs_np = np.asarray(tprs_arr, dtype=float)
                thr_np = np.asarray(thresholds_arr, dtype=float)
                if fprs_np.size == 0:
                    return None
                idx = int(np.argmin((fprs_np ** 2) + ((1.0 - tprs_np) ** 2)))
                return float(thr_np[idx])

            best_tau_case1 = _select_best_tau(
                gate_results["case1"]["fpr"], gate_results["case1"]["tpr"], gate_results["case1"]["thresholds"]
            )
            selected_tau_case1 = best_tau_case1 if best_tau_case1 is not None else CHALLENGE_PROB_TAU
            fpr_rows = compute_challenge_fpr_table(
                aggregated["targets"],
                aggregated["pred_mean"],
                aggregated["pred_log_var"],
                thresholds=challenge_thresholds,
                prob_threshold=selected_tau_case1,
                bins=CHALLENGE_BINS,
            )
            challenge_csv = output_dir / f"challenge_fpr_bins_n{group_size}.csv"
            with challenge_csv.open("w", encoding="utf-8") as fp:
                header = (
                    ["threshold"] + [label for label, _, _ in CHALLENGE_BINS] + ["total"]
                )
                fp.write(",".join(header) + "\n")
                for row in fpr_rows:
                    values = (
                        [f"{row['threshold']:.1f}"]
                        + [f"{row[label]:.6f}" for label, _, _ in CHALLENGE_BINS]
                        + [f"{row['total']:.6f}"]
                    )
                    fp.write(",".join(values) + "\n")
            print(f"Saved challenge FPR table to {challenge_csv}")
    else:
        print("Best model checkpoint not found; skipped challenge-threshold table.")
    print("Training complete. Best model saved on validation improvement.")


if __name__ == "__main__":
    main()
