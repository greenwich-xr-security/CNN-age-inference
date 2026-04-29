import argparse
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset.age import AgeDataset
from dataset.hand_metadata import get_dataset_root, load_combined_metadata, load_lucid_metadata, set_dataset_root
from dataset.samplers import DistributedGroupedBatchSampler
from dataset.transforms import build_normals_transform, build_transforms
from dataset.utils import (
    build_user_skin_color_series,
    compute_age_weight_map,
    dataset_composition_stats,
    filter_metadata,
    load_kfold_splits,
    load_test_split,
    map_user_series_to_array,
    oversample_by_age,
)
from displayUtils import DisplayUtils
from metrics import (
    CHALLENGE_BINS,
    CHALLENGE_FNR_BINS,
    CHALLENGE_PROB_TAU,
    embedding_contrastive_loss,
    embedding_variance_loss,
    LossWeights,
    aggregate_predictions_by_user,
    compute_age_gate_curves,
    compute_age_gate_curves_direct_threshold,
    compute_challenge_fnr_table_adult_gate,
    compute_challenge_fpr_table,
    compute_challenge_fpr_table_weighted,
    compute_group_summary_rows,
    intra_user_spread_loss,
    normals_reconstruction_loss,
    save_group_summary_csv,
    weighted_regression_loss,
)
from models import resolve_model_builder
from train_age import set_random_seed

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_LR = 3e-4
DEFAULT_MODEL_VARIANT = "b0"
DEFAULT_SEED = 42
DEFAULT_MIN_DELTA = 0.001
DEFAULT_PATIENCE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed training for EfficientNet age regressor.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--test-users-file",
        type=str,
        default=None,
        help="Path to held-out test split JSON (from make_test_split.py). Test users are excluded from train and val.",
    )
    parser.add_argument(
        "--fold-file",
        type=str,
        default=None,
        help="Path to a k-fold split JSON file. When set, --fold-index selects the validation fold.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Fold index to use as validation set (0-based). Required with --fold-file.",
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
            "Backbone to use. EfficientNet: b0-b7. ConvNeXt: convnext_{tiny,small,base,large,xlarge}. "
            "ResNet: resnet50. "
            "Swin: swin_{tiny,small,base,large,v2_tiny,v2_tiny_384,v2_tiny_512,v2_small,v2_base,v2_base_384,v2_large}. "
            "ViT: vit_{tiny_384,small_384}. Aliases: cnt,cns,cnb,cnl,cnx,rn50,sw2t,sw2t384,sw2t512,sw2s,sw2b,sw2b384,sw2l,vtt,vts,age_vit,age_vit_small."
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
        help="Per-process mini-batch size (default: 32).",
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
        default=0,
        help="Maximum samples per user after dorsal filtering (default: 0 = disabled).",
    )
    parser.add_argument(
        "--max-samples-per-age-bin",
        type=int,
        default=200,
        help="Maximum samples per integer age year (default: 200; set 0 to disable).",
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
        "--use-masks",
        action="store_true",
        help="Apply dataset masks (if available) to black out backgrounds during loading.",
    )
    parser.add_argument(
        "--no-hagrid",
        action="store_true",
        default=False,
        help="Exclude the HaGRIDv2 stop_inverted dataset from training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: 40).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience in epochs (default: {DEFAULT_PATIENCE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for RNGs (default: 42).",
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
        "--eval-age-gate-mode",
        choices=["probability", "age_threshold"],
        default="probability",
        help="How to compute the age gate ROC curve: 'probability' uses the Gaussian CDF "
             "(requires calibrated log_var); 'age_threshold' directly thresholds pred_mean "
             "against a sweep of age values (default: probability).",
    )
    parser.add_argument(
        "--age-gate-threshold-min",
        type=float,
        default=10.0,
        help="Lower bound of the age sweep for age_threshold eval mode (default: 10.0).",
    )
    parser.add_argument(
        "--age-gate-threshold-max",
        type=float,
        default=30.0,
        help="Upper bound of the age sweep for age_threshold eval mode (default: 30.0).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="Embedding head dimension (0 disables embedding head/losses).",
    )
    parser.add_argument(
        "--loss-weight-embed-var",
        type=float,
        default=0.0,
        help="Weight for embedding variance loss (same-user consistency).",
    )
    parser.add_argument(
        "--loss-weight-embed-contrast",
        type=float,
        default=0.0,
        help="Weight for embedding contrastive loss across users.",
    )
    parser.add_argument(
        "--embed-age-slack",
        type=float,
        default=5.0,
        help="Age slack for weighting same-user embedding variance (default: 5 years).",
    )
    parser.add_argument(
        "--embed-contrast-margin",
        type=float,
        default=1.0,
        help="Margin for embedding contrastive loss (default: 1.0).",
    )
    parser.add_argument(
        "--embed-contrast-age-thresh",
        type=float,
        default=5.0,
        help="Minimum age gap between different users to apply contrastive separation (default: 5 years).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers per process (default: 2).",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="torch.distributed backend to use.",
    )
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=1800,
        help="Timeout (in seconds) for torch.distributed initialization.",
    )
    parser.add_argument(
        "--find-unused-params",
        action="store_true",
        help="Enable DistributedDataParallel(find_unused_parameters=True).",
    )
    parser.add_argument(
        "--normals-aux",
        action="store_true",
        help="Attach auxiliary normal-map decoder to the backbone.",
    )
    parser.add_argument(
        "--loss-weight-normals",
        type=float,
        default=0.1,
        help="Weight for the auxiliary normal map reconstruction loss (default: 0.1).",
    )
    parser.add_argument(
        "--normals-privileged",
        action="store_true",
        help="Privileged normals input: expand first conv to 6 channels and supply normals during "
             "training (zeros at val/test time). Requires LUICIDHands.",
    )
    parser.add_argument(
        "--normals-dropout",
        type=float,
        default=0.5,
        help="Probability of zeroing out normals channels for a LUICID sample during training, "
             "teaching the model to work without normals (default: 0.5).",
    )
    parser.add_argument(
        "--lucid-fraction",
        type=float,
        default=0.4,
        help="Target fraction of each training batch from LUICIDHands (via oversampling; default: 0.4).",
    )
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    timeout = timedelta(seconds=int(args.dist_timeout))
    dist.init_process_group(backend=args.dist_backend, timeout=timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, device


def build_datasets(args: argparse.Namespace, seed: int, img_size: int):
    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()

    train_transform, test_transform = build_transforms(img_size)
    _needs_normals = getattr(args, "normals_aux", False) or getattr(args, "normals_privileged", False)
    normals_transform = build_normals_transform(img_size) if _needs_normals else None

    metadata = filter_metadata(
        load_combined_metadata(
            root=active_root,
            include_hagrid=not getattr(args, "no_hagrid", False),
        ),
        max_samples_per_user=args.max_samples_per_user or None,
        max_samples_per_age_bin=args.max_samples_per_age_bin or None,
    )

    test_ids: set = set()
    if args.test_users_file:
        test_data = load_test_split(args.test_users_file)
        test_ids = {str(uid) for uid in test_data["test_user_ids"]}
        before = metadata["user_id"].nunique()
        metadata = metadata[~metadata["user_id"].astype(str).isin(test_ids)]
        after = metadata["user_id"].nunique()
        print(f"[data] Excluded {before - after} held-out test users. Remaining: {after}")

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
        val_ids = {str(uid) for uid in folds[args.fold_index]}
        train_ids = set()
        for idx, fold in enumerate(folds):
            if idx == args.fold_index:
                continue
            train_ids.update(str(uid) for uid in fold)
        available_ids = set(metadata["user_id"].astype(str).unique())
        val_ids = [uid for uid in val_ids if uid in available_ids]
        train_ids = [uid for uid in train_ids if uid in available_ids and uid not in val_ids]
        fold_info = {
            "k": len(folds),
            "index": args.fold_index,
        }
    else:
        user_ids = metadata["user_id"].unique()
        train_ids, val_ids = train_test_split(user_ids, test_size=0.2, random_state=seed)
    train_meta = metadata[metadata["user_id"].isin(train_ids)].copy()
    val_meta = metadata[metadata["user_id"].isin(val_ids)].copy()

    if args.age_oversample:
        before = len(train_meta)
        train_meta = oversample_by_age(
            train_meta,
            target_per_age=args.age_oversample_target,
            max_multiplier=args.age_oversample_max_multiplier,
            seed=args.seed,
        )
        if len(train_meta) != before:
            print(f"[data] Oversampled train set from {before} to {len(train_meta)} samples.")

    # Merge LUICIDHands into training for both baseline and normals runs.
    # Strip the "lucid_" prefix to map LUICIDHands user IDs back to their HandRGBD integer IDs,
    # then exclude any whose raw ID appears in the val fold or held-out test split (leakage prevention).
    lucid_meta = filter_metadata(
        load_lucid_metadata(root=active_root),
        max_samples_per_user=args.max_samples_per_user or None,
    )
    if not lucid_meta.empty:
        excluded_ids = set(str(v) for v in val_ids) | test_ids
        raw_ids = lucid_meta["user_id"].astype(str).str.replace("^lucid_", "", regex=True)
        lucid_meta = lucid_meta[~raw_ids.isin(excluded_ids)].copy()
    if not lucid_meta.empty:
        frac = getattr(args, "lucid_fraction", 0.4)
        n_target = int(round(len(train_meta) * frac / max(1.0 - frac, 1e-6)))
        repeat = max(1, round(n_target / len(lucid_meta)))
        lucid_over = pd.concat([lucid_meta] * repeat, ignore_index=True).head(n_target)
        if normals_transform is None:
            lucid_over = lucid_over.copy()
            lucid_over["normals_path"] = None
        if "normals_path" not in train_meta.columns:
            train_meta["normals_path"] = None
        train_meta = pd.concat([train_meta, lucid_over], ignore_index=True)
        n_with_normals = int(lucid_over["normals_path"].notna().sum()) if "normals_path" in lucid_over.columns else 0
        suffix = f", {n_with_normals} with normals" if normals_transform is not None else ""
        print(
            f"[lucid] LUICIDHands merged: {len(lucid_over)} samples "
            f"({100 * len(lucid_over) / len(train_meta):.1f}% of combined train set{suffix})."
        )

    train_ds = AgeDataset(
        train_meta,
        transform=train_transform,
        normals_transform=normals_transform,
    )
    val_ds = AgeDataset(val_meta, transform=test_transform, use_masks=args.use_masks)
    val_user_skin = build_user_skin_color_series(val_meta)
    return train_ds, val_ds, active_root, len(train_meta), len(val_meta), fold_info, val_user_skin, normals_transform


def load_filtered_test_metadata(args: argparse.Namespace, active_root) -> pd.DataFrame | None:
    if not args.test_users_file:
        return None

    test_split_data = load_test_split(args.test_users_file)
    test_ids = {str(uid) for uid in test_split_data["test_user_ids"]}
    all_meta = filter_metadata(
        load_combined_metadata(
            root=active_root,
            include_hagrid=not getattr(args, "no_hagrid", False),
        ),
        max_samples_per_user=args.max_samples_per_user or None,
        max_samples_per_age_bin=args.max_samples_per_age_bin or None,
    )
    return all_meta[all_meta["user_id"].astype(str).isin(test_ids)].reset_index(drop=True)


def build_dataloaders(
    train_dataset,
    val_dataset,
    *,
    batch_size: int,
    group_size: int,
    num_workers: int,
    device: torch.device,
    world_size: int,
    rank: int,
    seed: int,
    collate_fn=None,
):
    pin_memory = device.type == "cuda"
    train_sampler = DistributedGroupedBatchSampler(
        train_dataset.records["user_id"].tolist(),
        batch_size=batch_size,
        group_size=group_size,
        shuffle=True,
        seed=seed,
        drop_last=False,
        num_replicas=world_size,
        rank=rank,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_sampler


def all_reduce_metrics(device: torch.device, sums: list[float]) -> list[float]:
    tensor = torch.tensor(sums, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def gather_all_lists(local_list, world_size: int):
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, list(local_list))
    merged = []
    for part in gather_list:
        if part:
            merged.extend(part)
    return merged


def _append_dataset_stats(config_path: Path, label: str, df: pd.DataFrame) -> None:
    stats = dataset_composition_stats(df)
    with config_path.open("a", encoding="utf-8") as fp:
        for key in sorted(stats):
            fp.write(f"{label}_{key}={stats[key]}\n")


def main() -> None:
    args = parse_args()
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
    rank, world_size, local_rank, device = init_distributed(args)
    is_main = rank == 0
    model_builder, default_size, model_desc, model_key = resolve_model_builder(
        args.model,
        embed_dim=args.embed_dim,
        normals_aux=getattr(args, "normals_aux", False),
        normals_privileged=getattr(args, "normals_privileged", False),
    )
    img_size = args.img_size if args.img_size is not None else default_size
    if args.img_size is not None and args.img_size != default_size and is_main:
        print(
            f"[train] Using requested --img-size {args.img_size} "
            f"(default for {model_desc} is {default_size})."
        )

    set_random_seed(args.seed + rank)
    train_dataset, val_dataset, active_root, train_len, val_len, fold_info, val_user_skin, normals_transform = build_datasets(
        args, args.seed, img_size
    )
    needs_normals = getattr(args, "normals_aux", False) or getattr(args, "normals_privileged", False)

    def _collate_with_normals(batch):
        rgbs, ages, user_ids, normals_list = zip(*batch)
        rgb_batch = torch.stack(rgbs)
        age_batch = torch.stack(ages)
        valid = [n for n in normals_list if n is not None]
        if not valid:
            return rgb_batch, age_batch, list(user_ids), None, None
        h, w = valid[0].shape[-2], valid[0].shape[-1]
        normals_batch = torch.stack(
            [n if n is not None else torch.zeros(3, h, w) for n in normals_list]
        )
        has_normals = torch.tensor([n is not None for n in normals_list], dtype=torch.bool)
        return rgb_batch, age_batch, list(user_ids), normals_batch, has_normals

    train_collate = _collate_with_normals if needs_normals else None

    age_weight_map: dict[int, float] | None = None
    if args.age_reweight_loss:
        age_weight_map = compute_age_weight_map(
            train_dataset.records["age"],
            eps=args.age_weight_eps,
            power=args.age_weight_power,
            min_w=args.age_weight_min,
            max_w=args.age_weight_max,
            normalise=True,
        )
        if is_main and age_weight_map:
            vals = list(age_weight_map.values())
            print(f"[loss] Age reweighting enabled (min={min(vals):.3f}, max={max(vals):.3f}).")
        elif is_main:
            print("[loss] Age reweighting requested but no weights were computed.")

    def _build_age_weight_tensor(batch_ages: torch.Tensor) -> torch.Tensor | None:
        if not age_weight_map:
            return None
        age_ints = torch.round(batch_ages).to(torch.int64).cpu().tolist()
        weights = [age_weight_map.get(int(a), 1.0) for a in age_ints]
        return batch_ages.new_tensor(weights)

    train_loader, val_loader, train_sampler = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        group_size=args.user_group_size,
        num_workers=args.num_workers,
        device=device,
        world_size=world_size,
        rank=rank,
        seed=args.seed,
        collate_fn=train_collate,
    )

    output_dir = Path(args.output_dir).expanduser()
    combined_records = None
    if is_main:
        combined_records = pd.concat(
            [train_dataset.records, val_dataset.records], ignore_index=True
        )
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.txt"
        run_started_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
        with config_path.open("w", encoding="utf-8") as fp:
            fp.write("# Training configuration\n")
            fp.write(f"run_started_at={run_started_at}\n")
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
            fp.write(f"resolved_loss_weight_spread={args.loss_weight_spread}\n")
            fp.write(f"resolved_embed_dim={args.embed_dim}\n")
            fp.write(f"resolved_loss_weight_embed_var={args.loss_weight_embed_var}\n")
            fp.write(f"resolved_loss_weight_embed_contrast={args.loss_weight_embed_contrast}\n")
            fp.write(f"resolved_embed_age_slack={args.embed_age_slack}\n")
            fp.write(f"resolved_embed_contrast_margin={args.embed_contrast_margin}\n")
            fp.write(f"resolved_embed_contrast_age_thresh={args.embed_contrast_age_thresh}\n")
            fp.write(f"resolved_age_reweight_loss={int(args.age_reweight_loss)}\n")
            fp.write(f"resolved_age_weight_eps={args.age_weight_eps}\n")
            fp.write(f"resolved_age_weight_power={args.age_weight_power}\n")
            fp.write(f"resolved_age_weight_min={args.age_weight_min}\n")
            fp.write(f"resolved_age_weight_max={args.age_weight_max}\n")
            fp.write(f"resolved_age_oversample={int(args.age_oversample)}\n")
            fp.write(f"resolved_age_oversample_target={args.age_oversample_target}\n")
            fp.write(f"resolved_age_oversample_max_multiplier={args.age_oversample_max_multiplier}\n")
            fp.write(f"resolved_use_masks={int(args.use_masks)}\n")
        if combined_records is not None:
            _append_dataset_stats(config_path, "dataset", combined_records)
        print(
            f"Using dataset root: {active_root}\n"
            f"Saving artifacts to: {output_dir}\n"
            f"Train images: {train_len}\n"
            f"Val images:   {val_len}\n"
            f"Model: {model_desc} | Image size: {img_size} | "
            f"Per-rank batch size: {args.batch_size} | User group size: {args.user_group_size}\n"
            f"Epochs: {args.epochs} | Learning rate: {args.lr:.2e} | Seed: {args.seed} | World size: {world_size}"
        )
        print(
            f"Loss weights -> NLL: {loss_weights.nll:.3f}, "
            f"MSE: {loss_weights.mse:.3f}, MAE: {loss_weights.mae:.3f}, "
            f"Spread: {args.loss_weight_spread:.3f}, "
            f"Normals: {getattr(args, 'loss_weight_normals', 0.0):.3f}"
        )
        if getattr(args, "normals_privileged", False):
            print(
                f"Privileged normals input: ON | dropout={getattr(args, 'normals_dropout', 0.5):.2f} "
                f"(normals zeroed at val/test time)"
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

    model = model_builder().to(device)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        find_unused_parameters=args.find_unused_params or getattr(args, "normals_aux", False),
    )
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_model_path = output_dir / f"{model_key}_age_regressor_ddp.pth"
    history_log_path = output_dir / "history_distributed.log"
    min_delta = DEFAULT_MIN_DELTA
    patience = max(1, int(args.patience))
    epochs_without_improvement = 0
    history_entries = [] if is_main else None

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        train_sample_count = 0.0
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_mse_sum = 0.0
        train_std_sum = 0.0
        train_spread_sum = 0.0
        train_emb_var_sum = 0.0
        train_emb_contrast_sum = 0.0
        train_normals_sum = 0.0
        _normals_aux = getattr(args, "normals_aux", False)
        _normals_privileged = getattr(args, "normals_privileged", False)
        _normals_dropout = getattr(args, "normals_dropout", 0.5)
        _loss_weight_normals = getattr(args, "loss_weight_normals", 0.0)

        progress = tqdm(
            train_loader,
            desc=f"[Rank {rank}] Epoch {epoch}/{args.epochs}",
            disable=not is_main,
        )
        for batch in progress:
            if _normals_aux or _normals_privileged:
                images, ages, batch_user_ids, normals_gt, has_normals = batch
                normals_gt = normals_gt.to(device, non_blocking=True) if normals_gt is not None else None
                has_normals = has_normals.to(device, non_blocking=True) if has_normals is not None else None
            else:
                images, ages, batch_user_ids = batch
                normals_gt = None
                has_normals = None
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)
            optimizer.zero_grad()
            if _normals_privileged:
                if normals_gt is None:
                    normals_gt = torch.zeros_like(images)
                    has_normals = torch.zeros(images.shape[0], dtype=torch.bool, device=device)
                # Apply per-sample dropout on LUICID normals: randomly zero them out
                # so the model learns to handle zero-normals (matching val/test).
                if _normals_dropout > 0 and has_normals is not None:
                    drop = (torch.rand(has_normals.shape[0], device=device) < _normals_dropout) & has_normals
                    normals_gt = normals_gt.clone()
                    normals_gt[drop] = 0.0
                images = torch.cat([images, normals_gt], dim=1)
            outputs = ddp_model(images)
            z = None
            pred_normals = None
            if isinstance(outputs, (tuple, list)):
                if _normals_aux and len(outputs) == 3:
                    pred_mean, pred_log_var, pred_normals = outputs
                elif _normals_aux and len(outputs) == 4:
                    pred_mean, pred_log_var, z, pred_normals = outputs
                elif len(outputs) == 3:
                    pred_mean, pred_log_var, z = outputs
                else:
                    pred_mean, pred_log_var = outputs
            else:
                pred_mean, pred_log_var = outputs
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
            else:
                spread_loss = None
                loss = base_loss
            embed_var_loss = None
            embed_contrast_loss = None
            if z is not None:
                if args.loss_weight_embed_var > 0:
                    embed_var_loss = embedding_variance_loss(
                        z, batch_user_ids, ages, age_slack=args.embed_age_slack
                    )
                    loss = loss + args.loss_weight_embed_var * embed_var_loss
                if args.loss_weight_embed_contrast > 0:
                    embed_contrast_loss = embedding_contrastive_loss(
                        z,
                        batch_user_ids,
                        ages,
                        margin=args.embed_contrast_margin,
                        age_thresh=args.embed_contrast_age_thresh,
                    )
                    loss = loss + args.loss_weight_embed_contrast * embed_contrast_loss
            norm_loss = None
            if pred_normals is not None and normals_gt is not None and _loss_weight_normals > 0:
                norm_loss = normals_reconstruction_loss(pred_normals, normals_gt, has_normals)
                loss = loss + _loss_weight_normals * norm_loss
            loss.backward()
            optimizer.step()

            batch_size = ages.size(0)
            train_sample_count += batch_size
            train_loss_sum += loss.item() * batch_size
            abs_err = torch.abs(pred_mean - ages)
            sq_err = (pred_mean - ages) ** 2
            batch_std = torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))
            train_mae_sum += torch.sum(abs_err).item()
            train_mse_sum += torch.sum(sq_err).item()
            train_std_sum += torch.sum(batch_std).item()
            if spread_loss is not None:
                train_spread_sum += spread_loss.item() * batch_size
            if embed_var_loss is not None:
                train_emb_var_sum += embed_var_loss.item() * batch_size
            if embed_contrast_loss is not None:
                train_emb_contrast_sum += embed_contrast_loss.item() * batch_size
            if norm_loss is not None:
                train_normals_sum += norm_loss.item() * batch_size

        train_totals = all_reduce_metrics(
            device,
            [
                train_loss_sum,
                train_mae_sum,
                train_mse_sum,
                train_std_sum,
                train_spread_sum,
                train_emb_var_sum,
                train_emb_contrast_sum,
                train_normals_sum,
                train_sample_count,
            ],
        )
        denom_train = max(1.0, train_totals[8])
        train_loss = train_totals[0] / denom_train
        train_mae = train_totals[1] / denom_train
        train_rmse = float(np.sqrt(train_totals[2] / denom_train))
        train_std = train_totals[3] / denom_train
        train_spread = train_totals[4] / denom_train
        train_emb_var = train_totals[5] / denom_train
        train_emb_contrast = train_totals[6] / denom_train
        train_normals = train_totals[7] / denom_train

        ddp_model.eval()
        val_sample_count = 0.0
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_mse_sum = 0.0
        val_std_sum = 0.0
        val_spread_sum = 0.0
        val_emb_var_sum = 0.0
        val_emb_contrast_sum = 0.0
        val_targets = []
        val_predictions = []
        val_log_vars = []
        val_user_ids = []

        with torch.no_grad():
            for images, ages, batch_user_ids in val_loader:
                images = images.to(device, non_blocking=True)
                ages = ages.to(device, non_blocking=True)
                if _normals_privileged:
                    zeros_n = torch.zeros(
                        images.shape[0], 3, images.shape[2], images.shape[3], device=device
                    )
                    images = torch.cat([images, zeros_n], dim=1)
                outputs = ddp_model(images)
                z = None
                if isinstance(outputs, (tuple, list)):
                    if _normals_aux and len(outputs) == 3:
                        pred_mean, pred_log_var, _pn = outputs
                    elif _normals_aux and len(outputs) == 4:
                        pred_mean, pred_log_var, z, _pn = outputs
                    elif len(outputs) == 3:
                        pred_mean, pred_log_var, z = outputs
                    else:
                        pred_mean, pred_log_var = outputs
                else:
                    pred_mean, pred_log_var = outputs
                batch_loss = weighted_regression_loss(pred_mean, pred_log_var, ages, loss_weights)

                batch_size = ages.size(0)
                val_sample_count += batch_size
                val_loss_sum += batch_loss.item() * batch_size
                val_mae_sum += torch.sum(torch.abs(pred_mean - ages)).item()
                val_mse_sum += torch.sum((pred_mean - ages) ** 2).item()
                val_std_sum += torch.sum(
                    torch.exp(0.5 * torch.clamp(pred_log_var.detach(), min=-10.0, max=10.0))
                ).item()
                if args.loss_weight_spread > 0:
                    val_spread_loss = intra_user_spread_loss(pred_mean, batch_user_ids)
                    val_spread_sum += val_spread_loss.item() * batch_size
                if z is not None and args.loss_weight_embed_var > 0:
                    val_emb_var_loss = embedding_variance_loss(
                        z, batch_user_ids, ages, age_slack=args.embed_age_slack
                    )
                    val_emb_var_sum += val_emb_var_loss.item() * batch_size
                if z is not None and args.loss_weight_embed_contrast > 0:
                    val_emb_contrast_loss = embedding_contrastive_loss(
                        z,
                        batch_user_ids,
                        ages,
                        margin=args.embed_contrast_margin,
                        age_thresh=args.embed_contrast_age_thresh,
                    )
                    val_emb_contrast_sum += val_emb_contrast_loss.item() * batch_size

                val_targets.extend(ages.detach().cpu().tolist())
                val_predictions.extend(pred_mean.detach().cpu().tolist())
                val_log_vars.extend(pred_log_var.detach().cpu().tolist())
                val_user_ids.extend(list(batch_user_ids))

        val_totals = all_reduce_metrics(
            device,
            [
                val_loss_sum,
                val_mae_sum,
                val_mse_sum,
                val_std_sum,
                val_spread_sum,
                val_emb_var_sum,
                val_emb_contrast_sum,
                val_sample_count,
            ],
        )
        denom_val = max(1.0, val_totals[7])
        val_loss = val_totals[0] / denom_val
        val_mae = val_totals[1] / denom_val
        val_rmse = float(np.sqrt(val_totals[2] / denom_val))
        val_std = val_totals[3] / denom_val
        val_spread = val_totals[4] / denom_val
        val_emb_var = val_totals[5] / denom_val
        val_emb_contrast = val_totals[6] / denom_val

        if is_main:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, "
                f"train_rmse={train_rmse:.4f}, train_std={train_std:.4f}, train_spread={train_spread:.4f}, "
                f"train_emb_var={train_emb_var:.4f}, train_emb_contrast={train_emb_contrast:.4f}, "
                f"train_normals={train_normals:.4f} | "
                f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, val_rmse={val_rmse:.4f}, "
                f"val_std={val_std:.4f}, val_spread={val_spread:.4f}, "
                f"val_emb_var={val_emb_var:.4f}, val_emb_contrast={val_emb_contrast:.4f}"
            )
            with history_log_path.open("a", encoding="utf-8") as log_fp:
                log_fp.write(
                    f"Epoch {epoch},train_loss={train_loss:.6f},train_mae={train_mae:.6f},train_rmse={train_rmse:.6f},"
                    f"train_std={train_std:.6f},train_spread={train_spread:.6f},"
                    f"train_emb_var={train_emb_var:.6f},train_emb_contrast={train_emb_contrast:.6f},"
                    f"train_normals={train_normals:.6f},"
                    f"val_loss={val_loss:.6f},val_mae={val_mae:.6f},val_rmse={val_rmse:.6f},"
                    f"val_std={val_std:.6f},val_spread={val_spread:.6f},"
                    f"val_emb_var={val_emb_var:.6f},val_emb_contrast={val_emb_contrast:.6f}\n"
                )
            history_entries.append(
                {
                    "epoch": epoch,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                }
            )

        previous_best = best_val_loss
        save_best = val_loss < best_val_loss
        if save_best:
            improvement = (
                float("inf") if previous_best == float("inf") else previous_best - val_loss
            )
            best_val_loss = val_loss
            if improvement == float("inf") or improvement >= min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if save_best:
            # All ranks must participate in the gather to avoid collective mismatch.
            targets_all = gather_all_lists(val_targets, world_size)
            preds_all = gather_all_lists(val_predictions, world_size)
            log_vars_all = gather_all_lists(val_log_vars, world_size)
            user_ids_all = gather_all_lists(val_user_ids, world_size)

        if save_best and is_main:
            model_to_save = ddp_model.module
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"[Rank 0] Saved best model to {best_model_path} (val_loss={val_loss:.4f})")

            targets_arr = np.asarray(targets_all, dtype=float)
            preds_arr = np.asarray(preds_all, dtype=float)
            log_vars_arr = np.asarray(log_vars_all, dtype=float)
            user_ids_list = list(user_ids_all)

            raw_preds_path = output_dir / "val_predictions_raw_ddp.npz"
            raw_skin_colors = map_user_series_to_array(user_ids_list, val_user_skin)
            np.savez(
                raw_preds_path,
                targets=targets_arr,
                pred_mean=preds_arr,
                pred_log_var=log_vars_arr,
                user_ids=np.asarray(user_ids_list, dtype=str),
                skin_color=raw_skin_colors,
                epoch=epoch,
            )

            saved_artifacts = []
            for group_size in eval_group_sizes:
                agg_rng = random.Random(eval_agg_seed + group_size)
                aggregated = aggregate_predictions_by_user(
                    user_ids_list,
                    targets_arr,
                    preds_arr,
                    log_vars_arr,
                    group_size=group_size,
                    rng=agg_rng,
                )
                suffix = f"n{group_size}_ddp"
                if args.eval_age_gate_mode == "age_threshold":
                    gate_results = compute_age_gate_curves_direct_threshold(
                        aggregated["targets"],
                        aggregated["pred_mean"],
                        age_min=args.age_gate_threshold_min,
                        age_max=args.age_gate_threshold_max,
                        num_thresholds=201,
                    )
                else:
                    gate_results = compute_age_gate_curves(
                        aggregated["targets"],
                        aggregated["pred_mean"],
                        aggregated["pred_log_var"],
                        age_threshold=18.0,
                        num_thresholds=201,
                    )
                adult_gate = gate_results["adult_gate"]

                # Select tau closest to top-left (0,1)
                def _select_best_tau(fprs_arr, tprs_arr, thresholds_arr):
                    fprs_np = np.asarray(fprs_arr, dtype=float)
                    tprs_np = np.asarray(tprs_arr, dtype=float)
                    thr_np = np.asarray(thresholds_arr, dtype=float)
                    if fprs_np.size == 0:
                        return None
                    idx = int(np.argmin((fprs_np ** 2) + ((1.0 - tprs_np) ** 2)))
                    return float(thr_np[idx])

                best_tau_adult_gate = _select_best_tau(
                    adult_gate["fpr"], adult_gate["tpr"], adult_gate["thresholds"]
                )
                selected_tau_adult_gate = (
                    best_tau_adult_gate if best_tau_adult_gate is not None else CHALLENGE_PROB_TAU
                )

                preds_dump_path = output_dir / f"val_predictions_{suffix}.npz"
                agg_skin_colors = map_user_series_to_array(aggregated["user_ids"], val_user_skin)
                np.savez(
                    preds_dump_path,
                    targets=aggregated["targets"],
                    pred_mean=aggregated["pred_mean"],
                    pred_log_var=aggregated["pred_log_var"],
                    adult_prob=gate_results["adult_prob"],
                    user_ids=aggregated["user_ids"],
                    skin_color=agg_skin_colors,
                    epoch=epoch,
                    group_size=group_size,
                )
                skin_summary_path = save_group_summary_csv(
                    output_dir / f"age_metrics_by_skin_color_{suffix}.csv",
                    compute_group_summary_rows(
                        agg_skin_colors,
                        aggregated["targets"],
                        aggregated["pred_mean"],
                        aggregated["pred_log_var"],
                        user_ids=aggregated["user_ids"],
                    ),
                    group_name="skin_color",
                )

                challenge_thresholds = np.arange(18, 31, 1, dtype=float)  # 18..30
                fpr_rows = compute_challenge_fpr_table(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    thresholds=challenge_thresholds,
                    prob_threshold=selected_tau_adult_gate,
                    bins=CHALLENGE_BINS,
                )
                challenge_csv = output_dir / f"challenge_fpr_bins_adult_gate_{suffix}.csv"
                with challenge_csv.open("w", encoding="utf-8") as fp:
                    header = ["threshold"] + [label for label, _, _ in CHALLENGE_BINS] + ["total"]
                    fp.write(",".join(header) + "\n")
                    for row in fpr_rows:
                        values = [f"{row['threshold']:.1f}"] + [
                            f"{row[label]:.6f}" for label, _, _ in CHALLENGE_BINS
                        ] + [f"{row['total']:.6f}"]
                        fp.write(",".join(values) + "\n")
                fnr_rows = compute_challenge_fnr_table_adult_gate(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    thresholds=challenge_thresholds,
                    prob_threshold=selected_tau_adult_gate,
                )
                fnr_csv = output_dir / f"challenge_fnr_bins_adult_gate_{suffix}.csv"
                with fnr_csv.open("w", encoding="utf-8") as fp:
                    header = ["threshold"] + [label for label, _, _ in CHALLENGE_FNR_BINS] + ["total"]
                    fp.write(",".join(header) + "\n")
                    for row in fnr_rows:
                        values = [f"{row['threshold']:.1f}"] + [
                            f"{row[label]:.6f}" for label, _, _ in CHALLENGE_FNR_BINS
                        ] + [f"{row['total']:.6f}"]
                        fp.write(",".join(values) + "\n")

                roc_adult_gate_path = output_dir / f"roc_adult_gate_{suffix}.png"
                DisplayUtils.plot_roc_curve(
                    adult_gate["fpr"],
                    adult_gate["tpr"],
                    thresholds=adult_gate["thresholds"],
                    save_path=roc_adult_gate_path,
                    title=f"ROC - Adult Gate (n={group_size}, DDP)",
                    auc_value=adult_gate["auc"],
                    highlight_tau=best_tau_adult_gate,
                    show=False,
                )

                metrics_csv_path = output_dir / f"age_gate_metrics_{suffix}.csv"
                with metrics_csv_path.open("w", encoding="utf-8") as metrics_fp:
                    metrics_fp.write("gate,tau,fpr,fnr,tpr,tnr\n")
                    for tau, fpr, fnr, tpr_val, tnr in zip(
                        adult_gate["thresholds"],
                        adult_gate["fpr"],
                        adult_gate["fnr"],
                        adult_gate["tpr"],
                        adult_gate["tnr"],
                    ):
                        metrics_fp.write(
                            f"adult_gate,{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n"
                        )

                challenge_thresholds = np.arange(18, 31, 1, dtype=float)
                fpr_rows_weighted_best = compute_challenge_fpr_table_weighted(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    thresholds=challenge_thresholds,
                    prob_threshold=selected_tau_adult_gate,
                    bins=CHALLENGE_BINS,
                )
                # First threshold with total <= 0.001 (ascending order); fallback to min-total threshold
                challenge_line = None
                for row in sorted(fpr_rows_weighted_best, key=lambda r: float(r.get("threshold", 1e9))):
                    if float(row.get("total", 1.0)) <= 0.001:
                        challenge_line = float(row["threshold"])
                        break
                if challenge_line is None and fpr_rows_weighted_best:
                    best = min(fpr_rows_weighted_best, key=lambda r: float(r.get("total", 1.0)))
                    challenge_line = float(best["threshold"])
                if is_main:
                    print(f"[debug] challenge_line (weighted total<=0.001) for n={group_size}: {challenge_line}")

                scatter_path = output_dir / f"age_val_scatter_{suffix}.png"
                if DisplayUtils.save_regression_scatter(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    save_path=scatter_path,
                    title=f"Validation Age Predictions (epoch {epoch}, n={group_size}, DDP)",
                    axis_limits=(0.0, 70.0),
                    point_size=20,
                    alpha=0.6,
                ):
                    saved_artifacts.append(
                        f"n={group_size} -> {roc_adult_gate_path.name}, {metrics_csv_path.name}, {preds_dump_path.name}, {challenge_csv.name}, {fnr_csv.name}, {skin_summary_path.name}, {scatter_path.name}"
                    )
                else:
                    saved_artifacts.append(
                        f"n={group_size} -> {roc_adult_gate_path.name}, {metrics_csv_path.name}, {preds_dump_path.name}, {challenge_csv.name}, {fnr_csv.name}, {skin_summary_path.name}"
                    )

                error_plot_path = output_dir / f"age_error_by_target_{suffix}.png"
                if DisplayUtils.save_error_by_age(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    save_path=error_plot_path,
                    title=f"Error vs Target Age (epoch {epoch}, n={group_size}, DDP)",
                ):
                    saved_artifacts.append(f"n={group_size} -> {error_plot_path.name}")

            print("[Rank 0] Saved/updated eval artifacts for: " + "; ".join(saved_artifacts))

        if epochs_without_improvement >= patience:
            if is_main:
                print(
                    f"Stopping early at epoch {epoch}: validation loss did not improve by at least "
                    f"{min_delta:.3f} for {patience} consecutive epochs."
                )
                history_plot_path = output_dir / "history_plot_ddp.png"
                DisplayUtils.plot_loss_history(
                    history_entries,
                    save_path=history_plot_path,
                    show=False,
                    title="Training History (MAE & RMSE, DDP)",
                )
                print(f"[Rank 0] Saved training history plot to {history_plot_path}")
            break

    if is_main:
        stale_users_hist = output_dir / "age_distribution_users_ddp.png"
        if stale_users_hist.exists():
            stale_users_hist.unlink()
            print(f"[Rank 0] Removed stale per-user age histogram {stale_users_hist}")
        test_meta = load_filtered_test_metadata(args, active_root)
        hist_samples_path = output_dir / "age_distribution_samples_ddp.png"
        saved_hist_samples = DisplayUtils.plot_age_histograms(
            train_dataset.records["age"].to_numpy(),
            val_dataset.records["age"].to_numpy(),
            save_path=hist_samples_path,
            show=False,
            title=(
                "Per-sample age distribution (train vs val vs test, DDP)"
                if test_meta is not None and not test_meta.empty
                else "Per-sample age distribution (train vs val, DDP)"
            ),
            train_title="Train samples",
            eval_title="Val samples",
            count_label="Number of samples",
            extra_ages=(
                test_meta["age"].to_numpy()
                if test_meta is not None and not test_meta.empty
                else None
            ),
            extra_title=(
                "Test samples"
                if test_meta is not None and not test_meta.empty
                else None
            ),
        )
        if saved_hist_samples:
            print(f"[Rank 0] Saved per-sample age histograms to {saved_hist_samples}")

        # Final challenge-threshold table using best checkpoint on rank 0
        if best_model_path.exists():
            print("[Rank 0] Computing challenge-threshold FPR table on val set using best model...")
            eval_model = model_builder()
            state = torch.load(best_model_path, map_location=device)
            eval_model.load_state_dict(state)
            eval_model = eval_model.to(device)
            eval_model.eval()

            eval_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            all_targets: list[float] = []
            all_means: list[float] = []
            all_log_vars: list[float] = []
            all_user_ids: list = []
            with torch.no_grad():
                for images, ages, batch_user_ids in eval_loader:
                    images = images.to(device)
                    ages = ages.to(device)
                    if getattr(args, "normals_privileged", False):
                        zeros_n = torch.zeros(
                            images.shape[0], 3, images.shape[2], images.shape[3], device=device
                        )
                        images = torch.cat([images, zeros_n], dim=1)
                    outputs = eval_model(images)
                    if isinstance(outputs, (tuple, list)):
                        if len(outputs) == 3:
                            mean, log_var, _ = outputs
                        else:
                            mean, log_var = outputs
                    else:
                        mean, log_var = outputs
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
                if args.eval_age_gate_mode == "age_threshold":
                    gate_results = compute_age_gate_curves_direct_threshold(
                        aggregated["targets"],
                        aggregated["pred_mean"],
                        age_min=args.age_gate_threshold_min,
                        age_max=args.age_gate_threshold_max,
                        num_thresholds=201,
                    )
                else:
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

                adult_gate = gate_results["adult_gate"]
                best_tau_adult_gate = _select_best_tau(
                    adult_gate["fpr"], adult_gate["tpr"], adult_gate["thresholds"]
                )
                selected_tau_adult_gate = (
                    best_tau_adult_gate if best_tau_adult_gate is not None else CHALLENGE_PROB_TAU
                )
                fpr_rows = compute_challenge_fpr_table(
                    aggregated["targets"],
                    aggregated["pred_mean"],
                    aggregated["pred_log_var"],
                    thresholds=challenge_thresholds,
                    prob_threshold=selected_tau_adult_gate,
                    bins=CHALLENGE_BINS,
                )
                challenge_csv = output_dir / f"challenge_fpr_bins_adult_gate_ddp_n{group_size}.csv"
                with challenge_csv.open("w", encoding="utf-8") as fp:
                    header = ["threshold"] + [label for label, _, _ in CHALLENGE_BINS] + ["total"]
                    fp.write(",".join(header) + "\n")
                    for row in fpr_rows:
                        values = [f"{row['threshold']:.1f}"] + [
                            f"{row[label]:.6f}" for label, _, _ in CHALLENGE_BINS
                        ] + [f"{row['total']:.6f}"]
                        fp.write(",".join(values) + "\n")
                print(f"[Rank 0] Saved challenge FPR table to {challenge_csv}")
        else:
            print("[Rank 0] Best checkpoint not found; skipped challenge-threshold table.")

        # ── Full evaluation on held-out test set ──────────────────────────
        if args.test_users_file and best_model_path.exists():
            print("[Rank 0] Running full evaluation on held-out test set...")
            _, test_transform = build_transforms(img_size)
            if test_meta is None:
                test_meta = load_filtered_test_metadata(args, active_root)

            if test_meta.empty:
                print("[Rank 0] No test samples found after filtering; skipping test evaluation.")
            else:
                print(f"[Rank 0] Test set: {test_meta['user_id'].nunique()} users, {len(test_meta)} samples.")
                test_dataset = AgeDataset(test_meta, transform=test_transform, use_masks=args.use_masks)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

                test_model = model_builder()
                test_user_skin = build_user_skin_color_series(test_meta)
                state = torch.load(best_model_path, map_location=device)
                test_model.load_state_dict(state)
                test_model = test_model.to(device)
                test_model.eval()

                t_targets: list[float] = []
                t_means: list[float] = []
                t_log_vars: list[float] = []
                t_user_ids: list = []
                with torch.no_grad():
                    for images, ages, batch_user_ids in test_loader:
                        images = images.to(device)
                        if getattr(args, "normals_privileged", False):
                            zeros_n = torch.zeros(
                                images.shape[0], 3, images.shape[2], images.shape[3], device=device
                            )
                            images = torch.cat([images, zeros_n], dim=1)
                        outputs = test_model(images)
                        if isinstance(outputs, (tuple, list)):
                            if len(outputs) == 3:
                                mean, log_var, _ = outputs
                            else:
                                mean, log_var = outputs
                        else:
                            mean, log_var = outputs
                        t_targets.extend(ages.tolist())
                        t_means.extend(mean.cpu().tolist())
                        t_log_vars.extend(log_var.cpu().tolist())
                        t_user_ids.extend(list(batch_user_ids))

                t_targets_arr = np.asarray(t_targets, dtype=float)
                t_means_arr = np.asarray(t_means, dtype=float)
                t_log_vars_arr = np.asarray(t_log_vars, dtype=float)

                np.savez(
                    output_dir / "test_predictions_raw_ddp.npz",
                    targets=t_targets_arr,
                    pred_mean=t_means_arr,
                    pred_log_var=t_log_vars_arr,
                    user_ids=np.asarray(t_user_ids, dtype=str),
                    skin_color=map_user_series_to_array(t_user_ids, test_user_skin),
                )

                def _select_best_tau_test(fprs_arr, tprs_arr, thresholds_arr):
                    fprs_np = np.asarray(fprs_arr, dtype=float)
                    tprs_np = np.asarray(tprs_arr, dtype=float)
                    thr_np = np.asarray(thresholds_arr, dtype=float)
                    if fprs_np.size == 0:
                        return None
                    idx = int(np.argmin((fprs_np ** 2) + ((1.0 - tprs_np) ** 2)))
                    return float(thr_np[idx])

                challenge_thresholds = np.arange(18, 31, 1, dtype=float)
                summary_rows = []

                for group_size in eval_group_sizes:
                    agg_rng = random.Random(eval_agg_seed + group_size)
                    aggregated = aggregate_predictions_by_user(
                        t_user_ids, t_targets_arr, t_means_arr, t_log_vars_arr,
                        group_size=group_size, rng=agg_rng,
                    )
                    suffix = f"n{group_size}_ddp"
                    if args.eval_age_gate_mode == "age_threshold":
                        gate_results = compute_age_gate_curves_direct_threshold(
                            aggregated["targets"],
                            aggregated["pred_mean"],
                            age_min=args.age_gate_threshold_min,
                            age_max=args.age_gate_threshold_max,
                            num_thresholds=201,
                        )
                    else:
                        gate_results = compute_age_gate_curves(
                            aggregated["targets"],
                            aggregated["pred_mean"],
                            aggregated["pred_log_var"],
                            age_threshold=18.0,
                            num_thresholds=201,
                        )
                    adult_gate = gate_results["adult_gate"]

                    best_tau_adult_gate = _select_best_tau_test(
                        adult_gate["fpr"], adult_gate["tpr"], adult_gate["thresholds"]
                    )
                    selected_tau_adult_gate = (
                        best_tau_adult_gate if best_tau_adult_gate is not None else CHALLENGE_PROB_TAU
                    )

                    np.savez(
                        output_dir / f"test_predictions_{suffix}.npz",
                        targets=aggregated["targets"],
                        pred_mean=aggregated["pred_mean"],
                        pred_log_var=aggregated["pred_log_var"],
                        adult_prob=gate_results["adult_prob"],
                        user_ids=aggregated["user_ids"],
                        skin_color=map_user_series_to_array(aggregated["user_ids"], test_user_skin),
                        group_size=group_size,
                    )
                    save_group_summary_csv(
                        output_dir / f"test_age_metrics_by_skin_color_{suffix}.csv",
                        compute_group_summary_rows(
                            map_user_series_to_array(aggregated["user_ids"], test_user_skin),
                            aggregated["targets"],
                            aggregated["pred_mean"],
                            aggregated["pred_log_var"],
                            user_ids=aggregated["user_ids"],
                        ),
                        group_name="skin_color",
                    )

                    with (output_dir / f"test_age_gate_metrics_{suffix}.csv").open("w", encoding="utf-8") as fp:
                        fp.write("gate,tau,fpr,fnr,tpr,tnr\n")
                        for tau, fpr, fnr, tpr_val, tnr in zip(
                            adult_gate["thresholds"], adult_gate["fpr"],
                            adult_gate["fnr"], adult_gate["tpr"], adult_gate["tnr"],
                        ):
                            fp.write(f"adult_gate,{tau:.4f},{fpr:.6f},{fnr:.6f},{tpr_val:.6f},{tnr:.6f}\n")

                    fpr_rows = compute_challenge_fpr_table(
                        aggregated["targets"], aggregated["pred_mean"], aggregated["pred_log_var"],
                        thresholds=challenge_thresholds, prob_threshold=selected_tau_adult_gate,
                        bins=CHALLENGE_BINS,
                    )
                    with (output_dir / f"test_challenge_fpr_bins_adult_gate_{suffix}.csv").open("w", encoding="utf-8") as fp:
                        header = ["threshold"] + [label for label, _, _ in CHALLENGE_BINS] + ["total"]
                        fp.write(",".join(header) + "\n")
                        for row in fpr_rows:
                            values = [f"{row['threshold']:.1f}"] + [
                                f"{row[label]:.6f}" for label, _, _ in CHALLENGE_BINS
                            ] + [f"{row['total']:.6f}"]
                            fp.write(",".join(values) + "\n")

                    fnr_rows = compute_challenge_fnr_table_adult_gate(
                        aggregated["targets"], aggregated["pred_mean"], aggregated["pred_log_var"],
                        thresholds=challenge_thresholds, prob_threshold=selected_tau_adult_gate,
                    )
                    with (output_dir / f"test_challenge_fnr_bins_adult_gate_{suffix}.csv").open("w", encoding="utf-8") as fp:
                        header = ["threshold"] + [label for label, _, _ in CHALLENGE_FNR_BINS] + ["total"]
                        fp.write(",".join(header) + "\n")
                        for row in fnr_rows:
                            values = [f"{row['threshold']:.1f}"] + [
                                f"{row[label]:.6f}" for label, _, _ in CHALLENGE_FNR_BINS
                            ] + [f"{row['total']:.6f}"]
                            fp.write(",".join(values) + "\n")

                    DisplayUtils.plot_roc_curve(
                        adult_gate["fpr"], adult_gate["tpr"],
                        thresholds=adult_gate["thresholds"],
                        save_path=output_dir / f"test_roc_adult_gate_{suffix}.png",
                        title=f"ROC - Adult Gate [TEST] (n={group_size}, DDP)",
                        auc_value=adult_gate["auc"],
                        highlight_tau=best_tau_adult_gate,
                        show=False,
                    )

                    DisplayUtils.save_regression_scatter(
                        aggregated["targets"], aggregated["pred_mean"],
                        save_path=output_dir / f"test_age_scatter_{suffix}.png",
                        title=f"Test Age Predictions (n={group_size}, DDP)",
                        axis_limits=(0.0, 70.0),
                        point_size=20,
                        alpha=0.6,
                    )

                    DisplayUtils.save_error_by_age(
                        aggregated["targets"], aggregated["pred_mean"],
                        save_path=output_dir / f"test_age_error_by_target_{suffix}.png",
                        title=f"Error vs Target Age [TEST] (n={group_size}, DDP)",
                    )

                    mae = float(np.mean(np.abs(aggregated["targets"] - aggregated["pred_mean"])))
                    rmse = float(np.sqrt(np.mean((aggregated["targets"] - aggregated["pred_mean"]) ** 2)))
                    summary_rows.append({
                        "group_size": group_size,
                        "mae": mae,
                        "rmse": rmse,
                        "auc_adult_gate": adult_gate["auc"],
                    })

                with (output_dir / "test_summary_ddp.csv").open("w", encoding="utf-8") as fp:
                    fp.write("group_size,mae,rmse,auc_adult_gate\n")
                    for row in summary_rows:
                        fp.write(
                            f"{row['group_size']},{row['mae']:.4f},{row['rmse']:.4f},"
                            f"{row['auc_adult_gate']:.4f}\n"
                        )
                print(f"[Rank 0] Test evaluation complete. Artifacts saved to {output_dir}")

        elif args.test_users_file and not best_model_path.exists():
            print("[Rank 0] Best checkpoint not found; skipped test evaluation.")

        # ── Normals reconstruction evaluation on LUICIDHands ─────────────────
        if is_main and getattr(args, "normals_aux", False) and best_model_path.exists():
            print("[Rank 0] Running normals reconstruction evaluation on LUICIDHands...")
            try:
                _, _test_tf = build_transforms(img_size)
                luicid_meta = filter_metadata(load_lucid_metadata(root=active_root))
                luicid_normals_meta = luicid_meta[luicid_meta["normals_path"].notna()].reset_index(drop=True)
                if luicid_normals_meta.empty:
                    print("[Rank 0] No LUICID normals found; skipping reconstruction evaluation.")
                else:
                    print(f"[Rank 0] LUICID normals samples: {len(luicid_normals_meta)}")
                    recon_ds = AgeDataset(
                        luicid_normals_meta,
                        transform=_test_tf,
                        normals_transform=normals_transform,
                    )

                    def _normals_collate(batch):
                        imgs = torch.stack([b[0] for b in batch])
                        ages = torch.stack([b[1] for b in batch])
                        uids = [b[2] for b in batch]
                        norms = [b[3] for b in batch]
                        valid = [n for n in norms if n is not None]
                        if valid:
                            h, w = valid[0].shape[1], valid[0].shape[2]
                            norms_t = torch.stack(
                                [n if n is not None else torch.zeros(3, h, w) for n in norms]
                            )
                        else:
                            norms_t = torch.zeros(len(batch), 3, 1, 1)
                        has_n = torch.tensor([n is not None for n in norms], dtype=torch.bool)
                        return imgs, ages, uids, norms_t, has_n

                    recon_loader = DataLoader(
                        recon_ds, batch_size=16, shuffle=False,
                        num_workers=0, collate_fn=_normals_collate,
                    )
                    recon_model = model_builder()
                    state = torch.load(best_model_path, map_location=device)
                    recon_model.load_state_dict(state)
                    recon_model = recon_model.to(device)
                    recon_model.eval()

                    all_rgbs, all_gt, all_pred = [], [], []
                    with torch.no_grad():
                        for imgs_b, _ages_b, _uids_b, gt_b, has_n_b in recon_loader:
                            imgs_b = imgs_b.to(device)
                            outputs = recon_model(imgs_b)
                            if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
                                print("[Rank 0] Model did not return normals output; aborting.")
                                break
                            pred_b = outputs[-1].cpu()  # pred_normals is last output
                            mask = has_n_b
                            if not mask.any():
                                continue
                            all_rgbs.append(imgs_b[mask].cpu())
                            all_gt.append(gt_b[mask])
                            all_pred.append(pred_b[mask])

                    if all_rgbs:
                        rgbs_t = torch.cat(all_rgbs, dim=0).numpy()    # [N, 3, H, W]
                        gt_t = torch.cat(all_gt, dim=0).numpy()        # [N, 3, H, W] in [-1,1]
                        pred_t = torch.cat(all_pred, dim=0).numpy()    # [N, 3, H, W] in [-1,1]

                        # Per-sample cosine similarity and per-pixel error maps
                        cos_sims_list, error_maps_list = [], []
                        for i in range(len(rgbs_t)):
                            p = pred_t[i].reshape(3, -1)   # [3, H*W]
                            g = gt_t[i].reshape(3, -1)
                            p_norm = p / (np.linalg.norm(p, axis=0, keepdims=True) + 1e-6)
                            g_norm = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-6)
                            cos_map = (p_norm * g_norm).sum(axis=0)    # [H*W]
                            h, w = pred_t[i].shape[1], pred_t[i].shape[2]
                            cos_sims_list.append(float(cos_map.mean()))
                            error_maps_list.append((1 - cos_map).reshape(h, w))

                        cos_sims_arr = np.array(cos_sims_list, dtype=np.float32)
                        error_maps_arr = np.stack(error_maps_list).astype(np.float32)

                        # Convert to display-ready arrays
                        _imnet_mean = np.array([0.485, 0.456, 0.406])
                        _imnet_std = np.array([0.229, 0.224, 0.225])
                        rgbs_disp = rgbs_t.transpose(0, 2, 3, 1) * _imnet_std + _imnet_mean
                        gt_disp = (gt_t.transpose(0, 2, 3, 1) + 1.0) / 2.0
                        pred_disp = (pred_t.transpose(0, 2, 3, 1) + 1.0) / 2.0

                        cos_csv = output_dir / "luicid_normals_cosine_sim.csv"
                        with cos_csv.open("w", encoding="utf-8") as fp:
                            fp.write("sample_idx,cos_sim\n")
                            for si, cs in enumerate(cos_sims_arr):
                                fp.write(f"{si},{cs:.6f}\n")

                        DisplayUtils.save_normals_reconstruction_grid(
                            rgbs_disp, gt_disp, pred_disp, error_maps_arr, cos_sims_arr,
                            save_path=output_dir / "luicid_normals_reconstruction.png",
                            n_samples=8,
                            title=f"LUICID Normal Reconstruction — {model_key}",
                        )
                        print(
                            f"[Rank 0] Normals reconstruction: "
                            f"mean cos-sim={cos_sims_arr.mean():.4f} ± {cos_sims_arr.std():.4f}"
                        )
                        print(f"[Rank 0] Saved: {cos_csv.name}, luicid_normals_reconstruction.png")
            except Exception as _exc:
                print(f"[Rank 0] Normals reconstruction evaluation failed: {_exc}")

        print("Distributed training complete. Best model saved based on validation improvement.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
