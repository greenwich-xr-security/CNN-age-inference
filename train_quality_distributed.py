#!/usr/bin/env python3
"""Distributed training for hand-image quality assessment for age regression."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataset.quality import QualityDataset
from dataset.transforms import build_transforms
from models import resolve_quality_model_builder
from train_age import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train quality assessor from generated quality targets.")
    parser.add_argument("--quality-targets", required=True, help="CSV produced by generate_quality_targets.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="b0", help="EfficientNet quality model variant.")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--fold-index", type=int, required=True, help="Validation fold index.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-masks", action="store_true")
    parser.add_argument("--dist-backend", default="nccl")
    return parser.parse_args()


def init_distributed(args: argparse.Namespace):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        backend = args.dist_backend if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def gather_objects(local_values: list) -> list:
    if not is_distributed():
        return list(local_values)
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, list(local_values))
    merged = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def drop_duplicate_prediction_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    deduped: list[dict[str, object]] = []
    for row in rows:
        image_path = str(row.get("image_path", ""))
        if image_path in seen:
            continue
        seen.add(image_path)
        deduped.append(row)
    removed = len(rows) - len(deduped)
    if removed:
        print(f"[quality] dropped {removed} duplicate validation prediction rows from DDP padding.")
    return deduped


def quality_loss(outputs: torch.Tensor, targets: torch.Tensor, _args: argparse.Namespace) -> torch.Tensor:
    pred_quality = torch.sigmoid(outputs[:, 0])
    return F.smooth_l1_loss(pred_quality, targets[:, 0])


def decode_outputs(outputs: torch.Tensor) -> dict[str, np.ndarray]:
    return {
        "pred_quality_score": torch.sigmoid(outputs[:, 0]).detach().cpu().numpy(),
    }


def build_loaders(args: argparse.Namespace, img_size: int, device: torch.device, rank: int, world_size: int):
    df = pd.read_csv(args.quality_targets)
    if "fold_index" not in df.columns:
        raise ValueError("quality targets must include fold_index.")
    folds = sorted(int(v) for v in df["fold_index"].dropna().unique())
    if args.fold_index not in folds:
        raise ValueError(f"fold_index {args.fold_index} not present in targets; available folds: {folds}")
    train_df = df[df["fold_index"].astype(int) != args.fold_index].reset_index(drop=True)
    val_df = df[df["fold_index"].astype(int) == args.fold_index].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Quality train/validation split is empty.")
    train_tf, val_tf = build_transforms(img_size)
    train_ds = QualityDataset(train_df, transform=train_tf, use_masks=args.use_masks)
    val_ds = QualityDataset(val_df, transform=val_tf, use_masks=args.use_masks)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader, train_sampler, len(train_ds), len(val_ds)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed(args)
    is_main = rank == 0
    set_random_seed(args.seed + rank)
    model_builder, default_size, model_desc, model_key = resolve_quality_model_builder(args.model)
    img_size = args.img_size if args.img_size is not None else default_size
    train_loader, val_loader, train_sampler, train_len, val_len = build_loaders(
        args, img_size, device, rank, world_size
    )
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "quality_config.txt").open("w", encoding="utf-8") as fp:
            for key, value in sorted(vars(args).items()):
                fp.write(f"{key}={value}\n")
            fp.write(f"resolved_model={model_desc}\nresolved_img_size={img_size}\n")
            fp.write(f"train_samples={train_len}\nval_samples={val_len}\n")

    model = model_builder().to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    stale_epochs = 0
    best_path = output_dir / f"{model_key}_quality_assessor_ddp.pth"
    history_path = output_dir / "quality_history.log"

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, targets, _user_ids, _paths in tqdm(train_loader, disable=not is_main, desc=f"quality {epoch}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = quality_loss(outputs, targets, args)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * images.shape[0]
            train_count += images.shape[0]

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        pred_rows: list[dict[str, object]] = []
        with torch.no_grad():
            for images, targets, user_ids, image_paths in val_loader:
                images = images.to(device, non_blocking=True)
                targets_dev = targets.to(device, non_blocking=True)
                outputs = model(images)
                loss = quality_loss(outputs, targets_dev, args)
                val_loss_sum += float(loss.item()) * images.shape[0]
                val_count += images.shape[0]
                decoded = decode_outputs(outputs)
                targets_np = targets.numpy()
                for i, path in enumerate(image_paths):
                    row = {
                        "image_path": str(path),
                        "user_id": str(user_ids[i]),
                        "target_quality_score": float(targets_np[i, 0]),
                    }
                    for key, values in decoded.items():
                        row[key] = float(values[i])
                    pred_rows.append(row)

        totals = torch.tensor([train_loss_sum, train_count, val_loss_sum, val_count], dtype=torch.float64, device=device)
        if is_distributed():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_loss = float(totals[0].item() / max(1.0, totals[1].item()))
        val_loss = float(totals[2].item() / max(1.0, totals[3].item()))
        gathered_rows = gather_objects(pred_rows)

        if is_main:
            with history_path.open("a", encoding="utf-8") as fp:
                fp.write(f"epoch={epoch},train_loss={train_loss:.6f},val_loss={val_loss:.6f}\n")
            print(f"[quality] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                stale_epochs = 0
                to_save = model.module if isinstance(model, DistributedDataParallel) else model
                torch.save(to_save.state_dict(), best_path)
                pd.DataFrame(drop_duplicate_prediction_rows(gathered_rows)).to_csv(
                    output_dir / "quality_predictions_val.csv", index=False
                )
                print(f"[quality] saved best checkpoint to {best_path}")
            else:
                stale_epochs += 1
        if is_distributed():
            stale_tensor = torch.tensor([stale_epochs], dtype=torch.int64, device=device)
            dist.broadcast(stale_tensor, src=0)
            stale_epochs = int(stale_tensor.item())
        if stale_epochs >= args.patience:
            if is_main:
                print(f"[quality] early stopping after {args.patience} stale epochs.")
            break
    cleanup_distributed()


if __name__ == "__main__":
    main()
