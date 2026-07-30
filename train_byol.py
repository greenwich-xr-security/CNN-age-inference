import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.ssl import HandSSLPairDataset
from dataset.ssl_transforms import DinoAugmentationConfig, DinoMultiCropTransform
from dataset.utils import filter_metadata_ssl
from models import resolve_backbone_builder
from models.byol import BYOLNetwork, BYOLPredictor, BYOLProjector, byol_loss

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MODEL_VARIANT = "v2_s"
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def cosine_schedule(start: float, end: float, total_steps: int) -> list[float]:
    if total_steps <= 1:
        return [end]
    values = []
    for step in range(total_steps):
        cosine = (1.0 + math.cos(math.pi * step / (total_steps - 1))) / 2.0
        values.append(end - (end - start) * cosine)
    return values


def forward_views(model, views: list[torch.Tensor]) -> list[torch.Tensor]:
    outputs = [None] * len(views)
    size_map: dict[tuple[int, int], list[int]] = {}
    for idx, view in enumerate(views):
        size_map.setdefault(tuple(view.shape[-2:]), []).append(idx)
    for indices in size_map.values():
        batch = torch.cat([views[i] for i in indices], dim=0)
        batch_out = model(batch)
        chunks = batch_out.chunk(len(indices))
        for idx, out in zip(indices, chunks):
            outputs[idx] = out
    return outputs


def set_bn_eval(module: nn.Module) -> None:
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        module.eval()


def disable_inplace_ops(module: nn.Module) -> None:
    if hasattr(module, "inplace"):
        try:
            module.inplace = False
        except Exception:
            pass


def compute_byol_view_loss(
    online_output: torch.Tensor,
    target_outputs: list[torch.Tensor],
    *,
    skip_target_index: int | None,
    total_terms: int,
) -> torch.Tensor:
    total = None
    for t_idx, t_out in enumerate(target_outputs):
        if skip_target_index is not None and t_idx == skip_target_index:
            continue
        term = byol_loss(online_output, t_out.detach()).mean()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0, device=online_output.device)
    return total / float(max(1, total_terms))


def update_target(online: nn.Module, target: nn.Module, momentum: float) -> None:
    for param_o, param_t in zip(online.encoder_parameters(), target.encoder_parameters()):
        param_t.data.mul_(momentum).add_(param_o.data, alpha=1.0 - momentum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BYOL SSL pretraining for hand images.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/byol",
        help="Directory where checkpoints and logs will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        help=(
            "Backbone to use. EfficientNet: b0-b7, v2_{s,m,l}. ConvNeXt: convnext_{tiny,small,base,large,xlarge} "
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
        help="Mini-batch size for SSL (default: 64).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of SSL epochs (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for RNGs (default: 42).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--pair-same-hand-prob",
        type=float,
        default=0.5,
        help="Probability of pairing same user/hand images (default: 0.5).",
    )
    parser.add_argument(
        "--num-local-crops",
        type=int,
        default=0,
        help=(
            "Number of extra local crops per sample, beyond the 2 global views BYOL compares "
            "(default: 0, the canonical BYOL 2-view recipe). Local views are only compared "
            "against the target's 2 global outputs, never against each other."
        ),
    )
    parser.add_argument(
        "--global-crop-scale",
        type=float,
        nargs=2,
        default=(0.6, 1.0),
        help="Scale range for global crops (default: 0.6 1.0).",
    )
    parser.add_argument(
        "--local-crop-scale",
        type=float,
        nargs=2,
        default=(0.4, 0.7),
        help="Scale range for local crops (default: 0.4 0.7).",
    )
    parser.add_argument(
        "--local-crop-size",
        type=int,
        default=None,
        help="Override local crop size (default: 0.6 * img_size).",
    )
    parser.add_argument(
        "--projector-hidden-dim",
        type=int,
        default=4096,
        help="BYOL projector MLP hidden dim (default: 4096).",
    )
    parser.add_argument(
        "--projector-dim",
        type=int,
        default=256,
        help="BYOL projector output dim (default: 256).",
    )
    parser.add_argument(
        "--predictor-hidden-dim",
        type=int,
        default=4096,
        help="BYOL predictor MLP hidden dim (default: 4096).",
    )
    parser.add_argument(
        "--target-momentum",
        type=float,
        default=0.996,
        help="Initial target-network EMA momentum (default: 0.996).",
    )
    parser.add_argument(
        "--target-momentum-end",
        type=float,
        default=1.0,
        help="Final target-network EMA momentum (default: 1.0).",
    )
    parser.add_argument(
        "--aug-ramp-fraction",
        type=float,
        default=0.2,
        help="Fraction of epochs to ramp augmentation strength (default: 0.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    model_builder, default_size, model_desc, model_key = resolve_backbone_builder(args.model)
    if args.img_size is not None and args.img_size != default_size:
        print(
            f"[byol] Ignoring requested --img-size {args.img_size}; {model_desc} uses {default_size}."
        )
    img_size = default_size
    local_size = args.local_crop_size if args.local_crop_size else max(64, int(img_size * 0.6))

    if args.data_root:
        set_dataset_root(args.data_root)
    active_root = get_dataset_root()

    metadata = filter_metadata_ssl(load_combined_metadata(root=active_root))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    aug_config = DinoAugmentationConfig()
    multi_crop = DinoMultiCropTransform(
        global_size=img_size,
        local_size=local_size,
        num_local_crops=args.num_local_crops,
        global_scale=tuple(args.global_crop_scale),
        local_scale=tuple(args.local_crop_scale),
        config=aug_config,
    )

    dataset = HandSSLPairDataset(
        metadata,
        transform=multi_crop,
        pair_same_hand_prob=args.pair_same_hand_prob,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    if args.num_workers > 0 and args.aug_ramp_fraction > 0:
        print("[byol] Augmentation ramp is most reliable with --num-workers 0.")

    print(
        f"Using dataset root: {active_root}\n"
        f"Saving artifacts to: {output_dir}\n"
        f"SSL images: {len(metadata)} | Users: {metadata['user_id'].nunique()}\n"
        f"Model: {model_desc} | Image size: {img_size} | Local crop: {local_size}\n"
        f"Batch size: {args.batch_size} | Epochs: {args.epochs} | Seed: {args.seed}"
    )

    online_backbone = model_builder()
    target_backbone = model_builder()
    embed_dim = getattr(online_backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Backbone must expose an embed_dim attribute for BYOL.")

    online = BYOLNetwork(
        online_backbone,
        BYOLProjector(embed_dim, hidden_dim=args.projector_hidden_dim, out_dim=args.projector_dim),
        BYOLPredictor(args.projector_dim, hidden_dim=args.predictor_hidden_dim),
    )
    target = BYOLNetwork(
        target_backbone,
        BYOLProjector(embed_dim, hidden_dim=args.projector_hidden_dim, out_dim=args.projector_dim),
    )

    online.apply(disable_inplace_ops)
    target.apply(disable_inplace_ops)

    target.load_state_dict(online.state_dict(), strict=False)
    for param in target.parameters():
        param.requires_grad = False

    online = online.to(DEVICE)
    target = target.to(DEVICE)

    optimizer = torch.optim.AdamW(online.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    momentum_schedule = cosine_schedule(args.target_momentum, args.target_momentum_end, total_steps)

    history_path = output_dir / "byol_history.log"
    checkpoint_path = output_dir / f"byol_{model_key}_pretrain.pth"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        if args.aug_ramp_fraction > 0:
            ramp_epochs = max(1, int(args.epochs * args.aug_ramp_fraction))
            strength = min(1.0, epoch / ramp_epochs)
            multi_crop.set_strength(strength)
        online.train()
        # Backbones start from ImageNet-pretrained weights; keep their BatchNorm running
        # stats frozen for stability (multi-crop SSL batches are small and non-i.i.d.).
        # The projector/predictor BatchNorm layers are freshly initialized and must be
        # allowed to update live -- BYOL relies on them tracking real batch statistics.
        online.backbone.apply(set_bn_eval)
        # The target network is never optimized directly (EMA-only), but it stays in
        # train() mode so its BatchNorm running stats keep evolving from its own forward
        # passes, same as the online projector/predictor; only its backbone is frozen.
        target.train()
        target.backbone.apply(set_bn_eval)

        running_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            views = [v.to(DEVICE, non_blocking=True) for v in batch]
            with torch.no_grad():
                target_outputs = forward_views(target, views[:2])
            total_terms = (len(views) * len(target_outputs)) - len(target_outputs)
            optimizer.zero_grad()
            batch_loss = 0.0
            for idx, view in enumerate(views):
                online_output = online(view)
                skip_idx = idx if idx < len(target_outputs) else None
                loss = compute_byol_view_loss(
                    online_output,
                    target_outputs,
                    skip_target_index=skip_idx,
                    total_terms=total_terms,
                )
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()

            momentum = momentum_schedule[global_step]
            update_target(online, target, momentum)
            running_loss += batch_loss
            global_step += 1

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch}: byol_loss={avg_loss:.4f}")
        with history_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"Epoch {epoch},loss={avg_loss:.6f}\n")

        checkpoint = {
            "backbone": online.backbone.state_dict(),
            "projector": online.projector.state_dict(),
            "model": args.model,
            "img_size": img_size,
            "projector_dim": args.projector_dim,
            "epoch": epoch,
        }
        torch.save(checkpoint, checkpoint_path)

    print(f"Saved SSL checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
