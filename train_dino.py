import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.ssl import HandSSLPairDataset
from dataset.ssl_transforms import DinoAugmentationConfig, DinoMultiCropTransform
from dataset.utils import filter_metadata_ssl
from models import resolve_backbone_builder
from models.dino import DINOHead, DinoNetwork

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MODEL_VARIANT = "b0"
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


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        *,
        student_temp: float,
        teacher_temp: float,
        center_momentum: float,
        num_global_crops: int,
    ) -> None:
        super().__init__()
        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.center_momentum = float(center_momentum)
        self.num_global_crops = int(num_global_crops)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def set_teacher_temp(self, temp: float) -> None:
        self.teacher_temp = float(temp)

    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_outputs: list[torch.Tensor],
        *,
        skip_teacher_index: int | None,
        total_terms: int,
    ) -> torch.Tensor:
        student_log_prob = F.log_softmax(student_output / self.student_temp, dim=-1)
        total_loss = None
        for t_idx, t_out in enumerate(teacher_outputs):
            if skip_teacher_index is not None and t_idx == skip_teacher_index:
                continue
            t_prob = F.softmax((t_out - self.center) / self.teacher_temp, dim=-1).detach()
            term = torch.sum(-t_prob * student_log_prob, dim=-1).mean()
            total_loss = term if total_loss is None else total_loss + term
        if total_loss is None:
            return torch.tensor(0.0, device=student_output.device)
        return total_loss / float(max(1, total_terms))

    @torch.no_grad()
    def update_center(self, teacher_outputs: list[torch.Tensor]) -> None:
        teacher_output = torch.cat(teacher_outputs, dim=0)
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data, alpha=1.0 - momentum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINO SSL pretraining for hand images.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/dino",
        help="Directory where checkpoints and logs will be saved.",
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
        default=4,
        help="Number of local crops per sample (default: 4).",
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
        "--out-dim",
        type=int,
        default=8192,
        help="DINO projection head output dim (default: 8192).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="DINO MLP hidden dim (default: 2048).",
    )
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=256,
        help="DINO bottleneck dim (default: 256).",
    )
    parser.add_argument(
        "--student-temp",
        type=float,
        default=0.1,
        help="Student temperature (default: 0.1).",
    )
    parser.add_argument(
        "--teacher-temp",
        type=float,
        default=0.04,
        help="Teacher temperature (default: 0.04).",
    )
    parser.add_argument(
        "--teacher-temp-warmup",
        type=float,
        default=0.04,
        help="Teacher warmup temperature (default: 0.04).",
    )
    parser.add_argument(
        "--teacher-temp-warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs for teacher temperature (default: 5).",
    )
    parser.add_argument(
        "--teacher-momentum",
        type=float,
        default=0.996,
        help="Initial teacher EMA momentum (default: 0.996).",
    )
    parser.add_argument(
        "--teacher-momentum-end",
        type=float,
        default=1.0,
        help="Final teacher EMA momentum (default: 1.0).",
    )
    parser.add_argument(
        "--center-momentum",
        type=float,
        default=0.9,
        help="Center momentum for DINO loss (default: 0.9).",
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
            f"[dino] Ignoring requested --img-size {args.img_size}; {model_desc} uses {default_size}."
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
        print("[dino] Augmentation ramp is most reliable with --num-workers 0.")

    print(
        f"Using dataset root: {active_root}\n"
        f"Saving artifacts to: {output_dir}\n"
        f"SSL images: {len(metadata)} | Users: {metadata['user_id'].nunique()}\n"
        f"Model: {model_desc} | Image size: {img_size} | Local crop: {local_size}\n"
        f"Batch size: {args.batch_size} | Epochs: {args.epochs} | Seed: {args.seed}"
    )

    student_backbone = model_builder()
    teacher_backbone = model_builder()
    embed_dim = getattr(student_backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Backbone must expose an embed_dim attribute for DINO.")

    student = DinoNetwork(
        student_backbone,
        DINOHead(
            embed_dim,
            out_dim=args.out_dim,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
        ),
    )
    teacher = DinoNetwork(
        teacher_backbone,
        DINOHead(
            embed_dim,
            out_dim=args.out_dim,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
        ),
    )

    student.apply(disable_inplace_ops)
    teacher.apply(disable_inplace_ops)

    teacher.load_state_dict(student.state_dict(), strict=True)
    for param in teacher.parameters():
        param.requires_grad = False

    student = student.to(DEVICE)
    teacher = teacher.to(DEVICE)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    momentum_schedule = cosine_schedule(args.teacher_momentum, args.teacher_momentum_end, total_steps)

    warmup_epochs = max(0, int(args.teacher_temp_warmup_epochs))
    warmup_steps = min(steps_per_epoch * warmup_epochs, total_steps)
    if warmup_steps > 0:
        teacher_temps = (
            np.linspace(args.teacher_temp_warmup, args.teacher_temp, warmup_steps).tolist()
            + [args.teacher_temp] * max(0, total_steps - warmup_steps)
        )
    else:
        teacher_temps = [args.teacher_temp] * total_steps

    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        center_momentum=args.center_momentum,
        num_global_crops=2,
    ).to(DEVICE)

    history_path = output_dir / "dino_history.log"
    checkpoint_path = output_dir / f"dino_{model_key}_pretrain.pth"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        if args.aug_ramp_fraction > 0:
            ramp_epochs = max(1, int(args.epochs * args.aug_ramp_fraction))
            strength = min(1.0, epoch / ramp_epochs)
            multi_crop.set_strength(strength)
        student.train()
        student.apply(set_bn_eval)
        teacher.eval()

        running_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            views = [v.to(DEVICE, non_blocking=True) for v in batch]
            with torch.no_grad():
                teacher_outputs = forward_views(teacher, views[:2])
            dino_loss.set_teacher_temp(teacher_temps[global_step])
            total_terms = (len(views) * len(teacher_outputs)) - len(teacher_outputs)
            optimizer.zero_grad()
            batch_loss = 0.0
            for idx, view in enumerate(views):
                student_output = student(view)
                skip_idx = idx if idx < len(teacher_outputs) else None
                loss = dino_loss.compute_loss(
                    student_output,
                    teacher_outputs,
                    skip_teacher_index=skip_idx,
                    total_terms=total_terms,
                )
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()

            momentum = momentum_schedule[global_step]
            update_teacher(student, teacher, momentum)
            dino_loss.update_center(teacher_outputs)
            running_loss += batch_loss
            global_step += 1

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch}: dino_loss={avg_loss:.4f}")
        with history_path.open("a", encoding="utf-8") as log_fp:
            log_fp.write(f"Epoch {epoch},loss={avg_loss:.6f}\n")

        checkpoint = {
            "backbone": student.backbone.state_dict(),
            "head": student.head.state_dict(),
            "model": args.model,
            "img_size": img_size,
            "out_dim": args.out_dim,
            "epoch": epoch,
        }
        torch.save(checkpoint, checkpoint_path)

    print(f"Saved SSL checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
