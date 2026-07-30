from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - best-effort lock
    FileLock = None

# DINOv2 variants available via torch.hub ("facebookresearch/dinov2").
DINOV2_IMG_SIZES = {
    "dinov2_vits14": 224,
    "dinov2_vitb14": 224,
    "dinov2_vitl14": 224,
    "dinov2_vitg14": 224,
    "dinov2_vits14_reg": 224,
    "dinov2_vitb14_reg": 224,
    "dinov2_vitl14_reg": 224,
    "dinov2_vitg14_reg": 224,
}

DINOV2_PATCH_SIZE = 14


def _hub_load(repo: str, variant: str) -> nn.Module:
    try:
        return torch.hub.load(repo, variant, pretrained=True, trust_repo=True)
    except TypeError:
        try:
            return torch.hub.load(repo, variant, pretrained=True)
        except TypeError:
            return torch.hub.load(repo, variant)


def _load_dinov2_backbone(variant: str) -> nn.Module:
    """Load a DINOv2 backbone via torch.hub, serializing downloads in multi-process runs."""
    repo = "facebookresearch/dinov2"
    cache_dir = torch.hub.get_dir()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    if FileLock is not None and cache_dir:
        lock_path = os.path.join(cache_dir, "dinov2_download.lock")
        with FileLock(lock_path):
            return _hub_load(repo, variant)
    return _hub_load(repo, variant)


def _first_tensor(values: Iterable) -> torch.Tensor | None:
    for value in values:
        if torch.is_tensor(value):
            return value
    return None


class DinoV2Backbone(nn.Module):
    """Feature-only DINOv2 ViT backbone for SSL pretraining (no age head)."""

    def __init__(self, variant: str):
        super().__init__()
        variant = variant.lower()
        if variant not in DINOV2_IMG_SIZES:
            raise ValueError(f"Unsupported DINOv2 variant '{variant}'.")

        self.backbone = _load_dinov2_backbone(variant)
        if hasattr(self.backbone, "mask_token"):
            # Mask token is only used for masked modeling; keep it frozen to avoid DDP unused-parameter errors.
            try:
                self.backbone.mask_token.requires_grad_(False)
            except Exception:
                pass

        self.variant = variant
        self.patch_size = DINOV2_PATCH_SIZE

        feature_dim = getattr(self.backbone, "embed_dim", None) or getattr(
            self.backbone, "num_features", None
        )
        if feature_dim is None:
            raise RuntimeError(f"Could not determine feature dimension for DINOv2 variant '{variant}'.")
        self.embed_dim = int(feature_dim)

    def freeze_blocks(self, num_blocks: int) -> int:
        """Freeze the patch embedding, position/cls tokens, and the first `num_blocks` transformer blocks.

        Intended for domain-adaptive continued pretraining, where fully unfreezing a ViT
        initialized from pretrained DINOv2 weights risks catastrophic forgetting on a small dataset.
        Returns the number of transformer blocks actually frozen (clamped to backbone depth).
        """
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative.")

        for param in self.backbone.patch_embed.parameters():
            param.requires_grad_(False)
        for name in ("cls_token", "pos_embed", "register_tokens"):
            tensor = getattr(self.backbone, name, None)
            if tensor is not None:
                tensor.requires_grad_(False)

        blocks = self.backbone.blocks
        frozen = min(num_blocks, len(blocks))
        for block in blocks[:frozen]:
            for param in block.parameters():
                param.requires_grad_(False)
        return frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)

        if isinstance(features, dict):
            for key in ("x_norm_clstoken", "x_clstoken", "x_cls_token"):
                if key in features:
                    features = features[key]
                    break
            else:
                candidate = _first_tensor(features.values())
                if candidate is None:
                    raise RuntimeError("DINOv2 forward_features returned no tensor outputs.")
                features = candidate

        if features.dim() == 3:
            # Take the class token if tokens are returned.
            features = features[:, 0]
        if features.dim() != 2:
            raise RuntimeError(
                "Unexpected DINOv2 feature tensor shape; expected [B, C] or [B, N, C]."
            )
        return features
