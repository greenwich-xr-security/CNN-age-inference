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
DINOv2_IMG_SIZES = {
    "dinov2_vits14": 224,
    "dinov2_vitb14": 224,
    "dinov2_vitl14": 224,
    "dinov2_vitg14": 224,
    "dinov2_vits14_reg": 224,
    "dinov2_vitb14_reg": 224,
    "dinov2_vitl14_reg": 224,
    "dinov2_vitg14_reg": 224,
}


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


class DinoV2AgeRegressor(nn.Module):
    """DINOv2 backbone that predicts age mean/log-variance pairs, optionally with an embedding head."""

    def __init__(self, variant: str, embed_dim: int = 0):
        super().__init__()
        variant = variant.lower()
        if variant not in DINOv2_IMG_SIZES:
            raise ValueError(f"Unsupported DINOv2 variant '{variant}'.")
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        self.backbone = _load_dinov2_backbone(variant)
        if hasattr(self.backbone, "mask_token"):
            # Mask token is only used for masked modeling; keep it frozen to avoid DDP unused-parameter errors.
            try:
                self.backbone.mask_token.requires_grad_(False)
            except Exception:
                pass
        self.variant = variant
        self.embed_dim = int(embed_dim)
        total_outputs = 2 + self.embed_dim

        feature_dim = getattr(self.backbone, "embed_dim", None) or getattr(
            self.backbone, "num_features", None
        )
        if feature_dim is None:
            # Fall back to lazy inference of feature size during first forward pass.
            self.head = nn.LazyLinear(total_outputs)
        else:
            self.head = nn.Linear(int(feature_dim), total_outputs)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor):
        feats = self._extract_features(x)
        preds = self.head(feats)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        if self.embed_dim > 0:
            mean = preds[:, 0]
            log_var = preds[:, 1]
            z = preds[:, 2:]
            return mean, log_var, z
        mean, log_var = preds.chunk(2, dim=1)
        return mean.squeeze(1), log_var.squeeze(1)
