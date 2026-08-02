from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.normals_head import NormalsHead

EFFICIENTNET_IMG_SIZES = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
    "b3": 300,
    "b4": 380,
    "b5": 456,
    "b6": 528,
    "b7": 600,
    "v2_s": 384,
    "v2_m": 480,
    "v2_l": 480,
}


def get_default_efficientnet_weights(variant: str):
    """Resolve the torchvision weights enum for the requested EfficientNet variant."""
    if variant.startswith("v2_"):
        weights_enum_name = f"EfficientNet_V2_{variant.split('_')[1].upper()}_Weights"
    else:
        weights_enum_name = f"EfficientNet_{variant.upper()}_Weights"
    weights_enum = getattr(models, weights_enum_name, None)
    if weights_enum is None:
        return None
    default_weights = getattr(weights_enum, "DEFAULT", None)
    if default_weights is not None:
        return default_weights
    try:
        return next(iter(weights_enum))
    except TypeError:
        return None


class EfficientNetAgeRegressor(nn.Module):
    """EfficientNet backbone for age regression with optional auxiliary normals head."""

    def __init__(
        self,
        variant: str,
        embed_dim: int = 0,
        normals_aux: bool = False,
        normals_privileged: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in EFFICIENTNET_IMG_SIZES:
            raise ValueError(f"Unsupported EfficientNet variant '{variant}'.")
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        model_name = f"efficientnet_{variant}"
        if not hasattr(models, model_name):
            raise ValueError(f"torchvision.models does not provide '{model_name}'.")

        backbone_builder = getattr(models, model_name)
        weights = get_default_efficientnet_weights(variant) if pretrained else None
        try:
            backbone = backbone_builder(weights=weights)
        except TypeError:
            backbone = backbone_builder(pretrained=pretrained)

        self.variant = variant
        self.embed_dim = int(embed_dim)
        self.img_size = EFFICIENTNET_IMG_SIZES[variant]
        total_outputs = 2 + self.embed_dim

        # Split backbone into three addressable parts so we can intercept spatial features.
        self.features = backbone.features
        self.pool = backbone.avgpool

        # Replace the final linear in the classifier.
        if isinstance(backbone.classifier, nn.Sequential) and len(backbone.classifier) >= 2:
            in_feats = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_feats, total_outputs)
        else:
            in_feats = getattr(backbone.classifier, "in_features", None)
            if in_feats is None:
                raise RuntimeError(f"Unexpected EfficientNet-{variant.upper()} classifier structure")
            backbone.classifier = nn.Linear(in_feats, total_outputs)
        self.classifier = backbone.classifier

        # Privileged normals input: expand first conv 3→6 channels (zero-init extra).
        if normals_privileged:
            from models import expand_first_conv_to_6ch
            expand_first_conv_to_6ch(self)

        # Lightweight normals decoder attached to spatial features (Option A).
        if normals_aux:
            # Infer feature channels by probing features with a dummy tensor.
            in_ch = 6 if normals_privileged else 3
            with torch.no_grad():
                dummy = torch.zeros(1, in_ch, self.img_size, self.img_size)
                feat_channels = self.features(dummy).shape[1]
            self.normals_head: nn.Module | None = NormalsHead(feat_channels)
        else:
            self.normals_head = None

    def forward(self, x: torch.Tensor):
        spatial = self.features(x)                          # [B, C, h, w]
        pooled = torch.flatten(self.pool(spatial), 1)       # [B, C]
        preds = self.classifier(pooled)                     # [B, 2 + embed_dim]

        if self.embed_dim > 0:
            mean = preds[:, 0]
            log_var = preds[:, 1]
            z = preds[:, 2:]
            if self.normals_head is not None:
                pred_normals = self.normals_head(spatial, self.img_size)
                return mean, log_var, z, pred_normals
            return mean, log_var, z

        mean, log_var = preds.chunk(2, dim=1)
        mean, log_var = mean.squeeze(1), log_var.squeeze(1)
        if self.normals_head is not None:
            pred_normals = self.normals_head(spatial, self.img_size)
            return mean, log_var, pred_normals
        return mean, log_var
