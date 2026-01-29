from __future__ import annotations

import torch.nn as nn
from torchvision import models

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
    # EfficientNet V1 variants are b0..b7, V2 variants are v2_s/m/l
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
    """EfficientNet backbone that predicts age mean/log-variance pairs, optionally with an embedding head."""

    def __init__(self, variant: str, embed_dim: int = 0):
        super().__init__()
        variant = variant.lower()
        if variant not in EFFICIENTNET_IMG_SIZES:
            raise ValueError(f"Unsupported EfficientNet variant '{variant}'.")
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        if variant.startswith("v2_"):
            model_name = f"efficientnet_{variant}"
        else:
            model_name = f"efficientnet_{variant}"
        if not hasattr(models, model_name):
            raise ValueError(f"torchvision.models does not provide '{model_name}'.")

        backbone_builder = getattr(models, model_name)

        weights = get_default_efficientnet_weights(variant)
        try:
            if weights is not None:
                backbone = backbone_builder(weights=weights)
            else:
                backbone = backbone_builder(pretrained=True)
        except TypeError:
            backbone = backbone_builder(pretrained=True)

        self.backbone = backbone
        self.variant = variant
        self.embed_dim = int(embed_dim)
        total_outputs = 2 + self.embed_dim
        # Replace the classifier to output mean/log-variance (+ optional embedding)
        if isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) >= 2:
            in_feats = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_feats, total_outputs)
        else:
            # Fallback: handle unexpected classifier structure
            in_feats = getattr(self.backbone.classifier, "in_features", None)
            if in_feats is None:
                raise RuntimeError(f"Unexpected EfficientNet-{variant.upper()} classifier structure")
            self.backbone.classifier = nn.Linear(in_feats, total_outputs)

    def forward(self, x):
        preds = self.backbone(x)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        if self.embed_dim > 0:
            mean = preds[:, 0]
            log_var = preds[:, 1]
            z = preds[:, 2:]
            return mean, log_var, z
        mean, log_var = preds.chunk(2, dim=1)
        return mean.squeeze(1), log_var.squeeze(1)
