from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Canonical ConvNeXt input resolutions for common variants.
CONVNEXT_IMG_SIZES = {
    "tiny": 224,
    "small": 224,
    "base": 224,
    "large": 224,
    "xlarge": 224,
}


class ConvNeXtAgeRegressor(nn.Module):
    """
    ConvNeXt backbone (from timm) that predicts age mean/log-variance pairs, optionally with an embedding head.

    The forward signature matches EfficientNetAgeRegressor:
        input  : tensor [B, 3, H, W]
        output : (mean [B], log_var [B]) or (mean, log_var, z[B, K]) when embed_dim>0
    """

    def __init__(self, variant: str = "base", embed_dim: int = 0):
        super().__init__()
        variant = variant.lower()
        model_name = f"convnext_{variant}"
        if variant not in CONVNEXT_IMG_SIZES:
            raise ValueError(f"Unsupported ConvNeXt variant '{variant}'. Expected one of {sorted(CONVNEXT_IMG_SIZES)}.")
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        if model_name not in timm.list_models(model_name):
            # timm.list_models returns patterns; ensure the exact name exists
            raise ValueError(f"timm does not provide model name '{model_name}'.")

        total_outputs = 2 + int(embed_dim)
        # num_classes controls head outputs
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=total_outputs)
        self.variant = variant
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor):
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
