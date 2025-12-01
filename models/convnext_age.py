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
    ConvNeXt backbone (from timm) that predicts age mean/log-variance pairs.

    The forward signature matches EfficientNetAgeRegressor:
        input  : tensor [B, 3, H, W]
        output : (mean [B], log_var [B])
    """

    def __init__(self, variant: str = "base"):
        super().__init__()
        variant = variant.lower()
        model_name = f"convnext_{variant}"
        if variant not in CONVNEXT_IMG_SIZES:
            raise ValueError(f"Unsupported ConvNeXt variant '{variant}'. Expected one of {sorted(CONVNEXT_IMG_SIZES)}.")

        if model_name not in timm.list_models(model_name):
            # timm.list_models returns patterns; ensure the exact name exists
            raise ValueError(f"timm does not provide model name '{model_name}'.")

        # num_classes=2 ensures the head outputs (mean, log_var) directly.
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=2)
        self.variant = variant

    def forward(self, x: torch.Tensor):
        preds = self.backbone(x)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        mean, log_var = preds.chunk(2, dim=1)
        return mean.squeeze(1), log_var.squeeze(1)
