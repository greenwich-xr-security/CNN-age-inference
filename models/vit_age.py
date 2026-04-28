from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Canonical ViT input resolutions for supported variants.
VIT_IMG_SIZES = {
    "tiny_384": 384,
    "small_384": 384,
}

_VIT_TIMM_MODEL_IDS = {
    "tiny_384": "timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k",
    "small_384": "timm/vit_small_patch16_384.augreg_in21k_ft_in1k",
}


class ViTAgeRegressor(nn.Module):
    """
    ViT backbone (from timm) that predicts age mean/log-variance pairs, optionally with an embedding head.

    The forward signature matches EfficientNetAgeRegressor:
        input  : tensor [B, 3, H, W]
        output : (mean [B], log_var [B]) or (mean, log_var, z[B, K]) when embed_dim>0
    """

    def __init__(self, variant: str = "tiny_384", embed_dim: int = 0, normals_privileged: bool = False):
        super().__init__()
        variant = variant.lower()
        if variant not in VIT_IMG_SIZES:
            raise ValueError(f"Unsupported ViT variant '{variant}'. Expected one of {sorted(VIT_IMG_SIZES)}.")
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        model_id = _VIT_TIMM_MODEL_IDS[variant]
        total_outputs = 2 + int(embed_dim)

        try:
            self.backbone = timm.create_model(model_id, pretrained=True, num_classes=total_outputs)
        except Exception as exc:
            raise ValueError(f"Unable to create pretrained ViT model '{model_id}' via timm.") from exc

        self.variant = variant
        self.model_id = model_id
        self.embed_dim = int(embed_dim)

        if normals_privileged:
            from models import expand_first_conv_to_6ch
            expand_first_conv_to_6ch(self)

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
