from __future__ import annotations

import timm
import torch
import torch.nn as nn

# Canonical input resolutions per Swin variant.
# Swin V1 uses 224px; Swin V2 variants use 256px (tiny/small/base) or 192px (large).
SWIN_IMG_SIZES: dict[str, int] = {
    # Swin Transformer V1
    "tiny":        224,
    "small":       224,
    "base":        224,
    "large":       224,
    # Swin Transformer V2
    "v2_tiny":     256,
    "v2_small":    256,
    "v2_base":     256,
    "v2_large":    256,
}

_SWIN_TIMM_IDS: dict[str, str] = {
    "tiny":     "swin_tiny_patch4_window7_224",
    "small":    "swin_small_patch4_window7_224",
    "base":     "swin_base_patch4_window7_224",
    "large":    "swin_large_patch4_window7_224",
    "v2_tiny":  "swinv2_tiny_window8_256",
    "v2_small": "swinv2_small_window8_256",
    "v2_base":  "swinv2_base_window8_256",
    "v2_large": "swinv2_large_window12to16_192to256_22kft1k",
}


class SwinAgeRegressor(nn.Module):
    """
    Swin Transformer backbone (V1 or V2, from timm) for probabilistic age regression.

    Forward signature matches EfficientNetAgeRegressor:
        input  : tensor [B, 3, H, W]
        output : (mean [B], log_var [B])
                 or (mean, log_var, z [B, embed_dim]) when embed_dim > 0

    Variant names
    -------------
    V1 : tiny | small | base | large
    V2 : v2_tiny | v2_small | v2_base | v2_large
    """

    def __init__(self, variant: str = "tiny", embed_dim: int = 0):
        super().__init__()
        variant = variant.lower()
        if variant not in SWIN_IMG_SIZES:
            raise ValueError(
                f"Unsupported Swin variant '{variant}'. "
                f"Expected one of {sorted(SWIN_IMG_SIZES)}."
            )
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        timm_id = _SWIN_TIMM_IDS[variant]
        total_outputs = 2 + int(embed_dim)

        try:
            self.backbone = timm.create_model(timm_id, pretrained=True, num_classes=total_outputs)
        except Exception as exc:
            raise ValueError(
                f"Unable to create pretrained Swin model '{timm_id}' via timm."
            ) from exc

        self.variant = variant
        self.timm_id = timm_id
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
