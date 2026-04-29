from __future__ import annotations

import timm
import torch
import torch.nn as nn

from models.normals_head import NormalsHead

# Canonical input resolutions per Swin variant.
# Swin V1 uses 224px; Swin V2 variants use 256px by default, with dedicated 384px/512px tiny variants.
SWIN_IMG_SIZES: dict[str, int] = {
    # Swin Transformer V1
    "tiny":        224,
    "small":       224,
    "base":        224,
    "large":       224,
    # Swin Transformer V2
    "v2_tiny":     256,
    "v2_tiny_384": 384,
    "v2_tiny_512": 512,
    "v2_small":    256,
    "v2_base":     256,
    "v2_base_384": 384,
    "v2_large":    256,
}

_SWIN_TIMM_IDS: dict[str, str] = {
    "tiny":     "swin_tiny_patch4_window7_224",
    "small":    "swin_small_patch4_window7_224",
    "base":     "swin_base_patch4_window7_224",
    "large":    "swin_large_patch4_window7_224",
    "v2_tiny":  "swinv2_tiny_window8_256",
    "v2_tiny_384": "swinv2_cr_tiny_384",
    "v2_tiny_512": "swinv2_tiny_window8_256",
    "v2_small": "swinv2_small_window8_256",
    "v2_base":  "swinv2_base_window8_256",
    "v2_base_384": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "v2_large": "swinv2_large_window12to16_192to256_22kft1k",
}

_SWIN_TIMM_KWARGS: dict[str, dict[str, int]] = {
    "v2_tiny_512": {"img_size": 512},
}


def _swin_features_to_bchw(features: torch.Tensor) -> torch.Tensor:
    """Convert common timm Swin feature layouts to BCHW for the normals head."""
    if features.dim() == 4:
        if features.shape[-1] >= features.shape[1]:
            return features.permute(0, 3, 1, 2).contiguous()
        return features
    if features.dim() == 3:
        batch, tokens, channels = features.shape
        side = int(tokens ** 0.5)
        if side * side != tokens:
            raise RuntimeError(f"Cannot reshape Swin token features with {tokens} tokens into a square map.")
        return features.transpose(1, 2).reshape(batch, channels, side, side).contiguous()
    raise RuntimeError(f"Unsupported Swin feature shape for normals aux: {tuple(features.shape)}")


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
    V2 : v2_tiny | v2_tiny_384 | v2_tiny_512 | v2_small | v2_base | v2_base_384 | v2_large
    """

    def __init__(
        self,
        variant: str = "tiny",
        embed_dim: int = 0,
        normals_aux: bool = False,
        normals_privileged: bool = False,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in SWIN_IMG_SIZES:
            raise ValueError(
                f"Unsupported Swin variant '{variant}'. "
                f"Expected one of {sorted(SWIN_IMG_SIZES)}."
            )
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        if variant == "v2_tiny_384":
            raise ValueError(
                "No pretrained Swin V2 tiny 384 checkpoint is available in the installed timm build. "
                "Use 'swin_v2_tiny' (256), 'swin_v2_tiny_512' (resized tiny), or "
                "'swin_v2_base_384' for a true pretrained 384px Swin V2 model."
            )

        timm_id = _SWIN_TIMM_IDS[variant]
        timm_kwargs = dict(_SWIN_TIMM_KWARGS.get(variant, {}))
        total_outputs = 2 + int(embed_dim)

        try:
            self.backbone = timm.create_model(
                timm_id,
                pretrained=True,
                num_classes=total_outputs,
                **timm_kwargs,
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to create pretrained Swin model '{timm_id}' via timm."
            ) from exc

        self.variant = variant
        self.timm_id = timm_id
        self.timm_kwargs = timm_kwargs
        self.embed_dim = int(embed_dim)
        self.img_size = SWIN_IMG_SIZES[variant]

        if normals_privileged:
            from models import expand_first_conv_to_6ch
            expand_first_conv_to_6ch(self)

        if normals_aux:
            in_ch = 6 if normals_privileged else 3
            with torch.no_grad():
                dummy = torch.zeros(1, in_ch, self.img_size, self.img_size)
                feat_channels = _swin_features_to_bchw(self.backbone.forward_features(dummy)).shape[1]
            self.normals_head: nn.Module | None = NormalsHead(feat_channels)
        else:
            self.normals_head = None

    def forward(self, x: torch.Tensor):
        features = self.backbone.forward_features(x)
        preds = self.backbone.forward_head(features)
        spatial = _swin_features_to_bchw(features) if self.normals_head is not None else None
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
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
