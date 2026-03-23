from __future__ import annotations

import torch.nn as nn
from torchvision import models

# Canonical input resolutions and torchvision weights enum names.
MOBILENET_IMG_SIZES: dict[str, int] = {
    "v2":       224,
    "v3_small": 224,
    "v3_large": 224,
}

_MOBILENET_WEIGHTS_NAMES: dict[str, str] = {
    "v2":       "MobileNet_V2_Weights",
    "v3_small": "MobileNet_V3_Small_Weights",
    "v3_large": "MobileNet_V3_Large_Weights",
}

_MOBILENET_MODEL_NAMES: dict[str, str] = {
    "v2":       "mobilenet_v2",
    "v3_small": "mobilenet_v3_small",
    "v3_large": "mobilenet_v3_large",
}


def _get_mobilenet_weights(variant: str):
    """Resolve the torchvision weights enum for the requested MobileNet variant."""
    weights_enum = getattr(models, _MOBILENET_WEIGHTS_NAMES[variant], None)
    if weights_enum is None:
        return None
    default = getattr(weights_enum, "DEFAULT", None)
    if default is not None:
        return default
    try:
        return next(iter(weights_enum))
    except TypeError:
        return None


class MobileNetAgeRegressor(nn.Module):
    """
    MobileNet backbone (torchvision) for probabilistic age regression.

    Forward signature matches EfficientNetAgeRegressor:
        input  : tensor [B, 3, H, W]
        output : (mean [B], log_var [B])
                 or (mean, log_var, z [B, embed_dim]) when embed_dim > 0

    Variant names
    -------------
    v2       — MobileNetV2  (~3.4M params, lightest)
    v3_small — MobileNetV3-Small (~2.5M params, lightest)
    v3_large — MobileNetV3-Large (~5.5M params, best accuracy/size trade-off)
    """

    def __init__(self, variant: str = "v3_large", embed_dim: int = 0):
        super().__init__()
        variant = variant.lower()
        if variant not in MOBILENET_IMG_SIZES:
            raise ValueError(
                f"Unsupported MobileNet variant '{variant}'. "
                f"Expected one of {sorted(MOBILENET_IMG_SIZES)}."
            )
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        model_name = _MOBILENET_MODEL_NAMES[variant]
        if not hasattr(models, model_name):
            raise ValueError(f"torchvision.models does not provide '{model_name}'.")

        weights = _get_mobilenet_weights(variant)
        builder = getattr(models, model_name)
        try:
            backbone = builder(weights=weights) if weights is not None else builder(pretrained=True)
        except TypeError:
            backbone = builder(pretrained=True)

        total_outputs = 2 + int(embed_dim)

        # Both MobileNetV2 and V3 expose a nn.Sequential classifier;
        # replace the final Linear layer with our regression head.
        if isinstance(backbone.classifier, nn.Sequential):
            in_feats = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_feats, total_outputs)
        else:
            in_feats = getattr(backbone.classifier, "in_features", None)
            if in_feats is None:
                raise RuntimeError(f"Unexpected MobileNet-{variant} classifier structure.")
            backbone.classifier = nn.Linear(in_feats, total_outputs)

        self.backbone = backbone
        self.variant = variant
        self.embed_dim = int(embed_dim)

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
