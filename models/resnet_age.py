from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.normals_head import NormalsHead

# Canonical input resolutions for supported ResNet variants.
RESNET_IMG_SIZES: dict[str, int] = {
    "50": 224,
}

_RESNET_WEIGHTS_NAMES: dict[str, str] = {
    "50": "ResNet50_Weights",
}

_RESNET_MODEL_NAMES: dict[str, str] = {
    "50": "resnet50",
}


def _get_resnet_weights(variant: str):
    """Resolve the torchvision weights enum for the requested ResNet variant."""
    weights_enum = getattr(models, _RESNET_WEIGHTS_NAMES[variant], None)
    if weights_enum is None:
        return None
    default = getattr(weights_enum, "DEFAULT", None)
    if default is not None:
        return default
    try:
        return next(iter(weights_enum))
    except TypeError:
        return None


class ResNetAgeRegressor(nn.Module):
    """
    ResNet backbone (torchvision) for probabilistic age regression.

    Forward signature matches EfficientNetAgeRegressor:
        input  : tensor [B, 3, H, W]
        output : (mean [B], log_var [B])
                 or (mean, log_var, z [B, embed_dim]) when embed_dim > 0
    """

    def __init__(
        self,
        variant: str = "50",
        embed_dim: int = 0,
        normals_aux: bool = False,
        normals_privileged: bool = False,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in RESNET_IMG_SIZES:
            raise ValueError(
                f"Unsupported ResNet variant '{variant}'. "
                f"Expected one of {sorted(RESNET_IMG_SIZES)}."
            )
        if embed_dim < 0:
            raise ValueError("embed_dim must be non-negative.")

        model_name = _RESNET_MODEL_NAMES[variant]
        if not hasattr(models, model_name):
            raise ValueError(f"torchvision.models does not provide '{model_name}'.")

        weights = _get_resnet_weights(variant)
        builder = getattr(models, model_name)
        try:
            backbone = builder(weights=weights) if weights is not None else builder(pretrained=True)
        except TypeError:
            backbone = builder(pretrained=True)

        total_outputs = 2 + int(embed_dim)
        in_feats = getattr(backbone.fc, "in_features", None)
        if in_feats is None:
            raise RuntimeError(f"Unexpected ResNet-{variant} classifier structure.")
        backbone.fc = nn.Linear(in_feats, total_outputs)

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = backbone.avgpool
        self.fc = backbone.fc
        self.backbone = backbone
        self.variant = variant
        self.embed_dim = int(embed_dim)
        self.img_size = RESNET_IMG_SIZES[variant]

        if normals_privileged:
            from models import expand_first_conv_to_6ch
            expand_first_conv_to_6ch(self)

        if normals_aux:
            in_ch = 6 if normals_privileged else 3
            with torch.no_grad():
                dummy = torch.zeros(1, in_ch, self.img_size, self.img_size)
                feat_channels = self.features(dummy).shape[1]
            self.normals_head: nn.Module | None = NormalsHead(feat_channels)
        else:
            self.normals_head = None

    def forward(self, x):
        spatial = self.features(x)
        pooled = torch.flatten(self.pool(spatial), 1)
        preds = self.fc(pooled)
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
