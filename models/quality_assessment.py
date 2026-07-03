from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .efficientnet_age import EFFICIENTNET_IMG_SIZES, get_default_efficientnet_weights


EFFICIENTNET_ALIASES = {
    "effnet_b0": "b0",
    "effnet_b1": "b1",
    "effnet_b2": "b2",
    "effnet_b3": "b3",
    "effnet_b4": "b4",
    "effnet_b5": "b5",
    "effnet_b6": "b6",
    "effnet_b7": "b7",
}


QUALITY_OUTPUT_NAMES = [
    "abs_error",
]


class EfficientNetQualityAssessor(nn.Module):
    """EfficientNet CNN for hand-image quality assessment for age regression."""

    def __init__(self, variant: str = "b0", num_outputs: int = len(QUALITY_OUTPUT_NAMES)):
        super().__init__()
        variant = EFFICIENTNET_ALIASES.get(variant.lower(), variant.lower())
        if variant not in EFFICIENTNET_IMG_SIZES:
            raise ValueError(
                f"Quality assessor currently supports EfficientNet variants: {sorted(EFFICIENTNET_IMG_SIZES)}"
            )
        model_name = f"efficientnet_{variant}"
        builder = getattr(models, model_name, None)
        if builder is None:
            raise ValueError(f"torchvision.models does not provide '{model_name}'.")
        weights = get_default_efficientnet_weights(variant)
        try:
            self.backbone = builder(weights=weights) if weights is not None else builder(pretrained=True)
        except TypeError:
            self.backbone = builder(pretrained=True)
        if isinstance(self.backbone.classifier, nn.Sequential):
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_outputs)
        else:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, num_outputs)
        self.variant = variant
        self.num_outputs = int(num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def resolve_quality_model_builder(model_name: str):
    name = EFFICIENTNET_ALIASES.get(model_name.lower(), model_name.lower())
    if name not in EFFICIENTNET_IMG_SIZES:
        raise ValueError(
            f"Unsupported quality model '{model_name}'. Use one of {sorted(EFFICIENTNET_IMG_SIZES)} "
            f"or an EfficientNet alias."
        )
    return (
        lambda: EfficientNetQualityAssessor(name),
        EFFICIENTNET_IMG_SIZES[name],
        f"EfficientNet-{name.upper()} quality assessor",
        name,
    )
