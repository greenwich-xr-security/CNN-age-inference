from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .efficientnet_age import EFFICIENTNET_IMG_SIZES, get_default_efficientnet_weights


class EfficientNetBackbone(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        variant = variant.lower()
        if variant not in EFFICIENTNET_IMG_SIZES:
            raise ValueError(f"Unsupported EfficientNet variant '{variant}'.")

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

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        classifier = backbone.classifier
        if isinstance(classifier, nn.Sequential) and len(classifier) >= 2:
            self.embed_dim = classifier[-1].in_features
        else:
            self.embed_dim = getattr(classifier, "in_features", None)
        if self.embed_dim is None:
            raise RuntimeError(f"Unexpected EfficientNet-{variant.upper()} classifier structure")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ConvNeXtBackbone(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        import timm

        model_name = f"convnext_{variant}"
        if model_name not in timm.list_models(model_name):
            raise ValueError(f"timm does not provide model name '{model_name}'.")

        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        self.backbone = backbone
        self.embed_dim = backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
