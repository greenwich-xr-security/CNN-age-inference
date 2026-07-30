from .efficientnet_age import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor
from .convnext_age import CONVNEXT_IMG_SIZES, ConvNeXtAgeRegressor
from .backbones import ConvNeXtBackbone, EfficientNetBackbone

__all__ = [
    "EFFICIENTNET_IMG_SIZES",
    "EfficientNetAgeRegressor",
    "CONVNEXT_IMG_SIZES",
    "ConvNeXtAgeRegressor",
    "ConvNeXtBackbone",
    "EfficientNetBackbone",
    "resolve_model_builder",
    "resolve_backbone_builder",
]

MODEL_ALIASES = {
    "cnt": "convnext_tiny",
    "cns": "convnext_small",
    "cnb": "convnext_base",
    "cnl": "convnext_large",
    "cnx": "convnext_xlarge",
}


def resolve_model_builder(model_name: str):
    """
    Resolve a model name (including aliases) to a builder, default image size, display label, and normalized key.
    """
    name = model_name.lower()
    name = MODEL_ALIASES.get(name, name)

    if name in EFFICIENTNET_IMG_SIZES:
        size = EFFICIENTNET_IMG_SIZES[name]
        return (
            lambda: EfficientNetAgeRegressor(name),
            size,
            f"EfficientNet-{name.upper()}",
            name,
        )

    if name.startswith("convnext_"):
        variant = name.split("_", 1)[1]
        if variant not in CONVNEXT_IMG_SIZES:
            raise ValueError(f"Unsupported ConvNeXt variant '{variant}'.")
        size = CONVNEXT_IMG_SIZES[variant]
        return (
            lambda: ConvNeXtAgeRegressor(variant),
            size,
            f"ConvNeXt-{variant}",
            name,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Expected one of {sorted(EFFICIENTNET_IMG_SIZES)} or convnext_{{tiny,small,base,large,xlarge}} "
        f"or aliases {sorted(MODEL_ALIASES)}."
    )


def resolve_backbone_builder(model_name: str):
    """Resolve a model name to a backbone builder and canonical image size."""
    name = model_name.lower()
    name = MODEL_ALIASES.get(name, name)

    if name in EFFICIENTNET_IMG_SIZES:
        size = EFFICIENTNET_IMG_SIZES[name]
        return (
            lambda: EfficientNetBackbone(name),
            size,
            f"EfficientNet-{name.upper()}",
            name,
        )

    if name.startswith("convnext_"):
        variant = name.split("_", 1)[1]
        if variant not in CONVNEXT_IMG_SIZES:
            raise ValueError(f"Unsupported ConvNeXt variant '{variant}'.")
        size = CONVNEXT_IMG_SIZES[variant]
        return (
            lambda: ConvNeXtBackbone(variant),
            size,
            f"ConvNeXt-{variant}",
            name,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Expected one of {sorted(EFFICIENTNET_IMG_SIZES)} or convnext_{{tiny,small,base,large,xlarge}} "
        f"or aliases {sorted(MODEL_ALIASES)}."
    )
