from .efficientnet_age import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor
from .convnext_age import CONVNEXT_IMG_SIZES, ConvNeXtAgeRegressor
from .vit_age import VIT_IMG_SIZES, ViTAgeRegressor

__all__ = [
    "EFFICIENTNET_IMG_SIZES",
    "EfficientNetAgeRegressor",
    "CONVNEXT_IMG_SIZES",
    "ConvNeXtAgeRegressor",
    "VIT_IMG_SIZES",
    "ViTAgeRegressor",
    "resolve_model_builder",
]

MODEL_ALIASES = {
    "cnt": "convnext_tiny",
    "cns": "convnext_small",
    "cnb": "convnext_base",
    "cnl": "convnext_large",
    "cnx": "convnext_xlarge",
    "vtt": "vit_tiny_384",
    "age_vit": "vit_tiny_384",
}


def resolve_model_builder(model_name: str, *, embed_dim: int = 0):
    """
    Resolve a model name (including aliases) to a builder, default image size, display label, and normalized key.
    """
    name = model_name.lower()
    name = MODEL_ALIASES.get(name, name)

    if name in EFFICIENTNET_IMG_SIZES:
        size = EFFICIENTNET_IMG_SIZES[name]
        return (
            lambda: EfficientNetAgeRegressor(name, embed_dim=embed_dim),
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
            lambda: ConvNeXtAgeRegressor(variant, embed_dim=embed_dim),
            size,
            f"ConvNeXt-{variant}",
            name,
        )

    if name.startswith("vit_"):
        variant = name.split("_", 1)[1]
        if variant not in VIT_IMG_SIZES:
            raise ValueError(f"Unsupported ViT variant '{variant}'.")
        size = VIT_IMG_SIZES[variant]
        return (
            lambda: ViTAgeRegressor(variant, embed_dim=embed_dim),
            size,
            f"ViT-{variant}",
            name,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Expected one of {sorted(EFFICIENTNET_IMG_SIZES)} or convnext_{{tiny,small,base,large,xlarge}} "
        f"or vit_{{tiny_384}} or aliases {sorted(MODEL_ALIASES)}."
    )
