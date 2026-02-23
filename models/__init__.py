from .efficientnet_age import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor
from .convnext_age import CONVNEXT_IMG_SIZES, ConvNeXtAgeRegressor
from .dinov2_age import DINOv2_IMG_SIZES, DinoV2AgeRegressor

__all__ = [
    "EFFICIENTNET_IMG_SIZES",
    "EfficientNetAgeRegressor",
    "CONVNEXT_IMG_SIZES",
    "ConvNeXtAgeRegressor",
    "DINOv2_IMG_SIZES",
    "DinoV2AgeRegressor",
    "resolve_model_builder",
]

MODEL_ALIASES = {
    "cnt": "convnext_tiny",
    "cns": "convnext_small",
    "cnb": "convnext_base",
    "cnl": "convnext_large",
    "cnx": "convnext_xlarge",
    "d2s": "dinov2_vits14",
    "d2b": "dinov2_vitb14",
    "d2l": "dinov2_vitl14",
    "d2g": "dinov2_vitg14",
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

    if name in DINOv2_IMG_SIZES:
        size = DINOv2_IMG_SIZES[name]
        label = name.replace("dinov2_", "DINOv2-").replace("_", "-")
        return (
            lambda: DinoV2AgeRegressor(name, embed_dim=embed_dim),
            size,
            label,
            name,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Expected one of {sorted(EFFICIENTNET_IMG_SIZES)} or convnext_{{tiny,small,base,large,xlarge}} "
        f"or DINOv2 variants {sorted(DINOv2_IMG_SIZES)} or aliases {sorted(MODEL_ALIASES)}."
    )
