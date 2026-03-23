from .efficientnet_age import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor
from .convnext_age import CONVNEXT_IMG_SIZES, ConvNeXtAgeRegressor
from .mobilenet_age import MOBILENET_IMG_SIZES, MobileNetAgeRegressor
from .swin_age import SWIN_IMG_SIZES, SwinAgeRegressor
from .vit_age import VIT_IMG_SIZES, ViTAgeRegressor

__all__ = [
    "EFFICIENTNET_IMG_SIZES",
    "EfficientNetAgeRegressor",
    "CONVNEXT_IMG_SIZES",
    "ConvNeXtAgeRegressor",
    "MOBILENET_IMG_SIZES",
    "MobileNetAgeRegressor",
    "SWIN_IMG_SIZES",
    "SwinAgeRegressor",
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
    # Swin V1
    "swt": "swin_tiny",
    "sws": "swin_small",
    "swb": "swin_base",
    "swl": "swin_large",
    # Swin V2
    "sw2t": "swin_v2_tiny",
    "sw2s": "swin_v2_small",
    "sw2b": "swin_v2_base",
    "sw2l": "swin_v2_large",
    # MobileNet
    "mnv2":  "mobilenet_v2",
    "mnv3s": "mobilenet_v3_small",
    "mnv3l": "mobilenet_v3_large",
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

    if name.startswith("mobilenet_"):
        variant = name[len("mobilenet_"):]
        if variant not in MOBILENET_IMG_SIZES:
            raise ValueError(f"Unsupported MobileNet variant '{variant}'.")
        size = MOBILENET_IMG_SIZES[variant]
        label = f"MobileNet-{variant.replace('_', '-').upper()}"
        return (
            lambda: MobileNetAgeRegressor(variant, embed_dim=embed_dim),
            size,
            label,
            name,
        )

    if name.startswith("swin_"):
        # Strip leading "swin_" prefix to get the variant key used in SWIN_IMG_SIZES
        variant = name[len("swin_"):]
        if variant not in SWIN_IMG_SIZES:
            raise ValueError(f"Unsupported Swin variant '{variant}'.")
        size = SWIN_IMG_SIZES[variant]
        label = f"Swin-{variant.replace('_', '-').upper()}"
        return (
            lambda: SwinAgeRegressor(variant, embed_dim=embed_dim),
            size,
            label,
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
        f"Expected one of {sorted(EFFICIENTNET_IMG_SIZES)} "
        f"or convnext_{{tiny,small,base,large,xlarge}} "
        f"or mobilenet_{{v2,v3_small,v3_large}} "
        f"or swin_{{tiny,small,base,large,v2_tiny,v2_small,v2_base,v2_large}} "
        f"or vit_{{tiny_384}} "
        f"or aliases {sorted(MODEL_ALIASES)}."
    )
