import torch.nn as nn

from .efficientnet_age import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor
from .convnext_age import CONVNEXT_IMG_SIZES, ConvNeXtAgeRegressor
from .mobilenet_age import MOBILENET_IMG_SIZES, MobileNetAgeRegressor
from .normals_head import NormalsHead
from .resnet_age import RESNET_IMG_SIZES, ResNetAgeRegressor
from .swin_age import SWIN_IMG_SIZES, SwinAgeRegressor
from .vit_age import VIT_IMG_SIZES, ViTAgeRegressor

__all__ = [
    "EFFICIENTNET_IMG_SIZES",
    "EfficientNetAgeRegressor",
    "CONVNEXT_IMG_SIZES",
    "ConvNeXtAgeRegressor",
    "MOBILENET_IMG_SIZES",
    "MobileNetAgeRegressor",
    "NormalsHead",
    "RESNET_IMG_SIZES",
    "ResNetAgeRegressor",
    "SWIN_IMG_SIZES",
    "SwinAgeRegressor",
    "VIT_IMG_SIZES",
    "ViTAgeRegressor",
    "expand_first_conv_to_6ch",
    "resolve_model_builder",
]


def expand_first_conv_to_6ch(model: nn.Module) -> None:
    """Expand the first Conv2d from 3→6 input channels in-place.

    The original RGB weights are copied to the first 3 channels.
    The extra 3 channels (normals) are zero-initialized, so the model is
    initially equivalent to a 3-channel model receiving zero-padded normals.
    This supports privileged-information training: normals are supplied during
    training and replaced with zeros at inference time.
    """
    import torch
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                6, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = module.weight
                new_conv.weight[:, 3:] = 0.0
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            return
    raise RuntimeError("expand_first_conv_to_6ch: no Conv2d with in_channels=3 found.")

MODEL_ALIASES = {
    "cnt": "convnext_tiny",
    "cns": "convnext_small",
    "cnb": "convnext_base",
    "cnl": "convnext_large",
    "cnx": "convnext_xlarge",
    "vtt": "vit_tiny_384",
    "vts": "vit_small_384",
    "age_vit": "vit_tiny_384",
    "age_vit_small": "vit_small_384",
    # Swin V1
    "swt": "swin_tiny",
    "sws": "swin_small",
    "swb": "swin_base",
    "swl": "swin_large",
    # Swin V2
    "sw2t": "swin_v2_tiny",
    "sw2t384": "swin_v2_tiny_384",
    "sw2t512": "swin_v2_tiny_512",
    "sw2s": "swin_v2_small",
    "sw2b": "swin_v2_base",
    "sw2b384": "swin_v2_base_384",
    "sw2l": "swin_v2_large",
    # MobileNet
    "mnv2":  "mobilenet_v2",
    "mnv3s": "mobilenet_v3_small",
    "mnv3l": "mobilenet_v3_large",
    # ResNet
    "rn50": "resnet50",
}


def resolve_model_builder(
    model_name: str,
    *,
    embed_dim: int = 0,
    normals_aux: bool = False,
    normals_privileged: bool = False,
):
    """
    Resolve a model name (including aliases) to a builder, default image size, display label, and normalized key.

    normals_aux: attach a lightweight normal-map decoder to the backbone (EfficientNet only).
    normals_privileged: expand the first conv to 6 channels for privileged normals input at train time.
    """
    name = model_name.lower()
    name = MODEL_ALIASES.get(name, name)

    if name in EFFICIENTNET_IMG_SIZES:
        size = EFFICIENTNET_IMG_SIZES[name]
        return (
            lambda: EfficientNetAgeRegressor(
                name, embed_dim=embed_dim,
                normals_aux=normals_aux, normals_privileged=normals_privileged,
            ),
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
            lambda: ConvNeXtAgeRegressor(variant, embed_dim=embed_dim, normals_privileged=normals_privileged),
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

    if name.startswith("resnet"):
        variant = name[len("resnet"):]
        if variant not in RESNET_IMG_SIZES:
            raise ValueError(f"Unsupported ResNet variant '{variant}'.")
        size = RESNET_IMG_SIZES[variant]
        label = f"ResNet-{variant}"
        return (
            lambda: ResNetAgeRegressor(variant, embed_dim=embed_dim),
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
            lambda: ViTAgeRegressor(variant, embed_dim=embed_dim, normals_privileged=normals_privileged),
            size,
            f"ViT-{variant}",
            name,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Expected one of {sorted(EFFICIENTNET_IMG_SIZES)} "
        f"or convnext_{{tiny,small,base,large,xlarge}} "
        f"or mobilenet_{{v2,v3_small,v3_large}} "
        f"or resnet50 "
        f"or swin_{{tiny,small,base,large,v2_tiny,v2_tiny_384,v2_tiny_512,v2_small,v2_base,v2_base_384,v2_large}} "
        f"or vit_{{tiny_384,small_384}} "
        f"or aliases {sorted(MODEL_ALIASES)}."
    )
