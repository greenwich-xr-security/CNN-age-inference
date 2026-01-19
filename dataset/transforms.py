"""Transform builders for hand age datasets."""
from __future__ import annotations

from dataclasses import dataclass, replace

from torchvision import transforms


@dataclass(frozen=True)
class AugmentConfig:
    color_jitter_brightness: float = 0.35
    color_jitter_contrast: float = 0.35
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.02
    random_grayscale_p: float = 0.15
    gaussian_blur_kernel_size: int = 3
    gaussian_blur_sigma_min: float = 0.1
    gaussian_blur_sigma_max: float = 1.2
    random_erasing_p: float = 0.25
    random_erasing_scale_min: float = 0.02
    random_erasing_scale_max: float = 0.12
    random_erasing_ratio_min: float = 0.3
    random_erasing_ratio_max: float = 3.3

    def validate(self) -> None:
        if self.color_jitter_brightness < 0:
            raise ValueError("color_jitter_brightness must be >= 0.")
        if self.color_jitter_contrast < 0:
            raise ValueError("color_jitter_contrast must be >= 0.")
        if self.color_jitter_saturation < 0:
            raise ValueError("color_jitter_saturation must be >= 0.")
        if not (0.0 <= self.color_jitter_hue <= 0.5):
            raise ValueError("color_jitter_hue must be in [0, 0.5].")
        if not (0.0 <= self.random_grayscale_p <= 1.0):
            raise ValueError("random_grayscale_p must be in [0, 1].")
        if self.gaussian_blur_kernel_size < 1 or self.gaussian_blur_kernel_size % 2 == 0:
            raise ValueError("gaussian_blur_kernel_size must be an odd integer >= 1.")
        if self.gaussian_blur_sigma_min <= 0 or self.gaussian_blur_sigma_max <= 0:
            raise ValueError("gaussian_blur_sigma_min/max must be > 0.")
        if self.gaussian_blur_sigma_min > self.gaussian_blur_sigma_max:
            raise ValueError("gaussian_blur_sigma_min must be <= gaussian_blur_sigma_max.")
        if not (0.0 <= self.random_erasing_p <= 1.0):
            raise ValueError("random_erasing_p must be in [0, 1].")
        if self.random_erasing_scale_min <= 0 or self.random_erasing_scale_max <= 0:
            raise ValueError("random_erasing_scale_min/max must be > 0.")
        if self.random_erasing_scale_min > self.random_erasing_scale_max:
            raise ValueError("random_erasing_scale_min must be <= random_erasing_scale_max.")
        if self.random_erasing_ratio_min <= 0 or self.random_erasing_ratio_max <= 0:
            raise ValueError("random_erasing_ratio_min/max must be > 0.")
        if self.random_erasing_ratio_min > self.random_erasing_ratio_max:
            raise ValueError("random_erasing_ratio_min must be <= random_erasing_ratio_max.")


PHOTOMETRIC_LEVELS = {
    "low": {
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.1,
        "hue": 0.01,
        "gray_p": 0.05,
    },
    "med": {
        "brightness": 0.35,
        "contrast": 0.35,
        "saturation": 0.2,
        "hue": 0.02,
        "gray_p": 0.15,
    },
    "high": {
        "brightness": 0.6,
        "contrast": 0.6,
        "saturation": 0.35,
        "hue": 0.06,
        "gray_p": 0.3,
    },
}

FOCUS_LEVELS = {
    "low": {
        "kernel": 3,
        "sigma_min": 0.05,
        "sigma_max": 0.6,
    },
    "med": {
        "kernel": 3,
        "sigma_min": 0.1,
        "sigma_max": 1.2,
    },
    "high": {
        "kernel": 5,
        "sigma_min": 0.3,
        "sigma_max": 2.0,
    },
}

OCCLUSION_LEVELS = {
    "tiny": {
        "erase_p": 0.15,
        "scale_min": 0.005,
        "scale_max": 0.03,
    },
    "small": {
        "erase_p": 0.25,
        "scale_min": 0.02,
        "scale_max": 0.12,
    },
    "big": {
        "erase_p": 0.35,
        "scale_min": 0.08,
        "scale_max": 0.3,
    },
}

PHOTOMETRIC_LEVEL_CHOICES = tuple(PHOTOMETRIC_LEVELS.keys())
FOCUS_LEVEL_CHOICES = tuple(FOCUS_LEVELS.keys())
OCCLUSION_LEVEL_CHOICES = tuple(OCCLUSION_LEVELS.keys())


def _interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _blend_levels(levels: dict[str, dict[str, float]], t: float, *, low_key: str, mid_key: str, high_key: str):
    if t <= 0.5:
        a = levels[low_key]
        b = levels[mid_key]
        alpha = t * 2.0
    else:
        a = levels[mid_key]
        b = levels[high_key]
        alpha = (t - 0.5) * 2.0
    return {key: _interp(a[key], b[key], alpha) for key in a}


def _nearest_odd(value: float) -> int:
    candidate = int(round(value))
    if candidate < 1:
        candidate = 1
    if candidate % 2 == 0:
        up = candidate + 1
        down = candidate - 1
        if down >= 1 and abs(value - down) <= abs(up - value):
            candidate = down
        else:
            candidate = up
    return candidate


def _level_to_strength(level: str | float, *, levels: tuple[str, ...], name: str) -> float:
    if isinstance(level, str):
        if level in levels:
            if level == "low" or level == "tiny":
                return 0.0
            if level == "med" or level == "small":
                return 0.5
            return 1.0
        try:
            strength = float(level)
        except ValueError as exc:
            raise ValueError(f"Unknown {name} level: {level}") from exc
    else:
        strength = float(level)
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"{name} level must be in [0, 1].")
    return strength


def apply_augmentation_levels(
    augment: AugmentConfig,
    *,
    photometric_level: str | float | None = None,
    focus_level: str | float | None = None,
    occlusion_level: str | float | None = None,
) -> AugmentConfig:
    updated = augment
    if photometric_level is not None:
        strength = _level_to_strength(
            photometric_level,
            levels=PHOTOMETRIC_LEVEL_CHOICES,
            name="photometric",
        )
        preset = _blend_levels(
            PHOTOMETRIC_LEVELS,
            strength,
            low_key="low",
            mid_key="med",
            high_key="high",
        )
        updated = replace(
            updated,
            color_jitter_brightness=preset["brightness"],
            color_jitter_contrast=preset["contrast"],
            color_jitter_saturation=preset["saturation"],
            color_jitter_hue=preset["hue"],
            random_grayscale_p=preset["gray_p"],
        )
    if focus_level is not None:
        strength = _level_to_strength(
            focus_level,
            levels=FOCUS_LEVEL_CHOICES,
            name="focus",
        )
        preset = _blend_levels(
            FOCUS_LEVELS,
            strength,
            low_key="low",
            mid_key="med",
            high_key="high",
        )
        updated = replace(
            updated,
            gaussian_blur_kernel_size=_nearest_odd(preset["kernel"]),
            gaussian_blur_sigma_min=preset["sigma_min"],
            gaussian_blur_sigma_max=preset["sigma_max"],
        )
    if occlusion_level is not None:
        strength = _level_to_strength(
            occlusion_level,
            levels=OCCLUSION_LEVEL_CHOICES,
            name="occlusion",
        )
        preset = _blend_levels(
            OCCLUSION_LEVELS,
            strength,
            low_key="tiny",
            mid_key="small",
            high_key="big",
        )
        updated = replace(
            updated,
            random_erasing_p=preset["erase_p"],
            random_erasing_scale_min=preset["scale_min"],
            random_erasing_scale_max=preset["scale_max"],
        )
    updated.validate()
    return updated


def build_transforms(img_size: int, augment: AugmentConfig | None = None):
    """Return train/test transforms for a given square image size."""
    if augment is None:
        augment = AugmentConfig()
    augment.validate()

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            
            # Photometric robustness
            transforms.ColorJitter(
                brightness=augment.color_jitter_brightness,
                contrast=augment.color_jitter_contrast,
                saturation=augment.color_jitter_saturation,
                hue=augment.color_jitter_hue,
            ),
            transforms.RandomGrayscale(p=augment.random_grayscale_p),

            # Camera / focus robustness
            transforms.GaussianBlur(
                kernel_size=augment.gaussian_blur_kernel_size,
                sigma=(augment.gaussian_blur_sigma_min, augment.gaussian_blur_sigma_max),
            ),

            transforms.ToTensor(),
            
            # Occlusion robustness (forces not relying on tiny regions)
            transforms.RandomErasing(
                p=augment.random_erasing_p,
                scale=(augment.random_erasing_scale_min, augment.random_erasing_scale_max),
                ratio=(augment.random_erasing_ratio_min, augment.random_erasing_ratio_max),
                value="random",
            ),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform
