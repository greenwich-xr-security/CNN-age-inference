"""Multi-crop transforms for DINO-style SSL on hand images."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@dataclass
class DinoAugmentationConfig:
    brightness: float = 0.05
    contrast: float = 0.05
    saturation: float = 0.02
    hue: float = 0.005
    color_jitter_prob: float = 0.2
    grayscale_prob: float = 0.0
    blur_prob_global: float = 0.05
    blur_prob_local: float = 0.05
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 0.2
    mask_prob: float = 0.1
    mask_scale_min: float = 0.9
    mask_scale_max: float = 0.98
    mask_offset: float = 0.02
    rotation_degrees: float = 360.0


class DinoMultiCropTransform:
    def __init__(
        self,
        *,
        global_size: int,
        local_size: int,
        num_local_crops: int,
        global_scale: Tuple[float, float],
        local_scale: Tuple[float, float],
        config: DinoAugmentationConfig,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.global_size = int(global_size)
        self.local_size = int(local_size)
        self.num_local_crops = int(num_local_crops)
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.config = config
        self.mean = mean
        self.std = std
        self.strength = 1.0
        self._global_transform = None
        self._local_transform = None
        self.set_strength(1.0)

    def set_strength(self, strength: float) -> None:
        strength = float(max(0.0, min(1.0, strength)))
        self.strength = strength

        brightness = self.config.brightness * strength
        contrast = self.config.contrast * strength
        saturation = self.config.saturation * strength
        hue = self.config.hue * strength
        jitter_prob = self.config.color_jitter_prob if strength > 0 else 0.0
        grayscale_prob = self.config.grayscale_prob * strength

        color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        jitter = transforms.RandomApply([color_jitter], p=jitter_prob)
        grayscale = transforms.RandomGrayscale(p=grayscale_prob)

        blur_sigma = (self.config.blur_sigma_min, self.config.blur_sigma_max)
        blur_global = transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=blur_sigma)],
            p=self.config.blur_prob_global * strength,
        )
        blur_local = transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=blur_sigma)],
            p=self.config.blur_prob_local * strength,
        )

        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self._global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.global_size,
                    scale=self.global_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                jitter,
                grayscale,
                blur_global,
                transforms.ToTensor(),
                normalize,
            ]
        )
        self._local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.local_size,
                    scale=self.local_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                jitter,
                grayscale,
                blur_local,
                transforms.ToTensor(),
                normalize,
            ]
        )

    def _apply_rotation(self, img: Image.Image, angle: float) -> Image.Image:
        rotated = F.rotate(
            img,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=True,
            fill=0,
        )
        width, height = img.size
        return F.center_crop(rotated, (height, width))

    def _sample_mask_params(self) -> dict:
        apply = random.random() < self.config.mask_prob * self.strength
        if not apply:
            return {"apply": False}
        scale = random.uniform(self.config.mask_scale_min, self.config.mask_scale_max)
        offset_x = random.uniform(-self.config.mask_offset, self.config.mask_offset)
        offset_y = random.uniform(-self.config.mask_offset, self.config.mask_offset)
        color = tuple(random.randint(32, 224) for _ in range(3))
        return {
            "apply": True,
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "color": color,
        }

    def _apply_background_mask(self, img: Image.Image, params: dict) -> Image.Image:
        if not params.get("apply", False):
            return img
        width, height = img.size
        scale = params["scale"]
        offset_x = params["offset_x"] * width
        offset_y = params["offset_y"] * height
        center_x = width * 0.5 + offset_x
        center_y = height * 0.5 + offset_y
        radius_x = width * 0.5 * scale
        radius_y = height * 0.5 * scale
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        bbox = [
            center_x - radius_x,
            center_y - radius_y,
            center_x + radius_x,
            center_y + radius_y,
        ]
        draw.ellipse(bbox, fill=255)
        bg = Image.new("RGB", (width, height), params["color"])
        return Image.composite(img, bg, mask)

    def __call__(self, img1: Image.Image, img2: Image.Image):
        angle1 = random.uniform(0.0, self.config.rotation_degrees)
        angle2 = random.uniform(0.0, self.config.rotation_degrees)
        mask_params = self._sample_mask_params()

        img1 = self._apply_rotation(img1, angle1)
        img2 = self._apply_rotation(img2, angle2)
        img1 = self._apply_background_mask(img1, mask_params)
        img2 = self._apply_background_mask(img2, mask_params)

        crops = [self._global_transform(img1), self._global_transform(img2)]
        local_from_first = int(math.ceil(self.num_local_crops / 2))
        for idx in range(self.num_local_crops):
            source = img1 if idx < local_from_first else img2
            crops.append(self._local_transform(source))
        return crops
