import os
import sys

import cv2
import numpy as np


DEFAULT_MASK_THRESHOLD = 128
DEFAULT_SHADOW_LUMA = 5
DEFAULT_HIGHLIGHT_LUMA = 250


def _to_luma(image: np.ndarray, color_space: str) -> np.ndarray:
    if image.ndim == 2:
        return image

    img = image
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if color_space.upper() == "RGB":
        code = cv2.COLOR_RGB2LAB
    elif color_space.upper() == "BGR":
        code = cv2.COLOR_BGR2LAB
    else:
        raise ValueError(f"Unsupported color_space: {color_space}")

    lab = cv2.cvtColor(img, code)
    return lab[:, :, 0]


def _to_mask(mask: np.ndarray | None, threshold: int) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bin = mask > threshold
    if not np.any(mask_bin):
        raise ValueError(f"Mask has no pixels above threshold {threshold}")
    return mask_bin


def compute_exposure_metrics(
    image: np.ndarray,
    mask: np.ndarray | None,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    shadow_luma: int = DEFAULT_SHADOW_LUMA,
    highlight_luma: int = DEFAULT_HIGHLIGHT_LUMA,
    color_space: str = "BGR",
) -> dict[str, float]:
    if image is None:
        raise ValueError("image is None")

    luma = _to_luma(image, color_space)
    mask_bin = _to_mask(mask, mask_threshold)
    if mask_bin is None:
        mask_bin = np.ones(luma.shape, dtype=bool)
    if mask_bin.shape != luma.shape:
        raise ValueError("mask must match image width and height")

    luma_roi = luma[mask_bin]
    mean_luma = float(np.mean(luma_roi))
    shadow_clip = float(np.mean(luma_roi <= shadow_luma))
    highlight_clip = float(np.mean(luma_roi >= highlight_luma))
    return {
        "mean": mean_luma,
        "shadow_clip": shadow_clip,
        "highlight_clip": highlight_clip,
        "count": float(luma_roi.size),
    }


def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


if __name__ == "__main__":
    input_path = "11899.png"
    input_dir, input_base = os.path.split(input_path)
    mask_path = os.path.join(input_dir, f"Mask_{input_base}")

    if not os.path.exists(mask_path):
        base_name, base_ext = os.path.splitext(input_base)
        if base_name.endswith("_out"):
            alt_base = f"{base_name[:-4]}{base_ext}"
            alt_mask_path = os.path.join(input_dir, f"Mask_{alt_base}")
            if os.path.exists(alt_mask_path):
                mask_path = alt_mask_path

    try:
        img = _load_image(input_path)
        mask = _load_mask(mask_path)
        metrics = compute_exposure_metrics(img, mask)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print(
        "mean={mean:.1f} shadow_clip={shadow_clip:.4f} highlight_clip={highlight_clip:.4f}".format(
            **metrics
        )
    )
