from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import cm
from PIL import Image, ImageOps
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from displayUtils import DisplayUtils
from models import resolve_model_builder
from onnx_tools.run_onnx import IMAGENET_MEAN, IMAGENET_STD

LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 10.0


def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if all(k.startswith("module.") for k in state.keys()):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint did not contain a valid state_dict.")
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not match model architecture. "
            f"Missing: {missing} | Unexpected: {unexpected}"
        )


def _apply_square_bbox_crop(
    image: Image.Image,
    mask: Image.Image | None,
    bbox: tuple[int, int, int, int] | None,
) -> tuple[Image.Image, Image.Image | None]:
    if bbox is None:
        width, height = image.size
        crop = min(width, height)
        left = (width - crop) // 2
        top = (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
        if mask is not None:
            mask = mask.crop((left, top, left + crop, top + crop))
        return image, mask

    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"bbox has invalid coordinates: {bbox}")

    sq_xmin, sq_ymin, sq_xmax, sq_ymax = DisplayUtils.make_square_bbox(
        (xmin, ymin, xmax, ymax)
    )
    width, height = image.size

    pad_left = max(0, -sq_xmin)
    pad_top = max(0, -sq_ymin)
    pad_right = max(0, sq_xmax - width)
    pad_bottom = max(0, sq_ymax - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0),
        )
        if mask is not None:
            mask = ImageOps.expand(
                mask,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=0,
            )
        sq_xmin += pad_left
        sq_xmax += pad_left
        sq_ymin += pad_top
        sq_ymax += pad_top

    sq_xmin = max(0, sq_xmin)
    sq_ymin = max(0, sq_ymin)
    sq_xmax = max(sq_xmin + 1, min(image.size[0], sq_xmax))
    sq_ymax = max(sq_ymin + 1, min(image.size[1], sq_ymax))
    image = image.crop((sq_xmin, sq_ymin, sq_xmax, sq_ymax))
    if mask is not None:
        mask = mask.crop((sq_xmin, sq_ymin, sq_xmax, sq_ymax))
    return image, mask


def _load_masked_base(
    image_path: Path,
    mask_path: Path | None,
    mask_threshold: int = 0,
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    img = Image.open(image_path).convert("RGB")
    mask = None
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)

    img, mask = _apply_square_bbox_crop(img, mask, bbox)

    img_arr = np.asarray(img, dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8) if mask is not None else None
    if mask_arr is not None:
        mask_bin = mask_arr > mask_threshold
        base_arr = np.array(img_arr, copy=True)
        base_arr[~mask_bin] = 0
    else:
        base_arr = img_arr

    return base_arr, mask_arr


def _prepare_tensor(base_arr: np.ndarray, size: int) -> tuple[torch.Tensor, Image.Image]:
    preview = Image.fromarray(base_arr, mode="RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(preview, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    tensor = torch.from_numpy(arr).float()
    return tensor, preview


def _get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current = model
    for part in path.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(current, (nn.Sequential, nn.ModuleList, list, tuple)):
                current = current[idx]
            else:
                raise ValueError(f"Cannot index into module '{part}' in path '{path}'.")
        else:
            current = getattr(current, part)
    return current


def _resolve_target_layer(
    model: torch.nn.Module, target_layer: str | None
) -> tuple[str, torch.nn.Module]:
    if target_layer:
        return target_layer, _get_module_by_path(model, target_layer)
    last_name = ""
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    if last_module is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return last_name, last_module


class _FeatureHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = module.register_forward_hook(self._forward_hook)
        try:
            self._backward_handle = module.register_full_backward_hook(self._backward_hook)
        except AttributeError:
            self._backward_handle = module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        if grad_output:
            self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


class GradCamRunner:
    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: Path,
        device: str = "cpu",
        target_layer: str | None = None,
        method: str = "gradcam",
        embed_dim: int = 0,
        target_mode: str = "age_mean",
        age_threshold: float = 18.0,
    ) -> None:
        method = method.lower().replace(" ", "")
        if method not in ("gradcam", "gradcam++"):
            raise ValueError("method must be 'gradcam' or 'gradcam++'.")
        target_mode = target_mode.lower().replace(" ", "_")
        if target_mode not in ("age_mean", "adult_prob"):
            raise ValueError("target_mode must be 'age_mean' or 'adult_prob'.")
        builder, default_size, _, _ = resolve_model_builder(model_name, embed_dim=embed_dim)
        model = builder()
        _load_checkpoint(model, checkpoint_path)
        model.eval()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.default_size = default_size
        self.method = method
        self.embed_dim = int(embed_dim)
        self.target_mode = target_mode
        self.age_threshold = float(age_threshold)
        self.layer_name, layer_module = _resolve_target_layer(self.model, target_layer)
        self._hook = _FeatureHook(layer_module)

    def close(self) -> None:
        self._hook.close()

    def _compute_target(
        self,
        outputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 3:
                pred_mean, pred_log_var, _ = outputs
            elif len(outputs) == 2:
                pred_mean, pred_log_var = outputs
            else:
                raise RuntimeError(f"Unsupported model output structure: {len(outputs)} tensors.")
        else:
            raise RuntimeError("Model output must be a tuple/list of (mean, log_var[, embedding]).")

        log_var = torch.clamp(pred_log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
        std = torch.exp(0.5 * log_var).clamp(min=1e-3)
        z = (self.age_threshold - pred_mean) / std
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        adult_prob = torch.clamp(1.0 - cdf, min=0.0, max=1.0)

        if self.target_mode == "adult_prob":
            target = adult_prob.sum()
        else:
            target = pred_mean.sum()
        return target, pred_mean, log_var, adult_prob

    def compute(
        self,
        image_path: Path,
        *,
        mask_path: Path | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        img_size: int | None = None,
        alpha: float = 0.45,
        mask_threshold: int = 0,
    ) -> dict[str, object]:
        size = img_size or self.default_size
        base_arr, _ = _load_masked_base(
            image_path,
            mask_path,
            mask_threshold,
            bbox=bbox,
        )
        tensor, preview = _prepare_tensor(base_arr, size)
        tensor = tensor.to(self.device)

        self.model.zero_grad(set_to_none=True)
        outputs = self.model(tensor)
        target, pred_mean, pred_log_var, adult_prob = self._compute_target(outputs)
        target.backward()

        activations = self._hook.activations
        gradients = self._hook.gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        if self.method == "gradcam":
            weights = gradients.mean(dim=(2, 3), keepdim=True)
        else:
            grads2 = gradients.pow(2)
            grads3 = gradients.pow(3)
            sum_a_grads3 = (activations * grads3).sum(dim=(2, 3), keepdim=True)
            denom = 2.0 * grads2 + sum_a_grads3
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha_weights = grads2 / denom
            positive_grads = torch.relu(gradients)
            weights = (alpha_weights * positive_grads).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam_np = cam[0].detach().cpu().numpy()
        cam_np -= cam_np.min()
        denom = cam_np.max()
        if denom > 0:
            cam_np /= denom

        cam_gray = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
        cam_gray = cam_gray.resize(preview.size, Image.BILINEAR)
        cam_arr = np.asarray(cam_gray, dtype=np.float32) / 255.0
        heatmap_arr = cm.get_cmap("jet")(cam_arr)[..., :3]
        heatmap_rgb = (heatmap_arr * 255).astype(np.uint8)

        base = np.asarray(preview, dtype=np.uint8)
        overlay = (1.0 - alpha) * base + alpha * heatmap_rgb
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return {
            "overlay": Image.fromarray(overlay, mode="RGB"),
            "heatmap": Image.fromarray(heatmap_rgb, mode="RGB"),
            "cam_gray": cam_gray,
            "cam_array": cam_arr,
            "preview": preview,
            "pred_mean": float(pred_mean.detach().cpu().reshape(-1)[0]),
            "pred_log_var": float(pred_log_var.detach().cpu().reshape(-1)[0]),
            "adult_prob": float(adult_prob.detach().cpu().reshape(-1)[0]),
            "target_score": float(target.detach().cpu().item()),
            "target_mode": self.target_mode,
            "layer_name": self.layer_name,
        }

    def render(
        self,
        image_path: Path,
        *,
        mask_path: Path | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        img_size: int | None = None,
        alpha: float = 0.45,
        mask_threshold: int = 0,
    ) -> Image.Image:
        result = self.compute(
            image_path,
            mask_path=mask_path,
            bbox=bbox,
            img_size=img_size,
            alpha=alpha,
            mask_threshold=mask_threshold,
        )
        return result["overlay"]


def _parse_bbox(raw_bbox: str | None) -> tuple[int, int, int, int] | None:
    if raw_bbox is None:
        return None
    parts = [part.strip() for part in raw_bbox.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must be four comma-separated integers: xmin,ymin,xmax,ymax")
    return tuple(int(round(float(part))) for part in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlay for a checkpoint.")
    parser.add_argument("--model", required=True, help="Model name (e.g. b2, convnext_base).")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--mask", type=str, default=None, help="Optional mask image path.")
    parser.add_argument("--bbox", type=str, default=None, help="Optional bbox xmin,ymin,xmax,ymax.")
    parser.add_argument("--size", type=int, default=None, help="Override input size.")
    parser.add_argument("--layer", type=str, default=None, help="Target layer path (dot notation).")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
    parser.add_argument("--embed-dim", type=int, default=0, help="Embedding dimension used during training.")
    parser.add_argument(
        "--target-mode",
        choices=["age_mean", "adult_prob"],
        default="age_mean",
        help="Quantity to attribute. 'adult_prob' is better aligned with age-gate analysis.",
    )
    parser.add_argument(
        "--age-threshold",
        type=float,
        default=18.0,
        help="Age threshold used when target-mode=adult_prob.",
    )
    parser.add_argument(
        "--method",
        choices=["gradcam", "gradcam++"],
        default="gradcam",
        help="Grad-CAM variant to use.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run Grad-CAM on.",
    )
    args = parser.parse_args()

    runner = GradCamRunner(
        model_name=args.model,
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
        target_layer=args.layer,
        method=args.method,
        embed_dim=args.embed_dim,
        target_mode=args.target_mode,
        age_threshold=args.age_threshold,
    )
    try:
        overlay = runner.render(
            Path(args.image),
            mask_path=Path(args.mask) if args.mask else None,
            bbox=_parse_bbox(args.bbox),
            img_size=args.size,
            alpha=args.alpha,
        )
    finally:
        runner.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    print(f"[grad-cam] saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
