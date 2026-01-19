from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import cm
from PIL import Image
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import resolve_model_builder
from onnx_tools.run_onnx import IMAGENET_MEAN, IMAGENET_STD


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


def _load_masked_base(
    image_path: Path, mask_path: Path | None, mask_threshold: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    img = Image.open(image_path).convert("RGB")
    mask = None
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)

    width, height = img.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    if mask is not None:
        mask = mask.crop((left, top, left + crop, top + crop))

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
    ) -> None:
        method = method.lower().replace(" ", "")
        if method not in ("gradcam", "gradcam++"):
            raise ValueError("method must be 'gradcam' or 'gradcam++'.")
        builder, default_size, _, _ = resolve_model_builder(model_name)
        model = builder()
        _load_checkpoint(model, checkpoint_path)
        model.eval()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.default_size = default_size
        self.method = method
        self.layer_name, layer_module = _resolve_target_layer(self.model, target_layer)
        self._hook = _FeatureHook(layer_module)

    def close(self) -> None:
        self._hook.close()

    def render(
        self,
        image_path: Path,
        *,
        mask_path: Path | None = None,
        img_size: int | None = None,
        alpha: float = 0.45,
        mask_threshold: int = 0,
    ) -> Image.Image:
        size = img_size or self.default_size
        base_arr, _ = _load_masked_base(image_path, mask_path, mask_threshold)
        tensor, preview = _prepare_tensor(base_arr, size)
        tensor = tensor.to(self.device)

        self.model.zero_grad(set_to_none=True)
        mean, _ = self.model(tensor)
        target = mean.sum()
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
            alpha = grads2 / denom
            positive_grads = torch.relu(gradients)
            weights = (alpha * positive_grads).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam_np = cam[0].detach().cpu().numpy()
        cam_np -= cam_np.min()
        denom = cam_np.max()
        if denom > 0:
            cam_np /= denom

        cam_img = Image.fromarray((cam_np * 255).astype(np.uint8))
        cam_img = cam_img.resize(preview.size, Image.BILINEAR)
        cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0
        heatmap = cm.get_cmap("jet")(cam_arr)[..., :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        base = np.asarray(preview, dtype=np.uint8)
        overlay = (1.0 - alpha) * base + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay, mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlay for a checkpoint.")
    parser.add_argument("--model", required=True, help="Model name (e.g. b2, convnext_base).")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--mask", type=str, default=None, help="Optional mask image path.")
    parser.add_argument("--size", type=int, default=None, help="Override input size.")
    parser.add_argument("--layer", type=str, default=None, help="Target layer path (dot notation).")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
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
    )
    try:
        overlay = runner.render(
            Path(args.image),
            mask_path=Path(args.mask) if args.mask else None,
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
