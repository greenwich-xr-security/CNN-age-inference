import argparse
import sys
from pathlib import Path

import torch

from models.efficientnet_age import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor

# Example commands:
#   python export_onnx.py --model b2 --checkpoint runs\b2_efficientnet\b2_age_regressor.pth --output runs\b2_efficientnet\b2_age_regressor.onnx
#   python export_onnx.py --model b2 --checkpoint runs\b2_efficientnet\b2_age_regressor.pth --output runs\b2_efficientnet\b2_age_regressor_fp16.onnx --precision fp16
#   python export_onnx.py --model b2 --checkpoint runs\b2_efficientnet\b2_age_regressor.pth --output runs\b2_efficientnet\b2_age_regressor_int8.onnx --precision int8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EfficientNet age regressor to ONNX (Sentis-friendly)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="b0",
        help="EfficientNet variant: b0-b7, v2_s, v2_m, v2_l.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth state_dict checkpoint. If omitted, exports the default-weight backbone with a random head.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="ONNX opset version (Sentis recommends 15).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override input size. Defaults to the canonical EfficientNet size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used for tracing (default: 1).",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch dimension.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Output precision / quantization mode.",
    )
    parser.add_argument(
        "--fp16-keep-io",
        action="store_true",
        help="Keep float32 input/output types when converting to fp16 (onnxconverter-common only).",
    )
    parser.add_argument(
        "--int8-per-channel",
        action="store_true",
        help="Use per-channel weights for INT8 dynamic quantization.",
    )
    parser.add_argument(
        "--int8-reduce-range",
        action="store_true",
        help="Use reduced int8 range (0-127) for INT8 quantization.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use during export.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate fp32 ONNX when producing fp16/int8 outputs.",
    )
    return parser.parse_args()


def _resolve_img_size(model_key: str, override: int | None) -> int:
    if override is not None:
        return override
    if model_key not in EFFICIENTNET_IMG_SIZES:
        raise ValueError(
            f"Unsupported EfficientNet variant '{model_key}'. "
            f"Expected one of {sorted(EFFICIENTNET_IMG_SIZES)}."
        )
    return EFFICIENTNET_IMG_SIZES[model_key]


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


def _export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> None:
    input_names = ["input"]
    output_names = ["mean", "log_var"]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "mean": {0: "batch"},
            "log_var": {0: "batch"},
        }
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def _convert_fp16(
    fp32_path: Path,
    output_path: Path,
    keep_io_types: bool,
) -> None:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            "FP16 conversion requires onnx and onnxconverter-common. "
            "Install them or export with --precision fp32."
        ) from exc
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
    )
    onnx.save(model_fp16, str(output_path))


def _quantize_int8(
    fp32_path: Path,
    output_path: Path,
    per_channel: bool,
    reduce_range: bool,
) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "INT8 quantization requires onnxruntime. Install it or export with --precision fp32."
        ) from exc
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )


def main() -> int:
    args = _parse_args()
    model_key = args.model.lower()
    img_size = _resolve_img_size(model_key, args.img_size)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = EfficientNetAgeRegressor(model_key)
    if args.checkpoint:
        _load_checkpoint(model, Path(args.checkpoint))
    model.eval()

    dummy_input = torch.randn(
        args.batch_size, 3, img_size, img_size, device=device
    )

    if args.precision == "fp32":
        model = model.to(device)
        with torch.no_grad():
            _export_onnx(model, dummy_input, output_path, args.opset, args.dynamic_batch)
        print(f"[export] Saved fp32 ONNX to {output_path}")
        return 0

    if args.precision == "fp16":
        fp32_path = output_path.with_suffix(".fp32.onnx")
        model = model.to(device)
        with torch.no_grad():
            _export_onnx(model, dummy_input, fp32_path, args.opset, args.dynamic_batch)
        try:
            _convert_fp16(fp32_path, output_path, keep_io_types=args.fp16_keep_io)
        except RuntimeError as exc:
            if device.type != "cuda":
                raise
            model = model.half()
            dummy_input = dummy_input.half()
            with torch.no_grad():
                _export_onnx(model, dummy_input, output_path, args.opset, args.dynamic_batch)
            print(f"[export] Saved fp16 ONNX to {output_path} (torch half export)")
        else:
            print(f"[export] Saved fp16 ONNX to {output_path}")
        if not args.keep_intermediate and fp32_path.exists():
            fp32_path.unlink()
        return 0

    if args.precision == "int8":
        fp32_path = output_path.with_suffix(".fp32.onnx")
        model = model.to(device)
        with torch.no_grad():
            _export_onnx(model, dummy_input, fp32_path, args.opset, args.dynamic_batch)
        _quantize_int8(
            fp32_path,
            output_path,
            per_channel=args.int8_per_channel,
            reduce_range=args.int8_reduce_range,
        )
        print(f"[export] Saved int8 ONNX to {output_path} (dynamic quantization)")
        if not args.keep_intermediate and fp32_path.exists():
            fp32_path.unlink()
        return 0

    raise ValueError(f"Unsupported precision '{args.precision}'.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[export] ERROR: {exc}", file=sys.stderr)
        raise
