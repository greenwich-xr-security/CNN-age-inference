import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import resolve_model_builder
from onnx_tools.generate_model_docs import generate_docs_artifacts

# Example commands:
#   python onnx_tools/export_onnx.py --model b2 --checkpoint runs\b2_bs32_seed204_nll0.5_mse0.25_mae0.25_ug2\b2_age_regressor_ddp.pth --output runs\b2_bs32_seed204_nll0.5_mse0.25_mae0.25_ug2\b2_age_regressor_ddp.onnx
#   python onnx_tools/export_onnx.py --model b2 --checkpoint runs\b2_bs32_seed204_nll0.5_mse0.25_mae0.25_ug2\b2_age_regressor_ddp.pth --output runs\b2_bs32_seed204_nll0.5_mse0.25_mae0.25_ug2\b2_age_regressor_ddp_fp16.onnx --precision fp16
#   python onnx_tools/export_onnx.py --model b2 --checkpoint runs\b2_bs32_seed204_nll0.5_mse0.25_mae0.25_ug2\b2_age_regressor_ddp.pth --output runs\b2_bs32_seed204_nll0.5_mse0.25_mae0.25_ug2\b2_age_regressor_ddp_int8.onnx --precision int8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export age regressor models (EfficientNet/ConvNeXt/ViT) to ONNX."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="b0",
        help=(
            "Backbone to export. EfficientNet: b0-b7, v2_s, v2_m, v2_l. "
            "ConvNeXt: convnext_{tiny,small,base,large,xlarge}. "
            "ViT: vit_tiny_384. Aliases: cnt,cns,cnb,cnl,cnx,vtt,age_vit."
        ),
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="Embedding head dimension used during training (default: 0).",
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
        help="Override input size. Defaults to the canonical size for the selected model.",
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
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate model_card.md and deployment_sheet.json next to the exported ONNX.",
    )
    parser.add_argument(
        "--docs-output-dir",
        type=str,
        default=None,
        help="Directory where model_card.md and deployment_sheet.json are written (default: ONNX parent).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to run config.txt used for docs and parameter inference (default: <checkpoint parent>/config.txt, then <ONNX parent>/config.txt).",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default=None,
        help="Path to training history log used to populate docs (default: auto-detect in ONNX folder).",
    )
    return parser.parse_args()


def _resolve_img_size(default_size: int, override: int | None) -> int:
    if override is not None:
        return override
    return default_size


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


def _infer_output_names(model: torch.nn.Module, dummy_input: torch.Tensor) -> list[str]:
    with torch.no_grad():
        outputs = model(dummy_input)
    if isinstance(outputs, (tuple, list)):
        count = len(outputs)
    else:
        count = 1
    if count == 2:
        return ["mean", "log_var"]
    if count == 3:
        return ["mean", "log_var", "embedding"]
    return [f"output_{idx}" for idx in range(count)]


def _export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
) -> None:
    input_names = ["input"]
    output_names = _infer_output_names(model, dummy_input)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}
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


def _resolve_default_history_path(folder: Path) -> Path | None:
    candidates = [
        folder / "history_distributed.log",
        folder / "history.log",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _parse_run_config(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _resolve_default_config_path(checkpoint_path: Path | None, output_path: Path) -> Path:
    if checkpoint_path is not None:
        candidate = checkpoint_path.parent / "config.txt"
        if candidate.is_file():
            return candidate
    return output_path.parent / "config.txt"


def _arg_supplied(flag: str) -> bool:
    return flag in sys.argv


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _generate_docs(
    args: argparse.Namespace,
    output_path: Path,
    img_size: int,
    *,
    effective_model: str,
    effective_embed_dim: int,
) -> None:
    docs_output_dir = (
        Path(args.docs_output_dir).expanduser().resolve()
        if args.docs_output_dir
        else output_path.parent
    )
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else (output_path.parent / "config.txt")
    )
    history_path = (
        Path(args.history_path).expanduser().resolve()
        if args.history_path
        else _resolve_default_history_path(output_path.parent)
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None

    model_card_path, deployment_path = generate_docs_artifacts(
        onnx_path=output_path.resolve(),
        output_dir=docs_output_dir,
        config_path=config_path if config_path.is_file() else None,
        history_path=history_path if history_path and history_path.is_file() else None,
        model_name=effective_model,
        checkpoint_path=checkpoint_path,
        precision=args.precision,
        opset=args.opset,
        img_size=img_size,
        batch_size=args.batch_size,
        embed_dim=effective_embed_dim,
        dynamic_batch=args.dynamic_batch,
    )
    print(f"[docs] Saved model card: {model_card_path}")
    print(f"[docs] Saved deployment sheet: {deployment_path}")


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else None
    config_path = (
        Path(args.config_path).expanduser()
        if args.config_path
        else _resolve_default_config_path(checkpoint_path, output_path)
    )
    run_cfg = _parse_run_config(config_path if config_path.is_file() else None)

    model_key = args.model.lower()
    if not _arg_supplied("--model"):
        cfg_model = run_cfg.get("model")
        if cfg_model:
            model_key = cfg_model.lower()

    embed_dim = args.embed_dim
    if not _arg_supplied("--embed-dim"):
        cfg_embed = _parse_int(run_cfg.get("resolved_embed_dim")) or _parse_int(run_cfg.get("embed_dim"))
        if cfg_embed is not None:
            embed_dim = cfg_embed

    img_override = args.img_size
    if not _arg_supplied("--img-size") and img_override is None:
        cfg_img = _parse_int(run_cfg.get("resolved_img_size")) or _parse_int(run_cfg.get("img_size"))
        if cfg_img is not None:
            img_override = cfg_img

    model_builder, default_img_size, _model_desc, resolved_model_key = resolve_model_builder(
        model_key, embed_dim=embed_dim
    )
    img_size = _resolve_img_size(default_img_size, img_override)

    device = torch.device(args.device)
    model = model_builder()
    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path)
    model.eval()
    if run_cfg:
        print(
            f"[export] Resolved from config {config_path}: "
            f"model={resolved_model_key}, embed_dim={embed_dim}, img_size={img_size}"
        )

    dummy_input = torch.randn(
        args.batch_size, 3, img_size, img_size, device=device
    )

    if args.precision == "fp32":
        model = model.to(device)
        with torch.no_grad():
            _export_onnx(model, dummy_input, output_path, args.opset, args.dynamic_batch)
        print(f"[export] Saved fp32 ONNX to {output_path}")
        if args.generate_docs:
            _generate_docs(
                args,
                output_path,
                img_size,
                effective_model=resolved_model_key,
                effective_embed_dim=embed_dim,
            )
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
        if args.generate_docs:
            _generate_docs(
                args,
                output_path,
                img_size,
                effective_model=resolved_model_key,
                effective_embed_dim=embed_dim,
            )
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
        if args.generate_docs:
            _generate_docs(
                args,
                output_path,
                img_size,
                effective_model=resolved_model_key,
                effective_embed_dim=embed_dim,
            )
        return 0

    raise ValueError(f"Unsupported precision '{args.precision}'.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[export] ERROR: {exc}", file=sys.stderr)
        raise
