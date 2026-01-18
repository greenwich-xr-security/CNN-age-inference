#!/usr/bin/env python3
"""Batch export ONNX files for all checkpoints under a directory."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from onnx_tools.export_onnx import (
    _convert_fp16,
    _export_onnx,
    _load_checkpoint,
    _quantize_int8,
    _resolve_img_size,
)
from models.efficientnet_age import EFFICIENTNET_IMG_SIZES, EfficientNetAgeRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively export ONNX models for checkpoints."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to search for .pth checkpoints.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pth",
        help="Glob pattern for checkpoints (default: *.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output root directory (mirrors the input tree).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="EfficientNet variant override (b0-b7, v2_s/m/l). If omitted, inferred from path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="ONNX opset version (default: 15).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override input size (default: model canonical size).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size for tracing (default: 1).",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing ONNX files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of checkpoints to export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List checkpoints without exporting.",
    )
    return parser.parse_args()


def infer_model_key(path: Path) -> str | None:
    haystack = str(path).lower()
    match = re.search(r"(v2_[sml]|b[0-7])", haystack)
    if match and match.group(1) in EFFICIENTNET_IMG_SIZES:
        return match.group(1)
    return None


def resolve_output_path(
    checkpoint: Path,
    root: Path,
    output_root: Path | None,
) -> Path:
    if output_root is None:
        return checkpoint.with_suffix(".onnx")
    try:
        rel = checkpoint.parent.relative_to(root)
    except ValueError:
        rel = Path()
    out_dir = output_root / rel
    return (out_dir / checkpoint.name).with_suffix(".onnx")


def export_checkpoint(
    *,
    checkpoint: Path,
    output_path: Path,
    model_key: str,
    args: argparse.Namespace,
) -> None:
    img_size = _resolve_img_size(model_key, args.img_size)
    device = torch.device(args.device)
    model = EfficientNetAgeRegressor(model_key)
    _load_checkpoint(model, checkpoint)
    model.eval()

    dummy_input = torch.randn(args.batch_size, 3, img_size, img_size, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.precision == "fp32":
        model = model.to(device)
        with torch.no_grad():
            _export_onnx(model, dummy_input, output_path, args.opset, args.dynamic_batch)
        print(f"[export] {checkpoint} -> {output_path}")
        return

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
            print(f"[export] {checkpoint} -> {output_path} (torch half export)")
        else:
            print(f"[export] {checkpoint} -> {output_path}")
        if not args.keep_intermediate and fp32_path.exists():
            fp32_path.unlink()
        return

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
        print(f"[export] {checkpoint} -> {output_path}")
        if not args.keep_intermediate and fp32_path.exists():
            fp32_path.unlink()
        return

    raise ValueError(f"Unsupported precision '{args.precision}'.")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    output_root = Path(args.output_dir).expanduser() if args.output_dir else None
    checkpoints = sorted(root.rglob(args.pattern))
    if args.max_files is not None:
        checkpoints = checkpoints[: args.max_files]

    if not checkpoints:
        print(f"[export] No checkpoints found under {root} (pattern: {args.pattern})")
        return 0

    count = 0
    skipped = 0
    for checkpoint in checkpoints:
        model_key = args.model or infer_model_key(checkpoint)
        if model_key is None:
            print(f"[export] Skipping {checkpoint}: unable to infer model key.")
            skipped += 1
            continue
        output_path = resolve_output_path(checkpoint, root, output_root)
        if output_path.exists() and not args.overwrite:
            print(f"[export] Skipping {checkpoint}: {output_path} already exists.")
            skipped += 1
            continue
        if args.dry_run:
            print(f"[export] Would export {checkpoint} -> {output_path}")
            continue
        try:
            export_checkpoint(
                checkpoint=checkpoint,
                output_path=output_path,
                model_key=model_key,
                args=args,
            )
            count += 1
        except Exception as exc:
            print(f"[export] ERROR: {checkpoint} -> {exc}")
            skipped += 1

    print(f"[export] Completed. Exported: {count}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[export] ERROR: {exc}", file=sys.stderr)
        raise
