import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def _read_kv_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not path.is_file():
        return cfg
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def _maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_history(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "epochs_seen": 0,
        "last_epoch": None,
        "last_metrics": {},
        "best_epoch": None,
        "best_by": None,
        "best_metrics": {},
    }
    if not path.is_file():
        return result

    entries: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("Epoch "):
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        try:
            epoch = int(parts[0].split()[1])
        except (IndexError, ValueError):
            continue
        rec: dict[str, float] = {"epoch": float(epoch)}
        for item in parts[1:]:
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            val = _maybe_float(v.strip())
            if val is not None:
                rec[k.strip()] = val
        entries.append(rec)

    if not entries:
        return result

    result["epochs_seen"] = len(entries)
    last = entries[-1]
    result["last_epoch"] = int(last["epoch"])
    result["last_metrics"] = {k: v for k, v in last.items() if k != "epoch"}

    best_key = None
    for candidate in ("val_loss", "val_mae", "val_rmse"):
        if any(candidate in e for e in entries):
            best_key = candidate
            break
    if best_key is not None:
        best_entry = min(entries, key=lambda e: e.get(best_key, float("inf")))
        result["best_by"] = best_key
        result["best_epoch"] = int(best_entry["epoch"])
        result["best_metrics"] = {k: v for k, v in best_entry.items() if k != "epoch"}
    return result


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_bool(value: str | int | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def generate_docs_artifacts(
    *,
    onnx_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    history_path: Path | None = None,
    model_name: str | None = None,
    checkpoint_path: Path | None = None,
    precision: str | None = None,
    opset: int | None = None,
    img_size: int | None = None,
    batch_size: int | None = None,
    embed_dim: int | None = None,
    dynamic_batch: bool | None = None,
) -> tuple[Path, Path]:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _read_kv_config(config_path) if config_path else {}
    history = _parse_history(history_path) if history_path else _parse_history(Path(""))

    model_id = onnx_path.stem
    created_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
    onnx_sha256 = _sha256_file(onnx_path)
    onnx_size_bytes = onnx_path.stat().st_size

    resolved_model = cfg.get("resolved_model") or cfg.get("model") or model_name or "unknown"
    resolved_img_size = cfg.get("resolved_img_size") or cfg.get("img_size") or img_size
    resolved_batch_size = cfg.get("resolved_batch_size") or cfg.get("batch_size") or batch_size
    resolved_embed_dim = cfg.get("resolved_embed_dim") or cfg.get("embed_dim") or embed_dim
    run_started_at = cfg.get("run_started_at", "unknown")
    requires_landmarks = _to_bool(cfg.get("requires_hand_landmarks"), default=True)
    requires_masking = _to_bool(cfg.get("requires_hand_masking"), default=True)

    deployment = {
        "schema_version": "1.0",
        "model_id": model_id,
        "created_at": created_at,
        "artifacts": {
            "onnx_path": str(onnx_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "onnx_sha256": onnx_sha256,
            "onnx_size_bytes": onnx_size_bytes,
            "precision": precision,
            "opset": opset,
        },
        "model": {
            "name": str(resolved_model),
            "embed_dim": int(resolved_embed_dim) if str(resolved_embed_dim).isdigit() else resolved_embed_dim,
            "dynamic_batch": bool(dynamic_batch) if dynamic_batch is not None else None,
        },
        "io_contract": {
            "inputs": [
                {
                    "name": "input",
                    "shape": [
                        "batch",
                        3,
                        int(resolved_img_size) if str(resolved_img_size).isdigit() else resolved_img_size,
                        int(resolved_img_size) if str(resolved_img_size).isdigit() else resolved_img_size,
                    ],
                    "dtype": "float32",
                }
            ],
            "outputs": ["mean", "log_var"] + (["embedding"] if str(resolved_embed_dim) not in ("0", "", "None") else []),
        },
        "serving": {
            "task": "age_regression",
            "default": False,
            "enabled": True,
        },
        "preprocessing": {
            "requires_hand_landmarks": requires_landmarks,
            "requires_hand_masking": requires_masking,
            "crop_strategy": "center_square",
            "resize": int(resolved_img_size) if str(resolved_img_size).isdigit() else resolved_img_size,
            "color_space": "rgb",
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "postprocessing": {
            "mean_output_name": "mean",
            "log_var_output_name": "log_var",
            "std_formula": "exp(0.5 * log_var)",
        },
        "runtime_constraints": {
            "min_onnxruntime": "1.16.0",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
        "training": {
            "run_started_at": run_started_at,
            "config_path": str(config_path) if config_path else None,
            "history_path": str(history_path) if history_path else None,
            "batch_size": resolved_batch_size,
            "img_size": resolved_img_size,
            "lr": cfg.get("resolved_lr") or cfg.get("lr"),
            "weight_decay": cfg.get("resolved_weight_decay") or cfg.get("weight_decay"),
            "epochs": cfg.get("epochs"),
            "patience": cfg.get("resolved_patience") or cfg.get("patience"),
            "seed": cfg.get("seed"),
            "user_group_size": cfg.get("resolved_user_group_size") or cfg.get("user_group_size"),
        },
        "dataset": {
            "data_root": cfg.get("data_root"),
            "total_samples": cfg.get("dataset_total_samples"),
            "total_users": cfg.get("dataset_total_users"),
            "sources": {
                k.replace("dataset_source_", ""): v
                for k, v in cfg.items()
                if k.startswith("dataset_source_")
            },
        },
        "evaluation": {
            "epochs_seen": history.get("epochs_seen"),
            "best_by": history.get("best_by"),
            "best_epoch": history.get("best_epoch"),
            "best_metrics": history.get("best_metrics"),
            "last_epoch": history.get("last_epoch"),
            "last_metrics": history.get("last_metrics"),
        },
        "limitations": [
            "Predictions may degrade under domain shift (camera, lighting, pose, demographics).",
            "Model card fields are partially auto-generated and should be reviewed before deployment.",
        ],
    }

    deployment_path = output_dir / "deployment_sheet.json"
    deployment_path.write_text(json.dumps(deployment, indent=2), encoding="utf-8")

    best_metrics = history.get("best_metrics") or {}
    last_metrics = history.get("last_metrics") or {}
    model_card = f"""# Model Card

## Model Identification
- Model ID: `{model_id}`
- Created at: `{created_at}`
- ONNX path: `{onnx_path}`
- ONNX SHA256: `{onnx_sha256}`
- ONNX size (bytes): `{onnx_size_bytes}`
- Checkpoint path: `{checkpoint_path if checkpoint_path else "n/a"}`

## Intended Use
- Primary use: age estimation from hand images.
- Out of scope: identity verification, medical diagnosis, legal age adjudication.

## Input / Output
- Input tensor: `input` with shape `[batch, 3, {resolved_img_size}, {resolved_img_size}]`
- Output tensors: `mean`, `log_var`{" , `embedding`" if str(resolved_embed_dim) not in ("0", "", "None") else ""}
- Dynamic batch: `{dynamic_batch}`
- Precision: `{precision}`
- Opset: `{opset}`

## Training Configuration
- Model: `{resolved_model}`
- Embed dim: `{resolved_embed_dim}`
- Batch size: `{resolved_batch_size}`
- Learning rate: `{cfg.get("resolved_lr") or cfg.get("lr")}`
- Weight decay: `{cfg.get("resolved_weight_decay") or cfg.get("weight_decay")}`
- Epochs: `{cfg.get("epochs")}`
- Patience: `{cfg.get("resolved_patience") or cfg.get("patience")}`
- Seed: `{cfg.get("seed")}`
- User group size: `{cfg.get("resolved_user_group_size") or cfg.get("user_group_size")}`
- Run started at: `{run_started_at}`

## Dataset Summary
- Data root: `{cfg.get("data_root")}`
- Total samples: `{cfg.get("dataset_total_samples")}`
- Total users: `{cfg.get("dataset_total_users")}`

## Evaluation Summary
- Epochs seen: `{history.get("epochs_seen")}`
- Best checkpoint criterion: `{history.get("best_by")}`
- Best epoch: `{history.get("best_epoch")}`
- Best metrics: `{json.dumps(best_metrics)}`
- Last epoch: `{history.get("last_epoch")}`
- Last metrics: `{json.dumps(last_metrics)}`

## Limitations
- Performance can vary for under-represented groups and unseen capture conditions.
- Use human review and uncertainty-aware thresholds for high-stakes decisions.
"""
    model_card_path = output_dir / "model_card.md"
    model_card_path.write_text(model_card, encoding="utf-8")

    return model_card_path, deployment_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model_card.md and deployment_sheet.json for an ONNX artifact.")
    parser.add_argument("--onnx-path", required=True, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--config-path", default=None, type=str)
    parser.add_argument("--history-path", default=None, type=str)
    parser.add_argument("--model-name", default=None, type=str)
    parser.add_argument("--checkpoint-path", default=None, type=str)
    parser.add_argument("--precision", default=None, type=str)
    parser.add_argument("--opset", default=None, type=int)
    parser.add_argument("--img-size", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--embed-dim", default=None, type=int)
    parser.add_argument("--dynamic-batch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else onnx_path.parent
    )
    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else None
    history_path = Path(args.history_path).expanduser().resolve() if args.history_path else None
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None

    model_card_path, deployment_path = generate_docs_artifacts(
        onnx_path=onnx_path,
        output_dir=output_dir,
        config_path=config_path,
        history_path=history_path,
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        precision=args.precision,
        opset=args.opset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        dynamic_batch=args.dynamic_batch,
    )
    print(f"[docs] Saved model card: {model_card_path}")
    print(f"[docs] Saved deployment sheet: {deployment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
