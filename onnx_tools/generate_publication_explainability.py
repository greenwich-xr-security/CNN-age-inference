from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.age import AgeDataset
from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import filter_metadata, load_kfold_splits, load_test_split
from metrics import compute_adult_probabilities
from onnx_tools.grad_cam import GradCamRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM explainability outputs for saved publication runs. "
            "Outputs are written under <publication-root>/explainability by default."
        )
    )
    parser.add_argument(
        "--publication-root",
        type=str,
        required=True,
        help="Root folder containing the publication run directories.",
    )
    parser.add_argument(
        "--run-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of run directory names to process.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Which saved prediction split to explain (default: test).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of fold indices to process. Defaults to all folds.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Destination root. Defaults to <publication-root>/explainability.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Local dataset root. Defaults to the repository's configured dataset root.",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=2,
        help="How many examples to export per selection category and fold (default: 2).",
    )
    parser.add_argument(
        "--target-mode",
        choices=["adult_prob", "age_mean"],
        default="adult_prob",
        help="Attribution target (default: adult_prob).",
    )
    parser.add_argument(
        "--age-threshold",
        type=float,
        default=18.0,
        help="Adult/minor threshold used for adult_prob attribution (default: 18).",
    )
    parser.add_argument(
        "--method",
        choices=["gradcam", "gradcam++"],
        default="gradcam",
        help="Grad-CAM variant to use (default: gradcam).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Torch device used for attribution (default: cuda).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional target layers. Pass multiple layers to export per-layer CAMs "
            "and comparison panels."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha (default: 0.45).",
    )
    return parser.parse_args()


def _parse_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    number = int(float(text))
    if number <= 0:
        return None
    return number


def _normalise_bbox(raw_bbox) -> tuple[int, int, int, int] | None:
    if raw_bbox is None or (isinstance(raw_bbox, float) and np.isnan(raw_bbox)):
        return None
    if isinstance(raw_bbox, str):
        text = (
            raw_bbox.strip()
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace(";", ",")
        )
        if not text:
            return None
        parts = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(raw_bbox, (list, tuple)):
        parts = list(raw_bbox)
    else:
        return None
    if len(parts) != 4:
        return None
    return tuple(int(round(float(part))) for part in parts)


def _resolve_mask_path(record: pd.Series, *, use_masks: bool) -> Path | None:
    if not use_masks:
        return None
    image_path = Path(record["image_path"])
    if any("handrgbd" in str(parent).lower() for parent in image_path.parents):
        return None
    explicit = record.get("mask_path")
    if explicit is not None and str(explicit).strip():
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
    resolved = AgeDataset._resolve_mask_path(image_path)
    if resolved is not None and resolved.is_file():
        return resolved
    return None


def _discover_run_dirs(publication_root: Path, run_names: list[str] | None) -> list[Path]:
    if run_names:
        run_dirs = [publication_root / name for name in run_names]
    else:
        run_dirs = [
            entry
            for entry in publication_root.iterdir()
            if entry.is_dir() and (entry / "fold_0" / "config.txt").is_file()
        ]
    run_dirs = [path for path in run_dirs if path.is_dir()]
    run_dirs.sort(key=lambda path: path.name.lower())
    return run_dirs


def _load_filtered_metadata(config: dict[str, str], data_root: Path) -> pd.DataFrame:
    include_handrgbd = _as_bool(config.get("include_handrgbd"), default=True)
    include_hagrid = _as_bool(config.get("include_hagrid"), default=False)
    include_prolific = _as_bool(config.get("include_prolific"), default=False)
    include_primary = _as_bool(config.get("include_primary"), default=False)
    include_archive = _as_bool(config.get("include_archive"), default=False)
    max_samples_per_user = _as_optional_int(config.get("max_samples_per_user"))
    max_samples_per_age_bin = _as_optional_int(config.get("max_samples_per_age_bin"))
    metadata = load_combined_metadata(
        root=data_root,
        include_handrgbd=include_handrgbd,
        include_hagrid=include_hagrid,
        include_prolific=include_prolific,
        include_primary=include_primary,
        include_archive=include_archive,
    )
    filtered = filter_metadata(
        metadata,
        max_samples_per_user=max_samples_per_user,
        max_samples_per_age_bin=max_samples_per_age_bin,
    )
    return filtered.reset_index(drop=True)


def _rebuild_records_for_split(
    run_dir: Path,
    metadata: pd.DataFrame,
    *,
    split: str,
    fold_index: int,
) -> pd.DataFrame:
    if split == "test":
        test_data = load_test_split(run_dir / "test_users.json")
        split_ids = {str(uid) for uid in test_data["test_user_ids"]}
    else:
        fold_data = load_kfold_splits(run_dir / "folds_k5.json")
        folds = fold_data.get("folds", [])
        if fold_index < 0 or fold_index >= len(folds):
            raise IndexError(f"Fold {fold_index} out of range for {run_dir.name}.")
        split_ids = {str(uid) for uid in folds[fold_index]}
    return metadata[metadata["user_id"].astype(str).isin(split_ids)].reset_index(drop=True)


def _load_prediction_frame(pred_path: Path) -> pd.DataFrame:
    data = np.load(pred_path)
    return pd.DataFrame(
        {
            "target_age_saved": data["targets"].astype(float),
            "pred_mean_saved": data["pred_mean"].astype(float),
            "pred_log_var_saved": data["pred_log_var"].astype(float),
            "user_id_saved": data["user_ids"].astype(str),
            "skin_color_saved": data["skin_color"].astype(str)
            if "skin_color" in data.files
            else np.repeat("unlabeled", len(data["targets"])),
        }
    )


def _align_records_with_predictions(records: pd.DataFrame, pred_path: Path) -> pd.DataFrame:
    preds = _load_prediction_frame(pred_path)
    if len(records) != len(preds):
        raise RuntimeError(
            f"Record/prediction length mismatch for {pred_path}: "
            f"{len(records)} records vs {len(preds)} predictions."
        )
    user_ids = records["user_id"].astype(str).to_numpy()
    pred_user_ids = preds["user_id_saved"].astype(str).to_numpy()
    if not np.array_equal(user_ids, pred_user_ids):
        mismatch_idx = int(np.flatnonzero(user_ids != pred_user_ids)[0])
        raise RuntimeError(
            f"User ordering mismatch at row {mismatch_idx} for {pred_path}: "
            f"{user_ids[mismatch_idx]} != {pred_user_ids[mismatch_idx]}."
        )

    out = records.copy()
    out["target_age_saved"] = preds["target_age_saved"].to_numpy()
    out["pred_mean_saved"] = preds["pred_mean_saved"].to_numpy()
    out["pred_log_var_saved"] = preds["pred_log_var_saved"].to_numpy()
    out["skin_color_saved"] = preds["skin_color_saved"].to_numpy()
    out["adult_prob_saved"] = compute_adult_probabilities(
        out["pred_mean_saved"].to_numpy(),
        out["pred_log_var_saved"].to_numpy(),
    )
    out["abs_error_saved"] = np.abs(out["pred_mean_saved"] - out["target_age_saved"])
    return out.reset_index(drop=True)


def _select_examples(records: pd.DataFrame, per_category: int) -> list[dict[str, object]]:
    categories: list[tuple[str, pd.DataFrame]] = [
        (
            "minor_low_adult_prob",
            records[records["target_age_saved"] < 18.0].sort_values(
                ["adult_prob_saved", "abs_error_saved"], ascending=[True, True]
            ),
        ),
        (
            "minor_high_adult_prob",
            records[records["target_age_saved"] < 18.0].sort_values(
                ["adult_prob_saved", "abs_error_saved"], ascending=[False, False]
            ),
        ),
        (
            "adult_high_adult_prob",
            records[records["target_age_saved"] >= 18.0].sort_values(
                ["adult_prob_saved", "abs_error_saved"], ascending=[False, True]
            ),
        ),
        (
            "adult_low_adult_prob",
            records[records["target_age_saved"] >= 18.0].sort_values(
                ["adult_prob_saved", "abs_error_saved"], ascending=[True, False]
            ),
        ),
        (
            "smallest_abs_error",
            records.sort_values(["abs_error_saved", "adult_prob_saved"], ascending=[True, False]),
        ),
        (
            "largest_abs_error",
            records.sort_values(["abs_error_saved", "adult_prob_saved"], ascending=[False, False]),
        ),
        (
            "boundary_minor_low_adult_prob",
            records[
                (records["target_age_saved"] >= 16.0) & (records["target_age_saved"] < 18.0)
            ].sort_values(["adult_prob_saved", "abs_error_saved"], ascending=[True, True]),
        ),
        (
            "boundary_minor_high_adult_prob",
            records[
                (records["target_age_saved"] >= 16.0) & (records["target_age_saved"] < 18.0)
            ].sort_values(["adult_prob_saved", "abs_error_saved"], ascending=[False, False]),
        ),
        (
            "boundary_adult_high_adult_prob",
            records[
                (records["target_age_saved"] >= 18.0) & (records["target_age_saved"] <= 21.0)
            ].sort_values(["adult_prob_saved", "abs_error_saved"], ascending=[False, True]),
        ),
        (
            "boundary_adult_low_adult_prob",
            records[
                (records["target_age_saved"] >= 18.0) & (records["target_age_saved"] <= 21.0)
            ].sort_values(["adult_prob_saved", "abs_error_saved"], ascending=[True, False]),
        ),
    ]

    chosen: list[dict[str, object]] = []
    used_indices: set[int] = set()
    for category_name, frame in categories:
        if frame.empty:
            continue
        rank = 0
        for row_index, row in frame.iterrows():
            if row_index in used_indices:
                continue
            used_indices.add(int(row_index))
            chosen.append(
                {
                    "category": category_name,
                    "rank_in_category": rank,
                    "row_index": int(row_index),
                }
            )
            rank += 1
            if rank >= per_category:
                break
    return chosen


def _sanitize_token(text: object) -> str:
    raw = str(text).strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "item"


def _build_output_stem(category: str, rank: int, row: pd.Series) -> str:
    age = f"{float(row['target_age_saved']):.1f}".replace(".", "p")
    pred = f"{float(row['pred_mean_saved']):.1f}".replace(".", "p")
    uid = _sanitize_token(row["user_id"])
    image_name = _sanitize_token(Path(row["image_path"]).stem)
    return (
        f"{category}__r{rank:02d}__uid_{uid}__age_{age}__pred_{pred}__img_{image_name}"
    )


def _make_layer_label(layer_name: str) -> str:
    parts = [part for part in str(layer_name).split(".") if part]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    if parts:
        return parts[-1]
    return "layer"


def _build_panel(items: list[tuple[str, Image.Image]]) -> Image.Image:
    if not items:
        raise ValueError("panel items must not be empty")

    margin = 16
    gap = 16
    caption_gap = 8
    background = (8, 10, 28)
    text_color = (232, 236, 247)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    caption_height = max(draw_probe.textbbox((0, 0), label)[3] for label, _ in items) + 2

    max_height = max(image.height for _, image in items)
    width = margin * 2 + sum(image.width for _, image in items) + gap * (len(items) - 1)
    height = margin * 2 + max_height + caption_gap + caption_height
    panel = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(panel)

    x = margin
    image_top = margin
    caption_top = image_top + max_height + caption_gap
    for label, image in items:
        y = image_top + max((max_height - image.height) // 2, 0)
        panel.paste(image, (x, y))
        text_bbox = draw.textbbox((0, 0), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + max((image.width - text_width) // 2, 0)
        draw.text((text_x, caption_top), label, fill=text_color)
        x += image.width + gap
    return panel


def _export_run_fold(
    *,
    run_dir: Path,
    fold_dir: Path,
    records: pd.DataFrame,
    config: dict[str, str],
    output_root: Path,
    args: argparse.Namespace,
) -> Path:
    model_name = config["model"]
    checkpoint_paths = list(fold_dir.glob("*.pth"))
    if len(checkpoint_paths) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint in {fold_dir}, found {len(checkpoint_paths)}.")
    checkpoint_path = checkpoint_paths[0]
    embed_dim = int(config.get("resolved_embed_dim", config.get("embed_dim", "0")))
    img_size = int(config.get("resolved_img_size", config.get("img_size", "224")))
    use_masks = _as_bool(config.get("resolved_use_masks", config.get("use_masks")), default=False)

    split_output = output_root / run_dir.name / fold_dir.name / args.split
    input_dir = split_output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    explicit_layers = list(args.layers) if args.layers else [None]
    multi_layer_mode = args.layers is not None
    panel_dir = split_output / "panels" if len(explicit_layers) > 1 else None
    if panel_dir is not None:
        panel_dir.mkdir(parents=True, exist_ok=True)

    runners: list[dict[str, object]] = []
    for requested_layer in explicit_layers:
        runner = GradCamRunner(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=args.device,
            target_layer=requested_layer,
            method=args.method,
            embed_dim=embed_dim,
            target_mode=args.target_mode,
            age_threshold=args.age_threshold,
        )
        layer_warning = ""
        if runner.layer_name.startswith("backbone.patch_embed"):
            layer_warning = (
                "Auto-selected layer is the patch-embedding convolution. "
                "This is a weak fallback for transformer attribution and should be treated cautiously."
            )
            print(f"[warn] {run_dir.name}/{fold_dir.name}: {layer_warning}", file=sys.stderr)

        if multi_layer_mode:
            layer_token = _sanitize_token(requested_layer or runner.layer_name)
            layer_root = split_output / "layers" / layer_token
            overlay_dir = layer_root / "overlays"
            heatmap_dir = layer_root / "heatmaps"
            grayscale_dir = layer_root / "cam_gray"
            array_dir = layer_root / "cam_arrays"
        else:
            overlay_dir = split_output / "overlays"
            heatmap_dir = split_output / "heatmaps"
            grayscale_dir = split_output / "cam_gray"
            array_dir = split_output / "cam_arrays"

        for folder in (overlay_dir, heatmap_dir, grayscale_dir, array_dir):
            folder.mkdir(parents=True, exist_ok=True)

        runners.append(
            {
                "requested_layer": requested_layer or "",
                "label": _make_layer_label(runner.layer_name),
                "runner": runner,
                "layer_warning": layer_warning,
                "overlay_dir": overlay_dir,
                "heatmap_dir": heatmap_dir,
                "grayscale_dir": grayscale_dir,
                "array_dir": array_dir,
            }
        )

    manifest_path = split_output / "manifest.csv"
    selected = _select_examples(records, args.examples_per_category)
    fieldnames = [
        "run",
        "fold",
        "split",
        "category",
        "rank_in_category",
        "row_index",
        "user_id",
        "source",
        "image_path",
        "skin_color",
        "target_age_saved",
        "pred_mean_saved",
        "pred_log_var_saved",
        "adult_prob_saved",
        "abs_error_saved",
        "pred_mean_cam",
        "pred_log_var_cam",
        "adult_prob_cam",
        "target_mode",
        "target_score",
        "model",
        "checkpoint_path",
        "requested_layer",
        "layer_name",
        "layer_warning",
        "bbox",
        "overlay_path",
        "heatmap_path",
        "cam_gray_path",
        "cam_array_path",
        "input_path",
        "comparison_panel_path",
    ]

    try:
        with manifest_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()

            for selection in selected:
                row_index = int(selection["row_index"])
                row = records.iloc[row_index]
                mask_path = _resolve_mask_path(row, use_masks=use_masks)
                bbox = _normalise_bbox(row.get("bbox"))
                stem = _build_output_stem(
                    str(selection["category"]),
                    int(selection["rank_in_category"]),
                    row,
                )
                input_path = input_dir / f"{stem}.png"
                panel_items: list[tuple[str, Image.Image]] = []
                layer_rows: list[dict[str, object]] = []
                input_saved = False

                for runner_info in runners:
                    runner = runner_info["runner"]
                    result = runner.compute(
                        Path(row["image_path"]),
                        mask_path=mask_path,
                        bbox=bbox,
                        img_size=img_size,
                        alpha=args.alpha,
                    )
                    if not input_saved:
                        result["preview"].save(input_path)
                        panel_items.append(("input", result["preview"]))
                        input_saved = True

                    overlay_dir = runner_info["overlay_dir"]
                    heatmap_dir = runner_info["heatmap_dir"]
                    grayscale_dir = runner_info["grayscale_dir"]
                    array_dir = runner_info["array_dir"]
                    layer_label = str(runner_info["label"])

                    overlay_path = Path(overlay_dir) / f"{stem}.png"
                    heatmap_path = Path(heatmap_dir) / f"{stem}.png"
                    cam_gray_path = Path(grayscale_dir) / f"{stem}.png"
                    cam_array_path = Path(array_dir) / f"{stem}.npy"

                    result["overlay"].save(overlay_path)
                    result["heatmap"].save(heatmap_path)
                    result["cam_gray"].save(cam_gray_path)
                    np.save(cam_array_path, result["cam_array"])
                    panel_items.append((layer_label, result["overlay"]))

                    layer_rows.append(
                        {
                            "run": run_dir.name,
                            "fold": fold_dir.name,
                            "split": args.split,
                            "category": selection["category"],
                            "rank_in_category": selection["rank_in_category"],
                            "row_index": row_index,
                            "user_id": str(row["user_id"]),
                            "source": str(row.get("source", "")),
                            "image_path": str(Path(row["image_path"]).resolve()),
                            "skin_color": str(
                                row.get("skin_color_saved", row.get("skin_color", "unlabeled"))
                            ),
                            "target_age_saved": float(row["target_age_saved"]),
                            "pred_mean_saved": float(row["pred_mean_saved"]),
                            "pred_log_var_saved": float(row["pred_log_var_saved"]),
                            "adult_prob_saved": float(row["adult_prob_saved"]),
                            "abs_error_saved": float(row["abs_error_saved"]),
                            "pred_mean_cam": float(result["pred_mean"]),
                            "pred_log_var_cam": float(result["pred_log_var"]),
                            "adult_prob_cam": float(result["adult_prob"]),
                            "target_mode": str(result["target_mode"]),
                            "target_score": float(result["target_score"]),
                            "model": model_name,
                            "checkpoint_path": str(checkpoint_path.resolve()),
                            "requested_layer": str(runner_info["requested_layer"]),
                            "layer_name": str(result["layer_name"]),
                            "layer_warning": str(runner_info["layer_warning"]),
                            "bbox": "" if bbox is None else ",".join(str(v) for v in bbox),
                            "overlay_path": str(overlay_path.resolve()),
                            "heatmap_path": str(heatmap_path.resolve()),
                            "cam_gray_path": str(cam_gray_path.resolve()),
                            "cam_array_path": str(cam_array_path.resolve()),
                            "input_path": str(input_path.resolve()),
                        }
                    )

                panel_path = ""
                if panel_dir is not None:
                    panel_path = str((panel_dir / f"{stem}.png").resolve())
                    _build_panel(panel_items).save(panel_path)

                for layer_row in layer_rows:
                    layer_row["comparison_panel_path"] = panel_path
                    writer.writerow(layer_row)
    finally:
        for runner_info in runners:
            runner_info["runner"].close()

    return manifest_path


def main() -> int:
    args = _parse_args()
    publication_root = Path(args.publication_root).expanduser()
    if not publication_root.is_dir():
        raise FileNotFoundError(f"Publication root not found: {publication_root}")

    if args.data_root:
        set_dataset_root(args.data_root)
    data_root = get_dataset_root()
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else publication_root / "explainability"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(publication_root, args.run_names)
    if not run_dirs:
        raise FileNotFoundError(f"No publication run directories found under {publication_root}")

    manifests: list[Path] = []
    for run_dir in run_dirs:
        config_path = run_dir / "fold_0" / "config.txt"
        config = _parse_config(config_path)
        metadata = _load_filtered_metadata(config, data_root)
        fold_data = load_kfold_splits(run_dir / "folds_k5.json")
        folds = fold_data.get("folds", [])
        selected_folds = args.folds if args.folds is not None else list(range(len(folds)))

        print(f"[explain] run={run_dir.name} split={args.split} folds={selected_folds}")
        for fold_index in selected_folds:
            fold_dir = run_dir / f"fold_{fold_index}"
            pred_name = f"{args.split}_predictions_raw_ddp.npz"
            pred_path = fold_dir / pred_name
            if not pred_path.is_file():
                print(f"[warn] Missing predictions file: {pred_path}", file=sys.stderr)
                continue
            records = _rebuild_records_for_split(
                run_dir,
                metadata,
                split=args.split,
                fold_index=fold_index,
            )
            aligned = _align_records_with_predictions(records, pred_path)
            manifest_path = _export_run_fold(
                run_dir=run_dir,
                fold_dir=fold_dir,
                records=aligned,
                config=config,
                output_root=output_root,
                args=args,
            )
            manifests.append(manifest_path)
            print(f"[explain] saved {manifest_path}")

    summary_path = output_root / f"manifest_index_{args.split}.txt"
    existing_manifests = sorted(
        {
            manifest.resolve()
            for manifest in output_root.rglob("manifest.csv")
            if manifest.parent.name == args.split
        },
        key=lambda path: str(path).lower(),
    )
    with summary_path.open("w", encoding="utf-8") as fp:
        for manifest in existing_manifests:
            fp.write(str(manifest) + "\n")
    print(
        f"[explain] indexed {len(existing_manifests)} manifests at {summary_path} "
        f"({len(manifests)} updated in this run)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
