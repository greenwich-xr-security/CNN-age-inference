import argparse
import io
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import load_kfold_splits
from models import CONVNEXT_IMG_SIZES, EFFICIENTNET_IMG_SIZES, MODEL_ALIASES
from onnx_tools.exposure_check import compute_exposure_metrics
from onnx_tools.run_onnx import IMAGENET_MEAN, IMAGENET_STD, _infer_img_size


MASK_THRESHOLD = 0
DATASET_OPTIONS = ("all", "primary", "archive", "handrgbd")
SPLIT_OPTIONS = ("all", "train", "val")
VIEW_OPTIONS = ("Gallery", "Full subset scatter")
HANDRGBD_RGB_ROOT_OPTIONS = (
    "auto",
    "rgb_jpg",
    "rotated_CLAHE_rgb_jpg",
    "rgb_xyz_jpg",
    "rotated_rgb_jpg",
)
HANDRGBD_MASK_ROOT_OPTIONS = ("auto", "rgb_mask", "rotated_rgb_mask", "xyz_mask")


def _scan_onnx_models(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    models = []
    for path in root.rglob("*.onnx"):
        if not path.is_file():
            continue
        if path.name.endswith(".fp32.onnx"):
            continue
        models.append(path)
    return sorted(models, key=lambda p: str(p).lower())


def _infer_fold_index(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("fold_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                continue
    return None


def _find_fold_file(path: Path) -> Path | None:
    for parent in [path] + list(path.parents):
        matches = sorted(parent.glob("folds_k*.json"))
        if matches:
            return matches[0]
    return None


def _infer_model_key_from_path(path: Path) -> str | None:
    parts = [path.stem] + [p for p in path.parts]
    tokens = re.split(r"[^a-z0-9]+", " ".join(parts).lower())
    token_set = {t for t in tokens if t}

    model_keys = set(EFFICIENTNET_IMG_SIZES.keys())
    model_keys.update(f"convnext_{k}" for k in CONVNEXT_IMG_SIZES.keys())
    model_keys.update(MODEL_ALIASES.keys())

    for key in sorted(model_keys, key=len, reverse=True):
        if key in token_set:
            return key

    for variant in CONVNEXT_IMG_SIZES:
        if "convnext" in token_set and variant in token_set:
            return f"convnext_{variant}"

    for suffix in ("s", "m", "l"):
        if "v2" in token_set and suffix in token_set:
            candidate = f"v2_{suffix}"
            if candidate in EFFICIENTNET_IMG_SIZES:
                return candidate

    return None


def _infer_checkpoint_path(model_path: Path) -> Path | None:
    if not model_path or not model_path.is_file():
        return None
    parent = model_path.parent
    stem = model_path.stem
    stem = re.sub(r"_(fp16|int8|fp32)$", "", stem)
    candidate = parent / f"{stem}.pth"
    if candidate.is_file():
        return candidate
    matches = sorted(parent.glob(f"{stem}*.pth"))
    if len(matches) == 1:
        return matches[0]
    matches = sorted(parent.glob("*.pth"))
    if len(matches) == 1:
        return matches[0]
    return None


def _build_cam_cache_path(
    cache_dir: Path,
    *,
    image_path: Path,
    mask_path: Path | None,
    model_key: str,
    checkpoint_path: Path,
    layer_name: str | None,
    img_size: int,
    alpha: float,
    method: str,
) -> Path:
    key = "|".join(
        [
            str(image_path.resolve()),
            str(mask_path.resolve()) if mask_path else "",
            model_key,
            str(checkpoint_path.resolve()),
            layer_name or "",
            str(img_size),
            f"{alpha:.4f}",
            method,
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.png"


def _load_metadata(data_root: str | None, aspect_filter: str | None) -> pd.DataFrame:
    if data_root:
        set_dataset_root(data_root)
    active_root = get_dataset_root()
    df = load_combined_metadata(root=active_root, handrgbd_include_wall3=True)
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["source"] = df["source"].astype(str).str.lower()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "wall_label" not in df.columns:
        df["wall_label"] = pd.NA
    df["wall_label"] = pd.to_numeric(df["wall_label"], errors="coerce")
    if "lights_label" not in df.columns:
        df["lights_label"] = pd.NA
    df["lights_label"] = df["lights_label"].astype(str).str.lower()
    df["image_path"] = df["image_path"].apply(Path)
    if aspect_filter:
        df = df[df["aspect"].str.contains(aspect_filter, case=False, na=False)]
    df = df[df["image_path"].apply(Path.exists)]
    return df.reset_index(drop=True)


def _apply_filters(
    df: pd.DataFrame,
    dataset_choice: str,
    split_choice: str,
    fold_data: dict | None,
    fold_index: int | None,
    handrgbd_walls: set[int] | None,
    handrgbd_lights: set[str] | None,
) -> pd.DataFrame:
    out = df
    if dataset_choice != "all":
        out = out[out["source"] == dataset_choice]
    if fold_data and split_choice in ("train", "val") and fold_index is not None:
        fold_ids = {str(uid) for uid in fold_data.get("folds", [])[fold_index]}
        if split_choice == "val":
            out = out[out["user_id"].isin(fold_ids)]
        else:
            out = out[~out["user_id"].isin(fold_ids)]
    if handrgbd_walls is not None:
        wall_set = {int(w) for w in handrgbd_walls}
        is_hand = out["source"] == "handrgbd"
        if wall_set:
            out = out[~is_hand | out["wall_label"].isin(wall_set)]
        else:
            out = out[~is_hand]
    if handrgbd_lights is not None:
        light_set = {str(val).lower() for val in handrgbd_lights}
        is_hand = out["source"] == "handrgbd"
        if light_set:
            out = out[~is_hand | out["lights_label"].isin(light_set)]
        else:
            out = out[~is_hand]
    return out.reset_index(drop=True)


def _build_user_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [(str(uid), group.reset_index(drop=True)) for uid, group in df.groupby("user_id")]
    groups.sort(key=lambda item: item[0])
    return groups


def _compute_true_age(records: pd.DataFrame) -> float | None:
    ages = records["age"].dropna()
    if ages.empty:
        return None
    return float(ages.mean())


def _build_items(
    records: pd.DataFrame,
    session: ort.InferenceSession,
    input_name: str,
    img_size: int,
    max_items: int | None,
    dataset_root: Path,
    *,
    handrgbd_rgb_root: str | None = None,
    handrgbd_mask_root: str | None = None,
    grad_cam_runner=None,
    cam_cache_dir: Path | None = None,
    cam_alpha: float = 0.45,
    cam_model_key: str | None = None,
    cam_checkpoint: Path | None = None,
    cam_layer_name: str | None = None,
    cam_method: str = "gradcam",
    noise_std: float = 0.0,
    noise_runs: int = 1,
) -> tuple[list[tuple], int, int, str, list[float], float | None]:
    shown_limit = len(records) if max_items is None else max(0, max_items)
    total_count = len(records)

    true_age_value = _compute_true_age(records)
    true_age = f"{true_age_value:.2f}" if true_age_value is not None else "NA"

    items = []
    predictions: list[float] = []
    noise_std = float(noise_std)
    noise_runs = int(noise_runs)
    use_noise = noise_std > 0 and noise_runs >= 1
    rng = np.random.default_rng() if use_noise else None
    for idx, row in records.iterrows():
        image_path = Path(row["image_path"])
        mask_path = None
        if str(row.get("source", "")).lower() == "handrgbd":
            image_path, mask_path = _resolve_handrgbd_paths(
                image_path,
                dataset_root,
                rgb_root_name=handrgbd_rgb_root,
                mask_root_name=handrgbd_mask_root,
            )
        if not image_path.is_file():
            continue
        try:
            if idx < shown_limit:
                arr, preview, metrics, hist, base_arr = _prepare_variant(
                    image_path, mask_path, img_size
                )
            else:
                base_arr, _ = _load_masked_base(image_path, mask_path)
                arr = _prepare_inference_input(base_arr, img_size)
                preview = None
                metrics = None
                hist = None
            if use_noise:
                age_pred, std_val = _run_noisy_inference(
                    session,
                    input_name,
                    base_arr,
                    img_size,
                    noise_std,
                    noise_runs,
                    rng,
                )
            else:
                age_pred, std_val = _run_inference(session, input_name, arr)
        except Exception as exc:
            print(f"[gallery] Skipping {image_path}: {exc}", file=sys.stderr)
            continue
        if (
            idx < shown_limit
            and preview is not None
            and grad_cam_runner is not None
            and cam_cache_dir is not None
            and cam_model_key
            and cam_checkpoint is not None
        ):
            cam_path = _build_cam_cache_path(
                cam_cache_dir,
                image_path=image_path,
                mask_path=mask_path,
                model_key=cam_model_key,
                checkpoint_path=cam_checkpoint,
                layer_name=cam_layer_name,
                img_size=img_size,
                alpha=cam_alpha,
                method=cam_method,
            )
            cam_preview = None
            if cam_path.is_file():
                try:
                    with Image.open(cam_path) as img:
                        cam_preview = img.convert("RGB").copy()
                except Exception as exc:
                    print(f"[gallery] Failed to load cached Grad-CAM: {exc}", file=sys.stderr)
            else:
                try:
                    cam_preview = grad_cam_runner.render(
                        image_path,
                        mask_path=mask_path,
                        img_size=img_size,
                        alpha=cam_alpha,
                    )
                    cam_path.parent.mkdir(parents=True, exist_ok=True)
                    cam_preview.save(cam_path)
                except Exception as exc:
                    print(f"[gallery] Grad-CAM failed for {image_path}: {exc}", file=sys.stderr)
            if cam_preview is not None:
                preview = cam_preview
        predictions.append(age_pred)
        if idx < shown_limit:
            items.append((image_path, mask_path, age_pred, std_val, true_age, preview, metrics, hist))

    return items, len(items), total_count, true_age, predictions, true_age_value


def _compute_luma_histogram(
    image_rgb: np.ndarray, mask: np.ndarray | None, mask_threshold: int
) -> np.ndarray:
    if image_rgb.ndim == 2:
        luma = image_rgb.astype(np.uint8)
    else:
        r = image_rgb[:, :, 0].astype(np.float32)
        g = image_rgb[:, :, 1].astype(np.float32)
        b = image_rgb[:, :, 2].astype(np.float32)
        luma = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_bin = mask > mask_threshold
        if mask_bin.shape != luma.shape:
            raise ValueError("mask must match image width and height")
        luma = luma[mask_bin]

    if luma.size == 0:
        return np.zeros(256, dtype=np.float32)

    return np.bincount(luma.ravel(), minlength=256).astype(np.float32)


def _load_masked_base(
    image_path: Path, mask_path: Path | None, mask_threshold: int = MASK_THRESHOLD
) -> tuple[np.ndarray, np.ndarray | None]:
    with Image.open(image_path) as img_file:
        img = img_file.convert("RGB")
    mask = None
    if mask_path is not None:
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")
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


def _resize_to_float(base_arr: np.ndarray, size: int) -> np.ndarray:
    preview = Image.fromarray(base_arr, mode="RGB")
    resized = preview.resize((size, size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _normalize_to_input(arr: np.ndarray) -> np.ndarray:
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


def _prepare_inference_input(base_arr: np.ndarray, size: int) -> np.ndarray:
    arr = _resize_to_float(base_arr, size)
    return _normalize_to_input(arr)


def _prepare_inference_input_with_noise(
    base_arr: np.ndarray, size: int, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    arr = _resize_to_float(base_arr, size)
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
    return _normalize_to_input(arr)


def _find_matching_file(root: Path, name: str) -> Path | None:
    if not root.is_dir():
        return None
    candidate = root / name
    if candidate.is_file():
        return candidate
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = root / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _resolve_handrgbd_paths(
    image_path: Path,
    dataset_root: Path,
    *,
    rgb_root_name: str | None = None,
    mask_root_name: str | None = None,
) -> tuple[Path, Path | None]:
    hand_root = dataset_root / "handRGBD"

    rgb_root = None
    if rgb_root_name and rgb_root_name != "auto":
        candidate = hand_root / rgb_root_name
        if candidate.is_dir():
            rgb_root = candidate
    else:
        for name in (
            "rgb",
            "rgb_xyz_jpg",
            "rgb_jpg",
            "rotated_CLAHE_rgb_jpg",
            "rotated_rgb_jpg",
        ):
            candidate = hand_root / name
            if candidate.is_dir():
                rgb_root = candidate
                break

    mask_root = None
    if mask_root_name and mask_root_name != "auto":
        candidate = hand_root / mask_root_name
        if candidate.is_dir():
            mask_root = candidate
    else:
        for name in ("xyz_mask", "rgb_mask", "rotated_rgb_mask"):
            candidate = hand_root / name
            if candidate.is_dir():
                mask_root = candidate
                break

    if rgb_root is None:
        rgb_root = hand_root
    if mask_root is None:
        mask_root = hand_root

    resolved_image = _find_matching_file(rgb_root, image_path.name) or image_path
    resolved_mask = _find_matching_file(mask_root, resolved_image.name) or _find_matching_file(
        mask_root, image_path.name
    )
    return resolved_image, resolved_mask


def _histogram_pixmap(hist: np.ndarray, width: int, height: int) -> QtGui.QPixmap:
    hist = np.asarray(hist, dtype=np.float32)
    max_val = float(hist.max()) if hist.size else 0.0
    if max_val <= 0.0:
        max_val = 1.0

    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(QtGui.QColor(0, 0, 0))
    painter = QtGui.QPainter(pixmap)
    painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))

    bins = int(hist.size)
    for x in range(width):
        start = int(x * bins / width)
        end = int((x + 1) * bins / width)
        if end <= start:
            end = start + 1
        value = float(hist[start:end].max())
        bar = int((value / max_val) * (height - 1))
        painter.drawLine(x, height - 1, x, height - 1 - bar)

    painter.end()
    return pixmap


def _prepare_variant(
    image_path: Path, mask_path: Path | None, size: int, mask_threshold: int = MASK_THRESHOLD
) -> tuple[np.ndarray, Image.Image, dict[str, float], np.ndarray, np.ndarray]:
    base_arr, mask_arr = _load_masked_base(image_path, mask_path, mask_threshold)
    preview = Image.fromarray(base_arr, mode="RGB")
    metrics = compute_exposure_metrics(
        base_arr,
        mask_arr,
        mask_threshold=mask_threshold,
        color_space="RGB",
    )
    hist = _compute_luma_histogram(base_arr, mask_arr, mask_threshold)

    arr = _prepare_inference_input(base_arr, size)
    return arr, preview, metrics, hist, base_arr


def _pixmap_from_pil(img: Image.Image) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    height, width = arr.shape[:2]
    bytes_per_line = 3 * width
    qimg = QtGui.QImage(arr.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _run_inference(session: ort.InferenceSession, input_name: str, arr: np.ndarray) -> tuple[float, float]:
    inputs = {input_name: arr}
    outputs = session.run(None, inputs)
    if len(outputs) < 2:
        raise RuntimeError("ONNX model did not return mean/log_var outputs.")
    mean, log_var = outputs[0], outputs[1]

    mean_val = float(np.asarray(mean).reshape(-1)[0])
    log_var_val = float(np.asarray(log_var).reshape(-1)[0])
    std_val = float(np.exp(0.5 * log_var_val))
    return mean_val, std_val


def _run_noisy_inference(
    session: ort.InferenceSession,
    input_name: str,
    base_arr: np.ndarray,
    size: int,
    noise_std: float,
    runs: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    if runs <= 1:
        if noise_std > 0:
            rng = rng or np.random.default_rng()
            arr = _prepare_inference_input_with_noise(base_arr, size, noise_std, rng)
        else:
            arr = _prepare_inference_input(base_arr, size)
        return _run_inference(session, input_name, arr)

    if noise_std <= 0:
        arr = _prepare_inference_input(base_arr, size)
        return _run_inference(session, input_name, arr)

    rng = rng or np.random.default_rng()
    means: list[float] = []
    stds: list[float] = []
    for _ in range(runs):
        arr = _prepare_inference_input_with_noise(base_arr, size, noise_std, rng)
        mean_val, std_val = _run_inference(session, input_name, arr)
        means.append(mean_val)
        stds.append(std_val)
    return float(np.mean(means)), float(np.mean(stds))


class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class BoxPlotWidget(QtWidgets.QWidget):
    def __init__(self, width: int = 220, height: int = 64) -> None:
        super().__init__()
        self._stats: tuple[float, float, float, float, float] | None = None
        self._scale: tuple[float, float] | None = None
        self._count = 0
        self._true_age: float | None = None
        self.setFixedSize(width, height)
        self.setToolTip("Predicted age distribution for user.")

    def set_data(self, values: list[float], true_age: float | None) -> None:
        vals = np.asarray([float(v) for v in values], dtype=np.float32)
        self._count = int(vals.size)
        self._true_age = float(true_age) if true_age is not None else None
        if vals.size:
            min_val = float(vals.min())
            max_val = float(vals.max())
            q1, median, q3 = [float(v) for v in np.percentile(vals, [25, 50, 75])]
            self._stats = (min_val, q1, median, q3, max_val)
            scale_min = min_val
            scale_max = max_val
            if self._true_age is not None:
                scale_min = min(scale_min, self._true_age)
                scale_max = max(scale_max, self._true_age)
            self._scale = (scale_min - 2.0, scale_max + 2.0)
            tip = (
                f"n={self._count} min={min_val:.2f} q1={q1:.2f} "
                f"median={median:.2f} q3={q3:.2f} max={max_val:.2f}"
            )
            if self._true_age is not None:
                tip += f" true_age={self._true_age:.2f} (marker)"
            self.setToolTip(tip)
        else:
            self._stats = None
            self._scale = None
            self.setToolTip("No predictions available.")
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.palette().window())

        rect = self.rect().adjusted(6, 6, -6, -6)
        if self._stats is None:
            painter.setPen(QtGui.QColor(120, 120, 120))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No data")
            return

        min_val, q1, median, q3, max_val = self._stats
        scale_min, scale_max = self._scale if self._scale is not None else (min_val, max_val)

        span = scale_max - scale_min
        if span <= 0:
            span = 1.0
            scale_min -= 0.5
            scale_max += 0.5

        info_bits = [f"median={median:.2f}"]
        if self._true_age is not None:
            info_bits.append(f"true={self._true_age:.2f}")
        if info_bits:
            painter.setPen(QtGui.QColor(90, 90, 90))
            painter.drawText(rect, QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft, " ".join(info_bits))
            rect = rect.adjusted(0, 12, 0, 0)

        def x_at(value: float) -> float:
            return rect.left() + (value - scale_min) / span * rect.width()

        y_mid = rect.center().y()
        x_min = x_at(min_val)
        x_max = x_at(max_val)
        x_q1 = x_at(q1)
        x_median = x_at(median)
        x_q3 = x_at(q3)

        whisker_pen = QtGui.QPen(QtGui.QColor(60, 60, 60), 1)
        painter.setPen(whisker_pen)
        painter.drawLine(int(x_min), int(y_mid), int(x_max), int(y_mid))

        cap_height = max(6, int(rect.height() * 0.45))
        painter.drawLine(
            int(x_min),
            int(y_mid - cap_height / 2),
            int(x_min),
            int(y_mid + cap_height / 2),
        )
        painter.drawLine(
            int(x_max),
            int(y_mid - cap_height / 2),
            int(x_max),
            int(y_mid + cap_height / 2),
        )

        box_height = max(8, int(rect.height() * 0.5))
        box_top = y_mid - box_height / 2
        box_rect = QtCore.QRectF(x_q1, box_top, max(1.0, x_q3 - x_q1), box_height)
        painter.setBrush(QtGui.QColor(210, 210, 210))
        painter.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 1))
        painter.drawRect(box_rect)

        painter.drawLine(
            int(x_median),
            int(box_top),
            int(x_median),
            int(box_top + box_height),
        )

        if self._true_age is not None:
            true_x = x_at(self._true_age)
            true_x = max(rect.left(), min(rect.right(), true_x))
            painter.setPen(QtGui.QPen(QtGui.QColor(200, 60, 60), 2))
            painter.drawLine(
                int(true_x),
                int(rect.top()),
                int(true_x),
                int(rect.bottom()),
            )


class ExplainDetailDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        image_path: Path,
        mask_path: Path | None,
        model_key: str | None,
        checkpoint: Path | None,
        img_size: int,
        alpha: float,
        device: str,
        method: str,
        initial_layer: str | None,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.model_key = model_key
        self.checkpoint = checkpoint
        self.img_size = img_size
        self.alpha = alpha
        self.device = device
        self.method = method
        self._runner = None
        self._runner_layer = None
        self._layer_names: list[str] = []

        self.setWindowTitle(f"Explain: {image_path.name}")
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Method:"))
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem("Grad-CAM", "gradcam")
        self.method_combo.addItem("Grad-CAM++", "gradcam++")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_idx = self.method_combo.findData(self.method)
        if method_idx >= 0:
            self.method_combo.setCurrentIndex(method_idx)
        controls.addWidget(self.method_combo)
        controls.addWidget(QtWidgets.QLabel("Layer:"))
        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        controls.addWidget(self.layer_combo, 1)
        layout.addLayout(controls)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(self.img_size, self.img_size)
        layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        base_arr, _ = _load_masked_base(image_path, mask_path)
        base_preview = Image.fromarray(base_arr, mode="RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        self._base_preview = base_preview
        self._set_pixmap(base_preview)

        self._init_layers(initial_layer)

    def _set_pixmap(self, img: Image.Image) -> None:
        pixmap = _pixmap_from_pil(img)
        self.image_label.setPixmap(pixmap)

    def _init_layers(self, initial_layer: str | None) -> None:
        if not self.model_key or not self.checkpoint:
            self.method_combo.setEnabled(False)
            self.layer_combo.setEnabled(False)
            self.status_label.setText("Grad-CAM unavailable: model/checkpoint missing.")
            return
        self._ensure_runner(initial_layer)
        if self._runner is None:
            self.method_combo.setEnabled(False)
            self.layer_combo.setEnabled(False)
            if not self.status_label.text():
                self.status_label.setText("Grad-CAM failed to initialize.")
            return
        self.method_combo.setEnabled(True)
        self._layer_names = self._list_conv_layers()
        self._populate_layers(initial_layer)
        self._render()

    def _list_conv_layers(self) -> list[str]:
        if self._runner is None:
            return []
        layers = []
        for name, module in self._runner.model.named_modules():
            if module.__class__.__name__ == "Conv2d":
                layers.append(name)
        return layers

    def _populate_layers(self, selected: str | None) -> None:
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItem("auto", None)
        for name in self._layer_names:
            self.layer_combo.addItem(name, name)
        if selected and selected in self._layer_names:
            idx = self.layer_combo.findData(selected)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)
        else:
            self.layer_combo.setCurrentIndex(0)
        self.layer_combo.blockSignals(False)

    def _ensure_runner(self, layer_name: str | None) -> None:
        if self._runner is not None and self._runner_layer == layer_name:
            return
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None
        try:
            from onnx_tools.grad_cam import GradCamRunner
        except Exception as exc:
            self.status_label.setText(f"Grad-CAM import failed: {exc}")
            return
        try:
            self._runner = GradCamRunner(
                model_name=self.model_key,
                checkpoint_path=self.checkpoint,
                device=self.device,
                target_layer=layer_name,
                method=self.method,
            )
            self._runner_layer = layer_name
        except Exception as exc:
            self.status_label.setText(f"Grad-CAM init failed: {exc}")
            self._runner = None

    def _render(self) -> None:
        if self._runner is None:
            self._set_pixmap(self._base_preview)
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            overlay = self._runner.render(
                self.image_path,
                mask_path=self.mask_path,
                img_size=self.img_size,
                alpha=self.alpha,
            )
        except Exception as exc:
            self.status_label.setText(f"Grad-CAM failed: {exc}")
            self._set_pixmap(self._base_preview)
        else:
            self.status_label.setText("")
            self._set_pixmap(overlay)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_layer_changed(self) -> None:
        selected = self.layer_combo.currentData()
        self._ensure_runner(selected if selected else None)
        self._render()

    def _on_method_changed(self) -> None:
        self.method = self.method_combo.currentData() or "gradcam"
        self._runner_layer = None
        self._ensure_runner(self.layer_combo.currentData())
        self._render()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None
        super().closeEvent(event)


class Gallery(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        metadata: pd.DataFrame,
        model_paths: list[Path],
        model_path: Path,
        optuna_root: Path | None,
        dataset_root: Path,
        dataset_choice: str,
        split_choice: str,
        fold_file: Path | None,
        fold_index: int | None,
        columns: int,
        thumb_size: int | None,
        max_items: int | None,
        aspect_filter: str | None,
        explain_model: str | None,
        explain_checkpoint: Path | None,
        explain_layer: str | None,
        explain_cache: Path | None,
        explain_alpha: float,
        explain_device: str,
        explain_method: str,
    ) -> None:
        super().__init__()
        self.metadata_all = metadata
        self.model_paths = model_paths
        self.optuna_root = optuna_root
        self.dataset_root = dataset_root
        self.dataset_choice = dataset_choice if dataset_choice in DATASET_OPTIONS else "all"
        self.split_choice = split_choice if split_choice in SPLIT_OPTIONS else "all"
        self.fold_file_override = fold_file
        self.fold_index_override = fold_index
        self.view_mode = VIEW_OPTIONS[0]
        self.scatter_timer = QtCore.QTimer(self)
        self.scatter_timer.setInterval(0)
        self.scatter_timer.timeout.connect(self._process_scatter_batch)
        self.scatter_session: ort.InferenceSession | None = None
        self.scatter_input_name = ""
        self.scatter_records: list[dict] = []
        self.scatter_index = 0
        self.scatter_preds: list[float] = []
        self.scatter_trues: list[float] = []
        self.scatter_user_acc: dict[str, tuple[float, float, int]] = {}
        self.scatter_running = False
        self.scatter_dirty = True
        self.scatter_aggregate = False
        self.handrgbd_rgb_root = HANDRGBD_RGB_ROOT_OPTIONS[0]
        self.handrgbd_mask_root = HANDRGBD_MASK_ROOT_OPTIONS[0]
        self.handrgbd_wall_filter = {1, 2, 3, 4}
        self.handrgbd_lights_filter = {"on", "off"}
        self.columns = columns
        self.thumb_size = thumb_size or 0
        self.thumb_size_override = thumb_size is not None
        self.max_items = max_items
        self.aspect_filter = aspect_filter
        self.user_groups: list[tuple[str, pd.DataFrame]] = []
        self.index = 0
        self.fold_data: dict | None = None
        self.fold_index: int | None = None
        self.model_path: Path | None = None
        self.session: ort.InferenceSession | None = None
        self.input_name = ""
        self.img_size = 224
        self.model_name = ""
        self.hist_height = 24
        self.explain_enabled = False
        self.explain_model_override = explain_model
        self.explain_checkpoint_override = explain_checkpoint
        self.explain_layer = explain_layer
        self.explain_cache_dir = explain_cache
        self.explain_alpha = explain_alpha
        self.explain_device = explain_device
        self.explain_method = explain_method
        self.explain_error: str | None = None
        self.explain_model_key: str | None = None
        self.explain_checkpoint: Path | None = None
        self.grad_cam_runner = None
        self._detail_windows: list[ExplainDetailDialog] = []

        layout = QtWidgets.QVBoxLayout(self)

        controls_row = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        self.dataset_combo = QtWidgets.QComboBox()
        for option in DATASET_OPTIONS:
            self.dataset_combo.addItem(option)
        self.dataset_combo.setCurrentText(self.dataset_choice)
        self.dataset_combo.currentIndexChanged.connect(self._on_filter_changed)

        self.split_combo = QtWidgets.QComboBox()
        for option in SPLIT_OPTIONS:
            self.split_combo.addItem(option)
        self.split_combo.setCurrentText(self.split_choice)
        self.split_combo.currentIndexChanged.connect(self._on_filter_changed)

        self.fold_combo = QtWidgets.QComboBox()
        self.fold_combo.currentIndexChanged.connect(self._on_fold_changed)

        self.mode_combo = QtWidgets.QComboBox()
        for option in VIEW_OPTIONS:
            self.mode_combo.addItem(option)
        self.mode_combo.setCurrentText(self.view_mode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.handrgbd_rgb_combo = QtWidgets.QComboBox()
        for option in HANDRGBD_RGB_ROOT_OPTIONS:
            self.handrgbd_rgb_combo.addItem(option)
        self.handrgbd_rgb_combo.setCurrentText(self.handrgbd_rgb_root)
        self.handrgbd_rgb_combo.currentIndexChanged.connect(self._on_handrgbd_root_changed)

        self.handrgbd_mask_combo = QtWidgets.QComboBox()
        for option in HANDRGBD_MASK_ROOT_OPTIONS:
            self.handrgbd_mask_combo.addItem(option)
        self.handrgbd_mask_combo.setCurrentText(self.handrgbd_mask_root)
        self.handrgbd_mask_combo.currentIndexChanged.connect(self._on_handrgbd_root_changed)

        self.handrgbd_wall_checks: dict[int, QtWidgets.QCheckBox] = {}
        for label in (1, 2, 3, 4):
            checkbox = QtWidgets.QCheckBox(str(label))
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._on_handrgbd_wall_changed)
            self.handrgbd_wall_checks[label] = checkbox

        self.handrgbd_light_checks: dict[str, QtWidgets.QCheckBox] = {}
        for label in ("on", "off"):
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._on_handrgbd_lights_changed)
            self.handrgbd_light_checks[label] = checkbox

        self.explain_button = QtWidgets.QPushButton("Explain")
        self.explain_button.setCheckable(True)
        self.explain_button.clicked.connect(self._toggle_explain)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem("Grad-CAM", "gradcam")
        self.method_combo.addItem("Grad-CAM++", "gradcam++")
        self.method_combo.currentIndexChanged.connect(self._on_explain_method_changed)
        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.currentIndexChanged.connect(self._on_explain_layer_changed)
        method_idx = self.method_combo.findData(self.explain_method)
        if method_idx >= 0:
            self.method_combo.setCurrentIndex(method_idx)

        controls_row.addWidget(QtWidgets.QLabel("Model:"))
        controls_row.addWidget(self.model_combo, 1)
        controls_row.addWidget(QtWidgets.QLabel("Dataset:"))
        controls_row.addWidget(self.dataset_combo)
        controls_row.addWidget(QtWidgets.QLabel("Split:"))
        controls_row.addWidget(self.split_combo)
        controls_row.addWidget(QtWidgets.QLabel("Fold:"))
        controls_row.addWidget(self.fold_combo)
        controls_row.addWidget(QtWidgets.QLabel("Mode:"))
        controls_row.addWidget(self.mode_combo)
        layout.addLayout(controls_row)

        handrgbd_row = QtWidgets.QHBoxLayout()
        handrgbd_row.addWidget(QtWidgets.QLabel("HandRGBD RGB:"))
        handrgbd_row.addWidget(self.handrgbd_rgb_combo)
        handrgbd_row.addWidget(QtWidgets.QLabel("Mask:"))
        handrgbd_row.addWidget(self.handrgbd_mask_combo)
        layout.addLayout(handrgbd_row)

        wall_row = QtWidgets.QHBoxLayout()
        wall_row.addWidget(QtWidgets.QLabel("HandRGBD wall:"))
        for label in (1, 2, 3, 4):
            wall_row.addWidget(self.handrgbd_wall_checks[label])
        wall_row.addStretch(1)
        layout.addLayout(wall_row)

        lights_row = QtWidgets.QHBoxLayout()
        lights_row.addWidget(QtWidgets.QLabel("HandRGBD lights:"))
        for label in ("on", "off"):
            lights_row.addWidget(self.handrgbd_light_checks[label])
        lights_row.addStretch(1)
        layout.addLayout(lights_row)

        self.explain_panel = QtWidgets.QWidget()
        explain_row = QtWidgets.QHBoxLayout(self.explain_panel)
        explain_row.setContentsMargins(0, 0, 0, 0)
        explain_row.addWidget(QtWidgets.QLabel("Method:"))
        explain_row.addWidget(self.method_combo)
        explain_row.addWidget(QtWidgets.QLabel("Layer:"))
        explain_row.addWidget(self.layer_combo, 1)
        explain_row.addWidget(self.explain_button)
        layout.addWidget(self.explain_panel)

        self.gallery_panel = QtWidgets.QWidget()
        gallery_layout = QtWidgets.QVBoxLayout(self.gallery_panel)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("Prev")
        self.next_button = QtWidgets.QPushButton("Next")
        self.prev_button.clicked.connect(self._prev)
        self.next_button.clicked.connect(self._next)

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.box_plot = BoxPlotWidget()
        self.metrics_label = QtWidgets.QLabel("MAE=n/a | RMSE=n/a")
        self.metrics_label.setAlignment(QtCore.Qt.AlignCenter)
        self.metrics_label.setToolTip("Per-user prediction error vs true age.")
        stats_host = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(stats_host)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(2)
        stats_layout.addWidget(self.box_plot, 0, QtCore.Qt.AlignCenter)
        stats_layout.addWidget(self.metrics_label, 0, QtCore.Qt.AlignCenter)

        top_row.addWidget(self.prev_button)
        top_row.addWidget(self.next_button)
        top_row.addWidget(self.header, 1)
        top_row.addWidget(stats_host)
        gallery_layout.addLayout(top_row)

        noise_row = QtWidgets.QHBoxLayout()
        noise_row.addWidget(QtWidgets.QLabel("Noise σ:"))
        self.noise_std_spin = QtWidgets.QDoubleSpinBox()
        self.noise_std_spin.setRange(0.0, 0.5)
        self.noise_std_spin.setDecimals(3)
        self.noise_std_spin.setSingleStep(0.01)
        self.noise_std_spin.setValue(0.0)
        self.noise_std_spin.setToolTip("Gaussian noise std in [0, 1] pixel space.")
        noise_row.addWidget(self.noise_std_spin)
        noise_row.addWidget(QtWidgets.QLabel("Runs:"))
        self.noise_runs_spin = QtWidgets.QSpinBox()
        self.noise_runs_spin.setRange(1, 50)
        self.noise_runs_spin.setValue(1)
        self.noise_runs_spin.setToolTip("Number of noisy inference attempts per image.")
        noise_row.addWidget(self.noise_runs_spin)
        self.noise_button = QtWidgets.QPushButton("Re-run noisy")
        self.noise_button.clicked.connect(self._run_noisy_current)
        noise_row.addWidget(self.noise_button)
        noise_row.addStretch(1)
        gallery_layout.addLayout(noise_row)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        gallery_layout.addWidget(self.scroll)

        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setSpacing(12)
        self.grid.setContentsMargins(12, 12, 12, 12)
        self.scroll.setWidget(self.grid_host)

        self.scatter_panel = QtWidgets.QWidget()
        scatter_layout = QtWidgets.QVBoxLayout(self.scatter_panel)
        scatter_layout.setContentsMargins(0, 0, 0, 0)
        scatter_layout.setSpacing(8)
        scatter_controls = QtWidgets.QHBoxLayout()
        self.scatter_run_button = QtWidgets.QPushButton("Run subset")
        self.scatter_run_button.clicked.connect(self._run_scatter)
        self.scatter_agg_check = QtWidgets.QCheckBox("Aggregate per user")
        self.scatter_agg_check.setChecked(self.scatter_aggregate)
        self.scatter_agg_check.stateChanged.connect(self._on_scatter_aggregate_changed)
        self.scatter_status = QtWidgets.QLabel("Click Run subset to plot.")
        self.scatter_status.setWordWrap(True)
        scatter_controls.addWidget(self.scatter_run_button)
        scatter_controls.addWidget(self.scatter_agg_check)
        scatter_controls.addWidget(self.scatter_status, 1)
        scatter_layout.addLayout(scatter_controls)
        self.scatter_stats = QtWidgets.QLabel("MAE=n/a | RMSE=n/a | n=0")
        self.scatter_stats.setAlignment(QtCore.Qt.AlignCenter)
        scatter_layout.addWidget(self.scatter_stats)
        self.scatter_plot = QtWidgets.QLabel("No scatter yet.")
        self.scatter_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.scatter_plot.setMinimumSize(320, 320)
        self.scatter_plot.setScaledContents(False)
        self.scatter_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.scatter_scroll = QtWidgets.QScrollArea()
        self.scatter_scroll.setWidgetResizable(False)
        self.scatter_scroll.setWidget(self.scatter_plot)
        self.scatter_scroll.setMinimumHeight(320)
        scatter_layout.addWidget(self.scatter_scroll, 1)

        layout.addWidget(self.gallery_panel)
        layout.addWidget(self.scatter_panel)

        self._apply_view_mode()
        self._load_model(model_path)
        self._resolve_fold_data()
        self._refresh_users()

    def _populate_model_combo(self) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        current_index = 0
        for idx, path in enumerate(self.model_paths):
            label = path.name
            if self.optuna_root:
                try:
                    label = str(path.relative_to(self.optuna_root))
                except ValueError:
                    label = path.name
            self.model_combo.addItem(label, path)
            if self.model_path and path == self.model_path:
                current_index = idx
        self.model_combo.setCurrentIndex(current_index)
        self.model_combo.blockSignals(False)

    def _load_model(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = _infer_img_size(self.session) or 224
        if not self.thumb_size_override:
            self.thumb_size = max(1, self.img_size // 2)
        self.hist_height = max(24, self.thumb_size // 3)
        self.model_name = model_path.name
        self._refresh_explain_runner()

    def _refresh_explain_runner(self) -> None:
        if self.grad_cam_runner is not None:
            try:
                self.grad_cam_runner.close()
            except Exception:
                pass
        self.grad_cam_runner = None
        self.explain_error = None
        self.explain_model_key = None
        self.explain_checkpoint = None

        if not self.explain_enabled or self.model_path is None:
            self._populate_explain_layers([])
            return

        model_key = self.explain_model_override or _infer_model_key_from_path(self.model_path)
        if not model_key:
            self.explain_error = "Grad-CAM disabled: could not infer model key."
            self._populate_explain_layers([])
            return

        checkpoint = self.explain_checkpoint_override or _infer_checkpoint_path(self.model_path)
        if not checkpoint or not checkpoint.is_file():
            self.explain_error = "Grad-CAM disabled: checkpoint not found."
            self._populate_explain_layers([])
            return

        if self.explain_cache_dir is not None:
            try:
                self.explain_cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"[gallery] Grad-CAM cache init failed: {exc}", file=sys.stderr)
                self.explain_cache_dir = None

        try:
            from onnx_tools.grad_cam import GradCamRunner
        except Exception as exc:
            self.explain_error = f"Grad-CAM import failed: {exc}"
            self._populate_explain_layers([])
            return

        try:
            self.grad_cam_runner = GradCamRunner(
                model_name=model_key,
                checkpoint_path=checkpoint,
                device=self.explain_device,
                target_layer=self.explain_layer,
                method=self.explain_method,
            )
        except Exception as exc:
            self.explain_error = f"Grad-CAM init failed: {exc}"
            self.grad_cam_runner = None
            self._populate_explain_layers([])
            return

        self.explain_model_key = model_key
        self.explain_checkpoint = checkpoint
        conv_layers = self._list_conv_layers()
        self._populate_explain_layers(conv_layers)

    def _list_conv_layers(self) -> list[str]:
        if self.grad_cam_runner is None:
            return []
        layers = []
        for name, module in self.grad_cam_runner.model.named_modules():
            if module.__class__.__name__ == "Conv2d":
                layers.append(name)
        return layers

    def _populate_explain_layers(self, layers: list[str]) -> None:
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItem("auto", None)
        if layers:
            for name in layers:
                self.layer_combo.addItem(name, name)
            self.layer_combo.setEnabled(True)
        else:
            self.layer_combo.setEnabled(False)
        if self.explain_layer and self.explain_layer in layers:
            idx = self.layer_combo.findData(self.explain_layer)
        else:
            idx = 0
        self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)

    def _resolve_fold_data(self) -> None:
        fold_file = self.fold_file_override or _find_fold_file(self.model_path or Path())
        fold_data = None
        if fold_file and fold_file.is_file():
            fold_data = load_kfold_splits(fold_file)
        fold_index = self.fold_index_override
        if fold_index is None and self.model_path:
            fold_index = _infer_fold_index(self.model_path)
        if fold_data:
            max_index = len(fold_data.get("folds", [])) - 1
            if fold_index is None or fold_index < 0 or fold_index > max_index:
                fold_index = 0
        else:
            fold_index = None
        self.fold_data = fold_data
        self.fold_index = fold_index
        self._update_fold_combo()

    def _update_fold_combo(self) -> None:
        self.fold_combo.blockSignals(True)
        self.fold_combo.clear()
        if self.fold_data:
            fold_count = len(self.fold_data.get("folds", []))
            for idx in range(fold_count):
                self.fold_combo.addItem(str(idx), idx)
            if self.fold_index is None:
                self.fold_index = 0
            self.fold_combo.setCurrentIndex(self.fold_index)
            self.fold_combo.setEnabled(True)
        else:
            self.fold_combo.addItem("n/a")
            self.fold_combo.setEnabled(False)
        self.fold_combo.blockSignals(False)

    def _clear_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_items(self, items: list[tuple]) -> None:
        self._clear_grid()
        if not items:
            empty_label = QtWidgets.QLabel("No matching samples.")
            empty_label.setAlignment(QtCore.Qt.AlignCenter)
            self.grid.addWidget(empty_label, 0, 0)
            return

        for idx, (path, mask_path, age, std_val, true_age, preview, metrics, hist) in enumerate(items):
            row, col = divmod(idx, self.columns)
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(6, 6, 6, 6)

            hist_label = QtWidgets.QLabel()
            hist_label.setFixedSize(self.thumb_size, self.hist_height)
            hist_label.setAlignment(QtCore.Qt.AlignCenter)
            if hist is None:
                hist = np.zeros(256, dtype=np.float32)
            hist_label.setPixmap(
                _histogram_pixmap(hist, self.thumb_size, self.hist_height)
            )

            thumb_label = ClickableLabel()
            thumb_label.setFixedSize(self.thumb_size, self.thumb_size)
            thumb_label.setAlignment(QtCore.Qt.AlignCenter)
            thumb_label.clicked.connect(lambda p=path, m=mask_path: self._open_detail_view(p, m))
            if preview is not None:
                pixmap = _pixmap_from_pil(preview)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        self.thumb_size,
                        self.thumb_size,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                    thumb_label.setPixmap(pixmap)
                else:
                    thumb_label.setText("Failed to load")
            else:
                thumb_label.setText("Preview N/A")

            if metrics:
                metrics_text = (
                    f"mean={metrics['mean']:.1f} "
                    f"sc={metrics['shadow_clip']:.3f} "
                    f"hc={metrics['highlight_clip']:.3f}"
                )
            else:
                metrics_text = "metrics=n/a"
            text_label = QtWidgets.QLabel(
                f"age={age:.2f}\nuncertainty={std_val:.2f}\ntrue_age={true_age}\n{metrics_text}"
            )
            text_label.setAlignment(QtCore.Qt.AlignCenter)

            cell_layout.addWidget(hist_label)
            cell_layout.addWidget(thumb_label)
            cell_layout.addWidget(text_label)
            self.grid.addWidget(cell, row, col)

    def _run_noisy_current(self) -> None:
        if self.session is None or not self.user_groups:
            return
        noise_std = float(self.noise_std_spin.value())
        noise_runs = int(self.noise_runs_spin.value())
        self._load_index(self.index, noise_std=noise_std, noise_runs=noise_runs)

    def _load_index(self, index: int, *, noise_std: float = 0.0, noise_runs: int = 1) -> None:
        if not self.user_groups:
            self._render_items([])
            self.header.setText("No samples for current filters.")
            self.box_plot.set_data([], None)
            self._update_metrics_label([], None)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        self.index = max(0, min(index, len(self.user_groups) - 1))
        user_id, records = self.user_groups[self.index]
        items, shown_count, total_count, true_age, predictions, true_age_value = _build_items(
            records,
            self.session,
            self.input_name,
            self.img_size,
            self.max_items,
            self.dataset_root,
            handrgbd_rgb_root=self.handrgbd_rgb_root,
            handrgbd_mask_root=self.handrgbd_mask_root,
            grad_cam_runner=self.grad_cam_runner,
            cam_cache_dir=self.explain_cache_dir,
            cam_alpha=self.explain_alpha,
            cam_model_key=self.explain_model_key,
            cam_checkpoint=self.explain_checkpoint,
            cam_layer_name=getattr(self.grad_cam_runner, "layer_name", None),
            cam_method=self.explain_method,
            noise_std=noise_std,
            noise_runs=noise_runs,
        )
        fold_label = f"{self.fold_index}" if self.fold_index is not None else "n/a"
        explain_status = ""
        if self.explain_enabled:
            if self.explain_error:
                explain_status = " | explain=error"
            elif self.grad_cam_runner is None:
                explain_status = " | explain=off"
            else:
                explain_status = " | explain=on"
        noise_status = ""
        if noise_std and noise_std > 0:
            noise_status = f" | noise=gauss(σ={noise_std:.3f})x{max(1, int(noise_runs))}"
        self._render_items(items)
        self.header.setText(
            f"user={user_id} | dataset={self.dataset_choice} | split={self.split_choice} "
            f"| fold={fold_label} | model={self.model_name} | input_size={self.img_size} "
            f"| showing {shown_count}/{total_count} | true_age={true_age}{explain_status}{noise_status}"
        )
        if self.explain_error:
            self.header.setToolTip(self.explain_error)
        else:
            self.header.setToolTip("")
        self.box_plot.set_data(predictions, true_age_value)
        self._update_metrics_label(predictions, true_age_value)
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.user_groups) - 1)
        self.scroll.verticalScrollBar().setValue(0)

    def _update_metrics_label(
        self, predictions: list[float], true_age_value: float | None
    ) -> None:
        if true_age_value is None or not predictions:
            self.metrics_label.setText("MAE=n/a | RMSE=n/a")
            return
        if not np.isfinite(true_age_value):
            self.metrics_label.setText("MAE=n/a | RMSE=n/a")
            return
        values = np.asarray(predictions, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            self.metrics_label.setText("MAE=n/a | RMSE=n/a")
            return
        errors = values - float(true_age_value)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        self.metrics_label.setText(f"MAE={mae:.2f} | RMSE={rmse:.2f}")

    def _refresh_users(self) -> None:
        self.dataset_choice = self.dataset_combo.currentText()
        self.split_choice = self.split_combo.currentText()
        filtered = _apply_filters(
            self.metadata_all,
            self.dataset_choice,
            self.split_choice,
            self.fold_data,
            self.fold_index,
            self.handrgbd_wall_filter,
            self.handrgbd_lights_filter,
        )
        self.user_groups = _build_user_groups(filtered)
        self._load_index(0)

    def _apply_view_mode(self) -> None:
        is_scatter = self.view_mode == VIEW_OPTIONS[1]
        self.gallery_panel.setVisible(not is_scatter)
        self.scatter_panel.setVisible(is_scatter)
        self.explain_panel.setVisible(not is_scatter)

    def _on_mode_changed(self) -> None:
        self.view_mode = self.mode_combo.currentText()
        self._apply_view_mode()
        if self.view_mode == VIEW_OPTIONS[1] and not self.scatter_running:
            if self.scatter_dirty:
                self.scatter_status.setText("Filters changed; click Run subset.")
            else:
                self.scatter_status.setText("Click Run subset to plot.")

    def _mark_scatter_dirty(self) -> None:
        self.scatter_dirty = True
        if not self.scatter_running:
            self.scatter_status.setText("Filters changed; click Run subset.")

    def _on_scatter_aggregate_changed(self) -> None:
        self.scatter_aggregate = bool(self.scatter_agg_check.isChecked())
        self._mark_scatter_dirty()

    def _set_scatter_placeholder(self, text: str) -> None:
        self.scatter_plot.setPixmap(QtGui.QPixmap())
        self.scatter_plot.setText(text)
        self.scatter_plot.setMinimumSize(320, 320)
        self.scatter_plot.adjustSize()

    def _run_scatter(self) -> None:
        if self.scatter_running:
            return
        if self.model_path is None:
            self.scatter_status.setText("Model not loaded.")
            return

        dataset_choice = self.dataset_combo.currentText()
        split_choice = self.split_combo.currentText()
        filtered = _apply_filters(
            self.metadata_all,
            dataset_choice,
            split_choice,
            self.fold_data,
            self.fold_index,
            self.handrgbd_wall_filter,
            self.handrgbd_lights_filter,
        )
        if filtered.empty:
            self.scatter_stats.setText("MAE=n/a | RMSE=n/a | n=0")
            self._set_scatter_placeholder("No samples to plot.")
            self.scatter_status.setText("No samples for current filters.")
            return

        records: list[dict] = []
        for _, row in filtered.iterrows():
            age_val = row.get("age")
            if age_val is None or (isinstance(age_val, float) and np.isnan(age_val)):
                continue
            if not np.isfinite(float(age_val)):
                continue
            records.append(
                {
                    "image_path": Path(row["image_path"]),
                    "age": float(age_val),
                    "source": str(row.get("source", "")).lower(),
                    "user_id": str(row.get("user_id", "")),
                }
            )

        total = len(records)
        if total == 0:
            self.scatter_stats.setText("MAE=n/a | RMSE=n/a | n=0")
            self._set_scatter_placeholder("No samples with valid ages.")
            self.scatter_status.setText("No samples with valid ages.")
            return

        self.scatter_running = True
        self.scatter_dirty = False
        self.scatter_run_button.setEnabled(False)
        self.scatter_stats.setText("MAE=n/a | RMSE=n/a | n=0")
        self.scatter_status.setText(f"Running 0/{total}...")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_mem_pattern = False
        try:
            self.scatter_session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            self.scatter_session = None
            self._on_scatter_failed(f"Session init failed: {exc}")
            return
        self.scatter_input_name = self.scatter_session.get_inputs()[0].name
        self.scatter_records = records
        self.scatter_index = 0
        self.scatter_preds = []
        self.scatter_trues = []
        self.scatter_user_acc = {}
        if not self.scatter_timer.isActive():
            self.scatter_timer.start()

    def _process_scatter_batch(self) -> None:
        if not self.scatter_running:
            if self.scatter_timer.isActive():
                self.scatter_timer.stop()
            return
        if self.scatter_session is None:
            self._on_scatter_failed("Scatter session missing.")
            return

        total = len(self.scatter_records)
        if total == 0:
            if self.scatter_timer.isActive():
                self.scatter_timer.stop()
            self._on_scatter_finished({"preds": [], "trues": [], "total": 0, "mode": "image"})
            return

        batch_size = 5
        processed = 0
        while processed < batch_size and self.scatter_index < total:
            rec = self.scatter_records[self.scatter_index]
            self.scatter_index += 1
            processed += 1
            image_path = Path(rec["image_path"])
            mask_path = None
            if rec.get("source") == "handrgbd":
                image_path, mask_path = _resolve_handrgbd_paths(
                    image_path,
                    self.dataset_root,
                    rgb_root_name=self.handrgbd_rgb_root,
                    mask_root_name=self.handrgbd_mask_root,
                )
            if not image_path.is_file():
                continue
            try:
                base_arr, _ = _load_masked_base(image_path, mask_path)
                arr = _prepare_inference_input(base_arr, self.img_size)
                mean_val, _ = _run_inference(self.scatter_session, self.scatter_input_name, arr)
            except Exception:
                continue
            pred_val = float(mean_val)
            true_val = float(rec["age"])
            if self.scatter_aggregate:
                user_id = str(rec.get("user_id", ""))
                sum_pred, sum_true, count = self.scatter_user_acc.get(user_id, (0.0, 0.0, 0))
                self.scatter_user_acc[user_id] = (
                    sum_pred + pred_val,
                    sum_true + true_val,
                    count + 1,
                )
            else:
                self.scatter_preds.append(pred_val)
                self.scatter_trues.append(true_val)

        if self.scatter_index >= total:
            if self.scatter_timer.isActive():
                self.scatter_timer.stop()
            if self.scatter_aggregate:
                preds: list[float] = []
                trues: list[float] = []
                for sum_pred, sum_true, count in self.scatter_user_acc.values():
                    if count <= 0:
                        continue
                    preds.append(sum_pred / count)
                    trues.append(sum_true / count)
                payload = {"preds": preds, "trues": trues, "total": total, "mode": "user"}
            else:
                payload = {
                    "preds": self.scatter_preds,
                    "trues": self.scatter_trues,
                    "total": total,
                    "mode": "image",
                }
            self._on_scatter_finished(payload)
        else:
            self._on_scatter_progress(self.scatter_index, total)

    def _on_scatter_progress(self, done: int, total: int) -> None:
        if not self.scatter_running:
            return
        self.scatter_status.setText(f"Running {done}/{total}...")

    def _on_scatter_finished(self, payload: object) -> None:
        self.scatter_running = False
        self.scatter_run_button.setEnabled(True)
        if self.scatter_timer.isActive():
            self.scatter_timer.stop()
        self.scatter_session = None
        self.scatter_input_name = ""
        self.scatter_records = []
        self.scatter_index = 0
        self.scatter_preds = []
        self.scatter_trues = []
        self.scatter_user_acc = {}

        data = payload if isinstance(payload, dict) else {}
        preds = data.get("preds", [])
        trues = data.get("trues", [])
        total = int(data.get("total", len(preds)))
        mode = data.get("mode", "image")
        mode_label = "users" if mode == "user" else "images"
        if not preds or not trues:
            self.scatter_stats.setText("MAE=n/a | RMSE=n/a | n=0")
            self._set_scatter_placeholder("No predictions to plot.")
            self.scatter_status.setText(f"Done. 0/{total} samples.")
            return

        preds_arr = np.asarray(preds, dtype=np.float32)
        trues_arr = np.asarray(trues, dtype=np.float32)
        mask = np.isfinite(preds_arr) & np.isfinite(trues_arr)
        preds_arr = preds_arr[mask]
        trues_arr = trues_arr[mask]
        if preds_arr.size == 0:
            self.scatter_stats.setText("MAE=n/a | RMSE=n/a | n=0")
            self._set_scatter_placeholder("No valid predictions to plot.")
            self.scatter_status.setText(f"Done. 0/{total} samples.")
            return

        count = int(preds_arr.size)
        errors = preds_arr - trues_arr
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        self.scatter_stats.setText(
            f"MAE={mae:.2f} | RMSE={rmse:.2f} | n={count} ({mode_label})"
        )
        if mode == "user":
            self.scatter_status.setText(f"Done. {count} users from {total} images.")
        else:
            self.scatter_status.setText(f"Done. {count}/{total} samples.")
        self._render_scatter_plot(trues_arr.tolist(), preds_arr.tolist())

    def _on_scatter_failed(self, message: str) -> None:
        self.scatter_running = False
        self.scatter_run_button.setEnabled(True)
        self.scatter_dirty = True
        if self.scatter_timer.isActive():
            self.scatter_timer.stop()
        self.scatter_session = None
        self.scatter_input_name = ""
        self.scatter_records = []
        self.scatter_index = 0
        self.scatter_preds = []
        self.scatter_trues = []
        self.scatter_user_acc = {}
        if message:
            self.scatter_status.setText(f"Scatter failed: {message}")
        else:
            self.scatter_status.setText("Scatter failed.")

    def _render_scatter_plot(self, trues: list[float], preds: list[float]) -> None:
        if not trues or not preds:
            self._set_scatter_placeholder("No data to plot.")
            return
        true_vals = np.asarray(trues, dtype=float)
        pred_vals = np.asarray(preds, dtype=float)
        mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
        if not mask.any():
            self._set_scatter_placeholder("No valid data to plot.")
            return
        true_vals = true_vals[mask]
        pred_vals = pred_vals[mask]

        min_val = float(np.min([true_vals.min(), pred_vals.min()]))
        max_val = float(np.max([true_vals.max(), pred_vals.max()]))
        padding = max(1.0, 0.05 * (max_val - min_val))
        axis_min = min_val - padding
        axis_max = max_val + padding

        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure(figsize=(6, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.scatter(true_vals, pred_vals, s=12, alpha=0.6, edgecolors="none")
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "r--", linewidth=1)
        ax.set_xlabel("True age")
        ax.set_ylabel("Predicted age")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(buf.getvalue(), "PNG")
        self.scatter_plot.setText("")
        self.scatter_plot.setPixmap(pixmap)
        self.scatter_plot.setFixedSize(pixmap.size())

    def _on_filter_changed(self) -> None:
        self._refresh_users()
        self._mark_scatter_dirty()

    def _on_handrgbd_root_changed(self) -> None:
        self.handrgbd_rgb_root = self.handrgbd_rgb_combo.currentText()
        self.handrgbd_mask_root = self.handrgbd_mask_combo.currentText()
        self._load_index(self.index)
        self._mark_scatter_dirty()

    def _on_handrgbd_wall_changed(self) -> None:
        selected = {label for label, cb in self.handrgbd_wall_checks.items() if cb.isChecked()}
        self.handrgbd_wall_filter = selected
        self._refresh_users()
        self._mark_scatter_dirty()

    def _on_handrgbd_lights_changed(self) -> None:
        selected = {label for label, cb in self.handrgbd_light_checks.items() if cb.isChecked()}
        self.handrgbd_lights_filter = selected
        self._refresh_users()
        self._mark_scatter_dirty()

    def _on_fold_changed(self) -> None:
        if not self.fold_data:
            return
        self.fold_index = int(self.fold_combo.currentText())
        self._refresh_users()
        self._mark_scatter_dirty()

    def _on_model_changed(self) -> None:
        model_path = self.model_combo.currentData()
        if not model_path:
            return
        self._load_model(Path(model_path))
        self._resolve_fold_data()
        self._refresh_users()
        self._mark_scatter_dirty()

    def _toggle_explain(self, checked: bool) -> None:
        self.explain_enabled = bool(checked)
        if not self.explain_enabled:
            if self.grad_cam_runner is not None:
                try:
                    self.grad_cam_runner.close()
                except Exception:
                    pass
            self.grad_cam_runner = None
            self.explain_error = None
            self._populate_explain_layers([])
        else:
            self._refresh_explain_runner()
        self._refresh_users()

    def _on_explain_layer_changed(self) -> None:
        selected = self.layer_combo.currentData()
        self.explain_layer = selected if selected else None
        if self.explain_enabled:
            self._refresh_explain_runner()
            self._refresh_users()

    def _on_explain_method_changed(self) -> None:
        self.explain_method = self.method_combo.currentData() or "gradcam"
        if self.explain_enabled:
            self._refresh_explain_runner()
            self._refresh_users()

    def _open_detail_view(self, image_path: Path, mask_path: Path | None) -> None:
        if self.model_path is None:
            return
        model_key = (
            self.explain_model_key
            or self.explain_model_override
            or _infer_model_key_from_path(self.model_path)
        )
        checkpoint = (
            self.explain_checkpoint
            or self.explain_checkpoint_override
            or _infer_checkpoint_path(self.model_path)
        )
        dialog = ExplainDetailDialog(
            image_path=image_path,
            mask_path=mask_path,
            model_key=model_key,
            checkpoint=checkpoint,
            img_size=self.img_size,
            alpha=self.explain_alpha,
            device=self.explain_device,
            method=self.explain_method,
            initial_layer=self.explain_layer,
        )
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(lambda _: self._on_detail_closed(dialog))
        self._detail_windows.append(dialog)
        dialog.show()

    def _on_detail_closed(self, dialog: ExplainDetailDialog) -> None:
        if dialog in self._detail_windows:
            self._detail_windows.remove(dialog)

    def _prev(self) -> None:
        if self.index > 0:
            self._load_index(self.index - 1)

    def _next(self) -> None:
        if self.index < len(self.user_groups) - 1:
            self._load_index(self.index + 1)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.scatter_timer.isActive():
            self.scatter_timer.stop()
        self.scatter_running = False
        self.scatter_session = None
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a grid of ONNX predictions from dataset metadata."
    )
    parser.add_argument(
        "--optuna-root",
        type=str,
        default=None,
        help="Root directory to scan for ONNX models.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to ONNX model (used when optuna root is not provided).",
    )
    parser.add_argument(
        "--fold-file",
        type=str,
        default=None,
        help="Path to folds_k*.json (optional override).",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Fold index override (0-based).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root directory (optional).",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_OPTIONS,
        default="all",
        help="Dataset source to display.",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_OPTIONS,
        default="val",
        help="Dataset split to display.",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default="dorsal",
        help="Filter by aspect label (default: dorsal).",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=5,
        help="Number of images per row.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=None,
        help="Thumbnail size in pixels; defaults to half the inferred model input size.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=15,
        help="Maximum images to show per user.",
    )
    parser.add_argument(
        "--explain-model",
        type=str,
        default=None,
        help="Model name for Grad-CAM (e.g. b2, convnext_base).",
    )
    parser.add_argument(
        "--explain-checkpoint",
        type=str,
        default=None,
        help="Path to .pth checkpoint for Grad-CAM.",
    )
    parser.add_argument(
        "--explain-layer",
        type=str,
        default=None,
        help="Target layer path for Grad-CAM (dot notation).",
    )
    parser.add_argument(
        "--explain-cache",
        type=str,
        default=str(_REPO_ROOT / "onnx_tools" / "grad_cam_cache"),
        help="Cache directory for Grad-CAM overlays.",
    )
    parser.add_argument(
        "--explain-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for Grad-CAM.",
    )
    parser.add_argument(
        "--explain-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run Grad-CAM on.",
    )
    parser.add_argument(
        "--explain-method",
        choices=["gradcam", "gradcam++"],
        default="gradcam",
        help="Grad-CAM variant to use.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_paths: list[Path] = []
    optuna_root = None
    if args.optuna_root:
        optuna_root = Path(args.optuna_root)
        model_paths = _scan_onnx_models(optuna_root)
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_file():
            print(f"ERROR: model not found: {model_path}", file=sys.stderr)
            return 2
        if model_path not in model_paths:
            model_paths.insert(0, model_path)
    if not model_paths:
        print("ERROR: no ONNX models found. Provide --optuna-root or --model.", file=sys.stderr)
        return 2

    model_path = model_paths[0]
    if args.model:
        model_path = Path(args.model)

    fold_file = Path(args.fold_file) if args.fold_file else None
    fold_index = args.fold_index

    metadata = _load_metadata(args.data_root, args.aspect)
    dataset_root = get_dataset_root()

    app = QtWidgets.QApplication(sys.argv)
    window = Gallery(
        metadata=metadata,
        model_paths=model_paths,
        model_path=model_path,
        optuna_root=optuna_root,
        dataset_root=dataset_root,
        dataset_choice=args.dataset,
        split_choice=args.split,
        fold_file=fold_file,
        fold_index=fold_index,
        columns=args.columns,
        thumb_size=args.thumb_size,
        max_items=args.max_items,
        aspect_filter=args.aspect,
        explain_model=args.explain_model,
        explain_checkpoint=Path(args.explain_checkpoint) if args.explain_checkpoint else None,
        explain_layer=args.explain_layer,
        explain_cache=Path(args.explain_cache) if args.explain_cache else None,
        explain_alpha=args.explain_alpha,
        explain_device=args.explain_device,
        explain_method=args.explain_method,
    )
    window.setWindowTitle("ONNX age gallery")
    window.resize(1200, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
