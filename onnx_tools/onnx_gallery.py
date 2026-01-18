import argparse
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
from onnx_tools.exposure_check import compute_exposure_metrics
from onnx_tools.run_onnx import IMAGENET_MEAN, IMAGENET_STD, _infer_img_size


MASK_THRESHOLD = 0
DATASET_OPTIONS = ("all", "primary", "archive", "handrgbd")
SPLIT_OPTIONS = ("all", "train", "val")


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


def _load_metadata(data_root: str | None, aspect_filter: str | None) -> pd.DataFrame:
    if data_root:
        set_dataset_root(data_root)
    active_root = get_dataset_root()
    df = load_combined_metadata(root=active_root)
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["source"] = df["source"].astype(str).str.lower()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
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
) -> tuple[list[tuple], int, int, str, list[float], float | None]:
    shown_limit = len(records) if max_items is None else max(0, max_items)
    total_count = len(records)

    true_age_value = _compute_true_age(records)
    true_age = f"{true_age_value:.2f}" if true_age_value is not None else "NA"

    items = []
    predictions: list[float] = []
    for idx, row in records.iterrows():
        image_path = Path(row["image_path"])
        mask_path = None
        if str(row.get("source", "")).lower() == "handrgbd":
            image_path, mask_path = _resolve_handrgbd_paths(image_path, dataset_root)
        if not image_path.is_file():
            continue
        try:
            if idx < shown_limit:
                arr, preview, metrics, hist = _prepare_variant(image_path, mask_path, img_size)
            else:
                base_arr, _ = _load_masked_base(image_path, mask_path)
                arr = _prepare_inference_input(base_arr, img_size)
                preview = None
                metrics = None
                hist = None
            age_pred, std_val = _run_inference(session, input_name, arr)
        except Exception as exc:
            print(f"[gallery] Skipping {image_path}: {exc}", file=sys.stderr)
            continue
        predictions.append(age_pred)
        if idx < shown_limit:
            items.append((image_path, age_pred, std_val, true_age, preview, metrics, hist))

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


def _prepare_inference_input(base_arr: np.ndarray, size: int) -> np.ndarray:
    preview = Image.fromarray(base_arr, mode="RGB")
    resized = preview.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


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


def _resolve_handrgbd_paths(image_path: Path, dataset_root: Path) -> tuple[Path, Path | None]:
    hand_root = dataset_root / "handRGBD"
    rgb_root = hand_root / "rgb"
    if not rgb_root.is_dir():
        alt_root = hand_root / "rgb_jpg"
        if alt_root.is_dir():
            rgb_root = alt_root
    mask_root = hand_root / "rgb_mask"
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
) -> tuple[np.ndarray, Image.Image, dict[str, float], np.ndarray]:
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
    return arr, preview, metrics, hist


def _pixmap_from_pil(img: Image.Image) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
    height, width = arr.shape[:2]
    bytes_per_line = 3 * width
    qimg = QtGui.QImage(arr.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _run_inference(session: ort.InferenceSession, input_name: str, arr: np.ndarray) -> tuple[float, float]:
    inputs = {input_name: arr}
    mean, log_var = session.run(None, inputs)

    mean_val = float(np.asarray(mean).reshape(-1)[0])
    log_var_val = float(np.asarray(log_var).reshape(-1)[0])
    std_val = float(np.exp(0.5 * log_var_val))
    return mean_val, std_val


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

        self._load_model(model_path)

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
        self._resolve_fold_data()

        controls_row.addWidget(QtWidgets.QLabel("Model:"))
        controls_row.addWidget(self.model_combo, 1)
        controls_row.addWidget(QtWidgets.QLabel("Dataset:"))
        controls_row.addWidget(self.dataset_combo)
        controls_row.addWidget(QtWidgets.QLabel("Split:"))
        controls_row.addWidget(self.split_combo)
        controls_row.addWidget(QtWidgets.QLabel("Fold:"))
        controls_row.addWidget(self.fold_combo)
        layout.addLayout(controls_row)

        top_row = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("Prev")
        self.next_button = QtWidgets.QPushButton("Next")
        self.prev_button.clicked.connect(self._prev)
        self.next_button.clicked.connect(self._next)

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.box_plot = BoxPlotWidget()

        top_row.addWidget(self.prev_button)
        top_row.addWidget(self.next_button)
        top_row.addWidget(self.header, 1)
        top_row.addWidget(self.box_plot)
        layout.addLayout(top_row)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setSpacing(12)
        self.grid.setContentsMargins(12, 12, 12, 12)
        self.scroll.setWidget(self.grid_host)

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

        for idx, (path, age, std_val, true_age, preview, metrics, hist) in enumerate(items):
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

            thumb_label = QtWidgets.QLabel()
            thumb_label.setFixedSize(self.thumb_size, self.thumb_size)
            thumb_label.setAlignment(QtCore.Qt.AlignCenter)
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
                f"age={age:.2f}\nstd_val={std_val:.2f}\ntrue_age={true_age}\n{metrics_text}"
            )
            text_label.setAlignment(QtCore.Qt.AlignCenter)

            cell_layout.addWidget(hist_label)
            cell_layout.addWidget(thumb_label)
            cell_layout.addWidget(text_label)
            self.grid.addWidget(cell, row, col)

    def _load_index(self, index: int) -> None:
        if not self.user_groups:
            self._render_items([])
            self.header.setText("No samples for current filters.")
            self.box_plot.set_data([], None)
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
        )
        fold_label = f"{self.fold_index}" if self.fold_index is not None else "n/a"
        self._render_items(items)
        self.header.setText(
            f"user={user_id} | dataset={self.dataset_choice} | split={self.split_choice} "
            f"| fold={fold_label} | model={self.model_name} | input_size={self.img_size} "
            f"| showing {shown_count}/{total_count} | true_age={true_age}"
        )
        self.box_plot.set_data(predictions, true_age_value)
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.user_groups) - 1)
        self.scroll.verticalScrollBar().setValue(0)

    def _refresh_users(self) -> None:
        self.dataset_choice = self.dataset_combo.currentText()
        self.split_choice = self.split_combo.currentText()
        filtered = _apply_filters(
            self.metadata_all,
            self.dataset_choice,
            self.split_choice,
            self.fold_data,
            self.fold_index,
        )
        self.user_groups = _build_user_groups(filtered)
        self._load_index(0)

    def _on_filter_changed(self) -> None:
        self._refresh_users()

    def _on_fold_changed(self) -> None:
        if not self.fold_data:
            return
        self.fold_index = int(self.fold_combo.currentText())
        self._refresh_users()

    def _on_model_changed(self) -> None:
        model_path = self.model_combo.currentData()
        if not model_path:
            return
        self._load_model(Path(model_path))
        self._resolve_fold_data()
        self._refresh_users()

    def _prev(self) -> None:
        if self.index > 0:
            self._load_index(self.index - 1)

    def _next(self) -> None:
        if self.index < len(self.user_groups) - 1:
            self._load_index(self.index + 1)


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
    )
    window.setWindowTitle("ONNX age gallery")
    window.resize(1200, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
