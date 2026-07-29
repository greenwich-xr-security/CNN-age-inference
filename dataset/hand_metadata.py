"""Utility helpers for loading and normalising hand datasets."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Make sure the repository root is on sys.path so the sibling handLandmarks package
# resolves when this module is executed directly (e.g., `python dataset/hand_metadata.py`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from handLandmarks.handLandmarksDetection import (
    MediaPipeTaskHandLandmarkDetector,
    SentisHandLandmarkDetector,
)
from dataset.utils import filter_metadata

# Base directory (can be overridden via env var or function argument)
_DEFAULT_ROOT = Path(r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets")
_ENV_VAR_NAME = "HANDS_DATASETS_ROOT"

PathLike = Union[str, Path]

SYNTHETIC_PROVENANCE_COLUMNS = (
    "skeleton_source_user_id",
    "skin_source_user_id",
    "sex_source_user_id",
    "lighting_source_user_id",
    "aspect_source_user_id",
)

_SYNTHETIC_SKIN_LABEL_MAP = {
    "very_light": "light",
    "light": "light",
    "intermediate": "tan",
    "tan_brown": "dark",
    "dark": "dark",
}

_DATA_ROOT = Path(os.environ.get(_ENV_VAR_NAME, _DEFAULT_ROOT))


def set_dataset_root(root: PathLike) -> Path:
    """Override the dataset root used by loader helpers."""
    global _DATA_ROOT
    _DATA_ROOT = Path(root).expanduser()
    return _DATA_ROOT


def get_dataset_root() -> Path:
    """Return the currently configured dataset root."""
    return _DATA_ROOT


def _resolve_root(root: Optional[PathLike] = None) -> Path:
    if root is None:
        return get_dataset_root()
    return Path(root).expanduser()


# ---------------------------------------------------------------------------
# Label normalisation helpers
VALID_LABELS = {
    "dorsal left",
    "dorsal right",
    "palmar left",
    "palmar right",
}

LABEL_ALIASES: Dict[str, str] = {
    "dorsal_left": "dorsal left",
    "dorsal-right": "dorsal right",
    "palmar_left": "palmar left",
    "palmar-right": "palmar right",
    "right dorsal": "dorsal right",
    "left dorsal": "dorsal left",
    "right palmar": "palmar right",
    "left palmar": "palmar left",
    "palmer left": "palmar left",
    "palmer right": "palmar right",
}


def _normalise_label(raw: object) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    label = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    label = " ".join(label.split())
    label = LABEL_ALIASES.get(label, label)

    # Accept labels regardless of token order (e.g., "right palmar" or "palmar right").
    tokens = label.split()
    side = None
    surface = None
    for tok in tokens:
        if tok in ("left", "right"):
            side = tok
        elif tok in ("dorsal", "palmar", "palmer", "palm"):
            surface = "palmar" if tok in ("palmer", "palm") else tok
    if surface and side:
        candidate = f"{surface} {side}"
        if candidate in VALID_LABELS:
            return candidate

    if label in VALID_LABELS:
        return label
    return None


def _normalise_gender(raw: object) -> Optional[str]:
    mapping = {
        "f": "female",
        "female": "female",
        "m": "male",
        "male": "male",
        0: "female",
        1: "female",  # some datasets use 1 for female
        2: "male",
    }
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        raw_lower = str(raw).strip().lower()
    except Exception:  # noqa: BLE001
        return None
    return mapping.get(raw_lower, mapping.get(raw, None))


# ---------------------------------------------------------------------------
# Bounding-box helpers

_MP_DETECTOR: Optional[MediaPipeTaskHandLandmarkDetector] = None
_SENTIS_DETECTOR: Optional[SentisHandLandmarkDetector] = None


def _get_mediapipe_detector() -> Optional[MediaPipeTaskHandLandmarkDetector]:
    global _MP_DETECTOR
    if _MP_DETECTOR is not None:
        return _MP_DETECTOR
    try:
        _MP_DETECTOR = MediaPipeTaskHandLandmarkDetector()
    except FileNotFoundError as exc:
        print(f"[bbox] MediaPipe detector unavailable: {exc}")
        _MP_DETECTOR = None
    return _MP_DETECTOR


def _get_sentis_detector() -> Optional[SentisHandLandmarkDetector]:
    global _SENTIS_DETECTOR
    if _SENTIS_DETECTOR is not None:
        return _SENTIS_DETECTOR
    try:
        _SENTIS_DETECTOR = SentisHandLandmarkDetector()
    except Exception as exc:  # noqa: BLE001
        print(f"[bbox] Sentis detector unavailable: {exc}")
        _SENTIS_DETECTOR = None
    return _SENTIS_DETECTOR


def _parse_bbox(raw: object) -> Optional[Tuple[int, int, int, int]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        cleaned = (
            raw.strip()
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
        )
        if not cleaned:
            return None
        cleaned = cleaned.replace(";", ",")
        if "," in cleaned:
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in cleaned.split() if p.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        return None

    if len(parts) != 4:
        return None
    try:
        values = tuple(int(round(float(p))) for p in parts)
    except (TypeError, ValueError):
        return None
    return values  # xmin, ymin, xmax, ymax


def _parse_landmarks(raw: object, *, expected_points: int = 21) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Parse a 2D landmark list into a stable tuple of xy pairs."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    value = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"[]", "nan", "none", "null"}:
            return None
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(value, (list, tuple)) or len(value) != expected_points:
        return None

    parsed = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        parsed.append((x, y))
    return tuple(parsed)


def _extract_landmarks(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].apply(_parse_landmarks)
    return pd.Series(index=df.index, data=[None] * len(df), dtype="object")


def _landmarks_to_tight_bbox(
    landmarks: Optional[Tuple[Tuple[float, float], ...]]
) -> Optional[Tuple[int, int, int, int]]:
    """Compute a tight xyxy bbox from landmark coordinates."""
    if not landmarks:
        return None
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x1, x2 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    y1, y2 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _hagrid_landmarks_to_crop(row) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Convert HaGRID normalized source-image landmarks into loaded crop pixels."""
    landmarks = _parse_landmarks(row.get("hand_landmarks"))
    bbox = _parse_bbox(row.get("square_crop_xyxy_px"))
    if landmarks is None or bbox is None:
        return None
    try:
        source_w = float(row.get("source_img_w"))
        source_h = float(row.get("source_img_h"))
        side = float(row.get("square_side_px"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(source_w) or not np.isfinite(source_h) or not np.isfinite(side) or side <= 0:
        return None

    crop_size = 500.0
    x1, y1, _x2, _y2 = bbox
    converted = []
    for x_norm, y_norm in landmarks:
        x = ((x_norm * source_w) - x1) * (crop_size / side)
        y = ((y_norm * source_h) - y1) * (crop_size / side)
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        converted.append((x, y))
    return tuple(converted)


def _drop_invalid_landmarks(df: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    missing_mask = df["landmarks"].isna()
    dropped = int(missing_mask.sum())
    if dropped:
        print(f"[{source_label}] Dropped {dropped} samples with missing/invalid landmarks.")
    return df[~missing_mask].copy()


def _bbox_to_string(bbox: Optional[Tuple[int, int, int, int]]) -> str:
    if bbox is None:
        return ""
    return ",".join(str(int(v)) for v in bbox)


def _compute_bbox_for_image(image_path: Path) -> Optional[Tuple[int, int, int, int]]:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[bbox] Warning: failed to read image '{image_path}'")
        return None

    height, width = img.shape[:2]

    def _detect_with(detector) -> Optional[np.ndarray]:
        if detector is None:
            return None
        try:
            landmarks_norm, _ = detector.detect(img)
        except Exception as exc:  # noqa: BLE001
            print(f"[bbox] {detector.__class__.__name__} failed on '{image_path}': {exc}")
            return None
        if landmarks_norm is None:
            return None
        arr = np.asarray(landmarks_norm, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr

    landmarks_norm = _detect_with(_get_mediapipe_detector())
    fallback_used = False

    if landmarks_norm is None:
        fallback_used = True
        landmarks_norm = _detect_with(_get_sentis_detector())

    if landmarks_norm is None:
        print(f"[bbox] Warning: no hand detected in '{image_path}'")
        return None

    xs = np.clip(landmarks_norm[:, 0], 0.0, 1.0) * max(width - 1, 0)
    ys = np.clip(landmarks_norm[:, 1], 0.0, 1.0) * max(height - 1, 0)

    xmin = int(np.floor(xs.min()))
    xmax = int(np.ceil(xs.max()))
    ymin = int(np.floor(ys.min()))
    ymax = int(np.ceil(ys.max()))

    if fallback_used:
        print(f"[bbox] Fallback detector succeeded for '{image_path}'")

    return xmin, ymin, xmax, ymax


def _center_square_bbox(width: int, height: int) -> Tuple[int, int, int, int]:
    """Return a square bbox centered within the given width/height."""
    side = min(width, height)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return x1, y1, x2, y2


def _resolve_relative_path(base: Path, raw: object) -> Optional[Path]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text.replace("\\", "/"))
    if not path.is_absolute():
        path = base / path
    return path if path.is_file() else None


def _ensure_bboxes(
    df: pd.DataFrame,
    image_path_col: str,
    *,
    csv_source: Optional[Path] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    if "bbox" in df.columns:
        parsed = df["bbox"].apply(_parse_bbox)
    else:
        parsed = pd.Series(index=df.index, data=[None] * len(df), dtype="object")

    missing_mask = parsed.isna()
    if not missing_mask.any():
        return parsed

    updates_for_csv: Dict[int, str] = {}

    if raw_df is not None and "bbox" not in raw_df.columns:
        raw_df["bbox"] = ""

    for idx, image_path in df.loc[missing_mask, image_path_col].items():
        bbox = _compute_bbox_for_image(Path(image_path))
        parsed.at[idx] = bbox
        if raw_df is not None:
            formatted = _bbox_to_string(bbox)
            current = raw_df.at[idx, "bbox"] if idx in raw_df.index and "bbox" in raw_df.columns else ""
            if formatted != current:
                raw_df.at[idx, "bbox"] = formatted
                updates_for_csv[idx] = formatted

    if updates_for_csv and csv_source is not None:
        raw_df.to_csv(csv_source, index=False)
        source_label = csv_source.name if hasattr(csv_source, "name") else str(csv_source)
        print(f"[bbox] Stored {len(updates_for_csv)} computed bounding boxes in {source_label}")

    return parsed


# ---------------------------------------------------------------------------
# Metadata loaders

def _build_archive_filename(
    person_no: int,
    age: int,
    gender: Optional[int],
    photo_no: int,
    *,
    archive_root: Path,
) -> Optional[Path]:
    parts = [str(person_no), str(age)]
    if gender is not None:
        parts.append(str(gender))
    parts.append(str(photo_no))
    stem = "_".join(parts)
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = archive_root / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_primary_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    dataset_root = _resolve_root(root)
    primary_root = dataset_root / "11kHands" / "Hands"
    primary_csv = dataset_root / "11kHands" / "HandInfo.csv"

    if not primary_csv.exists():
        raise FileNotFoundError(f"Primary CSV not found: {primary_csv}")

    raw_df = pd.read_csv(primary_csv)
    working_df = raw_df.copy()

    working_df["aspect_norm"] = working_df["aspectOfHand"].apply(_normalise_label)
    working_df = working_df[working_df["aspect_norm"].notna()]

    working_df["image_path"] = working_df["imageName"].apply(lambda name: primary_root / str(name))
    working_df = working_df[working_df["image_path"].apply(Path.exists)]

    working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    working_df["age_norm"] = working_df["age"].apply(lambda x: int(x) if pd.notna(x) else pd.NA)

    df_out = pd.DataFrame(
        {
            "source": "primary",
            "user_id": working_df["id"].apply(lambda x: f"primary_{int(x)}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"Primary dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_archive_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    dataset_root = _resolve_root(root)
    archive_root = dataset_root / "archive" / "Photos"
    archive_csv = dataset_root / "archive" / "annotated_dataset_details.csv"

    if not archive_csv.exists():
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path"])

    raw_df = pd.read_csv(archive_csv)
    if raw_df.empty or "aspectOfHand" not in raw_df.columns:
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path"])

    working_df = raw_df.copy()
    working_df["aspect_norm"] = working_df["aspectOfHand"].apply(_normalise_label)
    working_df = working_df[working_df["aspect_norm"].notna()]

    def resolve_path(row) -> Optional[Path]:
        try:
            person_no = int(row.get("Person No"))
            age_val = row.get("Age", -1)
            age = int(age_val) if pd.notna(age_val) else -1
            gender_val = row.get("Gender", None)
            gender = int(gender_val) if pd.notna(gender_val) else None
            photo_no = int(row.get("Photo No"))
        except (TypeError, ValueError):
            return None
        return _build_archive_filename(
            person_no,
            age,
            gender,
            photo_no,
            archive_root=archive_root,
        )

    working_df["image_path"] = working_df.apply(resolve_path, axis=1)
    working_df = working_df[working_df["image_path"].notna()]

    gender_map = {1: "female", 2: "male"}
    working_df["gender_norm"] = working_df["Gender"].apply(lambda g: gender_map.get(g) if pd.notna(g) else None)
    working_df["age_norm"] = working_df["Age"].apply(lambda a: int(a) if pd.notna(a) else pd.NA)

    df_out = pd.DataFrame(
        {
            "source": "archive",
            "user_id": working_df["Person No"].apply(lambda x: f"archive_{int(x)}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"].apply(Path),
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"Archive dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_handrgbd_metadata(
    root: Optional[PathLike] = None,
    *,
    include_wall3: bool = False,
) -> pd.DataFrame:
    dataset_root = _resolve_root(root)
    hand_root = dataset_root / "handRGBD"
    rgb_root = hand_root / "rgb_jpg"
    if not rgb_root.exists():
        alt_root = hand_root / "rgb"
        if alt_root.exists():
            rgb_root = alt_root
    if not rgb_root.exists():
        print(f"[handRGBD] RGB folder not found (tried 'rgb_jpg' and 'rgb' under {hand_root})")
        return pd.DataFrame(columns=["source", "user_id", "age", "gender", "aspect", "image_path", "bbox", "landmarks"])
    metadata_csv = dataset_root / "handRGBD" / "reference_table.csv"

    empty_cols = [
        "source",
        "user_id",
        "age",
        "gender",
        "aspect",
        "image_path",
        "bbox",
        "landmarks",
        "wall_label",
        "lights_label",
        "skin_color",
    ]
    if not metadata_csv.exists():
        return pd.DataFrame(columns=empty_cols)

    raw_df = pd.read_csv(metadata_csv)
    if raw_df.empty or "name" not in raw_df.columns or "user_id" not in raw_df.columns:
        return pd.DataFrame(columns=empty_cols)

    working_df = raw_df.copy()

    def parse_wall_label(name_val: object) -> int | None:
        if name_val is None or (isinstance(name_val, float) and pd.isna(name_val)):
            return None
        match = re.search(r"wall[-_\s]*(\d+)", str(name_val), re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def normalise_lights(val: object) -> str | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        text = str(val).strip().lower()
        if text in ("on", "1", "true", "yes"):
            return "on"
        if text in ("off", "0", "false", "no"):
            return "off"
        return None

    def parse_lights_label(name_val: object) -> str | None:
        if name_val is None or (isinstance(name_val, float) and pd.isna(name_val)):
            return None
        match = re.search(r"lights[-_\s]*(on|off)", str(name_val), re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower()

    working_df["wall_label"] = working_df["name"].apply(parse_wall_label)
    if "lights" in working_df.columns:
        working_df["lights_label"] = working_df["lights"].apply(normalise_lights)
    else:
        working_df["lights_label"] = None
    if working_df["lights_label"].isna().all():
        working_df["lights_label"] = working_df["name"].apply(parse_lights_label)
    if not include_wall3:
        working_df = working_df[working_df["wall_label"].ne(3)]

    if "aspect" in working_df.columns:
        working_df["aspect_norm"] = working_df["aspect"].apply(_normalise_label)
    else:
        working_df["aspect_norm"] = pd.NA
    working_df = working_df[working_df["aspect_norm"].notna()]

    def resolve_path(name_val: object) -> Optional[Path]:
        if name_val is None or (isinstance(name_val, float) and pd.isna(name_val)):
            return None
        name_str = str(name_val).strip()
        if not name_str:
            return None
        candidates = []
        if any(name_str.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
            candidates.append(rgb_root / name_str)
        else:
            candidates.append(rgb_root / f"{name_str}.png")
            candidates.append(rgb_root / f"{name_str}.jpg")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    working_df["image_path"] = working_df["name"].apply(resolve_path)
    working_df = working_df[working_df["image_path"].notna()]

    manifest_path = hand_root / "patches" / "manifest_with_ita.csv"
    working_df["skin_color"] = None
    if manifest_path.exists():
        try:
            manifest_df = pd.read_csv(manifest_path, usecols=["user_id", "skin color"])
        except (ValueError, pd.errors.EmptyDataError):
            manifest_df = pd.DataFrame(columns=["user_id", "skin color"])

        if not manifest_df.empty:
            manifest_df = manifest_df.rename(columns={"skin color": "skin_color"}).copy()
            manifest_df["user_id"] = manifest_df["user_id"].astype(str).str.strip()
            manifest_df["skin_color"] = (
                manifest_df["skin_color"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
            )
            manifest_df = manifest_df.drop_duplicates(subset="user_id", keep="first")

            working_df["user_id_str"] = working_df["user_id"].astype(str).str.strip()
            working_df = working_df.merge(
                manifest_df,
                left_on="user_id_str",
                right_on="user_id",
                how="left",
                suffixes=("", "_manifest"),
            )
            working_df["skin_color"] = working_df["skin_color_manifest"]
            working_df = working_df.drop(
                columns=[col for col in ("user_id_str", "user_id_manifest", "skin_color_manifest") if col in working_df.columns]
            )

    if "gender" in working_df.columns:
        working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    else:
        working_df["gender_norm"] = None

    if "age" in working_df.columns:
        working_df["age_norm"] = working_df["age"].apply(lambda a: int(a) if pd.notna(a) else pd.NA)
    else:
        working_df["age_norm"] = pd.NA

    working_df["landmarks"] = _extract_landmarks(working_df, "rgb_landmarks")
    working_df = _drop_invalid_landmarks(working_df, source_label="handRGBD")
    working_df["bbox_tuple"] = working_df["landmarks"].apply(_landmarks_to_tight_bbox)
    working_df = working_df[working_df["bbox_tuple"].notna()].copy()

    df_out = pd.DataFrame(
        {
            "source": "handrgbd",
            "user_id": working_df["user_id"].apply(lambda uid: f"handrgbd_{uid}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"].apply(Path),
            "bbox": working_df["bbox_tuple"],
            "landmarks": working_df["landmarks"],
            "wall_label": working_df["wall_label"],
            "lights_label": working_df["lights_label"],
            "skin_color": working_df["skin_color"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"HandRGBD dataset -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_hagrid_stop_inverted_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    """Load HaGRIDv2 stop_inverted gesture crops as a dorsal-hand source.

    The ``stop_inverted`` gesture (hand raised, back of hand facing camera)
    is a dorsal view suitable for age inference.  Images are already
    centre-cropped to 500 × 500 px and are loaded from ``rgb_masked/<name>.png``
    when available, falling back to ``rgb/<name>.jpg`` if needed.
    Only ``stop_inverted`` rows are included; ``no_gesture`` rows are dropped.
    """
    dataset_root = _resolve_root(root)
    hagrid_root = dataset_root / "HaGRIDv2_stop_inverted"
    csv_path = hagrid_root / "reference_hagrid_stop_inverted.csv"
    rgb_masked_root = hagrid_root / "rgb_masked"
    rgb_root = hagrid_root / "rgb"

    empty_cols = ["source", "user_id", "age", "gender", "aspect", "image_path", "bbox", "landmarks"]
    if not csv_path.exists():
        print(f"[HaGRID] CSV not found: {csv_path}")
        return pd.DataFrame(columns=empty_cols)

    raw_df = pd.read_csv(csv_path)
    if raw_df.empty or "name" not in raw_df.columns:
        return pd.DataFrame(columns=empty_cols)

    # Keep only stop_inverted gesture (dorsal view)
    working_df = raw_df[raw_df["label"].str.lower() == "stop_inverted"].copy()

    # Drop rows with missing age
    working_df = working_df[working_df["age"].notna()]

    # Resolve image paths
    def resolve_path(name_val: object) -> Optional[Path]:
        if name_val is None or (isinstance(name_val, float) and pd.isna(name_val)):
            return None
        name_str = str(name_val).strip()
        candidates = []
        if rgb_masked_root.exists():
            candidates.append(rgb_masked_root / f"{name_str}.png")
        if rgb_root.exists():
            candidates.append(rgb_root / f"{name_str}.jpg")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    working_df["image_path"] = working_df["name"].apply(resolve_path)
    working_df = working_df[working_df["image_path"].notna()]
    working_df["landmarks"] = working_df.apply(_hagrid_landmarks_to_crop, axis=1)
    working_df = _drop_invalid_landmarks(working_df, source_label="HaGRID")
    working_df["bbox_tuple"] = working_df["landmarks"].apply(_landmarks_to_tight_bbox)
    working_df = working_df[working_df["bbox_tuple"].notna()].copy()

    # Normalise gender: HaGRID uses "F" / "M"
    gender_map = {"f": "female", "m": "male"}
    working_df["gender_norm"] = working_df["gender"].apply(
        lambda g: gender_map.get(str(g).strip().lower()) if pd.notna(g) else None
    )

    # Age as int
    working_df["age_norm"] = working_df["age"].apply(
        lambda a: int(round(float(a))) if pd.notna(a) else pd.NA
    )

    # The stop_inverted gesture shows the dorsal side; assign "dorsal" so the
    # standard filter_metadata() dorsal filter accepts these rows.
    df_out = pd.DataFrame(
        {
            "source": "hagrid",
            "user_id": working_df["user_id"].apply(lambda uid: f"hagrid_{uid}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": "dorsal",
            "image_path": working_df["image_path"].apply(Path),
            "bbox": working_df["bbox_tuple"],
            "landmarks": working_df["landmarks"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"HaGRID stop_inverted -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def load_synthetic_dorsal_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    """Load SyntheticDorsalHands images into the shared metadata schema.

    Synthetic images are independent samples rather than repeated subjects, so
    each ``sample_id`` is used as its stable synthetic user ID.  The source CSV
    stores an RGB triplet/ITA in ``skin_color``; the categorical skin label is
    recovered from the canonical filename instead.
    """
    dataset_root = _resolve_root(root)
    synthetic_root = dataset_root / "SyntheticDorsalHands"
    csv_path = synthetic_root / "reference_synthetic.csv"
    empty_cols = [
        "source", "user_id", "age", "gender", "aspect", "image_path",
        "mask_path", "bbox", "skin_color", "synthetic_skin_label",
        *SYNTHETIC_PROVENANCE_COLUMNS,
    ]
    if not csv_path.exists():
        print(f"[SyntheticDorsalHands] CSV not found: {csv_path}")
        return pd.DataFrame(columns=empty_cols)

    raw_df = pd.read_csv(csv_path)
    required = {"sample_id", "image_path", "age", "gender", "aspect"}
    if raw_df.empty or not required.issubset(raw_df.columns):
        return pd.DataFrame(columns=empty_cols)

    working_df = raw_df.copy()
    working_df["sample_id"] = working_df["sample_id"].astype(str).str.strip()
    extracted = working_df["sample_id"].str.extract(
        r"^job\d+_age_\d+_(?:female|male)_(?P<skin_label>.+)_\d+$"
    )
    working_df["synthetic_skin_label"] = extracted["skin_label"].str.lower()
    working_df["skin_color_norm"] = working_df["synthetic_skin_label"].map(
        _SYNTHETIC_SKIN_LABEL_MAP
    )
    working_df["aspect_norm"] = working_df["aspect"].apply(_normalise_label)
    working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    working_df["age_norm"] = pd.to_numeric(working_df["age"], errors="coerce")
    working_df["image_path_abs"] = working_df["image_path"].apply(
        lambda value: synthetic_root / str(value).replace("\\", "/")
    )
    if "mask_path" in working_df.columns:
        working_df["mask_path_abs"] = working_df["mask_path"].apply(
            lambda value: synthetic_root / str(value).replace("\\", "/")
        )
    else:
        working_df["mask_path_abs"] = pd.NA
    working_df["bbox_tuple"] = working_df.get(
        "bbox", pd.Series(index=working_df.index, dtype="object")
    ).apply(_parse_bbox)

    valid = (
        working_df["aspect_norm"].notna()
        & working_df["gender_norm"].notna()
        & working_df["age_norm"].notna()
        & working_df["skin_color_norm"].notna()
        & working_df["image_path_abs"].apply(Path.is_file)
    )
    working_df = working_df.loc[valid].copy()
    working_df["mask_path_abs"] = working_df["mask_path_abs"].where(
        working_df["mask_path_abs"].apply(
            lambda path: isinstance(path, Path) and path.is_file()
        ),
        pd.NA,
    )
    for column in SYNTHETIC_PROVENANCE_COLUMNS:
        if column not in working_df.columns:
            working_df[column] = pd.NA

    df_out = pd.DataFrame(
        {
            "source": "synthetic_dorsal",
            "user_id": "synthetic_" + working_df["sample_id"],
            "age": working_df["age_norm"].astype(int),
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path_abs"],
            "mask_path": working_df["mask_path_abs"],
            "bbox": working_df["bbox_tuple"],
            "skin_color": working_df["skin_color_norm"],
            "synthetic_skin_label": working_df["synthetic_skin_label"],
        }
    )
    for column in SYNTHETIC_PROVENANCE_COLUMNS:
        df_out[column] = working_df[column].astype("string")
    df_out = df_out.reset_index(drop=True)
    print(
        f"SyntheticDorsalHands -> images: {len(df_out)} | "
        f"skin colours: {df_out['skin_color'].value_counts().to_dict()}"
    )
    return df_out


def load_synthetic_dorsal_hands2_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    """Load SyntheticDorsalHands2 images by parsing filenames in the rgb/ directory.

    Unlike V1 (which uses a reference CSV), V2 encodes all metadata in the filename:
    ``job{jobid}_age_{age}_{sex}_{ita_group}_{idx}.png``.  Each image is an
    independent generated sample, so each gets its own synthetic user ID.
    """
    dataset_root = _resolve_root(root)
    synthetic_root = dataset_root / "SyntheticDorsalHands2"
    rgb_root = synthetic_root / "rgb"

    empty_cols = [
        "source", "user_id", "age", "gender", "aspect", "image_path",
        "skin_color", "synthetic_skin_label",
    ]
    if not rgb_root.exists():
        print(f"[SyntheticDorsalHands2] rgb folder not found: {rgb_root}")
        return pd.DataFrame(columns=empty_cols)

    ita_groups = "|".join(re.escape(k) for k in _SYNTHETIC_SKIN_LABEL_MAP)
    pattern = re.compile(
        rf"^(job\d+)_age_(\d+)_(female|male)_({ita_groups})_(\d+)\.png$"
    )
    records = []
    for img_path in rgb_root.glob("*.png"):
        m = pattern.match(img_path.name)
        if m is None:
            continue
        age = int(m.group(2))
        sex = m.group(3)
        ita_group = m.group(4)
        records.append({
            "sample_id": img_path.stem,
            "image_path": img_path,
            "age": age,
            "gender": sex,
            "ita_group": ita_group,
        })

    if not records:
        print(f"[SyntheticDorsalHands2] No matching images found in {rgb_root}")
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(records)
    df["skin_color"] = df["ita_group"].map(_SYNTHETIC_SKIN_LABEL_MAP)
    df["synthetic_skin_label"] = df["ita_group"]

    df_out = pd.DataFrame(
        {
            "source": "synthetic_dorsal2",
            "user_id": "synthetic2_" + df["sample_id"],
            "age": df["age"],
            "gender": df["gender"],
            "aspect": "dorsal",
            "image_path": df["image_path"],
            "skin_color": df["skin_color"],
            "synthetic_skin_label": df["synthetic_skin_label"],
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"SyntheticDorsalHands2 -> images: {len(df_out)} | "
        f"skin colours: {df_out['skin_color'].value_counts().to_dict()}"
    )
    return df_out


def load_prolific_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    """Load the ProlificHands export as an optional hand age source."""
    dataset_root = _resolve_root(root)
    prolific_root = dataset_root / "ProlificHands"
    csv_path = prolific_root / "reference_prolific.csv"
    rgb_masked_root = prolific_root / "rgb_masked"

    empty_cols = [
        "source",
        "user_id",
        "age",
        "gender",
        "aspect",
        "image_path",
        "mask_path",
        "bbox",
        "landmarks",
        "skin_color",
        "skin_tone_self_reported",
    ]
    if not csv_path.exists():
        print(f"[ProlificHands] CSV not found: {csv_path}")
        return pd.DataFrame(columns=empty_cols)

    raw_df = pd.read_csv(csv_path)
    if raw_df.empty or "name" not in raw_df.columns or "participant_id" not in raw_df.columns:
        return pd.DataFrame(columns=empty_cols)

    working_df = raw_df.copy()
    working_df["aspect_norm"] = working_df["aspect"].apply(_normalise_label)
    working_df = working_df[working_df["aspect_norm"].notna()].copy()

    def resolve_image(row) -> Optional[Path]:
        name = str(row.get("name", "")).strip()
        if name and rgb_masked_root.exists():
            masked = rgb_masked_root / f"{name}.png"
            if masked.is_file():
                return masked
        return _resolve_relative_path(prolific_root, row.get("rgb_path"))

    working_df["image_path"] = working_df.apply(resolve_image, axis=1)
    working_df = working_df[working_df["image_path"].notna()].copy()
    if "mask_path" in working_df.columns:
        working_df["mask_path_resolved"] = working_df["mask_path"].apply(
            lambda value: _resolve_relative_path(prolific_root, value)
        )
    else:
        working_df["mask_path_resolved"] = None
    working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    working_df["age_norm"] = working_df["age"].apply(
        lambda a: int(round(float(a))) if pd.notna(a) else pd.NA
    )
    working_df["landmarks"] = _extract_landmarks(working_df, "rgb_landmarks")
    working_df = _drop_invalid_landmarks(working_df, source_label="ProlificHands")
    working_df["bbox_tuple"] = working_df["landmarks"].apply(_landmarks_to_tight_bbox)
    working_df = working_df[working_df["bbox_tuple"].notna()].copy()

    skin_color = (
        working_df["skin_color"]
        if "skin_color" in working_df.columns
        else pd.Series(index=working_df.index, data=pd.NA)
    )
    skin_color = (
        skin_color.astype("string")
        .str.strip()
        .str.lower()
        .replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
    )

    df_out = pd.DataFrame(
        {
            "source": "prolific",
            "user_id": working_df["participant_id"].apply(lambda uid: f"prolific_{uid}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"].apply(Path),
            "mask_path": working_df["mask_path_resolved"],
            "bbox": working_df["bbox_tuple"],
            "landmarks": working_df["landmarks"],
            "skin_color": skin_color,
            "skin_tone_self_reported": working_df.get("skin_tone_self_reported"),
        }
    )
    df_out = df_out.reset_index(drop=True)
    print(
        f"ProlificHands -> users: {df_out['user_id'].nunique()} | images: {len(df_out)}"
    )
    return df_out


def _limit_users_per_age(df: pd.DataFrame, *, max_users_per_year: int = 15) -> pd.DataFrame:
    """Cap unique users per age, preferring handRGBD/HaGRID when legacy sources are present."""
    required_cols = {"user_id", "age", "source"}
    if not required_cols.issubset(df.columns):
        return df

    age_known = df[df["age"].notna()].copy()
    if age_known.empty:
        return df

    age_known["age_year"] = age_known["age"].astype(float).round().astype(int)
    priority_map = {"handrgbd": 0, "hagrid": 1, "prolific": 2, "primary": 3, "archive": 4}
    age_known["priority"] = age_known["source"].map(priority_map).fillna(99).astype(int)

    keep_users: set[str] = set()
    dropped_users = 0
    for _age, group in age_known.groupby("age_year"):
        user_priorities = (
            group.groupby("user_id")["priority"]
                .min()
                .reset_index()
                .sort_values(["priority", "user_id"])
        )
        selected = user_priorities.head(max_users_per_year)["user_id"].tolist()
        keep_users.update(selected)
        dropped_users += max(0, len(user_priorities) - len(selected))

    filtered_known = age_known[age_known["user_id"].isin(keep_users)].drop(columns=["age_year", "priority"])
    if dropped_users:
        print(f"Per-age cap applied ({max_users_per_year} users/year): dropped {dropped_users} users.")

    age_unknown = df[df["age"].isna()]
    return pd.concat([age_unknown, filtered_known], ignore_index=True)


def load_lucid_metadata(root: Optional[PathLike] = None) -> pd.DataFrame:
    """Load LUICIDHands paired RGB + normal map metadata.

    Returns a DataFrame with the standard columns plus a ``normals_path``
    column pointing to the matching normal map image for each sample.
    Only rows where ``has_detection=true`` are included.
    """
    dataset_root = _resolve_root(root)
    lucid_root = dataset_root / "LUICIDHands"
    rgb_root = lucid_root / "rgb"
    normals_root = lucid_root / "normals"
    csv_path = lucid_root / "reference_LUCID.csv"

    empty_cols = ["source", "user_id", "age", "gender", "aspect", "image_path", "normals_path"]
    if not csv_path.exists():
        print(f"[LUICIDHands] CSV not found: {csv_path}")
        return pd.DataFrame(columns=empty_cols)

    raw_df = pd.read_csv(csv_path)
    if raw_df.empty or "name" not in raw_df.columns:
        return pd.DataFrame(columns=empty_cols)

    # Keep only detected samples.
    working_df = raw_df[raw_df["has_detection"].astype(str).str.lower() == "true"].copy()

    mask_root = lucid_root / "mask"

    def resolve_rgb(name: object) -> Optional[Path]:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        p = rgb_root / f"{name}.jpg"
        return p if p.is_file() else None

    def resolve_normals(name: object) -> Optional[Path]:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        p = normals_root / f"{name}.jpg"
        return p if p.is_file() else None

    def resolve_mask(name: object) -> Optional[Path]:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        p = mask_root / f"{name}.png"
        return p if p.is_file() else None

    working_df["image_path"] = working_df["name"].apply(resolve_rgb)
    working_df["normals_path"] = working_df["name"].apply(resolve_normals)
    working_df["mask_path"] = working_df["name"].apply(resolve_mask)
    working_df = working_df[working_df["image_path"].notna()].copy()

    working_df["aspect_norm"] = working_df["aspect"].apply(_normalise_label)
    working_df["gender_norm"] = working_df["gender"].apply(_normalise_gender)
    working_df["age_norm"] = working_df["age"].apply(
        lambda a: int(round(float(a))) if pd.notna(a) else pd.NA
    )

    df_out = pd.DataFrame(
        {
            "source": "lucid",
            "user_id": working_df["user_id"].apply(lambda uid: f"lucid_{uid}"),
            "age": working_df["age_norm"],
            "gender": working_df["gender_norm"],
            "aspect": working_df["aspect_norm"],
            "image_path": working_df["image_path"].apply(Path),
            "normals_path": working_df["normals_path"].apply(
                lambda p: Path(p) if p is not None else None
            ),
            "mask_path": working_df["mask_path"].apply(
                lambda p: Path(p) if p is not None else None
            ),
        }
    )
    df_out = df_out.reset_index(drop=True)
    n_with_normals = df_out["normals_path"].notna().sum()
    n_with_masks = df_out["mask_path"].notna().sum()
    print(
        f"LUICIDHands -> users: {df_out['user_id'].nunique()} | "
        f"images: {len(df_out)} | with normals: {n_with_normals} | with masks: {n_with_masks}"
    )
    return df_out


def load_combined_metadata(
    root: Optional[PathLike] = None,
    *,
    include_handrgbd: bool = True,
    handrgbd_include_wall3: bool = False,
    include_hagrid: bool = False,
    include_synthetic_dorsal: bool = False,
    include_synthetic_dorsal2: bool = False,
    include_prolific: bool = False,
    include_primary: bool = False,
    include_archive: bool = False,
    max_users_per_year: Optional[int] = None,
) -> pd.DataFrame:
    sources = []
    if include_handrgbd:
        sources.append(load_handrgbd_metadata(root=root, include_wall3=handrgbd_include_wall3))
    if include_primary:
        sources.insert(0, load_primary_metadata(root=root))
    if include_archive:
        insert_at = 1 if include_primary else 0
        sources.insert(insert_at, load_archive_metadata(root=root))
    if include_hagrid:
        hagrid_df = load_hagrid_stop_inverted_metadata(root=root)
        sources.append(hagrid_df)
    if include_synthetic_dorsal:
        sources.append(load_synthetic_dorsal_metadata(root=root))
    if include_synthetic_dorsal2:
        sources.append(load_synthetic_dorsal_hands2_metadata(root=root))
    if include_prolific:
        prolific_df = load_prolific_metadata(root=root)
        sources.append(prolific_df)
    if not sources:
        raise ValueError("At least one dataset must be included.")
    combined = pd.concat(sources, ignore_index=True)
    combined = combined.drop_duplicates(subset="image_path")
    if max_users_per_year:
        combined = _limit_users_per_age(combined, max_users_per_year=max_users_per_year)
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset class

class HandsDataset(Dataset):
    """Simple dataset that yields an image tensor, its label, and metadata."""

    def __init__(self, records: pd.DataFrame, transform=None):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401
        return len(self.records)

    def __getitem__(self, idx: int):  # noqa: D401
        row = self.records.iloc[idx]
        image_path: Path = row["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load image {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        label: str = row["aspect"]
        metadata = {
            "user_id": row["user_id"],
            "age": row["age"],
            "gender": row["gender"],
            "source": row["source"],
            "image_path": image_path,
        }
        return image, label, metadata


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect combined hands dataset metadata.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=f"Path to the dataset root (overrides env var {_ENV_VAR_NAME}).",
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=0,
        help="Maximum samples per user after dorsal filtering (default: 0 = disabled).",
    )
    parser.add_argument(
        "--max-samples-per-age-bin",
        type=int,
        default=200,
        help="Maximum total samples per integer age year (default: 200; set 0 to disable).",
    )
    parser.add_argument(
        "--max-users-per-year",
        type=int,
        default=0,
        help="Maximum users per age year (default: 0 = disabled).",
    )
    parser.add_argument(
        "--include-handrgbd",
        action="store_true",
        default=False,
        help="Include the HandRGBD dataset.",
    )
    parser.add_argument(
        "--include-hagrid",
        action="store_true",
        default=False,
        help="Include the HaGRIDv2 stop_inverted dataset.",
    )
    parser.add_argument(
        "--include-synthetic-dorsal",
        action="store_true",
        default=False,
        help="Include the SyntheticDorsalHands dataset.",
    )
    parser.add_argument(
        "--include-synthetic-dorsal2",
        action="store_true",
        default=False,
        help="Include the SyntheticDorsalHands2 dataset.",
    )
    parser.add_argument(
        "--include-prolific",
        action="store_true",
        default=False,
        help="Include the optional ProlificHands dataset.",
    )
    parser.add_argument(
        "--include-primary",
        action="store_true",
        default=False,
        help="Include the 11kHands primary dataset.",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        default=False,
        help="Include the archive dataset.",
    )
    args = parser.parse_args()

    combined = load_combined_metadata(
        root=args.root,
        include_handrgbd=args.include_handrgbd,
        include_hagrid=args.include_hagrid,
        include_synthetic_dorsal=args.include_synthetic_dorsal,
        include_synthetic_dorsal2=args.include_synthetic_dorsal2,
        include_prolific=args.include_prolific,
        include_primary=args.include_primary,
        include_archive=args.include_archive,
        max_users_per_year=args.max_users_per_year or None,
    )
    filtered = filter_metadata(
        combined,
        max_samples_per_user=args.max_samples_per_user or None,
        max_samples_per_age_bin=args.max_samples_per_age_bin or None,
    )
    active_root = _resolve_root(args.root)
    print(f"Using dataset root: {active_root}")
    print(f"Filtered samples: {len(filtered)}")
    print(filtered.groupby(["source", "aspect"]).size())

    # Plot age histogram for a quick sanity check.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping age histogram.")
    else:
        def _plot_histogram(
            age_series: pd.Series,
            stack_series: pd.Series,
            *,
            title: str,
            ylabel: str,
            output_path: Path,
            legend_title: str = "Source",
            colour_map: dict[str, str] | None = None,
            stack_order: list[str] | None = None,
        ) -> bool:
            if age_series.empty:
                return False
            default_colour_map = {
                "primary": "#4c72b0",
                "archive": "#dd8452",
                "handrgbd": "#55a868",
                "hagrid": "#c44e52",
                "synthetic_dorsal": "#937860",
                "synthetic_dorsal2": "#b5a48a",
                "prolific": "#8172b2",
            }
            palette = colour_map or default_colour_map
            stack_values = stack_series.fillna("unknown").astype(str)
            if stack_order:
                ordered = [label for label in stack_order if label in set(stack_values)]
                unique_stacks = ordered + [label for label in stack_values.unique() if label not in ordered]
            else:
                unique_stacks = list(stack_values.unique())

            min_age = float(np.floor(age_series.min()))
            max_age = float(np.ceil(age_series.max()))
            if min_age == max_age:
                bin_edges = np.array([min_age - 0.5, max_age + 0.5])
            else:
                bin_edges = np.arange(min_age - 0.5, max_age + 1.5, 1.0)  # one bin per year
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bar_width = 1.0

            plt.figure(figsize=(9, 5))
            cumulative = np.zeros_like(bin_centers, dtype=float)
            for stack_label in unique_stacks:
                mask = stack_values[stack_values == stack_label].index
                ages = age_series.loc[mask]
                if ages.empty:
                    continue
                counts, _ = np.histogram(ages, bins=bin_edges)
                plt.bar(
                    bin_centers,
                    counts,
                    width=bar_width * 0.9,
                    bottom=cumulative,
                    color=palette.get(stack_label, None),
                    edgecolor="black",
                    alpha=0.85,
                    label=stack_label,
                )
                cumulative = cumulative + counts
            if len(unique_stacks) > 1:
                plt.legend(title=legend_title)

            plt.xlabel("Age")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_path)
            print(f"Saved age histogram to {output_path}")
            return True

        per_user_age = (
            filtered.groupby("user_id")["age"]
                .first()
                .astype(float)
        )
        if per_user_age.empty:
            print("No age values available; histogram skipped.")
        else:
            default_colour_map = {
                "primary": "#4c72b0",
                "archive": "#dd8452",
                "handrgbd": "#55a868",
                "hagrid": "#c44e52",
                "prolific": "#8172b2",
            }
            skin_colour_map = {
                "light": "#f2d2b6",
                "tan": "#c68642",
                "dark": "#6f4e37",
                "unlabeled": "#9ea3a8",
            }
            use_skin_stacks = (
                filtered["source"].dropna().nunique() == 1
                and filtered["source"].dropna().iloc[0] == "handrgbd"
                and "skin_color" in filtered.columns
                and filtered["skin_color"].notna().any()
            )
            if use_skin_stacks:
                stack_title_prefix = "HandRGBD"
                legend_title = "Skin color"
                stack_order = ["light", "tan", "dark", "unlabeled"]
                user_stack = (
                    filtered.groupby("user_id")["skin_color"]
                    .first()
                    .fillna("unlabeled")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                sample_stack = (
                    filtered["skin_color"]
                    .fillna("unlabeled")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                plot_colour_map = skin_colour_map
            else:
                stack_title_prefix = "Hand datasets"
                legend_title = "Source"
                stack_order = None
                user_stack = filtered.groupby("user_id")["source"].first()
                sample_stack = filtered["source"]
                plot_colour_map = default_colour_map
            any_plot = False
            any_plot |= _plot_histogram(
                per_user_age,
                user_stack,
                title=f"{stack_title_prefix} age distribution (per user)",
                ylabel="User count",
                output_path=Path("age_histogram_users.png"),
                legend_title=legend_title,
                colour_map=plot_colour_map,
                stack_order=stack_order,
            )

            per_sample_age = filtered["age"].astype(float)
            any_plot |= _plot_histogram(
                per_sample_age,
                sample_stack,
                title=f"{stack_title_prefix} age distribution (per sample)",
                ylabel="Sample count",
                output_path=Path("age_histogram_samples.png"),
                legend_title=legend_title,
                colour_map=plot_colour_map,
                stack_order=stack_order,
            )

            if any_plot:
                plt.show()


if __name__ == "__main__":
    _cli_main()
