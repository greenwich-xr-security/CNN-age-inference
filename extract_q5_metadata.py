#!/usr/bin/env python3
"""
extract_q5_metadata.py — Q5 per-image metadata extraction.

Replicates evaluate_test.py's metadata loading order exactly (HandRGBD first,
ProlificHands second, same landmark-validity filtering, same dorsal/age filters),
then computes ITA from each image and assembles sex, lighting, and skin_color.

Output: runs/q5_test_image_metadata.csv with one row per test image, in the
same row order as test_predictions_raw_ddp.npz across all Q4 arms.

Usage (on HPC):
    source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
    conda activate greenwich-xr-security
    cd ~/CNN-age-inference
    python extract_q5_metadata.py

Verify afterwards:
    python -c "
    import numpy as np, pandas as pd
    npz = np.load('runs/minpair_seed43_real_from_real_lr2e5_v2s_384/fold_0/test_predictions_raw_ddp.npz', allow_pickle=True)
    meta = pd.read_csv('runs/q5_test_image_metadata.csv')
    assert len(meta) == len(npz['user_ids']), f'Row count mismatch: {len(meta)} vs {len(npz[\"user_ids\"])}'
    assert (meta['user_id'].values == npz['user_ids']).all(), 'user_id order mismatch'
    print('OK — metadata rows match NPZ order')
    "
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Paths — edit if run from a different working directory
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/rb3434w/HandsDatasets")
TEST_SPLIT = Path("/home/rb3434w/CNN-age-inference/splits/test_users_uncapped_20pct_seed42.json")
OUTPUT_CSV = Path("/home/rb3434w/CNN-age-inference/runs/q5_test_image_metadata.csv")

# ---------------------------------------------------------------------------
# ITA computation
# ---------------------------------------------------------------------------
_M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_D65_REF = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def compute_ita(
    image_path: Path,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_path: Optional[Path] = None,
) -> float:
    """Return ITA (degrees) computed only from foreground (hand) pixels.

    Foreground is determined by:
    - ``mask_path`` (binary PNG, white = foreground) when provided, or
    - non-black pixels (any channel > 10/255) for pre-masked rgb_masked images.

    Uses the landmark-derived tight bbox when available to limit the crop region
    before applying the mask.  Returns NaN if fewer than 100 foreground pixels
    survive masking, or on any loading / computation failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            pad_x = max(1, int((xmax - xmin) * 0.10))
            pad_y = max(1, int((ymax - ymin) * 0.10))
            xmin = max(0, xmin - pad_x)
            ymin = max(0, ymin - pad_y)
            xmax = min(w, xmax + pad_x)
            ymax = min(h, ymax + pad_y)
            crop_box = (xmin, ymin, xmax, ymax)
            img = img.crop(crop_box)
        else:
            crop_box = None

        arr = np.asarray(img, dtype=np.float64) / 255.0  # (H, W, 3)

        # --- Build foreground boolean mask (H×W) ---
        if mask_path is not None and mask_path.is_file():
            mask_img = Image.open(mask_path).convert("L")
            if crop_box is not None:
                mask_img = mask_img.crop(crop_box)
            fg = np.asarray(mask_img, dtype=np.uint8) > 128
        else:
            # rgb_masked images have background set to exactly black
            fg = np.any(arr > (10.0 / 255.0), axis=-1)

        fg_flat = fg.reshape(-1)
        arr_flat = arr.reshape(-1, 3)
        skin_pixels = arr_flat[fg_flat]  # (N_fg, 3)

        if len(skin_pixels) < 100:
            return float("nan")

        # sRGB -> linear light
        linear = np.where(
            skin_pixels <= 0.04045,
            skin_pixels / 12.92,
            ((skin_pixels + 0.055) / 1.055) ** 2.4,
        )
        # Linear RGB -> XYZ (D65)
        xyz = linear @ _M_RGB_TO_XYZ.T
        xyz /= _D65_REF

        # XYZ -> L*a*b*
        eps, kappa = 0.008856, 903.3
        f = np.where(xyz > eps, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
        L_star = 116.0 * f[:, 1] - 16.0
        b_star = 200.0 * (f[:, 1] - f[:, 2])

        L_mean = float(np.mean(L_star))
        b_mean = float(np.mean(b_star))
        return float(np.degrees(np.arctan2(L_mean - 50.0, b_mean)))
    except Exception as exc:  # noqa: BLE001
        print(f"  [ITA] Failed for {image_path}: {exc}")
        return float("nan")


def ita_to_chardon3(ita: float) -> str:
    """Map ITA (degrees) to Chardon 3-group label."""
    if math.isnan(ita):
        return "unknown"
    if ita > 41.0:
        return "light"
    if ita > -10.0:
        return "medium"
    return "dark"


# ---------------------------------------------------------------------------
# Landmark parsing (mirrors dataset/hand_metadata.py exactly)
# ---------------------------------------------------------------------------

def _parse_landmarks(raw) -> Optional[Tuple[Tuple[float, float], ...]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
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
    if not isinstance(value, (list, tuple)) or len(value) != 21:
        return None
    parsed = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        parsed.append((x, y))
    return tuple(parsed)


def _landmarks_to_tight_bbox(landmarks) -> Optional[Tuple[int, int, int, int]]:
    if not landmarks:
        return None
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x1, x2 = int(math.floor(min(xs))), int(math.ceil(max(xs)))
    y1, y2 = int(math.floor(min(ys))), int(math.ceil(max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Label normalisation (mirrors dataset/hand_metadata.py)
# ---------------------------------------------------------------------------
_VALID_LABELS = {"dorsal left", "dorsal right", "palmar left", "palmar right"}
_LABEL_ALIASES = {
    "dorsal_left": "dorsal left", "dorsal-right": "dorsal right",
    "palmar_left": "palmar left", "palmar-right": "palmar right",
    "right dorsal": "dorsal right", "left dorsal": "dorsal left",
    "right palmar": "palmar right", "left palmar": "palmar left",
    "palmer left": "palmar left", "palmer right": "palmar right",
}


def _normalise_label(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    label = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    label = " ".join(label.split())
    label = _LABEL_ALIASES.get(label, label)
    tokens = label.split()
    side = surface = None
    for tok in tokens:
        if tok in ("left", "right"):
            side = tok
        elif tok in ("dorsal", "palmar", "palmer", "palm"):
            surface = "palmar" if tok in ("palmer", "palm") else tok
    if surface and side:
        candidate = f"{surface} {side}"
        if candidate in _VALID_LABELS:
            return candidate
    return label if label in _VALID_LABELS else None


def _normalise_gender(raw) -> Optional[str]:
    mapping = {"f": "female", "female": "female", "m": "male", "male": "male"}
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    return mapping.get(str(raw).strip().lower(), None)


# ---------------------------------------------------------------------------
# HandRGBD loader  (mirrors load_handrgbd_metadata, include_wall3=False)
# ---------------------------------------------------------------------------

def load_handrgbd(data_root: Path) -> pd.DataFrame:
    hand_root = data_root / "handRGBD"
    rgb_root = hand_root / "rgb_jpg"
    if not rgb_root.exists():
        rgb_root = hand_root / "rgb"
    ref_csv = hand_root / "reference_table.csv"
    if not ref_csv.exists():
        print("[handRGBD] reference_table.csv not found")
        return pd.DataFrame()

    raw = pd.read_csv(ref_csv)
    print(f"[handRGBD] Loaded {len(raw)} rows from reference_table.csv")

    # Parse wall label and drop wall-3
    def parse_wall(name_val):
        if pd.isna(name_val):
            return None
        m = re.search(r"wall[-_\s]*(\d+)", str(name_val), re.IGNORECASE)
        return int(m.group(1)) if m else None

    raw["wall_label"] = raw["name"].apply(parse_wall)
    raw = raw[raw["wall_label"].ne(3)]

    # Parse lighting label from 'lights' column or filename
    def normalise_lights(val):
        if pd.isna(val):
            return None
        text = str(val).strip().lower()
        if text in ("on", "1", "true", "yes"):
            return "on"
        if text in ("off", "0", "false", "no"):
            return "off"
        return None

    def parse_lights_from_name(name_val):
        if pd.isna(name_val):
            return None
        m = re.search(r"lights[-_\s]*(on|off)", str(name_val), re.IGNORECASE)
        return m.group(1).lower() if m else None

    if "lights" in raw.columns:
        raw["lights_label"] = raw["lights"].apply(normalise_lights)
    else:
        raw["lights_label"] = None
    if raw["lights_label"].isna().all():
        raw["lights_label"] = raw["name"].apply(parse_lights_from_name)

    # Normalise aspect
    raw["aspect_norm"] = raw["aspect"].apply(_normalise_label)
    raw = raw[raw["aspect_norm"].notna()]

    # Resolve image paths
    def resolve_path(name_val):
        if pd.isna(name_val):
            return None
        name_str = str(name_val).strip()
        for ext in (".jpg", ".png"):
            candidate = rgb_root / f"{name_str}{ext}"
            if candidate.is_file():
                return candidate
        return None

    print("[handRGBD] Resolving image paths...")
    raw["image_path"] = raw["name"].apply(resolve_path)
    raw = raw[raw["image_path"].notna()]

    # Skin colour from manifest_with_ita.csv
    raw["skin_color"] = None
    manifest_path = hand_root / "patches" / "manifest_with_ita.csv"
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path, usecols=["user_id", "skin color"])
            manifest = manifest.rename(columns={"skin color": "skin_color"})
            manifest["user_id"] = manifest["user_id"].astype(str).str.strip()
            manifest["skin_color"] = (
                manifest["skin_color"].astype(str).str.strip().str.lower()
                .replace({"nan": None, "none": None, "": None})
            )
            manifest = manifest.drop_duplicates(subset="user_id", keep="first")
            raw["_uid_str"] = raw["user_id"].astype(str).str.strip()
            raw = raw.merge(manifest, left_on="_uid_str", right_on="user_id",
                            how="left", suffixes=("", "_m"))
            raw["skin_color"] = raw["skin_color_m"]
            raw = raw.drop(columns=[c for c in ("_uid_str", "user_id_m", "skin_color_m") if c in raw.columns])
        except Exception as exc:  # noqa: BLE001
            print(f"[handRGBD] manifest load failed: {exc}")

    # Gender and age
    if "gender" in raw.columns:
        raw["gender_norm"] = raw["gender"].apply(_normalise_gender)
    else:
        raw["gender_norm"] = None
    if "age" in raw.columns:
        raw["age_norm"] = pd.to_numeric(raw["age"], errors="coerce").apply(
            lambda a: int(round(a)) if pd.notna(a) else pd.NA
        )
    else:
        raw["age_norm"] = pd.NA

    # Landmarks (from CSV, no MediaPipe)
    raw["landmarks"] = raw["rgb_landmarks"].apply(_parse_landmarks) if "rgb_landmarks" in raw.columns else None
    before = len(raw)
    raw = raw[raw["landmarks"].notna()].copy()
    print(f"[handRGBD] Dropped {before - len(raw)} rows with missing landmarks")

    raw["bbox_tuple"] = raw["landmarks"].apply(_landmarks_to_tight_bbox)
    raw = raw[raw["bbox_tuple"].notna()].copy()

    df_out = pd.DataFrame({
        "source": "handrgbd",
        "user_id": raw["user_id"].apply(lambda uid: f"handrgbd_{uid}"),
        "age": raw["age_norm"],
        "gender": raw["gender_norm"],
        "aspect": raw["aspect_norm"],
        "image_path": raw["image_path"],
        "bbox": raw["bbox_tuple"],
        "wall_label": raw["wall_label"],
        "lights_label": raw["lights_label"],
        "skin_color": raw["skin_color"],
    }).reset_index(drop=True)

    print(f"[handRGBD] Final: {df_out['user_id'].nunique()} users, {len(df_out)} images")
    return df_out


# ---------------------------------------------------------------------------
# ProlificHands loader  (mirrors load_prolific_metadata)
# ---------------------------------------------------------------------------

def _resolve_relative_path(base: Path, rel) -> Optional[Path]:
    if rel is None or (isinstance(rel, float) and math.isnan(rel)):
        return None
    rel_str = str(rel).strip()
    if not rel_str:
        return None
    rel_str = rel_str.replace("\\", "/")
    candidate = base / rel_str
    return candidate if candidate.is_file() else None


def load_prolific(data_root: Path) -> pd.DataFrame:
    prolific_root = data_root / "ProlificHands"
    ref_csv = prolific_root / "reference_prolific.csv"
    rgb_masked_root = prolific_root / "rgb_masked"
    if not ref_csv.exists():
        print("[ProlificHands] reference_prolific.csv not found")
        return pd.DataFrame()

    raw = pd.read_csv(ref_csv)
    print(f"[ProlificHands] Loaded {len(raw)} rows from reference_prolific.csv")

    raw["aspect_norm"] = raw["aspect"].apply(_normalise_label)
    raw = raw[raw["aspect_norm"].notna()].copy()

    def resolve_image(row):
        name = str(row.get("name", "")).strip()
        if name and rgb_masked_root.exists():
            masked = rgb_masked_root / f"{name}.png"
            if masked.is_file():
                return masked
        return _resolve_relative_path(prolific_root, row.get("rgb_path"))

    print("[ProlificHands] Resolving image paths...")
    raw["image_path"] = raw.apply(resolve_image, axis=1)
    raw = raw[raw["image_path"].notna()].copy()

    raw["gender_norm"] = raw["gender"].apply(_normalise_gender)
    raw["age_norm"] = pd.to_numeric(raw["age"], errors="coerce").apply(
        lambda a: int(round(a)) if pd.notna(a) else pd.NA
    )

    raw["landmarks"] = raw["rgb_landmarks"].apply(_parse_landmarks) if "rgb_landmarks" in raw.columns else None
    before = len(raw)
    raw = raw[raw["landmarks"].notna()].copy()
    print(f"[ProlificHands] Dropped {before - len(raw)} rows with missing landmarks")

    raw["bbox_tuple"] = raw["landmarks"].apply(_landmarks_to_tight_bbox)
    raw = raw[raw["bbox_tuple"].notna()].copy()

    skin_color = pd.NA
    if "skin_color" in raw.columns:
        skin_color = (
            raw["skin_color"].astype("string").str.strip().str.lower()
            .replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
        )
    skin_tone_self = pd.NA
    if "skin_tone_self_reported" in raw.columns:
        skin_tone_self = raw["skin_tone_self_reported"]

    df_out = pd.DataFrame({
        "source": "prolific",
        "user_id": raw["participant_id"].apply(lambda uid: f"prolific_{uid}"),
        "age": raw["age_norm"],
        "gender": raw["gender_norm"],
        "aspect": raw["aspect_norm"],
        "image_path": raw["image_path"],
        "bbox": raw["bbox_tuple"],
        "wall_label": None,
        "lights_label": None,
        "skin_color": skin_color,
        "skin_tone_self_reported": skin_tone_self,
    }).reset_index(drop=True)

    print(f"[ProlificHands] Final: {df_out['user_id'].nunique()} users, {len(df_out)} images")
    return df_out


# ---------------------------------------------------------------------------
# Combine — mirrors load_combined_metadata + filter_metadata + test-user filter
# ---------------------------------------------------------------------------

def build_test_metadata(data_root: Path, test_ids: set[str]) -> pd.DataFrame:
    hrgbd = load_handrgbd(data_root)
    prolific = load_prolific(data_root)

    combined = pd.concat([hrgbd, prolific], ignore_index=True)
    combined = combined.drop_duplicates(subset="image_path").reset_index(drop=True)

    # filter_metadata: dorsal + known age (no caps since max=0 in all Q4 runs)
    combined = combined[combined["aspect"].str.contains("dorsal", case=False, na=False)]
    combined = combined[combined["age"].notna()].copy()
    combined["age"] = combined["age"].astype(float)

    # Filter to test users
    combined = combined[combined["user_id"].astype(str).isin(test_ids)].copy()

    # AgeDataset also filters dorsal (redundant here but keep for parity)
    combined = combined[combined["aspect"].str.contains("dorsal", case=False, na=False)]
    combined = combined.reset_index(drop=True)

    print(f"\nTest set metadata: {len(combined)} images from {combined['user_id'].nunique()} users")
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Test split : {TEST_SPLIT}")
    print(f"Data root  : {DATA_ROOT}")
    print(f"Output     : {OUTPUT_CSV}\n")

    with open(TEST_SPLIT) as fh:
        test_data = json.load(fh)
    test_ids = {str(uid) for uid in test_data["test_user_ids"]}
    print(f"Test users in split JSON: {len(test_ids)}")

    meta = build_test_metadata(DATA_ROOT, test_ids)

    if len(meta) != 1624:
        print(f"\nWARNING: expected 1624 rows, got {len(meta)}. "
              "Row order may not match NPZ. Investigate before proceeding.")
    else:
        print("Row count matches expected 1624. ")

    # ------------------------------------------------------------------
    # Compute ITA for each image — using binary masks to exclude background
    # HandRGBD: mask at rgb_mask/<stem>.png
    # ProlificHands: rgb_masked images have black background → non-black detection
    # ------------------------------------------------------------------
    hand_root = DATA_ROOT / "handRGBD"
    rgb_mask_root = hand_root / "rgb_mask"

    def infer_mask_path(image_path: Path, source: str) -> Optional[Path]:
        if source == "handrgbd":
            mask = rgb_mask_root / (image_path.stem + ".png")
            return mask if mask.is_file() else None
        # prolific: non-black detection in compute_ita (no separate mask file needed)
        return None

    print("\nComputing ITA scores (foreground pixels only)...")
    ita_scores = []
    for i, row in meta.iterrows():
        if i % 100 == 0:
            print(f"  {i}/{len(meta)}", end="\r", flush=True)
        mask_p = infer_mask_path(Path(row["image_path"]), row["source"])
        ita = compute_ita(Path(row["image_path"]), bbox=row.get("bbox"), mask_path=mask_p)
        ita_scores.append(ita)
    print(f"  {len(meta)}/{len(meta)} done.      ")

    meta["ITA_score"] = ita_scores
    meta["ITA_group"] = meta["ITA_score"].apply(ita_to_chardon3)

    # ------------------------------------------------------------------
    # Assemble output columns
    # ------------------------------------------------------------------
    out = pd.DataFrame({
        "row_index": range(len(meta)),
        "user_id": meta["user_id"],
        "age": meta["age"],
        "source": meta["source"],
        "gender": meta["gender"],
        "lights_label": meta["lights_label"],        # on/off for handRGBD; NaN for prolific
        "skin_color_metadata": meta["skin_color"],   # from manifest / reference CSV
        "ITA_score": meta["ITA_score"],
        "ITA_group": meta["ITA_group"],
        "image_path": meta["image_path"].astype(str),
        "bbox": meta["bbox"].astype(str),
    })
    if "skin_tone_self_reported" in meta.columns:
        out["skin_tone_self_reported"] = meta["skin_tone_self_reported"]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- ITA group counts ---")
    print(out["ITA_group"].value_counts().to_string())
    print("\n--- Lighting (handRGBD only) ---")
    print(out["lights_label"].value_counts(dropna=False).to_string())
    print("\n--- Gender ---")
    print(out["gender"].value_counts(dropna=False).to_string())
    print("\n--- Skin color (from metadata) ---")
    print(out["skin_color_metadata"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
