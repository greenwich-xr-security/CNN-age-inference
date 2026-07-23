"""Create the locked, image-level synthetic hold-out for Question 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


AGE_GROUPS = (
    ("10-12", 10, 12),
    ("13-15", 13, 15),
    ("16-17", 16, 17),
    ("18-19", 18, 19),
    ("20-24", 20, 24),
    ("25-29", 25, 29),
    ("30-39", 30, 39),
    ("40-49", 40, 49),
    ("50-59", 50, 59),
    ("60+", 60, None),
)

SKIN_LABEL_MAP = {
    "very_light": "light",
    "light": "light",
    "intermediate": "tan",
    "tan_brown": "dark",
    "dark": "dark",
}


def _balanced_strata_sample(frame: pd.DataFrame, *, count: int, seed: int) -> pd.DataFrame:
    """Select ``count`` images proportionally across gender/skin strata."""
    strata_counts = frame["_stratum"].value_counts().sort_index()
    exact = strata_counts * (count / len(frame))
    quotas = exact.astype(int)
    remainder = count - int(quotas.sum())
    fractions = (exact - quotas).sort_values(ascending=False, kind="stable")
    for stratum in fractions.index[:remainder]:
        quotas.loc[stratum] += 1

    selections: list[pd.DataFrame] = []
    for index, (stratum, quota) in enumerate(quotas.items()):
        if quota:
            group = frame[frame["_stratum"] == stratum]
            selections.append(group.sample(n=int(quota), random_state=seed + index))
    selected = pd.concat(selections, ignore_index=True)
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible images available; expected {count}.")
    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_split(data_root: Path, output_path: Path, *, seed: int, per_age_group: int) -> dict:
    synthetic_root = data_root / "SyntheticDorsalHands"
    manifest_path = synthetic_root / "reference_synthetic.csv"
    records = pd.read_csv(manifest_path)
    required = {"sample_id", "image_path", "age", "gender", "skin_color", "aspect"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Synthetic manifest is missing columns: {sorted(missing)}")

    records = records.copy()
    records["age"] = pd.to_numeric(records["age"], errors="coerce")
    records["image_path_abs"] = records["image_path"].map(lambda value: synthetic_root / str(value))
    records = records[
        records["aspect"].astype(str).str.contains("dorsal", case=False, na=False)
        & records["age"].notna()
        & records["image_path_abs"].map(Path.is_file)
    ].copy()
    records["age"] = records["age"].round().astype(int)
    records["gender"] = records["gender"].astype(str).str.strip().str.lower()
    sample_tokens = records["sample_id"].astype(str).str.extract(
        r"^job\d+_age_\d+_(?:female|male)_(?P<skin_label>.+)_\d+$"
    )
    records["skin_color"] = sample_tokens["skin_label"].map(SKIN_LABEL_MAP).fillna("unlabeled")
    records["_stratum"] = records["gender"] + "|" + records["skin_color"]

    selections: list[pd.DataFrame] = []
    age_group_summary: dict[str, dict[str, int]] = {}
    for index, (label, lower, upper) in enumerate(AGE_GROUPS):
        mask = records["age"] >= lower
        if upper is not None:
            mask &= records["age"] <= upper
        candidates = records.loc[mask]
        if len(candidates) < per_age_group:
            raise ValueError(f"Age group {label} has {len(candidates)} images; needs {per_age_group}.")
        chosen = _balanced_strata_sample(candidates, count=per_age_group, seed=seed + index)
        chosen = chosen.assign(age_group=label)
        selections.append(chosen)
        age_group_summary[label] = {
            "available_images": int(len(candidates)),
            "selected_images": int(len(chosen)),
        }

    selected = pd.concat(selections, ignore_index=True)
    if selected["sample_id"].duplicated().any():
        raise ValueError("Synthetic test selection contains duplicate sample IDs.")

    sample_records = [
        {
            "sample_id": str(row.sample_id),
            "user_id": f"synthetic_{row.sample_id}",
            "age": int(row.age),
            "age_group": str(row.age_group),
            "gender": str(row.gender),
            "skin_color": str(row.skin_color),
            "image_path": str(row.image_path),
        }
        for row in selected.itertuples(index=False)
    ]
    payload = {
        "schema_version": 1,
        "split_format": "synthetic_image_holdout",
        "seed": int(seed),
        "description": (
            "Locked image-level synthetic hold-out for Question 2. SyntheticDorsalHands "
            "has no repeated synthetic-person identity, so this is image-disjoint rather "
            "than user- or provenance-disjoint."
        ),
        "selection_method": {
            "age_groups": "10-12, 13-15, 16-17, 18-19, 20-24, 25-29, 30-39, 40-49, 50-59, 60+",
            "target_images_per_age_group": int(per_age_group),
            "secondary_strata": "proportional gender x skin_color",
        },
        "age_groups": age_group_summary,
        "test_user_ids": [record["user_id"] for record in sample_records],
        "samples": sample_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, default=Path("splits/synthetic_test_images.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-age-group", type=int, default=15)
    args = parser.parse_args()
    payload = build_split(args.data_root, args.out_file, seed=args.seed, per_age_group=args.per_age_group)
    print(f"Wrote {len(payload['test_user_ids'])} synthetic test images to {args.out_file}")


if __name__ == "__main__":
    main()
