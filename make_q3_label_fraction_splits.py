#!/usr/bin/env python3
"""Create Q3 real-label fraction manifests from an existing real k-fold split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import (
    build_user_age_skin_strata,
    build_user_age_strata,
    build_user_skin_color_series,
    filter_metadata,
    load_kfold_splits,
    load_test_split,
    normalise_user_skin_color_labels,
    summarise_user_stratification,
)

DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Q3 label-fraction train-user manifests. The held-out test "
            "split and validation folds stay fixed; only the real fine-tuning "
            "train users are subsampled."
        )
    )
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root override.")
    parser.add_argument("--fold-file", type=str, required=True, help="Existing real k-fold JSON.")
    parser.add_argument("--test-users-file", type=str, required=True, help="Locked real test split JSON.")
    parser.add_argument("--out-file", type=str, required=True, help="Output Q3 fraction manifest JSON.")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="Real-label fractions to sample from each fold's train pool.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=0,
        help="Metadata filtering cap used only for stratification QA (0 disables; default: 0).",
    )
    parser.add_argument(
        "--max-samples-per-age-bin",
        type=int,
        default=0,
        help="Metadata filtering cap used only for stratification QA (0 disables; default: 0).",
    )
    parser.add_argument("--include-handrgbd", action="store_true", default=True, help="Include HandRGBD.")
    parser.add_argument("--exclude-handrgbd", action="store_false", dest="include_handrgbd", help="Exclude HandRGBD.")
    parser.add_argument("--include-prolific", action="store_true", default=True, help="Include ProlificHands.")
    parser.add_argument("--exclude-prolific", action="store_false", dest="include_prolific", help="Exclude ProlificHands.")
    parser.add_argument("--include-primary", action="store_true", default=False, help="Include 11kHands primary.")
    parser.add_argument("--include-archive", action="store_true", default=False, help="Include archive.")
    parser.add_argument("--include-hagrid", action="store_true", default=False, help="Include HaGRID.")
    parser.add_argument(
        "--stratify-mode",
        type=str,
        default="age_bins_skin_color",
        choices=["age_bins_skin_color", "age_bins", "adult", "none"],
        help=(
            "Preferred user-level sampling stratification. Tiny fractions "
            "automatically fall back to coarser labels when necessary."
        ),
    )
    parser.add_argument("--age-bin-width", type=int, default=2)
    parser.add_argument("--age-bin-coarse-start", type=int, default=51)
    parser.add_argument("--age-bin-coarse-width", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output JSON/CSV set.")
    return parser.parse_args()


def format_fraction(value: float) -> str:
    return f"{float(value):.2f}"


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        uid = str(value)
        if uid in seen:
            continue
        seen.add(uid)
        out.append(uid)
    return out


def build_label_sets(metadata: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.Series | None]:
    user_age = metadata.groupby("user_id")["age"].mean()
    user_skin = build_user_skin_color_series(metadata).reindex(user_age.index)
    user_skin = normalise_user_skin_color_labels(user_skin.fillna("unlabeled"))

    age_labels = build_user_age_strata(
        user_age,
        fine_bin_width=args.age_bin_width,
        coarse_start_age=args.age_bin_coarse_start,
        coarse_bin_width=args.age_bin_coarse_width,
        min_count=1,
    )
    labels: dict[str, pd.Series | None] = {
        "age_bins_skin_color": build_user_age_skin_strata(
            user_age,
            user_skin,
            fine_bin_width=args.age_bin_width,
            coarse_start_age=args.age_bin_coarse_start,
            coarse_bin_width=args.age_bin_coarse_width,
            min_count=1,
        ),
        "age_bins": age_labels,
        "adult": (user_age >= 18.0).map(lambda is_adult: "adult" if is_adult else "minor"),
        "none": None,
    }
    for mode, series in list(labels.items()):
        if series is not None:
            series.index = series.index.astype(str)
            labels[mode] = series
    return labels


def _mode_order(preferred: str) -> list[str]:
    fallbacks = {
        "age_bins_skin_color": ["age_bins_skin_color", "age_bins", "adult", "none"],
        "age_bins": ["age_bins", "adult", "none"],
        "adult": ["adult", "none"],
        "none": ["none"],
    }
    return fallbacks[preferred]


def choose_sampling_labels(
    label_sets: dict[str, pd.Series | None],
    pool_ids: list[str],
    target_n: int,
    preferred: str,
) -> tuple[str, pd.Series | None]:
    for mode in _mode_order(preferred):
        labels = label_sets[mode]
        if labels is None:
            return mode, None
        pool_labels = labels.reindex(pool_ids).dropna()
        if pool_labels.empty:
            continue
        if pool_labels.nunique() <= max(1, target_n):
            return mode, pool_labels
    return "none", None


def proportional_sample(
    pool_ids: list[str],
    target_n: int,
    *,
    rng: np.random.Generator,
    labels: pd.Series | None,
) -> list[str]:
    pool_ids = _ordered_unique(pool_ids)
    target_n = min(max(1, int(target_n)), len(pool_ids))
    if target_n == len(pool_ids):
        return pool_ids
    if labels is None:
        return sorted(rng.choice(np.asarray(pool_ids, dtype=object), size=target_n, replace=False).tolist())

    labels = labels.reindex(pool_ids).dropna()
    counts = labels.value_counts()
    counts = counts[counts > 0]
    if counts.empty:
        return sorted(rng.choice(np.asarray(pool_ids, dtype=object), size=target_n, replace=False).tolist())

    strata = list(counts.index)
    ideal = counts.astype(float) / float(counts.sum()) * target_n
    alloc = np.floor(ideal).astype(int)
    if target_n >= len(strata):
        alloc[alloc == 0] = 1
    while int(alloc.sum()) > target_n:
        candidates = [s for s in strata if alloc[s] > 0]
        victim = min(candidates, key=lambda s: (ideal[s] - alloc[s], alloc[s], str(s)))
        alloc[victim] -= 1
    while int(alloc.sum()) < target_n:
        candidates = [s for s in strata if alloc[s] < counts[s]]
        winner = max(candidates, key=lambda s: (ideal[s] - alloc[s], counts[s], str(s)))
        alloc[winner] += 1

    selected: list[str] = []
    for stratum in strata:
        n = int(alloc[stratum])
        if n <= 0:
            continue
        ids = labels[labels == stratum].index.to_numpy(dtype=object)
        selected.extend(rng.choice(ids, size=n, replace=False).tolist())
    return sorted(str(uid) for uid in selected)


def add_qa_rows(
    *,
    label_series: pd.Series,
    split_to_user_ids: dict[str, list[str]],
    fold_index: int,
    fraction_key: str,
    stats_parts: list[pd.DataFrame],
    quality_parts: list[pd.DataFrame],
) -> None:
    stats_df, quality_df = summarise_user_stratification(label_series, split_to_user_ids)
    stats_df.insert(0, "label_fraction", fraction_key)
    stats_df.insert(0, "fold", fold_index)
    quality_df.insert(0, "label_fraction", fraction_key)
    quality_df.insert(0, "fold", fold_index)
    stats_parts.append(stats_df)
    quality_parts.append(quality_df)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_file)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path}")

    if args.data_root:
        set_dataset_root(args.data_root)

    metadata = filter_metadata(
        load_combined_metadata(
            root=get_dataset_root(),
            include_handrgbd=args.include_handrgbd,
            include_hagrid=args.include_hagrid,
            include_synthetic_dorsal=False,
            include_synthetic_dorsal2=False,
            include_prolific=args.include_prolific,
            include_primary=args.include_primary,
            include_archive=args.include_archive,
        ),
        max_samples_per_user=args.max_samples_per_user or None,
        max_samples_per_age_bin=args.max_samples_per_age_bin or None,
    )
    metadata["user_id"] = metadata["user_id"].astype(str)

    test_data = load_test_split(args.test_users_file)
    test_ids = _ordered_unique(test_data["test_user_ids"])
    metadata = metadata[~metadata["user_id"].isin(set(test_ids))].copy()

    fold_data = load_kfold_splits(args.fold_file)
    folds = [_ordered_unique(fold) for fold in fold_data["folds"]]
    all_fold_ids = _ordered_unique(uid for fold in folds for uid in fold)
    available_ids = set(metadata["user_id"].unique())
    missing = [uid for uid in all_fold_ids if uid not in available_ids]
    if missing:
        raise ValueError(f"{len(missing)} fold users are missing from filtered metadata; first: {missing[:5]}")

    label_sets = build_label_sets(metadata, args)
    age_labels = label_sets["age_bins"]
    skin_labels = normalise_user_skin_color_labels(build_user_skin_color_series(metadata))
    assert age_labels is not None

    fractions = sorted({float(frac) for frac in args.fractions})
    if not fractions or any(frac <= 0.0 or frac > 1.0 for frac in fractions):
        raise ValueError("Fractions must be in the interval (0, 1].")

    manifest_folds: list[dict] = []
    strat_stats_parts: list[pd.DataFrame] = []
    strat_quality_parts: list[pd.DataFrame] = []
    skin_stats_parts: list[pd.DataFrame] = []
    skin_quality_parts: list[pd.DataFrame] = []

    for fold_index, val_ids in enumerate(folds):
        train_pool_ids = _ordered_unique(
            uid
            for idx, fold in enumerate(folds)
            if idx != fold_index
            for uid in fold
        )
        fold_payload = {
            "fold_index": fold_index,
            "val_user_ids": val_ids,
            "train_pool_user_ids": train_pool_ids,
            "label_fractions": {},
        }
        for fraction in fractions:
            fraction_key = format_fraction(fraction)
            target_n = len(train_pool_ids) if fraction >= 1.0 else max(1, int(round(len(train_pool_ids) * fraction)))
            mode_used, sampling_labels = choose_sampling_labels(
                label_sets,
                train_pool_ids,
                target_n,
                args.stratify_mode,
            )
            rng = np.random.default_rng(args.seed + fold_index * 1009 + int(round(fraction * 10000)))
            selected_ids = proportional_sample(train_pool_ids, target_n, rng=rng, labels=sampling_labels)
            fold_payload["label_fractions"][fraction_key] = {
                "fraction": fraction,
                "target_train_users": target_n,
                "train_pool_users": len(train_pool_ids),
                "selected_train_users": len(selected_ids),
                "sampling_stratify_mode": mode_used,
                "train_user_ids": selected_ids,
            }

            split_to_user_ids = {
                "train_pool": train_pool_ids,
                "train_fraction": selected_ids,
                "val": val_ids,
            }
            add_qa_rows(
                label_series=age_labels,
                split_to_user_ids=split_to_user_ids,
                fold_index=fold_index,
                fraction_key=fraction_key,
                stats_parts=strat_stats_parts,
                quality_parts=strat_quality_parts,
            )
            add_qa_rows(
                label_series=skin_labels,
                split_to_user_ids=split_to_user_ids,
                fold_index=fold_index,
                fraction_key=fraction_key,
                stats_parts=skin_stats_parts,
                quality_parts=skin_quality_parts,
            )
        manifest_folds.append(fold_payload)

    payload = {
        "schema": "q3_label_fraction_splits_v1",
        "seed": int(args.seed),
        "source_fold_file": str(Path(args.fold_file)),
        "source_test_users_file": str(Path(args.test_users_file)),
        "test_user_ids": test_ids,
        "fractions": [format_fraction(frac) for frac in fractions],
        "sampling": {
            "preferred_stratify_mode": args.stratify_mode,
            "note": "Tiny fractions fall back to coarser stratification when the requested labels outnumber selected users.",
        },
        "real_sources": {
            "handrgbd": bool(args.include_handrgbd),
            "prolific": bool(args.include_prolific),
            "hagrid": bool(args.include_hagrid),
            "primary": bool(args.include_primary),
            "archive": bool(args.include_archive),
        },
        "metadata_filters": {
            "max_samples_per_user": int(args.max_samples_per_user),
            "max_samples_per_age_bin": int(args.max_samples_per_age_bin),
        },
        "folds": manifest_folds,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    stem = out_path.with_suffix("")
    pd.concat(strat_stats_parts, ignore_index=True).to_csv(f"{stem}_strat_stats.csv", index=False)
    pd.concat(strat_quality_parts, ignore_index=True).to_csv(f"{stem}_strat_quality.csv", index=False)
    pd.concat(skin_stats_parts, ignore_index=True).to_csv(f"{stem}_skin_color_stats.csv", index=False)
    pd.concat(skin_quality_parts, ignore_index=True).to_csv(f"{stem}_skin_color_quality.csv", index=False)

    print(f"Saved Q3 label-fraction manifest to: {out_path}")
    print(f"Fractions: {', '.join(payload['fractions'])}")
    print(f"Folds: {len(manifest_folds)}")
    print(f"QA CSV prefix: {stem}")


if __name__ == "__main__":
    main()
