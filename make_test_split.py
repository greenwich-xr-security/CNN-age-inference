#!/usr/bin/env python3
"""Create a stratified held-out test split and save to JSON.

This script must be run ONCE before any training begins.  The resulting
test_users.json is locked — these users must never appear in train or
validation sets.  Pass the file to make_kfold_splits.py (--test-users-file)
and to train_distributed.py (--test-users-file) to enforce the exclusion.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import (
    build_user_age_strata,
    build_user_age_skin_strata,
    build_held_out_test_split,
    filter_metadata,
    normalise_user_skin_color_labels,
    save_test_split,
    summarise_user_stratification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified held-out test split for age inference."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of users to reserve as held-out test set (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        required=True,
        help="Path to save the test split JSON file (e.g. splits/test_users.json).",
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=20,
        help="Maximum samples per user after dorsal filtering (default: 20; set 0 to disable).",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable split stratification entirely.",
    )
    parser.add_argument(
        "--include-handrgbd",
        action="store_true",
        default=False,
        help="Include the HandRGBD dataset when building the split.",
    )
    parser.add_argument(
        "--include-hagrid",
        action="store_true",
        default=False,
        help="Include the HaGRIDv2 stop_inverted dataset when building the split.",
    )
    parser.add_argument(
        "--include-prolific",
        action="store_true",
        default=False,
        help="Include the optional ProlificHands dataset when building the split.",
    )
    parser.add_argument(
        "--include-synthetic-dorsal",
        action="store_true",
        default=False,
        help="Include SyntheticDorsalHands when building the split.",
    )
    parser.add_argument(
        "--include-primary",
        action="store_true",
        default=False,
        help="Include the 11kHands primary dataset when building the split.",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        default=False,
        help="Include the archive dataset when building the split.",
    )
    parser.add_argument(
        "--stratify-mode",
        type=str,
        default="age_bins",
        choices=["age_bins", "age_bins_skin_color", "adult", "none"],
        help="User-level stratification mode (default: age_bins).",
    )
    parser.add_argument(
        "--age-bin-width",
        type=int,
        default=2,
        help="Width of the fine-grained user-age bins up to the coarse-age threshold (default: 2 years).",
    )
    parser.add_argument(
        "--age-bin-coarse-start",
        type=int,
        default=51,
        help="First age that uses coarser stratification bins (default: 51, so ages <= 50 stay in 2-year bins).",
    )
    parser.add_argument(
        "--age-bin-coarse-width",
        type=int,
        default=5,
        help="Width of the coarser user-age bins from --age-bin-coarse-start onward (default: 5 years).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing test split file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root:
        set_dataset_root(args.data_root)

    out_path = Path(args.out_file)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Test split file already exists: {out_path}\n"
            "Use --overwrite to replace it (only do this before any training has started)."
        )

    metadata = filter_metadata(
        load_combined_metadata(
            root=get_dataset_root(),
            include_handrgbd=args.include_handrgbd,
            include_hagrid=args.include_hagrid,
            include_synthetic_dorsal=args.include_synthetic_dorsal,
            include_prolific=args.include_prolific,
            include_primary=args.include_primary,
            include_archive=args.include_archive,
        ),
        max_samples_per_user=args.max_samples_per_user,
    )

    stratify_mode = "none" if args.no_stratify else args.stratify_mode
    user_age = metadata.groupby("user_id")["age"].mean()
    stratify_labels = None
    age_labels = None
    skin_labels = None
    if stratify_mode in {"age_bins", "age_bins_skin_color"}:
        min_bin_users = max(2, int(math.ceil(1.0 / min(args.test_size, 1.0 - args.test_size))))
        age_labels = build_user_age_strata(
            user_age,
            fine_bin_width=args.age_bin_width,
            coarse_start_age=args.age_bin_coarse_start,
            coarse_bin_width=args.age_bin_coarse_width,
            min_count=min_bin_users,
        )
        if stratify_mode == "age_bins_skin_color":
            user_skin = (
                metadata.groupby("user_id")["skin_color"].first()
                if "skin_color" in metadata.columns
                else user_age.map(lambda _age: "unlabeled")
            )
            skin_labels = normalise_user_skin_color_labels(user_skin)
            stratify_labels = build_user_age_skin_strata(
                user_age,
                user_skin,
                fine_bin_width=args.age_bin_width,
                coarse_start_age=args.age_bin_coarse_start,
                coarse_bin_width=args.age_bin_coarse_width,
                min_count=min_bin_users,
            )
        else:
            stratify_labels = age_labels
    elif stratify_mode == "adult":
        stratify_labels = (user_age >= 18.0).map(lambda x: "adult" if x else "minor")

    train_dev_ids, test_ids = build_held_out_test_split(
        metadata,
        test_size=args.test_size,
        random_state=args.seed,
        stratify_adult=False,
        stratify_labels=stratify_labels,
    )

    saved_path = save_test_split(
        test_ids,
        out_path,
        seed=args.seed,
        test_size=args.test_size,
        stratified=stratify_mode != "none",
    )

    total_users = metadata["user_id"].nunique()
    print(f"Saved test split to: {saved_path}")
    print(
        f"Total users: {total_users} | "
        f"Test: {len(test_ids)} ({len(test_ids)/total_users:.1%}) | "
        f"Train+dev: {len(train_dev_ids)} ({len(train_dev_ids)/total_users:.1%})"
    )

    if stratify_labels is not None:
        stats_df, quality_df = summarise_user_stratification(
            stratify_labels,
            {
                "train_dev": train_dev_ids,
                "test": test_ids,
            },
        )
        stats_path = saved_path.with_name(saved_path.stem + "_strat_stats.csv")
        quality_path = saved_path.with_name(saved_path.stem + "_strat_quality.csv")
        stats_df.to_csv(stats_path, index=False)
        quality_df.to_csv(quality_path, index=False)
        print(f"Stratification mode: {stratify_mode}")
        print(f"Stratification stats written to: {stats_path}")
        print(f"Stratification quality written to: {quality_path}")
        if stratify_mode == "age_bins":
            bins = ", ".join(str(label) for label in stratify_labels.cat.categories)
            print(f"User age bins: {bins}")
        elif stratify_mode == "age_bins_skin_color" and age_labels is not None:
            bins = ", ".join(str(label) for label in age_labels.cat.categories)
            print(f"User age bins: {bins}")
        for split_name in ("train_dev", "test"):
            row = quality_df[quality_df["split"] == split_name]
            if row.empty:
                continue
            row = row.iloc[0]
            print(
                f"{split_name} — max abs frac diff: {row['max_abs_frac_diff']:.4f}, "
                f"mean abs frac diff: {row['mean_abs_frac_diff']:.4f}"
            )

        if stratify_mode == "age_bins_skin_color" and skin_labels is not None:
            skin_stats_df, skin_quality_df = summarise_user_stratification(
                skin_labels,
                {
                    "train_dev": train_dev_ids,
                    "test": test_ids,
                },
            )
            skin_stats_path = saved_path.with_name(saved_path.stem + "_skin_color_stats.csv")
            skin_quality_path = saved_path.with_name(saved_path.stem + "_skin_color_quality.csv")
            skin_stats_df.to_csv(skin_stats_path, index=False)
            skin_quality_df.to_csv(skin_quality_path, index=False)
            print(f"Skin-color stats written to: {skin_stats_path}")
            print(f"Skin-color quality written to: {skin_quality_path}")


if __name__ == "__main__":
    main()
