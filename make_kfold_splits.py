#!/usr/bin/env python3
"""Create reproducible k-fold user splits and save to JSON."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import (
    build_kfold_user_splits,
    build_user_age_strata,
    filter_metadata,
    load_test_split,
    save_kfold_splits,
    summarise_user_stratification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create k-fold user splits for age inference.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to the dataset root directory. Overrides the default or env var.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of folds to create.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold assignment (default: 42).",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        required=True,
        help="Path to save the fold JSON file.",
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=16,
        help="Maximum samples per user after dorsal filtering (default: 16; set 0 to disable).",
    )
    parser.add_argument(
        "--test-users-file",
        type=str,
        default=None,
        help=(
            "Path to a held-out test split JSON (from make_test_split.py). "
            "Test users will be excluded from all folds."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing fold file.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable fold stratification entirely.",
    )
    parser.add_argument(
        "--stratify-mode",
        type=str,
        default="age_bins",
        choices=["age_bins", "adult", "none"],
        help="User-level fold stratification mode (default: age_bins).",
    )
    parser.add_argument(
        "--stratify-adult",
        action="store_true",
        help="Deprecated alias for --stratify-mode adult.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root:
        set_dataset_root(args.data_root)

    out_path = Path(args.out_file)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Fold file already exists: {out_path}")

    metadata = filter_metadata(
        load_combined_metadata(root=get_dataset_root()),
        max_samples_per_user=args.max_samples_per_user,
    )

    if args.test_users_file:
        test_data = load_test_split(args.test_users_file)
        test_ids = {str(uid) for uid in test_data["test_user_ids"]}
        before = metadata["user_id"].nunique()
        metadata = metadata[~metadata["user_id"].astype(str).isin(test_ids)]
        after = metadata["user_id"].nunique()
        print(
            f"Excluded {before - after} held-out test users "
            f"(from {args.test_users_file}). Remaining for k-fold: {after}"
        )

    stratify_mode = "none" if args.no_stratify else ("adult" if args.stratify_adult else args.stratify_mode)
    user_age = metadata.groupby("user_id")["age"].mean()
    stratify_labels = None
    stratify_col = None
    if stratify_mode == "adult":
        stratify_labels = (user_age >= 18.0).map(lambda x: "adult" if x else "minor")
    elif stratify_mode == "age_bins":
        stratify_labels = build_user_age_strata(
            user_age,
            fine_bin_width=args.age_bin_width,
            coarse_start_age=args.age_bin_coarse_start,
            coarse_bin_width=args.age_bin_coarse_width,
            min_count=args.k,
        )

    if stratify_labels is not None:
        metadata = metadata.merge(
            stratify_labels.rename("stratify_label").reset_index(),
            on="user_id",
            how="left",
        )
        stratify_col = "stratify_label"

    folds = build_kfold_user_splits(
        metadata,
        k=args.k,
        random_state=args.seed,
        stratify_on=stratify_col,
    )
    saved_path = save_kfold_splits(
        folds,
        out_path,
        seed=args.seed,
    )

    total_users = metadata["user_id"].nunique()
    fold_sizes = [len(fold) for fold in folds]
    print(f"Saved k-fold splits to: {saved_path}")
    print(f"Total users: {total_users} | Folds: {len(folds)} | Sizes: {fold_sizes}")
    if stratify_labels is not None:
        stats_df, quality_df = summarise_user_stratification(
            stratify_labels,
            {f"fold_{idx}": fold for idx, fold in enumerate(folds)},
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
        for idx in range(len(folds)):
            row = quality_df[quality_df["split"] == f"fold_{idx}"]
            if row.empty:
                continue
            row = row.iloc[0]
            print(
                f"fold_{idx} — max abs frac diff: {row['max_abs_frac_diff']:.4f}, "
                f"mean abs frac diff: {row['mean_abs_frac_diff']:.4f}"
            )


if __name__ == "__main__":
    main()
