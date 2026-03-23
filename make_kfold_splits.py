#!/usr/bin/env python3
"""Create reproducible k-fold user splits and save to JSON."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import build_kfold_user_splits, filter_metadata, load_test_split, save_kfold_splits


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
        "--stratify-adult",
        action="store_true",
        help="Stratify folds by adult/minor (>=18 vs <18) based on user mean age.",
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

    stratify_col = None
    if args.stratify_adult:
        user_age = metadata.groupby("user_id")["age"].mean()
        user_class = (user_age >= 18.0).map(lambda x: "adult" if x else "minor")
        metadata = metadata.merge(user_class.rename("adult_flag"), on="user_id", how="left")
        stratify_col = "adult_flag"
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
    if args.stratify_adult:
        # compute adult/minor counts per fold
        user_to_class = metadata.drop_duplicates(subset="user_id").set_index("user_id")["adult_flag"]
        rows = ["fold,adults,adults_frac,miners,miners_frac,total"]
        for idx, fold in enumerate(folds):
            labels = user_to_class.reindex(fold)
            adults = int((labels == "adult").sum())
            minors = int((labels == "minor").sum())
            total = len(fold)
            adults_frac = adults / total if total else 0
            minors_frac = minors / total if total else 0
            rows.append(f"{idx},{adults},{adults_frac:.4f},{minors},{minors_frac:.4f},{total}")
        stats_path = saved_path.with_name(saved_path.stem + "_strat_stats.csv")
        stats_path.write_text("\n".join(rows), encoding="utf-8")
        print(f"Adult/minor stratification stats written to: {stats_path}")


if __name__ == "__main__":
    main()
