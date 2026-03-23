#!/usr/bin/env python3
"""Create a stratified held-out test split and save to JSON.

This script must be run ONCE before any training begins.  The resulting
test_users.json is locked — these users must never appear in train or
validation sets.  Pass the file to make_kfold_splits.py (--test-users-file)
and to train_distributed.py (--test-users-file) to enforce the exclusion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import (
    build_held_out_test_split,
    filter_metadata,
    save_test_split,
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
        default=16,
        help="Maximum samples per user after dorsal filtering (default: 16; set 0 to disable).",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable adult/minor stratification (not recommended).",
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
        load_combined_metadata(root=get_dataset_root()),
        max_samples_per_user=args.max_samples_per_user,
    )

    stratify_adult = not args.no_stratify
    train_dev_ids, test_ids = build_held_out_test_split(
        metadata,
        test_size=args.test_size,
        random_state=args.seed,
        stratify_adult=stratify_adult,
    )

    saved_path = save_test_split(
        test_ids,
        out_path,
        seed=args.seed,
        test_size=args.test_size,
        stratified=stratify_adult,
    )

    total_users = metadata["user_id"].nunique()
    print(f"Saved test split to: {saved_path}")
    print(
        f"Total users: {total_users} | "
        f"Test: {len(test_ids)} ({len(test_ids)/total_users:.1%}) | "
        f"Train+dev: {len(train_dev_ids)} ({len(train_dev_ids)/total_users:.1%})"
    )

    if stratify_adult:
        user_age = metadata.groupby("user_id")["age"].mean()
        user_class = {str(uid): ("adult" if age >= 18.0 else "minor") for uid, age in user_age.items()}
        test_adults = sum(1 for uid in test_ids if user_class.get(str(uid)) == "adult")
        test_minors = len(test_ids) - test_adults
        dev_adults = sum(1 for uid in train_dev_ids if user_class.get(str(uid)) == "adult")
        dev_minors = len(train_dev_ids) - dev_adults
        print(
            f"Test  set — adults: {test_adults} ({test_adults/len(test_ids):.1%}), "
            f"minors: {test_minors} ({test_minors/len(test_ids):.1%})"
        )
        print(
            f"Dev   set — adults: {dev_adults} ({dev_adults/len(train_dev_ids):.1%}), "
            f"minors: {dev_minors} ({dev_minors/len(train_dev_ids):.1%})"
        )


if __name__ == "__main__":
    main()
