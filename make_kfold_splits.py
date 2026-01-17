#!/usr/bin/env python3
"""Create reproducible k-fold user splits and save to JSON."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import build_kfold_user_splits, filter_metadata, save_kfold_splits


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
        "--stratification",
        type=str,
        default="minorAdults",
        choices=["no", "minorAdults", "bins"],
        help="Stratification mode for the folds (default: minorAdults).",
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
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing fold file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root:
        set_dataset_root(args.data_root)

    out_path = Path(args.out_file)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Fold file already exists: {out_path}")

    metadata = filter_metadata(load_combined_metadata(root=get_dataset_root()))
    folds = build_kfold_user_splits(
        metadata,
        k=args.k,
        random_state=args.seed,
        stratification=args.stratification,
    )
    saved_path = save_kfold_splits(
        folds,
        out_path,
        stratification=args.stratification,
        seed=args.seed,
    )

    total_users = metadata["user_id"].nunique()
    fold_sizes = [len(fold) for fold in folds]
    print(f"Saved k-fold splits to: {saved_path}")
    print(f"Total users: {total_users} | Folds: {len(folds)} | Sizes: {fold_sizes}")


if __name__ == "__main__":
    main()
