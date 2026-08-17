#!/usr/bin/env python3
"""Create an archive-only evaluation split containing all filtered archive users."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dataset.hand_metadata import get_dataset_root, load_combined_metadata, set_dataset_root
from dataset.utils import filter_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an archive-only test-users JSON for evaluation-only runs."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root. Defaults to the repository dataset helper default.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="splits/archive_users_all.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-samples-per-user",
        type=int,
        default=0,
        help="Maximum samples per user after dorsal filtering (0 = disabled).",
    )
    parser.add_argument(
        "--max-samples-per-age-bin",
        type=int,
        default=0,
        help="Maximum samples per integer age after filtering (0 = disabled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing split JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root:
        set_dataset_root(args.data_root)

    out_path = Path(args.out_file)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Archive split already exists: {out_path}")

    metadata = filter_metadata(
        load_combined_metadata(
            root=get_dataset_root(),
            include_handrgbd=False,
            include_hagrid=False,
            include_synthetic_dorsal=False,
            include_synthetic_dorsal2=False,
            include_prolific=False,
            include_primary=False,
            include_archive=True,
        ),
        max_samples_per_user=args.max_samples_per_user or None,
        max_samples_per_age_bin=args.max_samples_per_age_bin or None,
    )
    if metadata.empty:
        raise RuntimeError("No archive samples found after filtering.")

    user_ids = sorted(str(uid) for uid in metadata["user_id"].unique())
    payload = {
        "dataset": "archive",
        "purpose": "archive-only evaluation split; all filtered archive users",
        "created_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "test_size": 1.0,
        "seed": None,
        "stratified": False,
        "test_user_ids": user_ids,
        "n_users": len(user_ids),
        "n_samples_after_filter": int(len(metadata)),
        "max_samples_per_user": int(args.max_samples_per_user),
        "max_samples_per_age_bin": int(args.max_samples_per_age_bin),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Saved archive evaluation split: {out_path}")
    print(f"Archive users: {len(user_ids)}")
    print(f"Archive samples after filter: {len(metadata)}")
    print("Age range:", float(metadata["age"].min()), "to", float(metadata["age"].max()))


if __name__ == "__main__":
    main()
