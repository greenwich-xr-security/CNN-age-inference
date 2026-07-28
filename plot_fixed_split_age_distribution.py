"""Plot train/validation/test age counts for a fixed user split and fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset.hand_metadata import load_combined_metadata, set_dataset_root
from dataset.utils import filter_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-users-file", required=True)
    parser.add_argument("--fold-file", required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--count-unit", choices=["images", "users"], default="images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_dataset_root(args.data_root)
    metadata = filter_metadata(
        load_combined_metadata(
            include_handrgbd=True,
            include_prolific=True,
        ),
        max_samples_per_user=None,
        max_samples_per_age_bin=None,
    )
    test_ids = {str(user) for user in json.load(open(args.test_users_file))["test_user_ids"]}
    folds = json.load(open(args.fold_file))["folds"]
    val_ids = {str(user) for user in folds[args.fold_index]}
    train_ids = {str(user) for index, fold in enumerate(folds) if index != args.fold_index for user in fold}
    user_ids = metadata["user_id"].astype(str)
    splits = {
        "Train samples": metadata[user_ids.isin(train_ids)],
        "Validation samples": metadata[user_ids.isin(val_ids)],
        "Test samples": metadata[user_ids.isin(test_ids)],
    }
    if args.count_unit == "users":
        splits = {
            name: frame.groupby("user_id", as_index=False)["age"].median()
            for name, frame in splits.items()
        }
        splits = {name.replace("samples", "users"): frame for name, frame in splits.items()}
    ages = np.arange(10, 76)
    bins = np.arange(9.5, 76.5, 1)
    histograms = [np.histogram(frame["age"].to_numpy(float), bins=bins)[0] for frame in splits.values()]
    y_max = max(int(counts.max()) for counts in histograms)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for axis, (title, frame), counts, color in zip(
        axes, splits.items(), histograms, ["#2b6cb0", "#dd6b20", "#38a169"]
    ):
        axis.bar(ages, counts, width=0.86, color=color)
        axis.set(title=title, xlabel="Age (years)", xlim=(9.5, 75.5), ylim=(0, y_max * 1.05))
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(f"Number of {args.count_unit}")
    fig.suptitle(
        f"Uncapped real-only age distribution ({args.count_unit}): 20% held-out test users, five-fold CV (fold 0)",
        y=1.02,
    )
    fig.tight_layout()
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print({name: len(frame) for name, frame in splits.items()})


if __name__ == "__main__":
    main()
