#!/usr/bin/env python3
"""Plot per-year dorsal-image frequencies for selected real datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

from dataset.hand_metadata import load_combined_metadata


COLOURS = {
    "handrgbd": "#55a868",
    "archive": "#dd8452",
    "prolific": "#4c9ac5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-year sample frequencies in separate dataset panels."
    )
    parser.add_argument("--root", required=True, help="HandsDatasets root directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["handrgbd", "archive", "prolific"],
        help="Dataset sources to include.",
    )
    parser.add_argument("--output-dir", default="reports/real_dataset_age_distribution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_combined_metadata(root=args.root, sources=args.datasets)
    metadata = metadata[metadata["aspect"].str.startswith("dorsal", na=False)].copy()
    metadata["age_year"] = metadata["age"].round().astype(int)
    counts = (
        metadata.groupby(["source", "age_year"], as_index=False)
        .size()
        .rename(columns={"size": "sample_count"})
    )
    counts.to_csv(output_dir / "age_frequency_per_year_by_dataset.csv", index=False)

    ages = range(int(counts["age_year"].min()), int(counts["age_year"].max()) + 1)
    y_max = int(counts["sample_count"].max())
    fig, axes = plt.subplots(
        len(args.datasets), 1, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True
    )
    if len(args.datasets) == 1:
        axes = [axes]
    for axis, source in zip(axes, args.datasets):
        source_counts = counts[counts["source"] == source].set_index("age_year")["sample_count"]
        axis.bar(
            list(ages),
            [int(source_counts.get(age, 0)) for age in ages],
            color=COLOURS.get(source, "#777777"),
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_title(source)
        axis.set_ylabel("Samples")
        axis.set_ylim(0, y_max * 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Age (years)")
    fig.suptitle("Dorsal-image frequency by age and dataset", fontsize=14)
    figure_path = output_dir / "age_frequency_per_year_by_dataset.png"
    fig.savefig(figure_path, dpi=180)
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
