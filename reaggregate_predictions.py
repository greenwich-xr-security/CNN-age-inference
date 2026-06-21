#!/usr/bin/env python3
"""
reaggregate_predictions.py

Re-aggregates per-image raw predictions into group-level prediction files
for aggregation sizes not produced during training.

For models trained on mixed datasets (e.g. EfficientNetV2-S + HAGRid),
use --handrgbd-only to restrict re-aggregation to HandRGBD participants.

Usage:
    # All HandRGBD-only models (Swin, ResNet-50, MobileNetV3):
    python reaggregate_predictions.py \\
        --run-dirs runs/swin-v2-base_384x384_ddp \\
                   runs/resnet-50_224x224_ddp \\
                   runs/mobilenetv3-large_224x224_ddp \\
        --group-sizes 2 4

    # EfficientNetV2-S (trained on HAGRid+HandRGBD — filter to HandRGBD only):
    python reaggregate_predictions.py \\
        --run-dirs runs/v2_s_ddp \\
        --group-sizes 2 4 \\
        --handrgbd-only
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metrics import aggregate_predictions_by_user


def reaggregate_split(
    fold_dir: Path,
    split: str,
    group_sizes: list[int],
    handrgbd_only: bool,
    seed: int,
) -> None:
    raw_path = fold_dir / f"{split}_predictions_raw_ddp.npz"
    if not raw_path.exists():
        print(f"    [skip] {raw_path.name} not found")
        return

    raw = np.load(raw_path)
    user_ids   = raw["user_ids"].astype(str)
    targets    = raw["targets"].astype(float)
    pred_mean  = raw["pred_mean"].astype(float)
    pred_log_var = raw["pred_log_var"].astype(float)
    skin_color = raw["skin_color"].astype(str) if "skin_color" in raw.files else np.full(len(user_ids), "", dtype=str)

    if handrgbd_only:
        mask = np.array([uid.startswith("handrgbd") for uid in user_ids])
        n_total = len(mask)
        user_ids    = user_ids[mask]
        targets     = targets[mask]
        pred_mean   = pred_mean[mask]
        pred_log_var = pred_log_var[mask]
        skin_color  = skin_color[mask]
        print(f"    handrgbd filter: {mask.sum()}/{n_total} samples kept")

    # per-user skin colour (first occurrence)
    user_skin: dict[str, str] = {}
    for uid, sc in zip(user_ids, skin_color):
        if uid not in user_skin:
            user_skin[uid] = sc

    for n in group_sizes:
        out_path = fold_dir / f"{split}_predictions_n{n}_ddp.npz"
        if out_path.exists():
            print(f"    [exists] {out_path.name} — skipping")
            continue

        rng = random.Random(seed + n)
        agg = aggregate_predictions_by_user(
            user_ids, targets, pred_mean, pred_log_var,
            group_size=n,
            rng=rng,
        )

        agg_skin = np.array(
            [user_skin.get(str(uid), "") for uid in agg["user_ids"]], dtype=str
        )

        np.savez(
            out_path,
            targets=agg["targets"],
            pred_mean=agg["pred_mean"],
            pred_log_var=agg["pred_log_var"],
            user_ids=agg["user_ids"],
            skin_color=agg_skin,
        )
        n_groups = len(agg["targets"])
        print(f"    saved {out_path.name}  ({n_groups} groups from {len(np.unique(agg['user_ids']))} users)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-aggregate raw fold predictions.")
    parser.add_argument(
        "--run-dirs", nargs="+", required=True, type=Path,
        help="One or more model run directories containing fold_* subdirectories.",
    )
    parser.add_argument(
        "--group-sizes", nargs="+", type=int, default=[2, 4],
        help="Aggregation group sizes to produce (default: 2 4).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["val", "test"],
        help="Splits to process (default: val test).",
    )
    parser.add_argument(
        "--handrgbd-only", action="store_true",
        help="Filter predictions to HandRGBD participants before aggregating.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for aggregation (default: 42).",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        print(f"\n{'='*60}")
        print(f"Run: {run_dir}")
        fold_dirs = sorted(run_dir.glob("fold_*"))
        if not fold_dirs:
            print(f"  No fold_* directories found — skipping.")
            continue
        for fold_dir in fold_dirs:
            print(f"  {fold_dir.name}")
            for split in args.splits:
                reaggregate_split(fold_dir, split, args.group_sizes, args.handrgbd_only, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
