#!/usr/bin/env python3
"""Compare k-fold runs by MAE per age decade."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


FOLD_PATTERN = re.compile(r"^fold_\\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare runs by MAE per age decade using val_predictions_*.npz files."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Parent directory containing run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots/tables (default: <runs-root>/compare_runs).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Aggregation group size n used in val_predictions_n{n}_ddp.npz (default: 1).",
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default=None,
        help="Optional subfolder name inside each fold (if outputs are nested).",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of run directory names to include.",
    )
    parser.add_argument(
        "--bin-start",
        type=float,
        default=10.0,
        help="Start age for decade bins (default: 10).",
    )
    parser.add_argument(
        "--bin-stop",
        type=float,
        default=100.0,
        help="Stop age for decade bins (default: 100).",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=10.0,
        help="Width of each age bin (default: 10).",
    )
    parser.add_argument(
        "--auto-bins",
        action="store_true",
        help="Derive bin start/stop from data instead of using --bin-start/--bin-stop.",
    )
    parser.add_argument(
        "--allow-raw",
        action="store_true",
        help="Fall back to val_predictions_raw_ddp.npz if aggregated file is missing.",
    )
    return parser.parse_args()


def discover_run_dirs(
    runs_root: Path,
    include: list[str] | None,
    exclude_names: set[str] | None = None,
) -> list[Path]:
    if include:
        include_set = {name.strip() for name in include if name.strip()}
    else:
        include_set = None
    run_dirs = []
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if exclude_names and entry.name in exclude_names:
            continue
        if include_set is not None and entry.name not in include_set:
            continue
        run_dirs.append(entry)
    run_dirs.sort(key=lambda p: p.name.lower())
    return run_dirs


def discover_folds(run_dir: Path) -> list[Path]:
    fold_dirs = []
    for entry in run_dir.iterdir():
        if entry.is_dir() and FOLD_PATTERN.match(entry.name):
            fold_dirs.append(entry)
    fold_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
    return fold_dirs


def _search_predictions(base_dir: Path, file_name: str) -> Path | None:
    direct = base_dir / file_name
    if direct.is_file():
        return direct
    for path in sorted(base_dir.glob(f"*/{file_name}")):
        if path.is_file():
            return path
    return None


def locate_predictions(
    fold_dir: Path,
    group_size: int,
    run_subdir: str | None,
    allow_raw: bool,
) -> Path | None:
    file_name = f"val_predictions_n{group_size}_ddp.npz"
    candidates: list[Path] = []
    if run_subdir:
        candidates.append(fold_dir / run_subdir)
    candidates.append(fold_dir)

    for cand in candidates:
        if cand.is_dir():
            found = _search_predictions(cand, file_name)
            if found:
                return found

    if not allow_raw:
        return None

    raw_name = "val_predictions_raw_ddp.npz"
    for cand in candidates:
        if cand.is_dir():
            found = _search_predictions(cand, raw_name)
            if found:
                return found
    return None


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "targets" not in data or "pred_mean" not in data:
        raise ValueError(f"Missing required keys in {path}")
    targets = np.asarray(data["targets"], dtype=float)
    preds = np.asarray(data["pred_mean"], dtype=float)
    mask = np.isfinite(targets) & np.isfinite(preds)
    return targets[mask], preds[mask]


def build_bin_edges(
    bin_start: float,
    bin_stop: float,
    bin_width: float,
) -> np.ndarray:
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0.")
    if bin_stop <= bin_start:
        raise ValueError("bin_stop must be greater than bin_start.")
    edges = np.arange(bin_start, bin_stop + bin_width, bin_width, dtype=float)
    if edges.size < 2:
        raise ValueError("Bin edges are invalid; check bin_start/bin_stop/bin_width.")
    return edges


def compute_mae_by_bin(
    targets: np.ndarray, preds: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    abs_err = np.abs(preds - targets)
    maes = []
    counts = []
    for idx in range(len(bin_edges) - 1):
        low = bin_edges[idx]
        high = bin_edges[idx + 1]
        if idx == len(bin_edges) - 2:
            mask = (targets >= low) & (targets <= high)
        else:
            mask = (targets >= low) & (targets < high)
        count = int(np.sum(mask))
        counts.append(count)
        if count == 0:
            maes.append(np.nan)
        else:
            maes.append(float(np.mean(abs_err[mask])))
    return np.asarray(maes, dtype=float), np.asarray(counts, dtype=int)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    output_dir = Path(args.output_dir) if args.output_dir else runs_root / "compare_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_names = set()
    try:
        output_dir_relative = output_dir.relative_to(runs_root)
        if len(output_dir_relative.parts) == 1:
            exclude_names.add(output_dir_relative.parts[0])
    except ValueError:
        pass

    run_dirs = discover_run_dirs(runs_root, args.include, exclude_names)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")

    run_data: dict[str, dict[str, np.ndarray]] = {}
    all_targets: list[np.ndarray] = []

    for run_dir in run_dirs:
        fold_dirs = discover_folds(run_dir)
        if not fold_dirs:
            fold_dirs = [run_dir]

        run_targets: list[np.ndarray] = []
        run_preds: list[np.ndarray] = []
        missing_folds: list[Path] = []

        for fold_dir in fold_dirs:
            pred_path = locate_predictions(
                fold_dir, args.group_size, args.run_subdir, args.allow_raw
            )
            if pred_path is None:
                missing_folds.append(fold_dir)
                continue
            targets, preds = load_predictions(pred_path)
            if targets.size == 0:
                continue
            run_targets.append(targets)
            run_preds.append(preds)

        if missing_folds:
            missing_list = ", ".join([p.name for p in missing_folds])
            print(f"[warn] {run_dir.name}: missing predictions in {missing_list}")

        if not run_targets:
            print(f"[warn] {run_dir.name}: no valid predictions found; skipping.")
            continue

        targets_all = np.concatenate(run_targets)
        preds_all = np.concatenate(run_preds)
        run_data[run_dir.name] = {"targets": targets_all, "preds": preds_all}
        all_targets.append(targets_all)

    if not run_data:
        raise RuntimeError("No runs had valid predictions to compare.")

    if args.auto_bins:
        targets_concat = np.concatenate(all_targets)
        min_age = float(np.min(targets_concat))
        max_age = float(np.max(targets_concat))
        bin_start = math.floor(min_age / args.bin_width) * args.bin_width
        bin_stop = math.ceil(max_age / args.bin_width) * args.bin_width
    else:
        bin_start = args.bin_start
        bin_stop = args.bin_stop

    bin_edges = build_bin_edges(bin_start, bin_stop, args.bin_width)
    bin_labels = [
        f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}"
        for i in range(len(bin_edges) - 1)
    ]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    rows = []
    for run_name, payload in run_data.items():
        targets = payload["targets"]
        preds = payload["preds"]
        maes, counts = compute_mae_by_bin(targets, preds, bin_edges)
        run_data[run_name]["mae"] = maes
        run_data[run_name]["counts"] = counts
        for label, low, high, mae, count in zip(
            bin_labels, bin_edges[:-1], bin_edges[1:], maes, counts
        ):
            rows.append(
                {
                    "run": run_name,
                    "bin_label": label,
                    "bin_start": float(low),
                    "bin_end": float(high),
                    "mae": float(mae) if np.isfinite(mae) else np.nan,
                    "count": int(count),
                }
            )

    csv_path = output_dir / "mae_by_decade_compare.csv"
    with csv_path.open("w", encoding="utf-8") as fp:
        fp.write("run,bin_label,bin_start,bin_end,mae,count\n")
        for row in rows:
            mae_val = f"{row['mae']:.6f}" if np.isfinite(row["mae"]) else ""
            fp.write(
                f"{row['run']},{row['bin_label']},{row['bin_start']:.1f},{row['bin_end']:.1f},"
                f"{mae_val},{row['count']}\n"
            )

    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name, payload in run_data.items():
        ax.plot(
            bin_centers,
            payload["mae"],
            marker="o",
            linewidth=1.6,
            label=run_name,
        )
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=0)
    ax.set_xlabel("Age bin (years)")
    ax.set_ylabel("MAE (years)")
    ax.set_title("MAE by Age Decade (aggregated across folds)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plot_path = output_dir / "mae_by_decade_compare.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {plot_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
