#!/usr/bin/env python3
"""Compare k-fold runs by MAE per age decade."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

import matplotlib
from metrics import compute_adult_probabilities


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


FOLD_PATTERN = re.compile(r"^fold_\d+$")
SKIN_TONE_ORDER = ("light", "tan", "dark")
SKIN_TONE_COLORS = {
    "light": "#f1d7c8",
    "tan": "#c89154",
    "dark": "#6f4122",
    "unlabeled": "#9a9a9a",
}
COMPACT_COMPARE_FIGSIZE = (5.25, 5.5)
OVERALL_COMPARE_FIGSIZE = (5.25, 4.125)
DECADE_COMPARE_FIGSIZE = (5.25, 1.925)
ROC_COMPARE_FIGSIZE = (5.25, 5.5)
DECADE_COMPARE_BIN_STOP = 70.0
MODEL_COLOR_OVERRIDES = {
    "RN50": "#4C78A8",
    "EffNetV2-M": "#F58518",
    "EffNetV2-S": "#F58518",
    "SwinV2-T": "#72B7B2",
    "SwinV2-B": "#54A24B",
    "MobNetV3-L": "#E45756",
    "ViT-S": "#B279A2",
}
FALLBACK_MODEL_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#72B7B2",
    "#FF9DA6",
)
RUN_LABEL_OVERRIDES = {
    "resnet50": "RN50",
    "resnet-50": "RN50",
    "efficientnet_v2_m": "EffNetV2-M",
    "efficientnetv2-m": "EffNetV2-M",
    "v2_m": "EffNetV2-M",
    "efficientnet_v2_s": "EffNetV2-S",
    "efficientnetv2-s": "EffNetV2-S",
    "v2_s": "EffNetV2-S",
    "mobilenet_v3_large": "MobNetV3-L",
    "mobilenetv3-large": "MobNetV3-L",
    "swin_v2_tiny": "SwinV2-T",
    "swin-v2-tiny": "SwinV2-T",
    "swin_v2_base": "SwinV2-B",
    "swin-v2-base": "SwinV2-B",
    "vit_small": "ViT-S",
    "vit-small": "ViT-S",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare runs by MAE per age decade using val/test prediction files."
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
        help="Aggregation group size n used in <split>_predictions_n{n}_ddp.npz (default: 1).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Prediction split to compare: validation or held-out test (default: val).",
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
        default=DECADE_COMPARE_BIN_STOP,
        help=f"Stop age for decade bins (default: {int(DECADE_COMPARE_BIN_STOP)}).",
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
        help="Fall back to <split>_predictions_raw_ddp.npz if aggregated file is missing.",
    )
    parser.add_argument(
        "--skip-variability",
        action="store_true",
        help="Skip intra-user variability comparison (requires <split>_predictions_raw_ddp.npz).",
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


def _base_display_label(run_name: str) -> str:
    name = run_name.strip()
    lowered = name.lower()
    for needle, label in RUN_LABEL_OVERRIDES.items():
        if needle in lowered:
            return label
    cleaned = re.sub(r"(_ddp|-ddp)$", "", name, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_-]\d{3,4}x\d{3,4}$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.replace("_", "-")


def _extract_resolution_label(run_name: str) -> str | None:
    match = re.search(r"[_-](\d{3,4})x(\d{3,4})(?:[_-]ddp)?$", run_name, flags=re.IGNORECASE)
    if not match:
        return None
    width, height = match.group(1), match.group(2)
    if width == height:
        return width
    return f"{width}x{height}"


def build_display_name_map(run_names: list[str]) -> dict[str, str]:
    display_map = {run_name: _base_display_label(run_name) for run_name in run_names}
    label_to_runs: dict[str, list[str]] = {}
    for run_name, label in display_map.items():
        label_to_runs.setdefault(label, []).append(run_name)
    for label, names in label_to_runs.items():
        if len(names) <= 1:
            continue
        for run_name in names:
            resolution = _extract_resolution_label(run_name)
            if resolution:
                display_map[run_name] = f"{label}\n{resolution}"
    return display_map


def get_model_color(run_name: str) -> str:
    base_label = _base_display_label(run_name)
    if base_label in MODEL_COLOR_OVERRIDES:
        return MODEL_COLOR_OVERRIDES[base_label]
    fallback_index = sum(ord(ch) for ch in base_label) % len(FALLBACK_MODEL_COLORS)
    return FALLBACK_MODEL_COLORS[fallback_index]


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
    split: str,
) -> Path | None:
    file_name = f"{split}_predictions_n{group_size}_ddp.npz"
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

    raw_name = f"{split}_predictions_raw_ddp.npz"
    for cand in candidates:
        if cand.is_dir():
            found = _search_predictions(cand, raw_name)
            if found:
                return found
    return None


def locate_raw_predictions(fold_dir: Path, run_subdir: str | None, split: str) -> Path | None:
    candidates: list[Path] = []
    if run_subdir:
        candidates.append(fold_dir / run_subdir)
    candidates.append(fold_dir)

    raw_name = f"{split}_predictions_raw_ddp.npz"
    for cand in candidates:
        if cand.is_dir():
            found = _search_predictions(cand, raw_name)
            if found:
                return found
    return None


def locate_skin_color_metrics(
    fold_dir: Path,
    group_size: int,
    run_subdir: str | None,
    split: str,
) -> Path | None:
    prefix = "" if split == "val" else "test_"
    file_name = f"{prefix}age_metrics_by_skin_color_n{group_size}_ddp.csv"
    candidates: list[Path] = []
    if run_subdir:
        candidates.append(fold_dir / run_subdir)
    candidates.append(fold_dir)

    for cand in candidates:
        if cand.is_dir():
            found = _search_predictions(cand, file_name)
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


def load_raw_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    if "targets" not in data or "pred_mean" not in data or "user_ids" not in data:
        raise ValueError(f"Missing required keys in {path}")
    targets = np.asarray(data["targets"], dtype=float)
    preds = np.asarray(data["pred_mean"], dtype=float)
    user_ids = np.asarray(data["user_ids"]).astype(str)
    mask = np.isfinite(targets) & np.isfinite(preds)
    return targets[mask], preds[mask], user_ids[mask]


def load_skin_color_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def load_adult_gate_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "targets" not in data:
        raise ValueError(f"Missing required keys in {path}")
    targets = np.asarray(data["targets"], dtype=float)
    if "adult_prob" in data:
        adult_prob = np.asarray(data["adult_prob"], dtype=float)
        # Binary adult_prob (age_threshold mode) gives a degenerate ROC — use
        # pred_mean directly (normalised to [0,1]) to recover a proper ranking score.
        unique_vals = np.unique(adult_prob[np.isfinite(adult_prob)])
        if set(unique_vals.tolist()).issubset({0.0, 1.0}) and "pred_mean" in data:
            pm = np.asarray(data["pred_mean"], dtype=float)
            lo, hi = pm.min(), pm.max()
            adult_prob = (pm - lo) / (hi - lo) if hi > lo else pm - lo
    elif "pred_mean" in data and "pred_log_var" in data:
        adult_prob = compute_adult_probabilities(
            np.asarray(data["pred_mean"], dtype=float),
            np.asarray(data["pred_log_var"], dtype=float),
            age_threshold=18.0,
        )
    else:
        raise ValueError(f"Missing adult gate scores in {path}")
    mask = np.isfinite(targets) & np.isfinite(adult_prob)
    return targets[mask], adult_prob[mask]


def compute_adult_gate_roc(
    targets: np.ndarray,
    adult_prob: np.ndarray,
    *,
    age_threshold: float = 18.0,
    num_thresholds: int = 201,
) -> tuple[np.ndarray, np.ndarray, float]:
    tau_values = np.linspace(0.0, 1.0, num=num_thresholds)
    is_adult = targets >= age_threshold
    is_minor = ~is_adult
    adult_total = int(np.sum(is_adult))
    minor_total = int(np.sum(is_minor))
    if adult_total <= 0 or minor_total <= 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), float("nan")

    fprs = []
    tprs = []
    for tau in tau_values:
        admit = adult_prob >= tau
        tp = int(np.logical_and(admit, is_adult).sum())
        fp = int(np.logical_and(admit, is_minor).sum())
        tpr = tp / adult_total if adult_total > 0 else 0.0
        fpr = fp / minor_total if minor_total > 0 else 0.0
        fprs.append(float(fpr))
        tprs.append(float(tpr))

    fprs_arr = np.asarray(fprs, dtype=float)
    tprs_arr = np.asarray(tprs, dtype=float)
    order = np.argsort(fprs_arr)
    fprs_sorted = fprs_arr[order]
    tprs_sorted = tprs_arr[order]
    uniq_fprs = np.unique(fprs_sorted)
    uniq_tprs = np.asarray(
        [np.max(tprs_sorted[fprs_sorted == fpr]) for fpr in uniq_fprs],
        dtype=float,
    )
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    auc = float(integrator(uniq_tprs, uniq_fprs))
    return uniq_fprs, uniq_tprs, auc


def compute_partial_auc(
    fprs: np.ndarray,
    tprs: np.ndarray,
    *,
    max_fpr: float = 0.1,
) -> tuple[float, float]:
    fprs_arr = np.asarray(fprs, dtype=float)
    tprs_arr = np.asarray(tprs, dtype=float)
    if fprs_arr.size == 0 or tprs_arr.size == 0 or max_fpr <= 0:
        return float("nan"), float("nan")

    order = np.argsort(fprs_arr)
    fprs_sorted = fprs_arr[order]
    tprs_sorted = tprs_arr[order]
    uniq_fprs = np.unique(fprs_sorted)
    uniq_tprs = np.asarray(
        [np.max(tprs_sorted[fprs_sorted == fpr]) for fpr in uniq_fprs],
        dtype=float,
    )

    if uniq_fprs[0] > 0.0:
        uniq_fprs = np.insert(uniq_fprs, 0, 0.0)
        uniq_tprs = np.insert(uniq_tprs, 0, 0.0)
    if uniq_fprs[-1] < max_fpr:
        uniq_fprs = np.append(uniq_fprs, max_fpr)
        uniq_tprs = np.append(uniq_tprs, uniq_tprs[-1])

    cutoff_tpr = float(np.interp(max_fpr, uniq_fprs, uniq_tprs))
    cutoff_mask = uniq_fprs < max_fpr
    fpr_clip = np.append(uniq_fprs[cutoff_mask], max_fpr)
    tpr_clip = np.append(uniq_tprs[cutoff_mask], cutoff_tpr)

    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    pauc = float(integrator(tpr_clip, fpr_clip))
    pauc_norm = pauc / max_fpr
    return pauc, pauc_norm


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


def compute_overall_metrics(targets: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    abs_err = np.abs(preds - targets)
    sq_err = (preds - targets) ** 2
    mae = float(np.mean(abs_err)) if abs_err.size else float("nan")
    rmse = float(np.sqrt(np.mean(sq_err))) if sq_err.size else float("nan")
    return mae, rmse


def collect_intra_user_variability(
    targets: np.ndarray, preds: np.ndarray, user_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[str, list[float]] = {}
    age_buckets: dict[str, list[float]] = {}
    for uid, pred, tgt in zip(user_ids, preds, targets):
        key = str(uid)
        buckets.setdefault(key, []).append(float(pred))
        age_buckets.setdefault(key, []).append(float(tgt))

    user_ages: list[float] = []
    user_stds: list[float] = []
    for key, preds_list in buckets.items():
        if len(preds_list) < 2:
            continue
        ages = age_buckets.get(key, [])
        if not ages:
            continue
        user_age = float(np.mean(ages))
        user_std = float(np.std(preds_list))
        user_ages.append(user_age)
        user_stds.append(user_std)
    return np.asarray(user_ages, dtype=float), np.asarray(user_stds, dtype=float)


def compute_stat_by_bin(
    ages: np.ndarray, values: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = []
    medians = []
    counts = []
    for idx in range(len(bin_edges) - 1):
        low = bin_edges[idx]
        high = bin_edges[idx + 1]
        if idx == len(bin_edges) - 2:
            mask = (ages >= low) & (ages <= high)
        else:
            mask = (ages >= low) & (ages < high)
        vals = values[mask]
        counts.append(int(vals.size))
        if vals.size == 0:
            means.append(np.nan)
            medians.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            medians.append(float(np.median(vals)))
    return (
        np.asarray(means, dtype=float),
        np.asarray(medians, dtype=float),
        np.asarray(counts, dtype=int),
    )


def compute_skin_tone_repeated_measures(
    skin_metric_rows: list[dict[str, object]],
    run_names: list[str],
) -> list[dict[str, object]]:
    try:
        import pandas as pd
        from scipy.stats import friedmanchisquare
        from statsmodels.stats.anova import AnovaRM
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Skin-tone repeated-measures analysis skipped: {exc}")
        return []

    results: list[dict[str, object]] = []
    for run_name in run_names:
        run_rows = [
            row
            for row in skin_metric_rows
            if row["run"] == run_name
            and row["skin_color"] in SKIN_TONE_ORDER
            and np.isfinite(row["mae"])
        ]
        if not run_rows:
            continue

        df = pd.DataFrame(run_rows)
        grouped = (
            df.groupby(["fold", "skin_color"], as_index=False)["mae"]
            .mean()
            .sort_values(["fold", "skin_color"])
        )
        pivot = grouped.pivot(index="fold", columns="skin_color", values="mae")
        tones = [tone for tone in SKIN_TONE_ORDER if tone in pivot.columns]
        if len(tones) < 2:
            continue
        pivot = pivot[tones].dropna(axis=0, how="any")
        if pivot.shape[0] < 2:
            continue

        long_df = (
            pivot.reset_index()
            .melt(id_vars="fold", var_name="skin_color", value_name="mae")
            .sort_values(["fold", "skin_color"])
        )

        result: dict[str, object] = {
            "run": run_name,
            "tones": "|".join(tones),
            "n_folds": int(pivot.shape[0]),
            "anova_f": np.nan,
            "anova_num_df": np.nan,
            "anova_den_df": np.nan,
            "anova_p": np.nan,
            "friedman_stat": np.nan,
            "friedman_p": np.nan,
        }
        for tone in SKIN_TONE_ORDER:
            vals = pivot[tone].to_numpy(dtype=float) if tone in pivot.columns else np.asarray([], dtype=float)
            result[f"{tone}_mae_mean"] = float(np.mean(vals)) if vals.size else np.nan
            result[f"{tone}_mae_median"] = float(np.median(vals)) if vals.size else np.nan

        try:
            aov = AnovaRM(long_df, depvar="mae", subject="fold", within=["skin_color"]).fit()
            row = aov.anova_table.loc["skin_color"]
            result["anova_f"] = float(row["F Value"])
            result["anova_num_df"] = float(row["Num DF"])
            result["anova_den_df"] = float(row["Den DF"])
            result["anova_p"] = float(row["Pr > F"])
        except Exception as exc:  # noqa: BLE001
            result["anova_error"] = str(exc)

        try:
            arrays = [pivot[tone].to_numpy(dtype=float) for tone in tones]
            fried_stat, fried_p = friedmanchisquare(*arrays)
            result["friedman_stat"] = float(fried_stat)
            result["friedman_p"] = float(fried_p)
        except Exception as exc:  # noqa: BLE001
            result["friedman_error"] = str(exc)

        results.append(result)

    return results


def compute_skin_tone_pairwise(
    skin_metric_rows: list[dict[str, object]],
    run_names: list[str],
) -> list[dict[str, object]]:
    try:
        import pandas as pd
        from itertools import combinations
        from scipy.stats import ttest_rel, wilcoxon
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Skin-tone pairwise analysis skipped: {exc}")
        return []

    results: list[dict[str, object]] = []
    for run_name in run_names:
        run_rows = [
            row
            for row in skin_metric_rows
            if row["run"] == run_name
            and row["skin_color"] in SKIN_TONE_ORDER
            and np.isfinite(row["mae"])
        ]
        if not run_rows:
            continue

        df = pd.DataFrame(run_rows)
        grouped = (
            df.groupby(["fold", "skin_color"], as_index=False)["mae"]
            .mean()
            .sort_values(["fold", "skin_color"])
        )
        pivot = grouped.pivot(index="fold", columns="skin_color", values="mae")
        tones = [tone for tone in SKIN_TONE_ORDER if tone in pivot.columns]
        if len(tones) < 2:
            continue

        tests: list[dict[str, object]] = []
        for tone_a, tone_b in combinations(tones, 2):
            pair = pivot[[tone_a, tone_b]].dropna(axis=0, how="any")
            if pair.shape[0] < 2:
                continue
            diff = pair[tone_a] - pair[tone_b]
            ttest_p = float(ttest_rel(pair[tone_a], pair[tone_b]).pvalue)
            try:
                wilcoxon_p = float(
                    wilcoxon(
                        pair[tone_a],
                        pair[tone_b],
                        zero_method="wilcox",
                        alternative="two-sided",
                    ).pvalue
                )
            except Exception:  # noqa: BLE001
                wilcoxon_p = float("nan")
            tests.append(
                {
                    "run": run_name,
                    "tone_a": tone_a,
                    "tone_b": tone_b,
                    "comparison": f"{tone_a} vs {tone_b}",
                    "n_folds": int(pair.shape[0]),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "ttest_p": ttest_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )

        if not tests:
            continue

        ttest_adj = multipletests([row["ttest_p"] for row in tests], method="holm")[1]
        valid_wilcoxon = [idx for idx, row in enumerate(tests) if np.isfinite(row["wilcoxon_p"])]
        wilcoxon_adj = []
        if valid_wilcoxon:
            wilcoxon_adj = multipletests(
                [tests[idx]["wilcoxon_p"] for idx in valid_wilcoxon],
                method="holm",
            )[1]

        wilcoxon_adj_idx = 0
        for idx, row in enumerate(tests):
            row["ttest_p_holm"] = float(ttest_adj[idx])
            if idx in valid_wilcoxon:
                row["wilcoxon_p_holm"] = float(wilcoxon_adj[wilcoxon_adj_idx])
                wilcoxon_adj_idx += 1
            else:
                row["wilcoxon_p_holm"] = float("nan")
            results.append(row)

    return results


def compute_overall_metric_repeated_measures(
    overall_fold_rows: list[dict[str, object]],
    run_names: list[str],
    metric_specs: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    try:
        import pandas as pd
        from scipy.stats import friedmanchisquare
        from statsmodels.stats.anova import AnovaRM
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Overall-metric repeated-measures analysis skipped: {exc}")
        return []

    results: list[dict[str, object]] = []
    for metric_key, metric_label, _ in metric_specs:
        metric_rows = [
            {
                "fold": str(row["fold"]),
                "run": str(row["run"]),
                "value": float(row[metric_key]),
            }
            for row in overall_fold_rows
            if np.isfinite(row[metric_key])
        ]
        if not metric_rows:
            continue

        df = pd.DataFrame(metric_rows)
        grouped = (
            df.groupby(["fold", "run"], as_index=False)["value"]
            .mean()
            .sort_values(["fold", "run"])
        )
        pivot = grouped.pivot(index="fold", columns="run", values="value")
        models = [run_name for run_name in run_names if run_name in pivot.columns]
        if len(models) < 2:
            continue
        pivot = pivot[models].dropna(axis=0, how="any")
        if pivot.shape[0] < 2:
            continue

        long_df = (
            pivot.reset_index()
            .melt(id_vars="fold", var_name="run", value_name="value")
            .sort_values(["fold", "run"])
        )

        result: dict[str, object] = {
            "metric": metric_key,
            "metric_label": metric_label,
            "models": "|".join(models),
            "n_folds": int(pivot.shape[0]),
            "anova_f": np.nan,
            "anova_num_df": np.nan,
            "anova_den_df": np.nan,
            "anova_p": np.nan,
            "friedman_stat": np.nan,
            "friedman_p": np.nan,
        }

        try:
            aov = AnovaRM(long_df, depvar="value", subject="fold", within=["run"]).fit()
            row = aov.anova_table.loc["run"]
            result["anova_f"] = float(row["F Value"])
            result["anova_num_df"] = float(row["Num DF"])
            result["anova_den_df"] = float(row["Den DF"])
            result["anova_p"] = float(row["Pr > F"])
        except Exception as exc:  # noqa: BLE001
            result["anova_error"] = str(exc)

        try:
            arrays = [pivot[run_name].to_numpy(dtype=float) for run_name in models]
            fried_stat, fried_p = friedmanchisquare(*arrays)
            result["friedman_stat"] = float(fried_stat)
            result["friedman_p"] = float(fried_p)
        except Exception as exc:  # noqa: BLE001
            result["friedman_error"] = str(exc)

        results.append(result)

    return results


def compute_overall_metric_pairwise(
    overall_fold_rows: list[dict[str, object]],
    run_names: list[str],
    metric_specs: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    try:
        import pandas as pd
        from itertools import combinations
        from scipy.stats import ttest_rel, wilcoxon
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Overall-metric pairwise analysis skipped: {exc}")
        return []

    results: list[dict[str, object]] = []
    for metric_key, metric_label, _ in metric_specs:
        metric_rows = [
            {
                "fold": str(row["fold"]),
                "run": str(row["run"]),
                "value": float(row[metric_key]),
            }
            for row in overall_fold_rows
            if np.isfinite(row[metric_key])
        ]
        if not metric_rows:
            continue

        df = pd.DataFrame(metric_rows)
        grouped = (
            df.groupby(["fold", "run"], as_index=False)["value"]
            .mean()
            .sort_values(["fold", "run"])
        )
        pivot = grouped.pivot(index="fold", columns="run", values="value")
        models = [run_name for run_name in run_names if run_name in pivot.columns]
        if len(models) < 2:
            continue

        tests: list[dict[str, object]] = []
        for model_a, model_b in combinations(models, 2):
            pair = pivot[[model_a, model_b]].dropna(axis=0, how="any")
            if pair.shape[0] < 2:
                continue
            diff = pair[model_a] - pair[model_b]
            ttest_p = float(ttest_rel(pair[model_a], pair[model_b]).pvalue)
            try:
                wilcoxon_p = float(
                    wilcoxon(
                        pair[model_a],
                        pair[model_b],
                        zero_method="wilcox",
                        alternative="two-sided",
                    ).pvalue
                )
            except Exception:  # noqa: BLE001
                wilcoxon_p = float("nan")
            tests.append(
                {
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparison": f"{model_a} vs {model_b}",
                    "n_folds": int(pair.shape[0]),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "ttest_p": ttest_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )

        if not tests:
            continue

        ttest_adj = multipletests([row["ttest_p"] for row in tests], method="holm")[1]
        valid_wilcoxon = [idx for idx, row in enumerate(tests) if np.isfinite(row["wilcoxon_p"])]
        wilcoxon_adj = []
        if valid_wilcoxon:
            wilcoxon_adj = multipletests(
                [tests[idx]["wilcoxon_p"] for idx in valid_wilcoxon],
                method="holm",
            )[1]

        wilcoxon_adj_idx = 0
        for idx, row in enumerate(tests):
            row["ttest_p_holm"] = float(ttest_adj[idx])
            if idx in valid_wilcoxon:
                row["wilcoxon_p_holm"] = float(wilcoxon_adj[wilcoxon_adj_idx])
                wilcoxon_adj_idx += 1
            else:
                row["wilcoxon_p_holm"] = float("nan")
            results.append(row)

    return results


def p_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def add_significance_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    h: float,
    label: str,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va="bottom", fontsize=10)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_name = "compare_runs" if args.split == "val" else f"compare_runs_{args.split}"
        output_dir = runs_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    split_label = "validation" if args.split == "val" else "held-out test"

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
    variability_data: dict[str, dict[str, np.ndarray]] = {}
    variability_targets: list[np.ndarray] = []
    skin_metric_rows: list[dict[str, object]] = []
    overall_fold_rows: list[dict[str, object]] = []
    roc_fold_data: dict[str, list[dict[str, object]]] = {}

    for run_dir in run_dirs:
        fold_dirs = discover_folds(run_dir)
        if not fold_dirs:
            fold_dirs = [run_dir]

        run_targets: list[np.ndarray] = []
        run_preds: list[np.ndarray] = []
        missing_folds: list[Path] = []
        missing_skin_folds: list[Path] = []

        for fold_dir in fold_dirs:
            pred_path = locate_predictions(
                fold_dir, args.group_size, args.run_subdir, args.allow_raw, args.split
            )
            if pred_path is None:
                missing_folds.append(fold_dir)
                continue
            targets, preds = load_predictions(pred_path)
            if targets.size == 0:
                continue
            run_targets.append(targets)
            run_preds.append(preds)

            fold_mae, fold_rmse = compute_overall_metrics(targets, preds)
            fold_intra_std = float("nan")
            if not args.skip_variability:
                raw_path = locate_raw_predictions(fold_dir, args.run_subdir, args.split)
                if raw_path is not None:
                    raw_targets_fold, raw_preds_fold, raw_user_ids_fold = load_raw_predictions(raw_path)
                    _, user_stds_fold = collect_intra_user_variability(
                        raw_targets_fold, raw_preds_fold, raw_user_ids_fold
                    )
                    if user_stds_fold.size:
                        fold_intra_std = float(np.mean(user_stds_fold))
            overall_fold_rows.append(
                {
                    "run": run_dir.name,
                    "fold": fold_dir.name,
                    "mae": float(fold_mae),
                    "rmse": float(fold_rmse),
                    "intra_user_std": float(fold_intra_std),
                }
            )
            try:
                gate_targets, adult_prob = load_adult_gate_scores(pred_path)
                roc_fpr, roc_tpr, roc_auc = compute_adult_gate_roc(gate_targets, adult_prob)
                roc_pauc, roc_pauc_norm = compute_partial_auc(roc_fpr, roc_tpr, max_fpr=0.1)
                roc_fold_data.setdefault(run_dir.name, []).append(
                    {
                        "fold": fold_dir.name,
                        "fpr": roc_fpr,
                        "tpr": roc_tpr,
                        "auc": float(roc_auc),
                        "pauc_fpr_0_1": float(roc_pauc),
                        "pauc_fpr_0_1_norm": float(roc_pauc_norm),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {run_dir.name}:{fold_dir.name} ROC skipped: {exc}")

            skin_path = locate_skin_color_metrics(
                fold_dir, args.group_size, args.run_subdir, args.split
            )
            if skin_path is None:
                missing_skin_folds.append(fold_dir)
            else:
                for row in load_skin_color_metrics(skin_path):
                    skin_metric_rows.append(
                        {
                            "run": run_dir.name,
                            "fold": fold_dir.name,
                            "skin_color": str(row.get("skin_color", "")).strip().lower(),
                            "samples": int(float(row.get("samples", "0") or 0)),
                            "users": int(float(row.get("users", "0") or 0)),
                            "adult_count": int(float(row.get("adult_count", "0") or 0)),
                            "minor_count": int(float(row.get("minor_count", "0") or 0)),
                            "target_age_mean": float(row.get("target_age_mean", "nan") or np.nan),
                            "pred_age_mean": float(row.get("pred_age_mean", "nan") or np.nan),
                            "mean_error": float(row.get("mean_error", "nan") or np.nan),
                            "mae": float(row.get("mae", "nan") or np.nan),
                            "rmse": float(row.get("rmse", "nan") or np.nan),
                            "auc_adult_gate": float(row.get("auc_adult_gate", "nan") or np.nan),
                        }
                    )

        if missing_folds:
            missing_list = ", ".join([p.name for p in missing_folds])
            print(f"[warn] {run_dir.name}: missing predictions in {missing_list}")
        if missing_skin_folds:
            missing_list = ", ".join([p.name for p in missing_skin_folds])
            print(f"[warn] {run_dir.name}: missing skin-color metrics in {missing_list}")

        if not run_targets:
            print(f"[warn] {run_dir.name}: no valid predictions found; skipping.")
            continue

        targets_all = np.concatenate(run_targets)
        preds_all = np.concatenate(run_preds)
        run_data[run_dir.name] = {"targets": targets_all, "preds": preds_all}
        all_targets.append(targets_all)

        if not args.skip_variability:
            raw_targets: list[np.ndarray] = []
            raw_preds: list[np.ndarray] = []
            raw_user_ids: list[np.ndarray] = []
            missing_raw: list[Path] = []
            for fold_dir in fold_dirs:
                raw_path = locate_raw_predictions(fold_dir, args.run_subdir, args.split)
                if raw_path is None:
                    missing_raw.append(fold_dir)
                    continue
                tgt_raw, pred_raw, uid_raw = load_raw_predictions(raw_path)
                if tgt_raw.size == 0:
                    continue
                raw_targets.append(tgt_raw)
                raw_preds.append(pred_raw)
                raw_user_ids.append(uid_raw)
            if missing_raw:
                missing_list = ", ".join([p.name for p in missing_raw])
                print(f"[warn] {run_dir.name}: missing raw predictions in {missing_list}")
            if raw_targets:
                raw_targets_all = np.concatenate(raw_targets)
                raw_preds_all = np.concatenate(raw_preds)
                raw_user_ids_all = np.concatenate(raw_user_ids)
                variability_data[run_dir.name] = {
                    "targets": raw_targets_all,
                    "preds": raw_preds_all,
                    "user_ids": raw_user_ids_all,
                }
                variability_targets.append(raw_targets_all)

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

    overall_rows = []
    for run_name, payload in run_data.items():
        mae, rmse = compute_overall_metrics(payload["targets"], payload["preds"])
        run_data[run_name]["overall_mae"] = mae
        run_data[run_name]["overall_rmse"] = rmse
        overall_rows.append({"run": run_name, "mae": mae, "rmse": rmse})

    overall_intra_rows = []
    for run_name, payload in variability_data.items():
        user_ages, user_stds = collect_intra_user_variability(
            payload["targets"], payload["preds"], payload["user_ids"]
        )
        if user_stds.size:
            overall_std = float(np.mean(user_stds))
        else:
            overall_std = float("nan")
        variability_data[run_name]["overall_std"] = overall_std
        overall_intra_rows.append({"run": run_name, "intra_std": overall_std})

    overall_csv = output_dir / "overall_metrics_compare.csv"
    with overall_csv.open("w", encoding="utf-8") as fp:
        fp.write("run,mae,rmse,intra_user_std\n")
        for run_name in run_data.keys():
            mae = run_data[run_name]["overall_mae"]
            rmse = run_data[run_name]["overall_rmse"]
            intra = (
                variability_data.get(run_name, {}).get("overall_std", float("nan"))
                if not args.skip_variability
                else float("nan")
            )
            mae_val = f"{mae:.6f}" if np.isfinite(mae) else ""
            rmse_val = f"{rmse:.6f}" if np.isfinite(rmse) else ""
            intra_val = f"{intra:.6f}" if np.isfinite(intra) else ""
            fp.write(f"{run_name},{mae_val},{rmse_val},{intra_val}\n")

    overall_per_fold_csv = output_dir / "overall_metrics_per_fold.csv"
    with overall_per_fold_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["run", "fold", "mae", "rmse", "intra_user_std"],
        )
        writer.writeheader()
        for row in overall_fold_rows:
            writer.writerow(row)

    run_names = list(run_data.keys())
    display_name_map = build_display_name_map(run_names)
    x = np.arange(len(run_names))
    metric_specs = [
        ("mae", "MAE", "#1f77b4"),
        ("rmse", "RMSE", "#ff7f0e"),
    ]
    if not args.skip_variability:
        metric_specs.append(("intra_user_std", "Intra-user STD", "#2ca02c"))

    overall_rm_results = compute_overall_metric_repeated_measures(
        overall_fold_rows, run_names, metric_specs
    )
    overall_rm_csv = output_dir / "overall_metrics_repeated_measures.csv"
    with overall_rm_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "metric",
                "metric_label",
                "models",
                "n_folds",
                "anova_f",
                "anova_num_df",
                "anova_den_df",
                "anova_p",
                "friedman_stat",
                "friedman_p",
                "anova_error",
                "friedman_error",
            ],
        )
        writer.writeheader()
        for row in overall_rm_results:
            writer.writerow(row)

    overall_pairwise_results = compute_overall_metric_pairwise(
        overall_fold_rows, run_names, metric_specs
    )
    overall_pairwise_csv = output_dir / "overall_metrics_pairwise.csv"
    with overall_pairwise_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "metric",
                "metric_label",
                "model_a",
                "model_b",
                "comparison",
                "n_folds",
                "mean_diff_a_minus_b",
                "median_diff_a_minus_b",
                "ttest_p",
                "wilcoxon_p",
                "ttest_p_holm",
                "wilcoxon_p_holm",
            ],
        )
        writer.writeheader()
        for row in overall_pairwise_results:
            writer.writerow(row)

    omnibus_significant: dict[str, bool] = {}
    for row in overall_rm_results:
        metric_key = str(row["metric"])
        anova_p = float(row["anova_p"]) if np.isfinite(row["anova_p"]) else float("nan")
        friedman_p = (
            float(row["friedman_p"]) if np.isfinite(row["friedman_p"]) else float("nan")
        )
        omnibus_significant[metric_key] = bool(
            (np.isfinite(anova_p) and anova_p < 0.05)
            or (np.isfinite(friedman_p) and friedman_p < 0.05)
        )

    fig, axes = plt.subplots(1, len(metric_specs), figsize=OVERALL_COMPARE_FIGSIZE, squeeze=False)
    axes = axes[0]
    for ax, (metric_key, metric_label, metric_color) in zip(axes, metric_specs):
        series = []
        series_runs = []
        positions = []
        labels = []
        run_max_values: dict[str, float] = {}
        for idx, run_name in enumerate(run_names):
            vals = [
                float(row[metric_key])
                for row in overall_fold_rows
                if row["run"] == run_name and np.isfinite(row[metric_key])
            ]
            if not vals:
                continue
            series.append(vals)
            series_runs.append(run_name)
            positions.append(float(x[idx]))
            labels.append(display_name_map.get(run_name, run_name))
            run_max_values[run_name] = float(max(vals))
        if not series:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(
            series,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for patch, run_name in zip(bp["boxes"], series_runs):
            patch.set_facecolor(get_model_color(run_name))
            patch.set_alpha(0.9)
        for flier, run_name in zip(bp["fliers"], series_runs):
            color = get_model_color(run_name)
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)
            flier.set_alpha(0.55)

        if omnibus_significant.get(metric_key):
            sig_rows = [
                row
                for row in overall_pairwise_results
                if row["metric"] == metric_key
                and np.isfinite(row["ttest_p_holm"])
                and row["ttest_p_holm"] < 0.05
            ]
            if sig_rows:
                model_positions = {run_name: float(idx) for idx, run_name in enumerate(run_names)}
                global_max = max(list(run_max_values.values()) + [1.0])
                bracket_gap = max(0.18, 0.04 * global_max)
                bracket_height = max(0.08, 0.02 * global_max)
                max_annotation_y = global_max
                sig_rows.sort(
                    key=lambda row: (
                        abs(
                            model_positions.get(str(row["model_a"]), 0.0)
                            - model_positions.get(str(row["model_b"]), 0.0)
                        ),
                        row["ttest_p_holm"],
                    )
                )
                base_y = global_max + bracket_gap
                for level, row in enumerate(sig_rows):
                    x1 = model_positions.get(str(row["model_a"]))
                    x2 = model_positions.get(str(row["model_b"]))
                    if x1 is None or x2 is None:
                        continue
                    y = base_y + level * (bracket_gap + bracket_height)
                    label = p_to_stars(float(row["ttest_p_holm"]))
                    if not label:
                        continue
                    add_significance_bracket(ax, x1, x2, y, bracket_height, label)
                    max_annotation_y = max(max_annotation_y, y + bracket_height)
                ax.set_ylim(top=max_annotation_y + bracket_gap)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, ha="center", va="top")
        ax.set_title(metric_label)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("Error (years)")
    fig.tight_layout()
    overall_plot = output_dir / "overall_metrics_compare.png"
    fig.savefig(overall_plot, dpi=150)
    fig.savefig(output_dir / "overall_metrics_compare.svg")
    plt.close(fig)

    if roc_fold_data:
        roc_grid = np.linspace(0.0, 1.0, 201)
        roc_csv = output_dir / "roc_adult_gate_compare.csv"
        with roc_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "run",
                    "fpr",
                    "tpr_mean",
                    "tpr_std",
                    "auc_mean",
                    "auc_std",
                    "pauc_fpr_0_1_mean",
                    "pauc_fpr_0_1_std",
                    "pauc_fpr_0_1_norm_mean",
                    "pauc_fpr_0_1_norm_std",
                    "folds",
                ],
            )
            writer.writeheader()

            fig, ax = plt.subplots(figsize=ROC_COMPARE_FIGSIZE)
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#777777", label="Chance")
            ax.axvline(0.1, linestyle=":", linewidth=1.0, color="#555555")
            ax.text(
                0.103,
                0.04,
                "FPR=0.1",
                rotation=90,
                fontsize=7,
                color="#555555",
                ha="left",
                va="bottom",
            )
            def _mean_pauc(rn: str) -> float:
                vals = [float(r.get("pauc_fpr_0_1", float("nan"))) for r in roc_fold_data.get(rn, [])]
                finite = [v for v in vals if np.isfinite(v)]
                return float(np.mean(finite)) if finite else float("-inf")

            roc_run_names = sorted(
                [rn for rn in run_names if roc_fold_data.get(rn)],
                key=_mean_pauc,
                reverse=True,
            )
            for run_name in roc_run_names:
                curves = roc_fold_data.get(run_name, [])
                if not curves:
                    continue
                interp_tprs = []
                aucs = []
                paucs = []
                paucs_norm = []
                for row in curves:
                    interp_tprs.append(
                        np.interp(
                            roc_grid,
                            np.asarray(row["fpr"], dtype=float),
                            np.asarray(row["tpr"], dtype=float),
                            left=0.0,
                            right=1.0,
                        )
                    )
                    if np.isfinite(float(row["auc"])):
                        aucs.append(float(row["auc"]))
                    if np.isfinite(float(row.get("pauc_fpr_0_1", float("nan")))):
                        paucs.append(float(row["pauc_fpr_0_1"]))
                    if np.isfinite(float(row.get("pauc_fpr_0_1_norm", float("nan")))):
                        paucs_norm.append(float(row["pauc_fpr_0_1_norm"]))
                if not interp_tprs:
                    continue
                tpr_matrix = np.vstack(interp_tprs)
                tpr_mean = np.mean(tpr_matrix, axis=0)
                tpr_std = np.std(tpr_matrix, axis=0)
                auc_mean = float(np.mean(aucs)) if aucs else float("nan")
                auc_std = float(np.std(aucs)) if aucs else float("nan")
                pauc_mean = float(np.mean(paucs)) if paucs else float("nan")
                pauc_std = float(np.std(paucs)) if paucs else float("nan")
                pauc_norm_mean = float(np.mean(paucs_norm)) if paucs_norm else float("nan")
                pauc_norm_std = float(np.std(paucs_norm)) if paucs_norm else float("nan")
                display_name = display_name_map.get(run_name, run_name)
                label = f"{display_name}\nAUC {auc_mean:.3f}"
                if np.isfinite(auc_std):
                    label += f" ± {auc_std:.3f}"
                if np.isfinite(pauc_mean):
                    label += f" | pAUC@0.1 {pauc_mean:.3f}"
                color = get_model_color(run_name)
                ax.plot(roc_grid, tpr_mean, linewidth=2.0, label=label, color=color)
                ax.fill_between(
                    roc_grid,
                    np.clip(tpr_mean - tpr_std, 0.0, 1.0),
                    np.clip(tpr_mean + tpr_std, 0.0, 1.0),
                    color=color,
                    alpha=0.12,
                )
                for fpr_val, tpr_mean_val, tpr_std_val in zip(roc_grid, tpr_mean, tpr_std):
                    writer.writerow(
                        {
                            "run": run_name,
                            "fpr": float(fpr_val),
                            "tpr_mean": float(tpr_mean_val),
                            "tpr_std": float(tpr_std_val),
                            "auc_mean": auc_mean,
                            "auc_std": auc_std,
                            "pauc_fpr_0_1_mean": pauc_mean,
                            "pauc_fpr_0_1_std": pauc_std,
                            "pauc_fpr_0_1_norm_mean": pauc_norm_mean,
                            "pauc_fpr_0_1_norm_std": pauc_norm_std,
                            "folds": int(len(curves)),
                        }
                    )

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"Adult-Gate ROC ({split_label}, mean across folds)")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_box_aspect(1)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            roc_plot = output_dir / "roc_adult_gate_compare.png"
            fig.savefig(roc_plot, dpi=150)
            roc_plot_svg = output_dir / "roc_adult_gate_compare.svg"
            fig.savefig(roc_plot_svg)
            plt.close(fig)
            print(f"Saved: {roc_plot}")
            print(f"Saved: {roc_plot_svg}")
            print(f"Saved: {roc_csv}")

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

    fig, ax = plt.subplots(figsize=DECADE_COMPARE_FIGSIZE)
    for run_name, payload in run_data.items():
        ax.plot(
            bin_centers,
            payload["mae"],
            marker="o",
            linewidth=1.6,
            label=display_name_map.get(run_name, run_name),
            color=get_model_color(run_name),
        )
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=0)
    ax.set_xlabel("Age bin (years)")
    ax.set_ylabel("MAE (years)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plot_path = output_dir / "mae_by_decade_compare.png"
    fig.savefig(plot_path, dpi=150)
    fig.savefig(output_dir / "mae_by_decade_compare.svg")
    plt.close(fig)

    print(f"Saved: {overall_plot}")
    print(f"Saved: {overall_csv}")
    print(f"Saved: {overall_per_fold_csv}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {csv_path}")

    if skin_metric_rows:
        skin_csv = output_dir / "mae_by_skin_color_compare.csv"
        with skin_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "run",
                    "fold",
                    "skin_color",
                    "samples",
                    "users",
                    "adult_count",
                    "minor_count",
                    "target_age_mean",
                    "pred_age_mean",
                    "mean_error",
                    "mae",
                    "rmse",
                    "auc_adult_gate",
                ],
            )
            writer.writeheader()
            for row in skin_metric_rows:
                writer.writerow(row)

        plot_skin_tones = [
            tone
            for tone in SKIN_TONE_ORDER
            if any(r["skin_color"] == tone and np.isfinite(r["mae"]) for r in skin_metric_rows)
        ]
        if plot_skin_tones:
            fig, ax = plt.subplots(figsize=OVERALL_COMPARE_FIGSIZE)
            x = np.arange(len(run_names), dtype=float)
            offsets = np.linspace(-0.26, 0.26, num=len(plot_skin_tones))
            box_width = 0.18 if len(plot_skin_tones) >= 3 else 0.24
            tone_positions: dict[str, dict[str, float]] = {run_name: {} for run_name in run_names}
            run_max_mae: dict[str, float] = {run_name: 0.0 for run_name in run_names}

            for offset, skin_tone in zip(offsets, plot_skin_tones):
                series = []
                positions = []
                for idx, run_name in enumerate(run_names):
                    vals = [
                        float(row["mae"])
                        for row in skin_metric_rows
                        if row["run"] == run_name
                        and row["skin_color"] == skin_tone
                        and np.isfinite(row["mae"])
                    ]
                    if not vals:
                        continue
                    tone_positions[run_name][skin_tone] = float(x[idx] + offset)
                    run_max_mae[run_name] = max(run_max_mae[run_name], float(np.max(vals)))
                    series.append(vals)
                    positions.append(x[idx] + offset)
                if not series:
                    continue
                bp = ax.boxplot(
                    series,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=True,
                    medianprops={"color": "black", "linewidth": 1.2},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                    boxprops={"linewidth": 1.0},
                )
                color = SKIN_TONE_COLORS.get(skin_tone, "#777777")
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.9)
                for flier in bp["fliers"]:
                    flier.set_markerfacecolor(color)
                    flier.set_markeredgecolor(color)
                    flier.set_alpha(0.55)

            pairwise_results = compute_skin_tone_pairwise(skin_metric_rows, run_names)
            pairwise_csv = output_dir / "mae_by_skin_color_pairwise.csv"
            with pairwise_csv.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "run",
                        "tone_a",
                        "tone_b",
                        "comparison",
                        "n_folds",
                        "mean_diff_a_minus_b",
                        "median_diff_a_minus_b",
                        "ttest_p",
                        "wilcoxon_p",
                        "ttest_p_holm",
                        "wilcoxon_p_holm",
                    ],
                )
                writer.writeheader()
                for row in pairwise_results:
                    writer.writerow(row)

            if pairwise_results:
                global_max = max(
                    [run_max_mae.get(run_name, 0.0) for run_name in run_names] + [1.0]
                )
                bracket_gap = max(0.18, 0.04 * global_max)
                bracket_height = max(0.08, 0.02 * global_max)
                max_annotation_y = global_max
                for run_name in run_names:
                    sig_rows = [
                        row
                        for row in pairwise_results
                        if row["run"] == run_name and np.isfinite(row["ttest_p_holm"]) and row["ttest_p_holm"] < 0.05
                    ]
                    if not sig_rows:
                        continue
                    sig_rows.sort(
                        key=lambda row: (
                            abs(
                                tone_positions[run_name].get(row["tone_a"], 0.0)
                                - tone_positions[run_name].get(row["tone_b"], 0.0)
                            ),
                            row["ttest_p_holm"],
                        )
                    )
                    base_y = run_max_mae.get(run_name, 0.0) + bracket_gap
                    for level, row in enumerate(sig_rows):
                        x1 = tone_positions[run_name].get(str(row["tone_a"]))
                        x2 = tone_positions[run_name].get(str(row["tone_b"]))
                        if x1 is None or x2 is None:
                            continue
                        y = base_y + level * (bracket_gap + bracket_height)
                        label = p_to_stars(float(row["ttest_p_holm"]))
                        if not label:
                            continue
                        add_significance_bracket(ax, x1, x2, y, bracket_height, label)
                        max_annotation_y = max(max_annotation_y, y + bracket_height)
                ax.set_ylim(top=max_annotation_y + bracket_gap)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [display_name_map.get(run_name, run_name) for run_name in run_names],
                rotation=90,
                ha="center",
                va="top",
            )
            ax.set_ylabel("Per-fold MAE (years)")
            ax.set_title("MAE by Skin Tone")
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
            legend = ax.legend(
                handles=[
                    Patch(facecolor=SKIN_TONE_COLORS.get(tone, "#777777"), edgecolor="black", label=tone)
                    for tone in plot_skin_tones
                ],
                title="Skin tone",
                ncol=len(plot_skin_tones),
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.52),
            )
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.42)
            skin_plot = output_dir / "mae_by_skin_color_compare.png"
            fig.savefig(skin_plot, dpi=150, bbox_extra_artists=(legend,))
            fig.savefig(output_dir / "mae_by_skin_color_compare.svg", bbox_extra_artists=(legend,))
            plt.close(fig)
            print(f"Saved: {skin_plot}")
            print(f"Saved: {pairwise_csv}")

            summary_rows = []
            for run_name in run_names:
                for skin_tone in plot_skin_tones:
                    vals = np.asarray(
                        [
                            float(row["mae"])
                            for row in skin_metric_rows
                            if row["run"] == run_name
                            and row["skin_color"] == skin_tone
                            and np.isfinite(row["mae"])
                        ],
                        dtype=float,
                    )
                    if vals.size == 0:
                        continue
                    summary_rows.append(
                        {
                            "run": run_name,
                            "skin_color": skin_tone,
                            "folds": int(vals.size),
                            "mae_mean": float(np.mean(vals)),
                            "mae_median": float(np.median(vals)),
                            "mae_std": float(np.std(vals)),
                            "mae_min": float(np.min(vals)),
                            "mae_max": float(np.max(vals)),
                        }
                    )
            summary_csv = output_dir / "mae_by_skin_color_summary.csv"
            with summary_csv.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(
                    fp,
                    fieldnames=[
                        "run",
                        "skin_color",
                        "folds",
                        "mae_mean",
                        "mae_median",
                        "mae_std",
                        "mae_min",
                        "mae_max",
                    ],
                )
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)
            print(f"Saved: {skin_csv}")
            print(f"Saved: {summary_csv}")

            rm_results = compute_skin_tone_repeated_measures(skin_metric_rows, run_names)
            if rm_results:
                anova_csv = output_dir / "mae_by_skin_color_repeated_measures.csv"
                with anova_csv.open("w", encoding="utf-8", newline="") as fp:
                    writer = csv.DictWriter(
                        fp,
                        fieldnames=[
                            "run",
                            "tones",
                            "n_folds",
                            "light_mae_mean",
                            "tan_mae_mean",
                            "dark_mae_mean",
                            "light_mae_median",
                            "tan_mae_median",
                            "dark_mae_median",
                            "anova_f",
                            "anova_num_df",
                            "anova_den_df",
                            "anova_p",
                            "friedman_stat",
                            "friedman_p",
                            "anova_error",
                            "friedman_error",
                        ],
                    )
                    writer.writeheader()
                    for row in rm_results:
                        writer.writerow(row)
                print(f"Saved: {anova_csv}")
        else:
            print("[warn] No finite light/tan/dark skin-tone MAE rows found; skin-tone boxplot skipped.")

    if not args.skip_variability and variability_data:
        rows = []
        for run_name, payload in variability_data.items():
            targets = payload["targets"]
            preds = payload["preds"]
            user_ids = payload["user_ids"]
            user_ages, user_stds = collect_intra_user_variability(targets, preds, user_ids)
            if user_ages.size == 0:
                print(f"[warn] {run_name}: no multi-sample users found for variability.")
                continue
            mean_vals, median_vals, counts = compute_stat_by_bin(
                user_ages, user_stds, bin_edges
            )
            variability_data[run_name]["std_mean"] = mean_vals
            variability_data[run_name]["std_median"] = median_vals
            variability_data[run_name]["user_counts"] = counts
            for label, low, high, mean_v, med_v, count in zip(
                bin_labels,
                bin_edges[:-1],
                bin_edges[1:],
                mean_vals,
                median_vals,
                counts,
            ):
                rows.append(
                    {
                        "run": run_name,
                        "bin_label": label,
                        "bin_start": float(low),
                        "bin_end": float(high),
                        "std_mean": float(mean_v) if np.isfinite(mean_v) else np.nan,
                        "std_median": float(med_v) if np.isfinite(med_v) else np.nan,
                        "user_count": int(count),
                    }
                )

        if rows:
            csv_path = output_dir / "intra_user_std_by_decade_compare.csv"
            with csv_path.open("w", encoding="utf-8") as fp:
                fp.write("run,bin_label,bin_start,bin_end,std_mean,std_median,user_count\n")
                for row in rows:
                    mean_val = f"{row['std_mean']:.6f}" if np.isfinite(row["std_mean"]) else ""
                    med_val = f"{row['std_median']:.6f}" if np.isfinite(row["std_median"]) else ""
                    fp.write(
                        f"{row['run']},{row['bin_label']},{row['bin_start']:.1f},{row['bin_end']:.1f},"
                        f"{mean_val},{med_val},{row['user_count']}\n"
                    )

            fig, ax = plt.subplots(figsize=DECADE_COMPARE_FIGSIZE)
            for run_name, payload in variability_data.items():
                if "std_mean" not in payload:
                    continue
                ax.plot(
                    bin_centers,
                    payload["std_mean"],
                    marker="o",
                    linewidth=1.6,
                    label=display_name_map.get(run_name, run_name),
                    color=get_model_color(run_name),
                )
            ax.set_xticks(bin_centers)
            ax.set_xticklabels(bin_labels, rotation=0)
            ax.set_xlabel("Age bin (years)")
            ax.set_ylabel("Intra-user prediction STD (years)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            plot_path = output_dir / "intra_user_std_by_decade_compare.png"
            fig.savefig(plot_path, dpi=150)
            fig.savefig(output_dir / "intra_user_std_by_decade_compare.svg")
            plt.close(fig)

            print(f"Saved: {plot_path}")
            print(f"Saved: {csv_path}")
    elif not args.skip_variability:
        print("[warn] No raw predictions found; intra-user variability plot skipped.")


if __name__ == "__main__":
    main()
