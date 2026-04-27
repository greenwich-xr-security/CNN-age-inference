#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from compare_runs import (
    ROC_COMPARE_FIGSIZE,
    SKIN_TONE_COLORS,
    SKIN_TONE_ORDER,
    compute_adult_gate_roc,
    compute_partial_auc,
)
from metrics import compute_adult_probabilities


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate held-out test adult-gate ROC-by-skin-color plots from saved fold predictions."
        )
    )
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Directory containing fold_* subdirectories for a single run.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Prediction split to use (default: test).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Prediction aggregation size n used in *_predictions_n{n}_ddp.npz (default: 1).",
    )
    parser.add_argument(
        "--age-threshold",
        type=float,
        default=18.0,
        help="Adult/minor threshold in years (default: 18).",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.1,
        help="Maximum FPR for partial AUC reporting and the vertical guide (default: 0.1).",
    )
    return parser.parse_args()


def _title_case_skin(tone: str) -> str:
    return {"light": "Light", "tan": "Medium", "dark": "Dark", "unlabeled": "Unlabeled"}.get(
        tone, str(tone).title()
    )


def _discover_prediction_files(run_root: Path, split: str, group_size: int) -> list[Path]:
    pred_name = f"{split}_predictions_n{group_size}_ddp.npz"
    fold_paths = sorted(
        [
            fold_dir / pred_name
            for fold_dir in run_root.glob("fold_*")
            if fold_dir.is_dir() and (fold_dir / pred_name).is_file()
        ],
        key=lambda path: path.parent.name.lower(),
    )
    return fold_paths


def _load_fold_predictions(
    pred_path: Path,
    *,
    age_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(pred_path)
    if "targets" not in data or "skin_color" not in data:
        raise ValueError(f"{pred_path} is missing required targets/skin_color arrays.")

    targets = np.asarray(data["targets"], dtype=float)
    skin_color = np.asarray(data["skin_color"]).astype(str)
    if "adult_prob" in data:
        adult_prob = np.asarray(data["adult_prob"], dtype=float)
    elif "pred_mean" in data and "pred_log_var" in data:
        adult_prob = compute_adult_probabilities(
            np.asarray(data["pred_mean"], dtype=float),
            np.asarray(data["pred_log_var"], dtype=float),
            age_threshold=age_threshold,
        )
    else:
        raise ValueError(f"{pred_path} is missing adult gate scores.")

    mask = np.isfinite(targets) & np.isfinite(adult_prob)
    return targets[mask], adult_prob[mask], skin_color[mask]


def _safe_int_mean(values: list[int]) -> int:
    if not values:
        return 0
    rounded = int(round(float(np.mean(values))))
    return rounded


def _write_csv(
    output_csv: Path,
    roc_grid: np.ndarray,
    tone_stats: dict[str, dict[str, object]],
    *,
    max_fpr: float,
) -> None:
    fieldnames = [
        "skin_color",
        "fpr",
        "tpr_mean",
        "tpr_std",
        "auc_mean",
        "auc_std",
        f"pauc_fpr_0_{int(round(max_fpr * 10))}_mean",
        f"pauc_fpr_0_{int(round(max_fpr * 10))}_std",
        f"pauc_fpr_0_{int(round(max_fpr * 10))}_norm_mean",
        f"pauc_fpr_0_{int(round(max_fpr * 10))}_norm_std",
        "folds",
        "samples_per_fold",
        "adults_per_fold",
        "minors_per_fold",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for tone in [tone for tone in SKIN_TONE_ORDER if tone in tone_stats]:
            stats = tone_stats[tone]
            for fpr_val, tpr_mean, tpr_std in zip(roc_grid, stats["tpr_mean"], stats["tpr_std"]):
                writer.writerow(
                    {
                        "skin_color": tone,
                        "fpr": float(fpr_val),
                        "tpr_mean": float(tpr_mean),
                        "tpr_std": float(tpr_std),
                        "auc_mean": float(stats["auc_mean"]),
                        "auc_std": float(stats["auc_std"]),
                        f"pauc_fpr_0_{int(round(max_fpr * 10))}_mean": float(stats["pauc_mean"]),
                        f"pauc_fpr_0_{int(round(max_fpr * 10))}_std": float(stats["pauc_std"]),
                        f"pauc_fpr_0_{int(round(max_fpr * 10))}_norm_mean": float(
                            stats["pauc_norm_mean"]
                        ),
                        f"pauc_fpr_0_{int(round(max_fpr * 10))}_norm_std": float(
                            stats["pauc_norm_std"]
                        ),
                        "folds": int(stats["folds"]),
                        "samples_per_fold": int(stats["samples_per_fold"]),
                        "adults_per_fold": int(stats["adults_per_fold"]),
                        "minors_per_fold": int(stats["minors_per_fold"]),
                    }
                )


def _write_summary_md(
    output_md: Path,
    tone_stats: dict[str, dict[str, object]],
    *,
    run_label: str,
    split: str,
    max_fpr: float,
) -> None:
    summary_rows = sorted(
        tone_stats.items(),
        key=lambda item: float(item[1]["pauc_norm_mean"]) if np.isfinite(item[1]["pauc_norm_mean"]) else -np.inf,
        reverse=True,
    )
    max_fpr_text = f"{max_fpr:.1f}".rstrip("0").rstrip(".")
    split_label = {
        "held-out test": "Held-out test",
        "validation": "Validation",
    }.get(split.strip().lower(), split)
    lines = [
        "# Adult-Gate ROC by Skin Tone Summary",
        "",
        f"Model: `{run_label}`",
        "",
        f"{split_label} summary across {sum(1 for _ in tone_stats.values()) and max(int(stats['folds']) for stats in tone_stats.values())} folds. "
        f"`pAUC@{max_fpr_text}` is the unnormalized partial AUC over `FPR <= {max_fpr_text}`.",
        "",
        "## AUC and pAUC Summary",
        "",
        "| Skin Tone | Folds | Samples/fold | Adults/fold | Minors/fold | AUC mean | AUC std | "
        f"pAUC@{max_fpr_text} mean | pAUC@{max_fpr_text} std | "
        f"pAUC@{max_fpr_text} norm mean | pAUC@{max_fpr_text} norm std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tone, stats in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _title_case_skin(tone),
                    str(int(stats["folds"])),
                    str(int(stats["samples_per_fold"])),
                    str(int(stats["adults_per_fold"])),
                    str(int(stats["minors_per_fold"])),
                    f"{float(stats['auc_mean']):.4f}",
                    f"{float(stats['auc_std']):.4f}",
                    f"{float(stats['pauc_mean']):.5f}",
                    f"{float(stats['pauc_std']):.5f}",
                    f"{float(stats['pauc_norm_mean']):.4f}",
                    f"{float(stats['pauc_norm_std']):.4f}",
                ]
            )
            + " |"
        )

    try:
        from scipy.stats import ttest_rel, wilcoxon
        from statsmodels.stats.multitest import multipletests

        tests: list[dict[str, object]] = []
        ordered_tones = [tone for tone in SKIN_TONE_ORDER if tone in tone_stats]
        for tone_a, tone_b in itertools.combinations(ordered_tones, 2):
            fold_to_pauc_a = tone_stats[tone_a]["pauc_by_fold"]
            fold_to_pauc_b = tone_stats[tone_b]["pauc_by_fold"]
            common_folds = sorted(set(fold_to_pauc_a) & set(fold_to_pauc_b))
            if len(common_folds) < 2:
                continue
            a_vals = np.asarray([fold_to_pauc_a[fold] for fold in common_folds], dtype=float)
            b_vals = np.asarray([fold_to_pauc_b[fold] for fold in common_folds], dtype=float)
            mask = np.isfinite(a_vals) & np.isfinite(b_vals)
            a_vals = a_vals[mask]
            b_vals = b_vals[mask]
            if a_vals.size < 2:
                continue
            delta = a_vals - b_vals
            ttest_p = float(ttest_rel(a_vals, b_vals).pvalue)
            try:
                wilcoxon_p = float(wilcoxon(a_vals, b_vals, zero_method="wilcox").pvalue)
            except ValueError:
                wilcoxon_p = float("nan")
            tests.append(
                {
                    "tone_a": tone_a,
                    "tone_b": tone_b,
                    "mean_delta": float(np.mean(delta)),
                    "median_delta": float(np.median(delta)),
                    "ttest_p": ttest_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )

        if tests:
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

            lines.extend(
                [
                    "",
                    "## Pairwise pAUC Comparisons",
                    "",
                    "Paired comparisons use the same held-out test folds for each skin-tone pair. "
                    "Holm-corrected p-values are reported for both the paired t-test and Wilcoxon signed-rank test.",
                    "",
                    "| Skin Tone A | Skin Tone B | Mean Delta pAUC (A-B) | Median Delta pAUC | Paired t p | Paired t Holm p | Wilcoxon p | Wilcoxon Holm p | Sig. after Holm (t) |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in tests:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _title_case_skin(str(row["tone_a"])),
                            _title_case_skin(str(row["tone_b"])),
                            f"{float(row['mean_delta']):.5f}",
                            f"{float(row['median_delta']):.5f}",
                            f"{float(row['ttest_p']):.6f}",
                            f"{float(row['ttest_p_holm']):.6f}",
                            f"{float(row['wilcoxon_p']):.6f}" if np.isfinite(float(row["wilcoxon_p"])) else "nan",
                            f"{float(row['wilcoxon_p_holm']):.6f}" if np.isfinite(float(row["wilcoxon_p_holm"])) else "nan",
                            "yes" if float(row["ttest_p_holm"]) < 0.05 else "no",
                        ]
                    )
                    + " |"
                )
    except Exception as exc:  # noqa: BLE001
        lines.extend(
            [
                "",
                "## Pairwise pAUC Comparisons",
                "",
                f"Pairwise statistical comparison could not be generated automatically: `{exc}`",
            ]
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).expanduser()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    pred_paths = _discover_prediction_files(run_root, args.split, args.group_size)
    if not pred_paths:
        raise FileNotFoundError(
            f"No {args.split}_predictions_n{args.group_size}_ddp.npz files found under {run_root}"
        )

    tone_curves: dict[str, list[dict[str, object]]] = {}
    for pred_path in pred_paths:
        fold_name = pred_path.parent.name
        targets, adult_prob, skin_color = _load_fold_predictions(
            pred_path,
            age_threshold=args.age_threshold,
        )
        for tone in SKIN_TONE_ORDER + ("unlabeled",):
            tone_mask = skin_color.astype(str) == tone
            if not np.any(tone_mask):
                continue
            tone_targets = targets[tone_mask]
            tone_probs = adult_prob[tone_mask]
            adult_total = int(np.sum(tone_targets >= args.age_threshold))
            minor_total = int(np.sum(tone_targets < args.age_threshold))
            if adult_total <= 0 or minor_total <= 0:
                continue
            fpr, tpr, auc = compute_adult_gate_roc(
                tone_targets,
                tone_probs,
                age_threshold=args.age_threshold,
                num_thresholds=201,
            )
            pauc, pauc_norm = compute_partial_auc(fpr, tpr, max_fpr=args.max_fpr)
            tone_curves.setdefault(tone, []).append(
                {
                    "fold": fold_name,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": float(auc),
                    "pauc": float(pauc),
                    "pauc_norm": float(pauc_norm),
                    "samples": int(tone_targets.size),
                    "adults": adult_total,
                    "minors": minor_total,
                }
            )

    if not tone_curves:
        raise RuntimeError("No valid skin-color ROC curves could be computed.")

    roc_grid = np.linspace(0.0, 1.0, 201)
    tone_stats: dict[str, dict[str, object]] = {}
    for tone in [tone for tone in SKIN_TONE_ORDER + ("unlabeled",) if tone in tone_curves]:
        rows = tone_curves[tone]
        interp_tprs = [
            np.interp(roc_grid, np.asarray(row["fpr"], dtype=float), np.asarray(row["tpr"], dtype=float), left=0.0, right=1.0)
            for row in rows
        ]
        tpr_matrix = np.vstack(interp_tprs)
        aucs = np.asarray([float(row["auc"]) for row in rows], dtype=float)
        paucs = np.asarray([float(row["pauc"]) for row in rows], dtype=float)
        paucs_norm = np.asarray([float(row["pauc_norm"]) for row in rows], dtype=float)
        tone_stats[tone] = {
            "tpr_mean": np.mean(tpr_matrix, axis=0),
            "tpr_std": np.std(tpr_matrix, axis=0),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "pauc_mean": float(np.mean(paucs)),
            "pauc_std": float(np.std(paucs)),
            "pauc_norm_mean": float(np.mean(paucs_norm)),
            "pauc_norm_std": float(np.std(paucs_norm)),
            "folds": int(len(rows)),
            "samples_per_fold": _safe_int_mean([int(row["samples"]) for row in rows]),
            "adults_per_fold": _safe_int_mean([int(row["adults"]) for row in rows]),
            "minors_per_fold": _safe_int_mean([int(row["minors"]) for row in rows]),
            "pauc_by_fold": {str(row["fold"]): float(row["pauc"]) for row in rows},
        }

    prefix = f"{args.split}_roc_adult_gate_by_skin_color_kfold_n{args.group_size}"
    csv_path = run_root / f"{prefix}.csv"
    png_path = run_root / f"{prefix}.png"
    pdf_path = run_root / f"{prefix}.pdf"
    md_path = run_root / f"{prefix}_summary.md"

    _write_csv(csv_path, roc_grid, tone_stats, max_fpr=args.max_fpr)

    fig, ax = plt.subplots(figsize=ROC_COMPARE_FIGSIZE)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#777777", label="Chance")
    ax.axvline(args.max_fpr, linestyle=":", linewidth=1.0, color="#555555")
    ax.text(
        args.max_fpr + 0.003,
        0.04,
        f"FPR={args.max_fpr:.1f}",
        rotation=90,
        fontsize=7,
        color="#555555",
        ha="left",
        va="bottom",
    )

    for tone in [tone for tone in SKIN_TONE_ORDER if tone in tone_stats]:
        stats = tone_stats[tone]
        label = f"{_title_case_skin(tone)}\nAUC {float(stats['auc_mean']):.3f}"
        if np.isfinite(float(stats["auc_std"])):
            label += f" ± {float(stats['auc_std']):.3f}"
        if np.isfinite(float(stats["pauc_mean"])):
            label += f" | pAUC@{args.max_fpr:.1f} {float(stats['pauc_mean']):.3f}"
        color = SKIN_TONE_COLORS.get(tone, "#7f7f7f")
        ax.plot(roc_grid, stats["tpr_mean"], linewidth=2.0, label=label, color=color)
        ax.fill_between(
            roc_grid,
            np.clip(stats["tpr_mean"] - stats["tpr_std"], 0.0, 1.0),
            np.clip(stats["tpr_mean"] + stats["tpr_std"], 0.0, 1.0),
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Adult-Gate ROC by Skin Tone [{args.split.upper()}] (k-fold mean, n={args.group_size})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_box_aspect(1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    run_label = run_root.name
    if run_label == "swin-v2-base_384x384_ddp":
        run_label = "SwinV2-B"
    _write_summary_md(md_path, tone_stats, run_label=run_label, split="held-out test" if args.split == "test" else "validation", max_fpr=args.max_fpr)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
