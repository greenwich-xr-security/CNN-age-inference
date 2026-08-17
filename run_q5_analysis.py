"""Q5 Stratified Residual Analysis.

Loads Q4 test-set predictions for all 21 arm×seed combinations (3 arms × 7
seeds, 5 folds each), computes per-image seed-averaged |error|, joins with
image-level metadata (ITA, sex, lighting, age), fits a Linear Mixed-Effects
Model with arm × covariate interactions, runs stratified Wilcoxon signed-rank
tests, and generates Figures 5a–5e.

Outputs: runs/q5_residual_analysis/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
Q5_ART = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich"
    r"\slurm_runs\CNN age inference"
    r"\publication_IJCB_2026\q5_artifacts"
)
META_CSV = Q5_ART / "q5_test_image_metadata.csv"
OUT_DIR = Path(r"C:\Users\Staff\Documents\GitHub\CNN-age-inference\runs\q5_residual_analysis")

N_IMAGES = 1624
ARMS = ["R0", "Pooled", "S-age"]
SEEDS = list(range(42, 49))

# Consistent colour scheme matching Q4 figures
ARM_COLORS = {"R0": "#888888", "Pooled": "#1f77b4", "S-age": "#ff7f0e"}

# Age-bin definitions (right-inclusive upper edge uses strict < for minor)
AGE_BINS  = [0, 18, 30, 50, 200]
AGE_LABELS = ["minor", "young-adult", "mid-adult", "older-adult"]

# ---------------------------------------------------------------------------
# Arm → directory mapping
# ---------------------------------------------------------------------------
# Each entry maps (arm, seed) → list of (run_dir_name, [fold_indices]).
# Seed 42 S-age is split across two directories (fold 4 from the non-rerun
# directory, folds 0-3 from the rerun directory).
ARM_SEED_DIRS: dict[tuple[str, int], list[tuple[str, list[int]]]] = {
    ("R0", 42): [
        ("q3_r0_from1050753_real_finetune_lr2e5_seed42_v2s_384", list(range(5)))
    ],
    ("Pooled", 42): [
        ("v2s_hrgbd_real_finetuning_nll_lr2e5_uncapped_test20_k5_4gpu_16cpu_20260730_384",
         list(range(5)))
    ],
    ("S-age", 42): [
        # Rerun dir has folds 0-2; fold 3 is missing from both dirs; original has fold 4
        ("q3_sage_realfrac_f100_lr2e5_seed42_v2s_384_rerun_20260804", [0, 1, 2]),
        ("q3_sage_realfrac_f100_lr2e5_seed42_v2s_384", [4]),
    ],
}
for _seed in range(43, 49):
    ARM_SEED_DIRS[("R0", _seed)] = [
        (f"minpair_seed{_seed}_real_from_real_lr2e5_v2s_384", list(range(5)))
    ]
    ARM_SEED_DIRS[("Pooled", _seed)] = [
        (f"minpair_seed{_seed}_real_from_syn2_lr2e5_v2s_384", list(range(5)))
    ]
    ARM_SEED_DIRS[("S-age", _seed)] = [
        (f"q4_sage_seed{_seed}_real_ft_lr2e5_v2s_384", list(range(5)))
    ]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_fold_errors(arm: str, seed: int) -> np.ndarray:
    """Return per-image |error| averaged across all 5 folds for one (arm, seed)."""
    entries = ARM_SEED_DIRS[(arm, seed)]
    fold_arrays: list[np.ndarray] = []
    for dir_name, fold_indices in entries:
        for fi in fold_indices:
            npz_path = Q5_ART / dir_name / f"fold_{fi}" / "test_predictions_raw_ddp.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing NPZ: {npz_path}")
            npz = np.load(npz_path, allow_pickle=True)
            fold_arrays.append(np.abs(npz["pred_mean"] - npz["targets"]))
    if len(fold_arrays) != 5:
        warnings.warn(f"({arm}, seed={seed}): expected 5 folds, got {len(fold_arrays)} — averaging over available folds.")
    return np.mean(fold_arrays, axis=0)  # (N_IMAGES,)


def build_error_long(meta: pd.DataFrame) -> pd.DataFrame:
    """Build long-format DataFrame: (row_index, arm, abs_error) × 3 arms."""
    rows = []
    for arm in ARMS:
        # Average per-image |error| across 7 seeds for this arm
        seed_errors = np.stack([load_fold_errors(arm, s) for s in SEEDS], axis=0)
        arm_mean = seed_errors.mean(axis=0)  # (N_IMAGES,)
        for i in range(N_IMAGES):
            rows.append({"row_index": i, "arm": arm, "abs_error": float(arm_mean[i])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def signed_rank_biserial(diff: np.ndarray) -> float:
    """Signed rank-biserial correlation for paired Wilcoxon (positive = diff > 0 dominates)."""
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    t_pos = float(ranks[nonzero > 0].sum())
    t_neg = float(ranks[nonzero < 0].sum())
    return (t_pos - t_neg) / (t_pos + t_neg)


def wilcoxon_pair(
    wide: pd.DataFrame, arm_a: str, arm_b: str, factor: str, level: str, n_bonf: int
) -> dict:
    """Run paired Wilcoxon (arm_a vs arm_b) on rows filtered to a stratum level."""
    diff = wide[arm_a].values - wide[arm_b].values  # positive → arm_a larger (worse)
    n = len(diff)
    if n < 5:
        return {}
    stat, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    return {
        "factor": factor,
        "level": str(level),
        "comparison": f"{arm_a} vs {arm_b}",
        "n_images": n,
        "W": float(stat),
        "p_two_sided": float(p),
        "p_bonf": min(float(p) * n_bonf, 1.0),
        "rank_biserial_r": signed_rank_biserial(diff),
        "mean_diff_years": float(np.mean(diff)),
        "significant_bonf": bool(float(p) * n_bonf < 0.05),
    }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def violin_by_factor(
    long_df: pd.DataFrame,
    factor_col: str,
    factor_levels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Violin plot: one panel per factor level, three violins per panel (one per arm)."""
    fig, axes = plt.subplots(
        1, len(factor_levels),
        figsize=(3.0 * len(factor_levels), 4.2),
        sharey=True,
    )
    if len(factor_levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, factor_levels):
        sub = long_df[long_df[factor_col] == level]
        sns.violinplot(
            data=sub,
            x="arm",
            y="abs_error",
            order=ARMS,
            palette=ARM_COLORS,
            inner="quartile",
            linewidth=0.8,
            ax=ax,
        )
        n_imgs = len(sub) // len(ARMS)
        ax.set_title(f"{level}\n(n={n_imgs})", fontsize=9)
        ax.set_xlabel("")
        ax.set_xticklabels(ARMS, rotation=15, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)

    axes[0].set_ylabel("|error| (years)", fontsize=9)
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Load metadata
    # -------------------------------------------------------------------
    print("Loading metadata...")
    meta = pd.read_csv(META_CSV)
    assert len(meta) == N_IMAGES, f"Expected {N_IMAGES} rows, got {len(meta)}"

    meta["age_bin"] = pd.cut(
        meta["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    # lights_label (on/off/NaN from prolific) → lighting ('on'/'off'/'unknown')
    meta["lighting"] = meta["lights_label"].fillna("unknown")
    meta["sex"] = meta["gender"]  # male / female
    meta["subject_id"] = meta["user_id"]
    meta["true_age"] = meta["age"]
    meta["ITA"] = meta["ITA_score"].astype(float)

    # -------------------------------------------------------------------
    # Load per-image errors
    # -------------------------------------------------------------------
    print("Loading NPZ prediction arrays (21 arm×seed combinations × 5 folds)...")
    err_long = build_error_long(meta)

    # Merge metadata into long frame
    keep_cols = [
        "row_index", "subject_id", "true_age", "age_bin",
        "sex", "lighting", "ITA", "ITA_group", "source", "image_path",
    ]
    long_df = err_long.merge(meta[keep_cols], on="row_index")
    long_df["arm"] = pd.Categorical(long_df["arm"], categories=ARMS)

    long_df.to_csv(OUT_DIR / "per_image_errors.csv", index=False)
    print(f"  Saved per_image_errors.csv ({len(long_df)} rows = {N_IMAGES} images × 3 arms)")

    # Wide format (one row per image, columns per arm) — convenient for Wilcoxon
    wide_df = (
        err_long.pivot(index="row_index", columns="arm", values="abs_error")
        .reset_index()
        .merge(meta[keep_cols], on="row_index")
    )
    wide_df.to_csv(OUT_DIR / "per_image_errors_wide.csv", index=False)

    # -------------------------------------------------------------------
    # Descriptive counts
    # -------------------------------------------------------------------
    print("\nDescriptive counts...")
    one_arm = wide_df.copy()  # one row per image

    desc_rows = []
    for factor, col, levels in [
        ("age_bin",   "age_bin",  AGE_LABELS),
        ("sex",       "sex",      ["male", "female"]),
        ("lighting",  "lighting", ["on", "off", "unknown"]),
        ("ITA_group", "ITA_group", ["light", "medium", "dark"]),
    ]:
        for level in levels:
            sub = one_arm[one_arm[col].astype(str) == str(level)]
            desc_rows.append({
                "factor":     factor,
                "level":      str(level),
                "n_images":   len(sub),
                "n_subjects": sub["subject_id"].nunique(),
            })

    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(OUT_DIR / "descriptive_counts.csv", index=False)
    print(desc_df.to_string(index=False))

    # Flag underpowered groups
    for _, row in desc_df.iterrows():
        if row["n_subjects"] < 10:
            print(f"  WARNING: {row['factor']}={row['level']} has only "
                  f"{row['n_subjects']} subjects — underpowered for LMM interaction.")

    # -------------------------------------------------------------------
    # Figures 5a–5d: violin plots per factor
    # -------------------------------------------------------------------
    print("\nGenerating violin plots...")

    violin_by_factor(
        long_df[long_df["age_bin"].notna()],
        factor_col="age_bin",
        factor_levels=AGE_LABELS,
        title="|Error| by age bin × arm",
        out_path=OUT_DIR / "fig_5a_age_violin.png",
    )
    violin_by_factor(
        long_df[long_df["sex"].isin(["male", "female"])],
        factor_col="sex",
        factor_levels=["male", "female"],
        title="|Error| by sex × arm",
        out_path=OUT_DIR / "fig_5b_sex_violin.png",
    )
    violin_by_factor(
        long_df[long_df["lighting"].isin(["on", "off"])],
        factor_col="lighting",
        factor_levels=["on", "off"],
        title="|Error| by lighting × arm",
        out_path=OUT_DIR / "fig_5c_lighting_violin.png",
    )
    violin_by_factor(
        long_df,
        factor_col="ITA_group",
        factor_levels=["light", "medium", "dark"],
        title="|Error| by ITA group (Chardon 3-class) × arm",
        out_path=OUT_DIR / "fig_5d_ita_violin.png",
    )

    # -------------------------------------------------------------------
    # Figure 5e: |error| vs continuous ITA with LOWESS per arm
    # -------------------------------------------------------------------
    print("Generating Figure 5e...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
    for ax, arm in zip(axes, ARMS):
        sub = long_df[(long_df["arm"] == arm) & long_df["ITA"].notna()].copy()
        ax.scatter(sub["ITA"], sub["abs_error"],
                   alpha=0.12, s=7, color=ARM_COLORS[arm], rasterized=True)
        xy = sub[["ITA", "abs_error"]].dropna()
        if len(xy) > 30:
            smoothed = lowess(xy["abs_error"].values, xy["ITA"].values,
                              frac=0.35, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1.8, label="LOWESS")
        ax.axvline(x=41,  ls="--", lw=0.9, color="#666666", alpha=0.7)
        ax.axvline(x=-10, ls=":",  lw=0.9, color="#666666", alpha=0.7)
        ax.set_xlabel("ITA (degrees)", fontsize=9)
        ax.set_title(arm, color=ARM_COLORS[arm], fontweight="bold", fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("|error| (years)", fontsize=9)
    # Threshold annotations on the rightmost panel
    axes[2].text(41 + 0.5, axes[2].get_ylim()[1] * 0.95, "41° (light/med)",
                 fontsize=7, va="top", color="#555555")
    axes[2].text(-10 + 0.5, axes[2].get_ylim()[1] * 0.85, "-10° (med/dark)",
                 fontsize=7, va="top", color="#555555")
    fig.suptitle("|Error| vs ITA (skin tone) per arm  [LOWESS frac=0.35]", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_5e_ita_scatter_lowess.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_5e_ita_scatter_lowess.png")

    # -------------------------------------------------------------------
    # Linear Mixed-Effects Model
    # -------------------------------------------------------------------
    print("\nFitting Linear Mixed-Effects Model...")
    # Restrict to HandRGBD rows (known lighting) for the full model
    lmm_df = long_df[long_df["lighting"].isin(["on", "off"])].copy()
    lmm_df = lmm_df.dropna(subset=["ITA", "sex", "lighting", "true_age", "abs_error"])
    lmm_df["arm"] = pd.Categorical(lmm_df["arm"].astype(str), categories=ARMS)
    print(f"  LMM dataset: {len(lmm_df)} rows, "
          f"{lmm_df['row_index'].nunique()} images, "
          f"{lmm_df['subject_id'].nunique()} subjects")

    formula = "abs_error ~ arm * (true_age + ITA + C(sex) + C(lighting))"
    lmm_ok = False
    try:
        model = smf.mixedlm(formula, data=lmm_df, groups=lmm_df["subject_id"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method="lbfgs", maxiter=1000)
        summary_txt = str(result.summary())
        (OUT_DIR / "lmm_full_summary.txt").write_text(summary_txt)
        print("  Converged. Summary saved to lmm_full_summary.txt")
        lmm_ok = True

        # Extract fixed effects (bse_fe is fixed-effects-only SE; tvalues/pvalues
        # include variance components in some statsmodels versions, so compute manually)
        fe = result.fe_params
        se = result.bse_fe
        zvals = fe / se
        from scipy.stats import norm as _norm
        pvals = pd.Series(2 * _norm.sf(np.abs(zvals.values)), index=fe.index)
        coef_df = pd.DataFrame({
            "term": fe.index,
            "coef": fe.values,
            "se":   se.values,
            "z":    zvals.values,
            "p":    pvals.values,
        })
        coef_df["ci_lower"] = coef_df["coef"] - 1.96 * coef_df["se"]
        coef_df["ci_upper"] = coef_df["coef"] + 1.96 * coef_df["se"]

        # Interaction terms only, Bonferroni across 4 interaction groups
        ix = coef_df[coef_df["term"].str.contains(":")].copy()
        ix["p_bonf"] = np.minimum(ix["p"] * 4, 1.0)
        ix.to_csv(OUT_DIR / "lmm_interactions.csv", index=False)

        print(f"\n  Interaction terms ({len(ix)} terms, Bonferroni × 4):")
        pd.set_option("display.width", 140)
        print(ix[["term", "coef", "se", "z", "p", "p_bonf"]].to_string(index=False))

    except Exception as exc:
        msg = f"Full LMM failed: {exc}"
        print(f"  {msg} — running fallback two-way models.")
        (OUT_DIR / "lmm_full_summary.txt").write_text(msg + "\n")

    if not lmm_ok:
        for covar, formula_fb in [
            ("true_age", "abs_error ~ arm * true_age"),
            ("ITA",      "abs_error ~ arm * ITA"),
            ("sex",      "abs_error ~ arm * C(sex)"),
            ("lighting", "abs_error ~ arm * C(lighting)"),
        ]:
            sub = lmm_df.dropna(subset=[covar, "abs_error"])
            try:
                r = smf.mixedlm(formula_fb, data=sub,
                                groups=sub["subject_id"]).fit(method="lbfgs", maxiter=500)
                (OUT_DIR / f"lmm_fallback_{covar}.txt").write_text(str(r.summary()))
                print(f"  Fallback arm×{covar}: saved")
            except Exception as exc2:
                print(f"  Fallback arm×{covar} also failed: {exc2}")

    # -------------------------------------------------------------------
    # Post-hoc stratified Wilcoxon signed-rank
    # -------------------------------------------------------------------
    print("\nRunning stratified Wilcoxon signed-rank tests...")

    wilcox_rows: list[dict] = []
    strata = [
        ("age_bin",   "age_bin",   AGE_LABELS,              4),
        ("sex",       "sex",       ["male", "female"],       2),
        ("lighting",  "lighting",  ["on", "off"],            2),
        ("ITA_group", "ITA_group", ["light", "medium", "dark"], 3),
    ]
    for factor, col, levels, n_bonf in strata:
        for level in levels:
            sub = wide_df[wide_df[col].astype(str) == str(level)]
            if len(sub) < 5:
                continue
            for arm_a, arm_b in [("R0", "Pooled"), ("S-age", "Pooled")]:
                rec = wilcoxon_pair(sub, arm_a, arm_b, factor, level, n_bonf)
                if rec:
                    wilcox_rows.append(rec)

    wilcox_df = pd.DataFrame(wilcox_rows)
    wilcox_df.to_csv(OUT_DIR / "posthoc_wilcoxon.csv", index=False)
    pd.set_option("display.max_colwidth", 30)
    print(wilcox_df[[
        "factor","level","comparison","n_images",
        "W","p_two_sided","p_bonf","rank_biserial_r","mean_diff_years","significant_bonf"
    ]].to_string(index=False))

    print(f"\nDone. All outputs written to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
