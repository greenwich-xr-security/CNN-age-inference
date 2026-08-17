# Q5 Stratified Residual Analysis Plan

## Purpose

Q5 is a post-hoc stratified error analysis on the locked test set, using the
Q4 arm results. The central question is whether the `Pooled` advantage over `R0`
(and `S-age`) observed at the aggregate level is uniform across subpopulations,
or whether it is concentrated in — or driven by — specific groups.

This is not a new training experiment. No new checkpoints are required. The
analysis consumes the per-image predictions already produced by `evaluate_test.py`
for each Q4 arm × seed combination.

**Research questions:**
- Does the `Pooled` MAE advantage hold across age bins, or is it dominated by a
  specific range (e.g. borderline adults 16–24)?
- Is the model more biased for darker skin tones, and does synthetic pretraining
  close or widen that gap?
- Does lighting condition (lights-on vs lights-off) interact with arm?
- Does the `S-age` AUC degradation (visible in Q4 aggregate) come from a specific
  subgroup, e.g. subjects near the 18-year boundary?

---

## Test Set

- File: `splits/test_users_uncapped_20pct_seed42.json`
- Size: **1,624 images**, ~20% of total users (locked, never touched during training)
- Model: EfficientNet-V2-S, 384 px, pure Gaussian NLL
- Arms × seeds: `R0`, `Pooled`, `S-age` × seeds 42–48 (7 seeds per arm)

---

## Stratification Factors

### 1. Age bin

Derived from ground-truth age column in metadata. Four bins:

| Bin label | Age range | Rationale |
|---|---|---|
| `minor` | < 18 | Protected group (FPR denominator for Case 1) |
| `young-adult` | 18–29 | Borderline adults; highest confusion with minors |
| `mid-adult` | 30–49 | Model comfortable zone |
| `older-adult` | ≥ 50 | Sparse training data, potentially higher error |

### 2. Sex

Binary (`male` / `female`) from subject metadata. Available directly in the
combined metadata DataFrame. Run Mann-Whitney U test per arm within each bin.

### 3. Lighting

Derived from the image filename. Filenames encode `lights-on` or `lights-off`
as a literal token. Parse via regex on the `image_path` column:

```python
import re
def parse_lighting(path: str) -> str:
    if re.search(r'lights[-_]off', path, re.IGNORECASE):
        return 'lights-off'
    if re.search(r'lights[-_]on', path, re.IGNORECASE):
        return 'lights-on'
    return 'unknown'
```

Rows with `unknown` lighting are excluded from the lighting-stratified analysis
but retained in all other strata.

### 4. Skin tone — ITA score (Chardon 3-group)

**Computation.** For each image, extract a hand-ROI patch (using the existing
binary mask where available, otherwise the full crop), convert to CIE L\*a\*b\*,
and compute:

```
ITA = arctan2(L* - 50, b*) × (180 / π)
```

where `arctan2` is used to handle the quadrant correctly (though in practice
L\* and b\* values for skin keep the angle in a well-defined range with plain
`arctan`).

**Implementation.**

```python
from PIL import Image
import numpy as np

def compute_ita(image_path: str, mask_path: str | None = None) -> float:
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0

    if mask_path is not None:
        mask = np.array(Image.open(mask_path).convert('L')) > 128
        arr[~mask] = np.nan

    # sRGB -> linear RGB (approximate gamma)
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    # Linear RGB -> XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = linear.reshape(-1, 3) @ M.T

    # XYZ -> L*a*b* (D65 reference white)
    ref = np.array([0.95047, 1.00000, 1.08883])
    xyz /= ref
    eps, kappa = 0.008856, 903.3
    f = np.where(xyz > eps, xyz ** (1/3), (kappa * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    b = 200 * (f[:, 1] - f[:, 2])

    # Mask NaN (background) pixels
    valid = np.isfinite(L) & np.isfinite(b)
    if valid.sum() == 0:
        return float('nan')

    L_mean = float(np.nanmean(L[valid]))
    b_mean = float(np.nanmean(b[valid]))
    return float(np.degrees(np.arctan2(L_mean - 50, b_mean)))
```

**Chardon 3-group thresholds** (collapsed from Chardon et al. 1991, 6 groups):

| Group | ITA range | Chardon original |
|---|---|---|
| `light` | ITA > 41° | Very light + Light |
| `medium` | −10° < ITA ≤ 41° | Intermediate + Tan |
| `dark` | ITA ≤ −10° | Brown + Dark |

These thresholds are fixed prior to any examination of the test set distribution.
Report per-group N alongside results; if `dark` contains fewer than 10 subjects,
note this as a limitation and collapse `dark` + `medium` for the LMM (retain the
3-group split for descriptive plots).

**Practical note.** ITA is computed from the dorsal hand, not the face. Values
will be systematically lower than face-based ITA for the same subject due to
knuckle texture lowering L\*. This is acceptable for within-dataset relative
stratification but should not be compared to face-ITA norms in external work.

---

## Statistical Analysis

### Primary model — Linear Mixed-Effects Model (LMM)

Unit of analysis: **per-image absolute residual**, averaged across the 7 seeds
for that image before model fitting.

```
|error|_ij ~ arm × (true_age + ITA_continuous + sex + lighting) + (1 | subject_j)
```

| Term | Type | Levels / range |
|---|---|---|
| `arm` | Fixed, categorical | `R0`, `Pooled`, `S-age` |
| `true_age` | Fixed, continuous | raw years |
| `ITA_continuous` | Fixed, continuous | degrees (raw ITA, not binned) |
| `sex` | Fixed, categorical | `male`, `female` |
| `lighting` | Fixed, categorical | `lights-on`, `lights-off` |
| `arm × covariate` | Interaction (fixed) | all four two-way interactions |
| `(1 \| subject)` | Random intercept | one per unique user |

The interaction terms are the primary estimands. `arm × ITA_continuous` tests
whether the `Pooled` advantage is larger for darker skin. `arm × true_age`
tests whether the advantage concentrates near the 18-year boundary.

**Implementation.** Use `statsmodels.MixedLM` (pure Python):

```python
import statsmodels.formula.api as smf

# df has columns: abs_error, arm, true_age, ITA, sex, lighting, subject_id
df['arm'] = pd.Categorical(df['arm'], categories=['R0', 'Pooled', 'S-age'])

model = smf.mixedlm(
    "abs_error ~ arm * (true_age + ITA + C(sex) + C(lighting))",
    data=df,
    groups=df['subject_id'],
)
result = model.fit(method='lbfgs')
print(result.summary())
```

Report: coefficient estimates, z-scores, p-values (two-sided), and partial η²
for each fixed effect. Apply Bonferroni correction across the four interaction
terms (multiplicity = 4).

**If the full interaction model fails to converge** (likely if `dark` N < 10 and
ITA has low variance), fall back to separate two-way LMMs:

```
|error| ~ arm × true_age + (1 | subject)
|error| ~ arm × ITA + (1 | subject)
|error| ~ arm × C(sex) + (1 | subject)
|error| ~ arm × C(lighting) + (1 | subject)
```

### Post-hoc — Stratified Wilcoxon signed-rank

For each factor bin, test `Pooled` vs `R0` (primary comparison) and
`Pooled` vs `S-age` (secondary) using paired Wilcoxon signed-rank on the
per-image seed-averaged absolute errors.

- Pairing: by image (same test image, different arm)
- Effect size: rank-biserial correlation r
- Correction: Bonferroni across bins within each factor (e.g. 4 bins for age
  → α = 0.05 / 4 = 0.0125 per test)

---

## Figures

| Figure | Content |
|---|---|
| 5a | Violin plots of \|error\| per age bin, one panel per arm, arms overlaid |
| 5b | Violin plots of \|error\| per sex × arm |
| 5c | Violin plots of \|error\| per lighting condition × arm |
| 5d | Violin plots of \|error\| per ITA group (3-group) × arm |
| 5e | Scatter: \|error\| vs continuous ITA, per arm, with LOWESS smoother |
| 5f | LMM coefficient plot: interaction terms arm × covariate, with 95% CI |

For 5a–5d: use a consistent colour scheme — `R0` grey, `Pooled` blue,
`S-age` orange — matching any Q4 figures already produced.

For 5e: this is the main visual contribution for skin tone. Plot each arm as a
separate panel or overlay with transparency. The LOWESS smoother reveals whether
error rises monotonically as ITA decreases (darker skin harder) or whether there
is a non-linear pattern.

---

## Implementation Steps

1. **Collect predictions.** For each arm × seed, locate the `evaluate_test.py`
   output CSV (per-image `mu`, `log_var`, `true_age`, `user_id`, `image_path`).
   Compute `|error| = |mu - true_age|` per image. Average across the 7 seeds to
   get one `abs_error` per image.

2. **Parse lighting.** Apply the regex to `image_path`. Exclude `unknown` rows
   from lighting analyses.

3. **Compute ITA.** Run `compute_ita()` on each unique image (not per seed —
   ITA is image-level). Store in a lookup keyed by `image_path`. Apply binary
   mask where available via `AgeDataset._resolve_mask_path()`.

4. **Join factors.** Merge `abs_error` table with ITA lookup and metadata
   (sex, true_age, user_id). Assign Chardon group from continuous ITA.

5. **Descriptive counts.** Report N images and N unique subjects per ITA group,
   per lighting condition, per sex, per age bin. Flag any group with < 10 subjects.

6. **Fit LMM.** Run the primary model. If singular/non-convergent, run the four
   separate two-way models.

7. **Post-hoc Wilcoxon.** Run within each factor bin, Bonferroni-correct.

8. **Figures.** Generate 5a–5f. Save to `runs/q5_residual_analysis/`.

---

## Outputs

All outputs written to `runs/q5_residual_analysis/`:

```
runs/q5_residual_analysis/
├── per_image_errors.csv          # image_path, arm, abs_error (seed-avg), ITA, group, sex, lighting, true_age, subject_id
├── ita_scores.csv                # image_path, ITA_raw, ITA_group
├── descriptive_counts.csv        # N images, N subjects per stratum
├── lmm_full_summary.txt          # statsmodels result.summary() output
├── lmm_interactions.csv          # interaction term coefficients + CIs + p-values
├── posthoc_wilcoxon.csv          # per-bin Wilcoxon results, effect sizes
├── fig_5a_age_violin.png
├── fig_5b_sex_violin.png
├── fig_5c_lighting_violin.png
├── fig_5d_ita_violin.png
└── fig_5e_ita_scatter_lowess.png
```

---

## Results (run 2026-08-12)

Script: `run_q5_analysis.py`. Outputs: `runs/q5_residual_analysis/`.

Note: seed 42 S-age fold 3 was missing from both the rerun and original directories;
seed 42 S-age is averaged over 4 folds (0, 1, 2, 4) instead of 5.

### Descriptive counts

| Factor | Level | N images | N subjects |
|---|---:|---:|---:|
| age_bin | minor | 439 | 37 |
| age_bin | young-adult | 352 | 30 |
| age_bin | mid-adult | 556 | 42 |
| age_bin | older-adult | 277 | 19 |
| sex | male | 776 | 62 |
| sex | female | 848 | 66 |
| lighting | on | 529 | 88 |
| lighting | off | 496 | 86 |
| lighting | unknown (ProlificHands) | 599 | 40 |
| ITA_group | light (>41°) | 37 | **7** |
| ITA_group | medium | 516 | 90 |
| ITA_group | dark (≤−10°) | 1071 | 120 |

`light` ITA group has only 7 subjects — underpowered. Reported descriptively only.

### LMM results

Full model: `abs_error ~ arm × (true_age + ITA + C(sex) + C(lighting)) + (1|subject)`.
Restricted to HandRGBD rows (known lighting), n=3075 rows, 88 subjects. Converged.

| Interaction term | Coef | SE | z | p (raw) | p (Bonf×4) | Sig? |
|---|---:|---:|---:|---:|---:|---|
| arm[Pooled]:true_age | −0.031 | 0.008 | −3.71 | 0.00021 | **0.00084** | ✓ |
| arm[S-age]:true_age | −0.072 | 0.008 | −8.73 | 2.6e-18 | **1.0e-17** | ✓ |
| arm[Pooled]:ITA | +0.002 | 0.004 | +0.52 | 0.606 | 1.00 | ✗ |
| arm[S-age]:ITA | +0.005 | 0.004 | +1.39 | 0.164 | 0.655 | ✗ |
| arm[Pooled]:sex[male] | −0.042 | 0.203 | −0.21 | 0.837 | 1.00 | ✗ |
| arm[S-age]:sex[male] | −0.440 | 0.203 | −2.16 | 0.031 | 0.122 | ✗ |
| arm[Pooled]:lighting[on] | +0.031 | 0.202 | +0.15 | 0.880 | 1.00 | ✗ |
| arm[S-age]:lighting[on] | +0.154 | 0.202 | +0.76 | 0.447 | 1.00 | ✗ |

**The only significant interaction after Bonferroni correction is arm × true_age.** Both Pooled
and S-age reduce |error| more strongly for older subjects. Neither ITA (skin tone) nor
sex nor lighting shows a significant arm interaction.

At the mean test-set age (~32 years): Pooled advantage over R0 ≈ 0.56 − 0.031×32 = −0.43 years
(aligns with Q4 aggregate −0.36 year advantage). The crossover age (where Pooled starts to beat R0)
is approximately 0.561 / 0.031 ≈ 18 years.

### Post-hoc Wilcoxon (stratified, Bonferroni-corrected within factor)

Primary comparison: `R0 vs Pooled`. Positive mean_diff = Pooled is better.

**Age bin (Bonf × 4):**

| Level | Comparison | mean diff (yr) | rank-biserial r | p_bonf | Sig? |
|---|---|---:|---:|---:|---|
| minor | R0 vs Pooled | 0.041 | 0.205 | 0.00079 | ✓ |
| minor | S-age vs Pooled | 0.479 | 0.283 | 1.2e-6 | ✓ |
| young-adult | R0 vs Pooled | 0.027 | 0.149 | 0.061 | ✗ |
| young-adult | S-age vs Pooled | 0.603 | 0.347 | 6.9e-8 | ✓ |
| mid-adult | R0 vs Pooled | 0.107 | 0.131 | 0.030 | ✓ |
| mid-adult | S-age vs Pooled | 0.291 | 0.146 | 0.012 | ✓ |
| older-adult | R0 vs Pooled | **1.798** | **0.9999** | **1.4e-46** | ✓ |
| older-adult | S-age vs Pooled | **−1.013** | **−0.501** | **2.0e-12** | ✓ (S-age WINS) |

Key finding: the Pooled advantage is almost entirely concentrated in `older-adult` (≥50 years).
Across minors and young adults, the absolute advantage is tiny (0.04–0.03 yr) despite being
statistically significant. Counterintuitively, **S-age beats Pooled for subjects ≥50**.

**Sex (Bonf × 2):**

| Level | Comparison | mean diff (yr) | p_bonf | Sig? |
|---|---|---:|---:|---|
| male | R0 vs Pooled | 0.408 | 2.5e-30 | ✓ |
| male | S-age vs Pooled | 0.012 | 1.00 | ✗ |
| female | R0 vs Pooled | 0.316 | 2.2e-14 | ✓ |
| female | S-age vs Pooled | 0.347 | 1.1e-5 | ✓ |

S-age is competitive with Pooled for male subjects but not female.

**Lighting (Bonf × 2):**

| Level | Comparison | mean diff (yr) | p_bonf | Sig? |
|---|---|---:|---:|---|
| on | R0 vs Pooled | 0.212 | 1.4e-9 | ✓ |
| on | S-age vs Pooled | 0.372 | 8.2e-5 | ✓ |
| off | R0 vs Pooled | 0.252 | 4.9e-10 | ✓ |
| off | S-age vs Pooled | 0.232 | 0.0082 | ✓ |

Lighting condition does not differentially modulate the Pooled advantage (LMM ns).

**ITA group (Bonf × 3) — dataset thresholds (dark ≤ 10°, medium 10–28°, light > 28°):**

Thresholds updated from Chardon (−10°/41°) to match the HandRGBD dataset's own classification
(`patches/manifest_with_ita.csv`). The LMM uses continuous ITA and is unaffected by this change.

| Level | Comparison | N images | N subjects | mean diff (yr) | p_bonf | Sig? |
|---|---|---:|---:|---:|---:|---|
| light (ITA > 28°) | R0 vs Pooled | 72 | 19 | +0.246 | 0.141 | ✗ |
| light | S-age vs Pooled | 72 | 19 | −0.001 | 1.00 | ✗ |
| medium (10–28°) | R0 vs Pooled | 196 | 56 | +0.411 | 2.7e-10 | ✓ |
| medium | S-age vs Pooled | 196 | 56 | +0.343 | 0.272 | ✗ |
| dark (ITA ≤ 10°) | R0 vs Pooled | 1356 | 127 | +0.359 | 5.0e-32 | ✓ |
| dark | S-age vs Pooled | 1356 | 127 | +0.174 | 0.0018 | ✓ |

Pooled advantage is significant for medium and dark ITA groups with similar effect sizes
(0.41 vs 0.36 yr). The light group is borderline underpowered (19 subjects, p=0.141).
LMM arm×ITA interaction remains non-significant.

### Interpretation summary

1. **Age drives the Pooled advantage.** The arm×true_age interaction is the only
   significant LMM interaction. Pooled's aggregate MAE advantage is dominated by subjects
   ≥50 years (mean 1.80 yr advantage, nearly perfect directional consistency r≈1). For
   young adults and minors, the advantage is marginal (<0.05 yr).

2. **S-age beats Pooled for subjects ≥50.** Despite losing overall, S-age
   outperforms Pooled for the oldest subjects. This is consistent with the very large
   `arm[S-age]:true_age` coefficient (−0.072 vs −0.031 for Pooled): S-age's error
   decreases faster with age than Pooled, crossing below Pooled somewhere around age 50.

3. **No skin-tone interaction.** The Pooled advantage is statistically equivalent
   for medium and dark ITA groups (0.41 vs 0.36 yr, LMM arm×ITA ns). This is reassuring
   for fairness: synthetic pretraining does not preferentially help or hurt subjects by
   skin tone. The `light` group (19 subjects, dataset threshold ITA > 28°) is borderline
   underpowered (p_bonf=0.141).

4. **No lighting interaction.** The advantage is consistent under lights-on and
   lights-off conditions.

5. **Sex: marginal.** Pooled beats R0 for both sexes. S-age is unexpectedly competitive
   with Pooled for male subjects only; this is not significant after Bonferroni.

---

## Scope and Limitations

- ITA is computed from the dorsal hand under lab lighting; values are not
  comparable to face-ITA norms.
- If `dark` ITA group has < 10 test subjects, the skin-tone interaction is
  underpowered and should be reported descriptively only.
- The LMM uses random intercepts only. If subjects have multiple lighting
  conditions (both lights-on and lights-off), a random slope `(1 + lighting |
  subject)` would be more appropriate but likely over-parameterised at this N.
  Use random intercepts only and note the limitation.
- No new training is performed. Q5 is purely analytical and does not alter any
  checkpoint or split file.
