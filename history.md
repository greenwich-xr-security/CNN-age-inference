# Age Regression Loss History

## Branch Purpose

Branch: `feature/age-assurance-severity-losses`

This branch is about the age-regression model.

The purpose is to compare and document two age-regression training objectives:

- The existing age-regression probability loss, implemented as the negative log-likelihood (`NLL`) age loss.
- The age-assurance loss family, which adds explicit pressure around the minor/adult decision boundary.

The age-assurance objective is meant to improve behaviour around the `18` year threshold. It focuses on the false-positive and false-negative adult/minor errors that matter most for age assurance:

- Minor predicted as adult.
- Adult predicted as minor.
- Boundary mistakes with larger age-side violations receiving stronger penalty through the severity loss.

This branch also runs the comparison with age reweighting enabled, so the model is not dominated by the most common age ranges in the training set.

## Loss Formulations Compared

| Training recipe | Age probability loss | Age-assurance BCE | Severity loss | Age reweighting | Purpose |
|---|---:|---:|---:|---:|---|
| NLL-only baseline | Enabled | Disabled | Disabled | Enabled | Baseline age-regression model using the existing probabilistic age loss. |
| NLL + age assurance | Enabled | Enabled | Enabled | Enabled | Tests whether boundary-aware adult/minor supervision improves age assurance while preserving regression accuracy. |

## Current Loss Weights

These are the weights used by the completed `NLL + age assurance` run.

| Component | Value | Notes |
|---|---:|---|
| Age probability loss (`NLL`) | `1.0` | Main age-regression objective. |
| Age-assurance BCE | `0.25` | Adult/minor classification pressure around the `18` threshold. |
| Age-assurance severity | `0.5` | Extra penalty for adult/minor boundary mistakes, scaled by severity. |
| Age-assurance threshold | `18.0` | Boundary used for minor/adult supervision. |
| Severity radius | `5.0` | Age window used to scale boundary-error severity. |
| Age reweight min/max | `0.25` / `4.0` | Per-age loss reweighting bounds used for imbalance. |

## Completed HPC Jobs

Date checked: 2026-07-02

| Job ID | Run name | Training recipe | State | Runtime | Exit code |
|---|---|---|---|---:|---:|
| `1049170` | `v2s_nll_age_assurance_age_reweight_shared20_r3` | NLL + age-assurance BCE + severity loss | Completed | `08:04:35` | `0:0` |
| `1049171` | `v2s_nll_only_age_reweight_shared20_r3` | NLL-only baseline | Completed | `09:06:10` | `0:0` |

HPC artifact roots:

- `/home/rb3434w/CNN-age-inference/runs/multitasking/v2s_nll_age_assurance_age_reweight_shared20_r3`
- `/home/rb3434w/CNN-age-inference/runs/multitasking/v2s_nll_only_age_reweight_shared20_r3`

Local copied artifact roots:

- `C:\Users\Staff\OneDrive - University of Greenwich\slurm_runs\CNN age inference\multitasking\v2s_nll_age_assurance_age_reweight_shared20_r3`
- `C:\Users\Staff\OneDrive - University of Greenwich\slurm_runs\CNN age inference\multitasking\v2s_nll_only_age_reweight_shared20_r3`

## Held-Out Test Results

Mean across the 5 folds on the shared held-out test split.

MAE:

| Training recipe | n=1 | n=2 | n=3 | n=4 |
|---|---:|---:|---:|---:|
| NLL + age assurance | `5.1050` | `4.7485` | `4.6166` | `4.5462` |
| NLL only | `5.3137` | `4.9986` | `4.8872` | `4.7979` |

Adult AUC:

| Training recipe | n=1 | n=2 | n=3 | n=4 |
|---|---:|---:|---:|---:|
| NLL + age assurance | `0.9450` | `0.9652` | `0.9712` | `0.9765` |
| NLL only | `0.9381` | `0.9537` | `0.9553` | `0.9627` |

Boundary error rate by age bin:

Computed from held-out predictions using the `tau` value that gives the lowest total minor false-positive rate for each training recipe. The search used `tau = 10.0..30.0` in `0.1` steps; if multiple thresholds tied, the lowest tied `tau` was kept. Counts are from the held-out split once; rates are mean percentages across the 5 folds.

For minor bins, the rate is false-positive rate: minor predicted as adult. For adult bins, the rate is false-negative rate: adult predicted as minor. The blank column separates the minor and adult age groups.

`n=1` selected `tau`: `29.8` for NLL + age assurance, `29.9` for NLL only.

| Row | 10-12 | 13-15 | 16-17 |  | 18-19 | 20-24 | 25-29 | 30-39 | 40-49 | 50+ |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Samples | `102` | `115` | `78` |  | `70` | `203` | `155` | `310` | `290` | `306` |
| Users | `10` | `18` | `9` |  | `5` | `14` | `11` | `22` | `20` | `19` |
| NLL + age assurance | `1.0%` | `2.6%` | `2.8%` |  | `98.3%` | `92.5%` | `51.1%` | `30.3%` | `4.9%` | `0.6%` |
| NLL only | `0.8%` | `2.6%` | `3.8%` |  | `98.3%` | `91.9%` | `53.7%` | `33.7%` | `6.5%` | `1.3%` |

`n=4` selected `tau`: `29.8` for NLL + age assurance, `29.9` for NLL only.

| Row | 10-12 | 13-15 | 16-17 |  | 18-19 | 20-24 | 25-29 | 30-39 | 40-49 | 50+ |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Samples | `22` | `27` | `18` |  | `17` | `50` | `37` | `73` | `67` | `72` |
| Users | `10` | `18` | `9` |  | `5` | `14` | `11` | `22` | `19` | `19` |
| NLL + age assurance | `0.0%` | `0.0%` | `0.0%` |  | `100.0%` | `95.6%` | `53.5%` | `32.1%` | `2.7%` | `0.0%` |
| NLL only | `0.0%` | `0.7%` | `2.2%` |  | `100.0%` | `94.4%` | `55.7%` | `34.2%` | `6.3%` | `0.6%` |

## Interpretation

The age-assurance recipe is better than the NLL-only age-reweighted baseline on the held-out split for every aggregation size checked (`n=1..4`).

It improves:

- Age MAE.
- Adult/minor gate AUC.
- Total minor false-positive rate at the recipe-specific lowest-FPR operating point.

The current evidence suggests that adding adult/minor boundary supervision and severity weighting is useful for this age-regression branch. The improvement is not only in the adult/minor gate metric; it also improves ordinary age-regression error on the held-out split.

The lowest-FPR operating points are intentionally conservative and sit near `tau = 30`. They reduce minor false positives, but create high false-negative rates for young adults, especially `18-19` and `20-24`. A practical operating threshold should therefore be chosen later by balancing minor FPR against adult FNR.

## Future Direction

Likely next decisions:

- Inspect false positives and false negatives around the 18-year threshold in detail.
- Compare per-age-bin and near-boundary performance, not only global aggregate metrics.
- Tune `loss_weight_age_assurance_bce`, `loss_weight_age_assurance_severity`, and `age_assurance_severity_radius` to reduce minor false positives without pushing young-adult false negatives too high.
- Check whether the age-assurance loss improves safety around the boundary without harming calibration away from it.
- Decide whether `NLL + age assurance + severity + age reweighting` should become the preferred age-regression training recipe.

Recommended next sweep:

Keep `loss_weight_nll = 1.0` fixed first, then vary the auxiliary age-assurance losses. This keeps the main age-regression objective anchored while testing whether the boundary losses can be tuned for a better FPR/FNR trade-off.

| Run | NLL | Age-assurance BCE | Severity | Purpose |
|---|---:|---:|---:|---|
| Current | `1.0` | `0.25` | `0.5` | Baseline severity-loss recipe from this branch. |
| Softer assurance | `1.0` | `0.10` | `0.25` | Check if lower boundary pressure reduces young-adult FNR. |
| Stronger BCE | `1.0` | `0.50` | `0.5` | Test stronger adult/minor classification pressure. |
| Stronger severity | `1.0` | `0.25` | `1.0` | Test whether severity specifically improves near-boundary safety. |
| Balanced stronger | `1.0` | `0.50` | `1.0` | Test a stronger full age-assurance recipe. |

Compare each run on MAE, adult AUC, minor FPR by age bin, adult FNR by age bin, and threshold trade-off curves rather than only the lowest-FPR `tau`.
