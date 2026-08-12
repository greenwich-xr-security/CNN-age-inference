# Q4 Minimum Paired Retraining Plan

## Purpose

Q4 is a secondary, standalone question: once all available real labels are used,
does synthetic data still add value, and does the way it is incorporated matter?

This tests three ways of initialising the same low-learning-rate real-only
fine-tuning stage at full real-label supervision: 100% of HandRGBD +
ProlificHands, with no fraction subsampling.

- `R0`: real-only initialisation; no synthetic data anywhere.
- `S-age`: staged initialisation; pretrain on SyntheticDorsalHands2 only, then
  fine-tune on real. This matches Q3's canonical staged arm, replicated here at
  the 100% point with its own paired seed set.
- `Pooled`: pooled initialisation; pretrain on HandRGBD + ProlificHands +
  SyntheticDorsalHands2 combined, then fine-tune on real.

The minimum design was three paired repeats. Seeds 42-48 are complete for the
primary seed-level comparison of `R0` versus `Pooled`; `S-age` has now been
completed as a third arm on the same seeds.

Cannot claim: equivalent-real-labels, because this is the 100% real-label point
and there is no remaining label gap to convert into a head start. Also cannot
claim an attribution ladder for the pooled arm, because a pooled stage-1 run has
no separable pretraining signal to attribute. `S-age` inherits the Q3
stage-1 data composition and epoch budget, so the Q3 S-shuffle/S-ssl/U-ssl
attribution ladder remains relevant for that staged arm, but is not re-run here.

## Design

Each repeat contains up to six jobs:

| Stage | Arm | Training data | Initialisation | LR | Max epochs |
| --- | --- | --- | --- | ---: | ---: |
| 1 | `R0` checkpoint | HandRGBD + ProlificHands | ImageNet/default V2-S | `2e-4` | 240 |
| 1 | `S-age` checkpoint | SyntheticDorsalHands2 | ImageNet/default V2-S | `2e-4` | 240 |
| 1 | `Pooled` checkpoint | HandRGBD + ProlificHands + SyntheticDorsalHands2 | ImageNet/default V2-S | `2e-4` | 240 |
| 2 | `R0` fine-tune | HandRGBD + ProlificHands | matching repeat's `R0` checkpoint | `2e-5` | 120 |
| 2 | `S-age` fine-tune | HandRGBD + ProlificHands | matching repeat's `S-age` checkpoint | `2e-5` | 120 |
| 2 | `Pooled` fine-tune | HandRGBD + ProlificHands | matching repeat's `Pooled` checkpoint | `2e-5` | 120 |

All jobs use EfficientNet-V2-S at 384 px, pure Gaussian NLL, five folds,
image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), the
locked real test split `splits/test_users_uncapped_20pct_seed42.json`, and the
same requested resources: 4 GPUs, 16 CPU cores, 64 GB RAM, batch size 16/GPU.

The fine-tuning jobs must depend on their matching checkpoint jobs. Reusing a
checkpoint across repeats would only test fine-tuning-stage variability; this
plan tests the full retraining path.

## Relation To Q3

`S-age` is the same staged idea as Q3's canonical supervised synthetic-age arm:
SyntheticDorsalHands2-only stage-1 training followed by real-only fine-tuning.
For seed 42, the existing Q3 LR-control pair is treated as the available Q4
`S-age` seed-42 point: synthetic-only checkpoint `1050939`, then 100% real
low-LR fine-tune `1050975`.

For seeds 43-48, the `S-age` arm was submitted on 2026-08-11 with the same
locked real test split, same real fold manifest for fine-tuning, and the same
resources as the existing Q4 `R0` and `Pooled` arms.

## S-Age Training

Assuming the existing seed-42 Q3 `S-age` pair (`1050939` -> `1050975`) is reused
as the seed-42 Q4 staged point, no additional Slurm jobs are missing. The 12
seed 43-48 S-age replacement jobs were launched on 2026-08-11 and are complete:

| Arm | Seeds | Jobs per seed | Slurm training jobs |
| --- | --- | ---: | ---: |
| `S-age` synthetic-only checkpoint | 43-48 | 1 | 6 completed |
| `S-age` real-only fine-tune | 43-48 | 1 | 6 completed |
| Total | 43-48 | 2 | 12 completed |

If seed 42 must be rerun as a fresh Q4-only `S-age` repeat rather than reusing
`1050939` -> `1050975`, add two more jobs.

## Completed Repeat

| Repeat seed | `R0` checkpoint | `Pooled` checkpoint | `R0` fine-tune | `Pooled` fine-tune | Main result |
| ---: | --- | --- | --- | --- | --- |
| 42 | `1050753` | `1050812` | `1054887` | `1050819` | `1050819` beat `1054887` on MAE: 4.514 vs 4.718 years; paired fold t-test p=0.0167 |

## Planned Repeats

| Repeat seed | `R0` checkpoint | `Pooled` checkpoint | `S-age` checkpoint | `R0` fine-tune | `Pooled` fine-tune | `S-age` fine-tune | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 42 | `1050753` | `1050812` | `1050939` | `1054887` | `1050819` | `1050975` | `R0`/`Pooled` complete; `S-age` available from Q3 |
| 43 | `1054892` | `1054893` | `1055190` | `1054894`, after `1054892` | `1054895`, after `1054893` | `1055191`, after `1055190` | All three arms complete |
| 44 | `1054896` | `1054897` | `1055192` | `1054898`, after `1054896` | `1054899`, after `1054897` | `1055193`, after `1055192` | All three arms complete |
| 45 | `1054912` | `1054913` | `1055194` | `1054914`, after `1054912` | `1054915`, after `1054913` | `1055195`, after `1055194` | All three arms complete |
| 46 | `1054944` | `1054945` | `1055196` | `1054946`, after `1054944` | `1054947`, after `1054945` | `1055197`, after `1055196` | All three arms complete |
| 47 | `1055120` | `1055121` | `1055198` | `1055122`, after `1055120` | `1055123`, after `1055121` | `1055199`, after `1055198` | All three arms complete |
| 48 | `1055124` | `1055125` | `1055200` | `1055126`, after `1055124` | `1055127`, after `1055125` | `1055201`, after `1055200` | All three arms complete |

The first submission attempt (`1055178`-`1055189`) was superseded: its
checkpoint jobs failed immediately because the generic launcher requires a real
test-dataset flag, and the dependent fine-tunes were cancelled before start.
The replacement uses a Q4-specific synthetic-only checkpoint launcher.

## Results

All rows are five-fold unweighted `n=1` aggregates over the fixed 1,624-sample
held-out real test set.

| Repeat seed | `R0` FT MAE | `Pooled` FT MAE | `S-age` FT MAE | Best MAE arm | `R0` FT RMSE | `Pooled` FT RMSE | `S-age` FT RMSE |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 42 | 4.718 | 4.514 | 4.817 | `Pooled` | 6.413 | 6.246 | 6.571 |
| 43 | 5.240 | 4.779 | 4.861 | `Pooled` | 7.313 | 6.567 | 6.564 |
| 44 | 5.542 | 4.731 | 4.919 | `Pooled` | 7.920 | 6.512 | 6.647 |
| 45 | 4.690 | 4.574 | 4.834 | `Pooled` | 6.465 | 6.304 | 6.591 |
| 46 | 4.808 | 4.740 | 4.799 | `Pooled` | 6.630 | 6.618 | 6.498 |
| 47 | 4.957 | 4.574 | 4.863 | `Pooled` | 6.877 | 6.319 | 6.567 |
| 48 | 5.095 | 4.617 | 4.798 | `Pooled` | 6.931 | 6.315 | 6.507 |
| Mean | 5.007 | 4.647 | 4.842 | `Pooled` | 6.936 | 6.412 | 6.564 |

Read: across seeds 42-48, `Pooled` has the lowest MAE in all seven completed
three-arm repeats. `S-age` is better than `R0` on MAE in five of seven seeds,
but remains worse than `Pooled` in every seed.

## Overall Paired Student T-Test

Paired by repeat seed (`42`, `43`, `44`, `45`, `46`, `47`, `48`). The comparison
is `R0` fine-tune versus `Pooled` fine-tune.

For MAE/RMSE/Adult FNR, positive mean difference means the `Pooled` fine-tune is
better. For AUC, negative mean difference means the `Pooled` fine-tune is
better.

| Metric | Mean `R0` FT | Mean `Pooled` FT | Mean difference | t | df | p two-sided | p one-sided, pooled better |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | 5.007 | 4.647 | 0.360 | 3.704 | 6 | 0.0100 | 0.0050 |
| RMSE | 6.936 | 6.412 | 0.524 | 2.910 | 6 | 0.0270 | 0.0135 |
| Adult-gate AUC | 0.9657 | 0.9678 | -0.0021 | -1.755 | 6 | 0.130 | 0.0649 |
| Adult FNR | 15.38% | 15.13% | 0.25 pp | 0.579 | 6 | 0.584 | 0.292 |

Interpretation: the `Pooled` fine-tune wins MAE and RMSE in all
seven seeds. MAE is now significant on both the two-sided paired Student t-test
(`p=0.0100`) and the directional one-sided test (`p=0.0050`).

## Three-Arm Pairwise Checks

These are exploratory pairwise checks for the completed Q4 three-arm table,
paired by seed. Differences are oriented so a positive value means the second
named arm is better; for AUC, higher is better.

| Comparison | Metric | Mean advantage | Wins | Paired t-test p two-sided | Wilcoxon p two-sided |
| --- | --- | ---: | ---: | ---: | ---: |
| `S-age` over `R0` | MAE | 0.166 years | 5/7 | 0.168 | 0.297 |
| `Pooled` over `S-age` | MAE | 0.195 years | 7/7 | 0.0018 | 0.015625 |
| `S-age` over `R0` | RMSE | 0.372 years | 5/7 | 0.101 | 0.109 |
| `Pooled` over `S-age` | RMSE | 0.152 years | 5/7 | 0.0475 | 0.0781 |
| `Pooled` over `R0` | Adult-gate AUC | 0.0021 | 5/7 | 0.128 | 0.109375 |
| `S-age` over `R0` | Adult-gate AUC | -0.0130 | 0/7 | 0.000111 | 0.015625 |
| `Pooled` over `S-age` | Adult-gate AUC | 0.0151 | 7/7 | 0.000066 | 0.015625 |

Interpretation: `S-age` trends better than `R0`, but is not significant at this
seed count for MAE/RMSE. For AUC, `S-age` is lower than `R0` in every seed, and
`Pooled` is higher than `S-age` in every seed.

## Overall Wilcoxon Signed-Rank Test

Paired by repeat seed (`42`, `43`, `44`, `45`, `46`, `47`, `48`). Differences
are oriented so a positive value means the `Pooled` fine-tune is better.

With seven paired seeds, the Wilcoxon p-values are still discrete, but a
seven-of-seven directional win reaches two-sided significance.

| Metric | Pooled advantage by seed | Mean advantage | Wilcoxon W+ | p two-sided | p one-sided, pooled better |
| --- | --- | ---: | ---: | ---: | ---: |
| MAE | 0.204, 0.461, 0.811, 0.116, 0.068, 0.383, 0.478 years | 0.360 years | 28.0 | 0.015625 | 0.0078125 |
| RMSE | 0.167, 0.746, 1.408, 0.161, 0.012, 0.559, 0.616 years | 0.524 years | 28.0 | 0.015625 | 0.0078125 |
| Adult-gate AUC | 0.0028, -0.0008, 0.0082, 0.0022, 0.0027, 0.0012, -0.0015 | 0.0021 | 24.0 | 0.109375 | 0.0546875 |
| Adult FNR | -0.35, -0.19, 2.15, -0.86, 0.35, 1.45, -0.79 pp | 0.25 pp | 16.0 | 0.8125 | 0.40625 |

Interpretation: the non-parametric paired test now reaches both two-sided and
directional significance for MAE and RMSE because pooled-initialised fine-tuning
wins all seven seed-level pairs.

## S-Age Real-Test Detail

All metrics are five-fold unweighted `n=1` aggregates over the fixed
1,624-sample held-out real test set.

| Seed | `S-age` fine-tune job | MAE | RMSE | Adult-gate AUC | Mean FPR | Adult FNR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | `1050975` | 4.817 | 6.571 | 0.9526 | 4.69% | 18.46% |
| 43 | `1055191` | 4.861 | 6.564 | 0.9499 | 4.74% | 18.55% |
| 44 | `1055193` | 4.919 | 6.647 | 0.9517 | 4.74% | 19.41% |
| 45 | `1055195` | 4.834 | 6.591 | 0.9558 | 4.74% | 18.40% |
| 46 | `1055197` | 4.799 | 6.498 | 0.9521 | 4.74% | 19.12% |
| 47 | `1055199` | 4.863 | 6.567 | 0.9507 | 4.74% | 19.97% |
| 48 | `1055201` | 4.798 | 6.507 | 0.9562 | 4.74% | 18.40% |
| Mean |  | 4.842 | 6.564 | 0.9527 | 4.73% | 18.90% |

## Seed-Level Effect Sizes

Effect sizes are computed on the paired seed-level differences, where positive
values mean the `Pooled` fine-tune is better.

| Metric | Mean reduction | Relative reduction | Median reduction | Paired Cohen's dz | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAE | 0.360 years | 7.19% | 0.383 years | 1.40 | 7/7 |
| RMSE | 0.524 years | 7.56% | 0.559 years | 1.10 | 7/7 |

Interpretation: the improvement is not only statistically detectable; it is
also practically meaningful. The pooled-initialised fine-tune reduces MAE
by about 0.36 years, or roughly 4.3 months, relative to the matched real-only
initialisation.

## Analysis

For each repeat, compare the three Stage 2 fine-tuned arms:

- `Pooled` fine-tune
- `R0` fine-tune
- `S-age fine-tune`

Primary endpoint: held-out real-test MAE at image-level aggregation (`n=1`).

Secondary endpoints: RMSE, adult-gate AUC, mean FPR, and Adult FNR at each
fold's best operating point with FPR <= 5%.

The repeat set is complete; paired comparisons are run across seed-level means.
