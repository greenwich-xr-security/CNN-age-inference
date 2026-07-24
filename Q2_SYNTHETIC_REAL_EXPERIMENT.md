# Question 2: Real/Synthetic Distribution Relationship

## Objective

Determine how the SyntheticDorsalHands distribution relates to real dorsal-hand
data. Treat the real/synthetic train/test matrix as a diagnostic pattern, not
as four isolated benchmark scores.

The analysis asks two distinct questions:

- **Coverage:** can a model trained on synthetic data generalise to real hands?
- **Fidelity:** does a model trained on real data generalise to synthetic hands?

The asymmetry between these results is evidence about the relationship between
the two distributions.

## Matched conditions

| ID | Train distribution | Test distribution | Role |
| --- | --- | --- | --- |
| RR | Real | Locked real test | Reference performance |
| SS | Synthetic | Held-out synthetic test | Shortcut probe |
| SR | Synthetic | Locked real test | TSTR: coverage of real-world variation |
| RS | Real | Held-out synthetic test | TRTS: fidelity of the synthetic distribution |

The same backbone, input size, loss, optimiser, training schedule, early
stopping rule, augmentation policy, and matched training-set size must be used
for every condition. Report MAE, RMSE, adult-gate AUC, calibration, and
per-age-bin error.

### Training distribution and loss balancing

Keep every eligible dorsal image: there is **no per-age image cap** and no
age oversampling. The held-out split is fixed first; training and validation
are then selected only from the remaining users.

Age imbalance is handled in the training loss, never by dropping data. For
each run, calculate integer-age counts `n_a` from that run's training fold
only and use the normalised, clipped sample weight:

```text
raw_a = (n_a + 1)^(-p)
w_a = clip(raw_a, quantile_5(raw), quantile_95(raw)) / mean_b(clip(raw_b, quantile_5(raw), quantile_95(raw)))
```

The default quantiles are 5% and 95%; they are computed independently from
the age-frequency weights in each training fold. Record the resulting minimum
and maximum weight in the run configuration.

Apply `w_a` to every enabled regression-loss component (Gaussian NLL, MSE,
and MAE). Validation and held-out metrics remain unweighted and report both
the natural sample-weighted result and per-age-bin results.

Calibrate `p` on the RR validation folds only, before examining any locked
test result: compare `p = 0` (empirical-frequency baseline), `p = 0.5`
(square-root correction), and `p = 1` (full inverse-frequency correction).
Select the setting with the best mean validation **macro age-bin MAE**, using
ordinary MAE as a tie-breaker. Freeze the selected value and use the same
rule for RR, SS, SR, and RS; the counts are recalculated from each condition's
own training fold, so the rule adapts to its training distribution without
using validation or test data.

## Input image size

Use **384 × 384 pixels** for every condition, backbone, training run, and
evaluation run. SyntheticDorsalHands images are natively 384 × 384; real
images must use the existing hand crop/mask pipeline and then be resized to
384 × 384 with the same interpolation policy.

Do not change input resolution between RR, SS, SR, and RS. This makes the
real/synthetic performance matrix comparable and prevents resolution from
becoming a distributional confound.

### 384-pixel backbone set

Use `v2_s` (**EfficientNetV2-S**) as the primary backbone. It has a native
384 x 384 input configuration in the current pipeline, is pretrained through
torchvision, and is a more appropriate first model than EfficientNet-B0 for
this native-resolution protocol.

| Pipeline model name | Backbone | Native input size | Use in Question 2 |
| --- | --- | --- | --- |
| `v2_s` | EfficientNetV2-S | 384 x 384 | **Primary model; run the full RR/SS/SR/RS matrix.** |
| `vit_tiny_384` | ViT-Tiny/16 | 384 x 384 | Architecture-sensitivity repeat after the primary matrix. |
| `vit_small_384` | ViT-Small/16 | 384 x 384 | Optional higher-capacity repeat if GPU time permits. |
| `swin_v2_base_384` | SwinV2-Base | 384 x 384 | Optional high-capacity repeat; use a reduced batch size. |

Do not use `v2_m` or `v2_l` in the 384-pixel matrix: their canonical size is
480 x 480. Do not use ConvNeXt for this fixed-size protocol because the
implemented variants are canonical 224 x 224. `swin_v2_tiny_384` is also
excluded because the installed timm build has no pretrained checkpoint for
it. These exclusions avoid changing a backbone's intended resolution or
comparing pretrained models under materially different input settings.

## Dataset roles and split rules

### Dataset membership

| Category | Pipeline source name | Dataset | Role in this experiment |
| --- | --- | --- | --- |
| Real | `handrgbd` | handRGBD | Real training and locked real evaluation |
| Real | `archive` | Archive | Real training and locked real evaluation |
| Real | `primary` | 11kHands | Real training and fallback locked-test source |
| Real | `prolific` | ProlificHands | Real training and locked real evaluation |
| Synthetic | `synthetic_dorsal` | SyntheticDorsalHands | Synthetic training and separate synthetic hold-out evaluation |
| Excluded | `hagrid` | HaGRIDv2 | Not part of the paper protocol; do not use in Question 2 |

`SyntheticDorsalHands` is generated data. All other sources in the table are
real-image corpora. The real distribution for this experiment is therefore
handRGBD + Archive + 11kHands + ProlificHands.

### Real test set

`splits/held_out_test.json` is the canonical 150-subject, subject-disjoint
real test set. Its SHA-256 is:

```text
A21F339DF2454E52EFD84A51D3A7A7BC80050398C72183801BD5A625951C17E5
```

It is used only by RR and SR. Its subjects must be removed from all real
training data and from every synthetic conditioning pool.

### Required held-out exclusion flow

1. The canonical JSON is loaded by `dataset.utils.load_test_split()`, which
   normalises its subject records to `test_user_ids` and records the exact
   split SHA-256.
2. `train_distributed.py` removes those IDs from `train_metadata` before
   constructing batches.
3. The same IDs are removed from `eval_metadata` before validation folds are
   created, so the locked test subjects cannot enter validation or early
   stopping.
4. The held-out test loader re-loads only `eval_metadata` and selects exactly
   those IDs for final testing.
5. K-fold files must be generated with the real `--eval-datasets` list and
   the same `--test-users-file`, so no locked test ID can appear in a fold.

### Pipeline support review

| Requirement | Status | Review finding |
| --- | --- | --- |
| Exclude held-out IDs from real training | Supported | `train_distributed.py` removes `test_user_ids` from training metadata. |
| Exclude held-out IDs from real validation | Supported | Validation IDs are selected only after the held-out IDs are removed from evaluation metadata. |
| Evaluate on the locked real IDs only | Supported | Final test metadata uses `--eval-datasets` and filters to the canonical IDs. |
| Record which locked split was used | Supported | Distributed configuration records path, format, user count, and SHA-256. |
| Include all 150 canonical real subjects | Supported | The `prolific` loader resolves the three ProlificHands subjects in the canonical split. |
| Exclude held-out real people from synthetic conditioning provenance | **Gap** | Synthetic provenance columns are loaded but are not checked against `test_user_ids` by the CNN pipeline. Generation must guarantee this exclusion and the training loader should validate it before SR/SS runs. |
| Prevent synthetic data from entering the locked real test | Supported when configured | Use real-only `--eval-datasets` for RR/SR and the canonical real split. Do not rely on the legacy shared `--datasets` default. |

### Synthetic test set

Create and lock a separate `splits/synthetic_test_images.json` from
`SyntheticDorsalHands`. Use it only for SS and RS.

The synthetic training/test partition must be image-disjoint. This dataset has
one unique generated image per `sample_id`, rather than repeated images for a
synthetic person, so it cannot support a user-disjoint synthetic split.

Provenance-disjoint partitioning is also impossible in the current manifest:
all generated images form one connected component through the source identity
fields below. Therefore SS and RS are explicitly image-level tests, not
synthetic-subject or provenance-generalisation tests:

- `skeleton_source_user_id`
- `skin_source_user_id`
- `sex_source_user_id`
- `lighting_source_user_id`
- `aspect_source_user_id`

RR and SR evaluate only against real data. SS and RS evaluate only
against the synthetic hold-out. Synthetic images must never be mixed into the
locked real test set.

### Source selections

The distributed pipeline accepts independent source lists:

```text
--train-datasets ...
--eval-datasets ...
```

For this paper, the real source list is:

```text
handrgbd archive primary prolific
```

Do not include HaGRID in this experiment unless the protocol is formally
updated to include it.

| ID | `TRAIN_DATASETS` | `EVAL_DATASETS` | Split file |
| --- | --- | --- | --- |
| RR | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `splits/held_out_test.json` |
| SS | `synthetic_dorsal` | `synthetic_dorsal` | `splits/synthetic_test_images.json` |
| SR | `synthetic_dorsal` | `handrgbd archive primary prolific` | `splits/held_out_test.json` |
| RS | `handrgbd archive primary prolific` | `synthetic_dorsal` | `splits/synthetic_test_images.json` |

## HPC campaign

### Stage 1: smoke test

Run all four conditions with EfficientNetV2-S (`v2_s`), one GPU, one seed (42), and one
fold. Confirm dataset membership, split checksums, output files, and metric
generation before scaling up.

### Completed smoke-test record

The following four image-level smoke jobs completed on `gpu-beast`. Each used
one GPU, 16 allocated CPU cores, 64 GB memory, seed 42, no K-fold file
(`KFOLDS=1`), two epochs, patience 2, per-process batch size 4, four data
workers, no mask application, and no age oversampling. The model was
EfficientNetV2-S (`v2_s`, native 384 x 384 input). The loss was Gaussian NLL
only (`NLL=1`, `MSE=0`, `MAE=0`) with square-root inverse-frequency age loss
weights (`p=0.5`), clipped at the training-fold 5th/95th percentiles.

| ID | Slurm job | Started (BST) | Elapsed | Train datasets | Evaluation datasets | Locked test split | Run directory | Result |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| SS | 1050642 | 2026-07-23 16:37:51 | 00:04:21 | `synthetic_dorsal` | `synthetic_dorsal` | `splits/synthetic_test_images.json` | `runs/q2_smoke_ss_v2s_seed42` | Completed, exit 0 |
| RR | 1050645 | 2026-07-23 16:43:04 | 00:01:59 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `splits/held_out_test.json` | `runs/q2_smoke_rr_v2s_seed42_retry` | Completed, exit 0 |
| SR | 1050646 | 2026-07-23 16:43:05 | 00:04:55 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `splits/held_out_test.json` | `runs/q2_smoke_sr_v2s_seed42_retry` | Completed, exit 0 |
| RS | 1050647 | 2026-07-23 16:43:34 | 00:02:38 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `splits/synthetic_test_images.json` | `runs/q2_smoke_rs_v2s_seed42_retry` | Completed, exit 0 |

The first RR/SR/RS submissions used the same default rendezvous port and were
cancelled before training. The recorded retry jobs use distinct master ports
and are the valid smoke-test results.

### Stage 2: main matrix

Run EfficientNetV2-S (`v2_s`) with **two GPUs per job**, seed 42, and five
folds. This produces **20 training runs** (4 conditions x 5 folds). Submit one
two-GPU DDP Slurm job per condition; each scheduler job executes its five folds
sequentially. The 8-GPU `gpu-beast` node can therefore run the four condition
jobs concurrently. Request each job with `sbatch --gres=gpu:2`, allocate 16
CPU cores, and assign every simultaneous job a distinct `MASTER_PORT`.

### Main-matrix job and result snapshot

The jobs below were submitted on 23 July 2026. Each job uses two GPUs, 16 CPU
cores, seed 42, EfficientNetV2-S (`v2_s`), 384 x 384 input, per-process batch
size 4, eight data workers, up to 240 epochs, and patience 10. A scheduler job
runs its five folds sequentially; a metric is recorded only when that fold's
held-out evaluation has completed. The live HPC monitor writes the current
version to `runs/q2_main_results.md` and `runs/q2_main_results.csv`.

### Corrected full-real main-matrix job and result snapshot

The first RR/SR/RS replacement jobs include `primary` (11kHands). They are
preserved below as a distinct 11kHands-inclusive result set. The current
RR/SR/RS jobs use `handrgbd archive prolific` wherever real data is required;
`primary` is excluded from both training and evaluation. All runs use
EfficientNetV2-S (`v2_s`), two GPUs, seed 42, five sequential folds, 384 x 384
input, NLL-only loss, and age-weight exponent `p=0.5`. The no-11kHands reruns
use the Slurm-job-specific rendezvous port/ID fix from commit `6ffb99f`.

| Condition | Slurm job | Train datasets | Evaluation datasets | Run directory | Fold | Status | MAE | Adult-gate AUC |
| --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: |
| SS | 1050659 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 0 | Completed | 6.4643 | 0.8819 |
| SS | 1050659 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 1 | Completed | 5.3637 | 0.8974 |
| SS | 1050659 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 2 | Completed | 5.5609 | 0.8959 |
| SS | 1050659 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 3 | Completed | 6.1841 | 0.8912 |
| SS | 1050659 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 4 | Completed | 5.4613 | 0.8902 |
| RR | 1050658 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 0 | Completed (includes 11k) | 10.7384 | 0.8010 |
| RR | 1050658 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 1 | Completed (includes 11k) | 10.2701 | 0.8031 |
| RR | 1050658 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 2 | Completed (includes 11k) | 9.4030 | 0.8296 |
| RR | 1050658 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 3 | Completed (includes 11k) | 10.6489 | 0.7772 |
| RR | 1050658 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 4 | Completed (includes 11k) | 10.5339 | 0.7926 |
| SR | 1050660 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 0 | Completed (includes 11k) | 10.6536 | 0.7449 |
| SR | 1050660 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 1 | Completed (includes 11k) | 12.2485 | 0.7117 |
| SR | 1050660 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 2 | Completed (includes 11k) | 11.9015 | 0.7265 |
| SR | 1050660 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 3 | Completed (includes 11k) | 11.4352 | 0.7184 |
| SR | 1050660 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 4 | Completed (includes 11k) | 11.4304 | 0.7470 |
| RS | 1050661 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 0 | Completed (includes 11k) | 15.0097 | 0.6539 |
| RS | 1050661 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 1 | Completed (includes 11k) | 13.0946 | 0.6374 |
| RS | 1050661 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 2 | Completed (includes 11k) | 11.2639 | 0.8126 |
| RS | 1050661 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 3 | Completed (includes 11k) | 14.5272 | 0.7514 |
| RS | 1050661 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 4 | Completed (includes 11k) | 14.8008 | 0.6920 |


### Corrected full-real main-matrix summary (with sample counts)

Held-out test subjects/images are excluded from a condition's training pool
only when that condition is *evaluated* against them — RR/SR exclude the 150
real held-out subjects from real training; RS does not, since it is
evaluated on synthetic. The mirror holds for SS (excludes the synthetic
held-out images) versus RS/SR's use of the full synthetic pool. This is why
train size differs between conditions that otherwise train on the same data.

| Condition | Train data | Train samples/fold | Eval data | Test samples | Mean MAE | Mean AUC |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| RR | real | ~6,327 | real held-out (150 subj.) | 2,289 | 8.21 | 0.838 |
| SS | synthetic | ~10,641 | synthetic held-out (150 img.) | 150 | 5.81 | 0.891 |
| SR | synthetic (full pool) | 13,451 | real held-out (150 subj.) | 2,289 | 10.54 | 0.713 |
| RS | real (full pool) | 10,198 | synthetic held-out (150 img.) | 150 | 13.62 | 0.794 |

Train samples/fold is the training-pool size after the fold's validation
slice is removed (RR/SS vary slightly by fold; SR/RS train on the full pool
every fold since their validation slice is drawn from the *other* domain).
Mean MAE/AUC are unweighted averages of the per-fold values in the table
above.

These are the current, no-11kHands numbers (`q2_no11k_rr_v2s_seed42_retry`,
`q2_no11k_sr_v2s_seed42_retry`, `q2_no11k_rs_v2s_seed42`, and the unaffected
`q2_fullreal_ss_v2s_seed42`). See the **Result** section below for the
recomputed coverage/fidelity ratios, per-fold significance tests, and
per-age-bin/skin-tone breakdowns.

### Stage 3: architecture sensitivity

Repeat the main matrix for ViT-Tiny-384 after the EfficientNetV2-S results
have passed data and split validation. Run ViT-Small-384 and SwinV2-Base-384
only if the primary results warrant the extra GPU time.

## Run metadata

Every output directory must record:

```text
condition_id
train_datasets
eval_datasets
test_split_path
test_split_sha256
synthetic_manifest_checksum
backbone
seed
fold_index
training_sample_count
evaluation_sample_count
code_commit
```

Use explicit run names, for example:

```text
q2_rr_b0_seed42_fold0
q2_ss_b0_seed42_fold0
q2_sr_b0_seed42_fold0
q2_rs_b0_seed42_fold0
```

## Interpretation

- **High SS but low SR:** synthetic data contains shortcuts that do not cover
  real variation.
- **High RS relative to RR:** synthetic images preserve features learned from
  real data; this supports distributional fidelity but is not sufficient alone.
- **Low RS and low SR:** substantial distribution mismatch.
Report fold-to-fold and seed-to-seed variation. Do not make a conclusion from
a single aggregate score.

### Result (seed 42, corrected full-real matrix, 20/20 folds)

All 20 folds (4 conditions x 5 folds) completed. RR/SR/RS use the current
no-11kHands real pool (`handrgbd archive prolific`); SS is unaffected by that
change since it never trains or evaluates on real data. Source runs:
`q2_no11k_rr_v2s_seed42_retry`, `q2_no11k_sr_v2s_seed42_retry`,
`q2_no11k_rs_v2s_seed42`, `q2_fullreal_ss_v2s_seed42`; comparison artefacts in
`runs/q2_no11k_comparison/`.

#### Headline metrics (mean ± SD across 5 folds)

| Condition | Role | MAE (yrs) | RMSE (yrs) | Adult-gate AUC | Test n/fold |
| --- | --- | ---: | ---: | ---: | ---: |
| RR | Reference | 8.21 ± 0.67 | 11.65 ± 1.24 | 0.838 ± 0.014 | 2,289 |
| SS | Shortcut probe | 5.81 ± 0.49 | 8.01 ± 0.18 | 0.891 ± 0.006 | 150 |
| SR | Coverage (TSTR) | 10.54 ± 0.73 | 13.38 ± 0.73 | 0.713 ± 0.025 | 2,289 |
| RS | Fidelity (TRTS) | 13.62 ± 0.54 | 15.59 ± 0.48 | 0.794 ± 0.026 | 150 |

An omnibus repeated-measures test across all four conditions confirms a real
condition effect and not just fold noise: MAE (ANOVA F(3,12)=145.3,
p=1.07e-9; Friedman χ²=15.0, p=0.0018), RMSE (F(3,12)=68.6, p=8.0e-8;
Friedman p=0.0029).

#### Coverage: SR vs RR (can synthetic training generalise to real hands?)

Training on synthetic and testing on the locked real set is worse than the
real-trained reference at every fold: MAE +2.33 yrs (10.54 vs 8.21, a
**1.28x** increase), RMSE +1.73 yrs, adult-gate AUC drops 0.125 (0.838 →
0.713, a 14.9% relative drop). Paired t-test p=0.0020 (Holm p=0.0041,
n=5 folds); every fold shows SR worse than RR (Wilcoxon p=0.0625, the floor
for n=5 paired folds).

Coverage is partial, not absent: SR (0.713 AUC) is well above chance and not
catastrophically far from RR (0.838), so synthetic pretraining does carry
real age-relevant signal. But it does not substitute for real training data
at this scale — the gap is consistent and significant, not noise.

#### Fidelity: RS vs SS (does synthetic imagery preserve real age features?)

Training on real and testing on the synthetic hold-out is far worse than the
synthetic-trained reference: MAE +7.81 yrs (13.62 vs 5.81, a **2.34x**
increase), RMSE +7.59 yrs, adult-gate AUC drops 0.097 (0.891 → 0.794, a 10.9%
relative drop). Paired t-test p=0.00013 (Holm p=0.00064); every fold shows RS
worse than SS.

The fidelity gap (2.34x MAE) is substantially larger than the coverage gap
(1.28x MAE), even though the AUC drops are closer in size (14.9% vs 10.9%).
Read against the interpretation guide above: this is the **"low RS relative
to a strong SS, with only partial SR coverage"** pattern — not the "high RS"
signature that would indicate strong fidelity. Real-trained features do not
transfer cleanly onto synthetic images; synthetic images retain enough
real-world age signal to be partially learnable *from* (coverage), but their
appearance differs enough from real images that a real-trained model
struggles to *read* them (fidelity). The direction of the asymmetry (fidelity
gap > coverage gap) suggests the synthetic domain gap is more about
image-level appearance (texture, lighting, rendering) than about missing
age-relevant structure.

#### Cross-condition ranking

SS is the best-performing condition overall (lowest MAE, highest AUC) — the
expected shortcut-probe signature: evaluating a model on held-out data from
its own training distribution is the easiest case and is not informative
about generalisation by itself. RS is the worst on MAE/RMSE, but SR — not
RS — is the worst on AUC; RS pairwise comparisons on RMSE (RR vs RS,
RS vs SR) and MAE all reach significance, while RR vs SR on RMSE alone
narrowly misses the Holm-corrected threshold (p=0.067). Full pairwise table:
`runs/q2_no11k_comparison/overall_metrics_pairwise.csv`.

#### Per-age-decade error

| Age bin | n (RR/SR, per matrix) | RR MAE | SR MAE | Δ (SR−RR) | n (SS/RS) | SS MAE | RS MAE | Δ (RS−SS) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10–20 | 4,790 | 4.29 | 7.69 | +3.40 | 300 | 5.16 | 17.86 | +12.70 |
| 20–30 | 2,245 | 5.97 | 6.87 | +0.90 | 150 | 4.37 | 14.04 | +9.67 |
| 30–40 | 1,215 | 9.41 | 9.46 | +0.05 | 75 | 6.56 | 8.31 | +1.75 |
| 40–50 | 1,215 | 11.00 | 15.89 | +4.89 | 75 | 5.96 | 4.64 | −1.32 |
| 50–60 | 995 | 14.14 | 17.28 | +3.14 | 75 | 8.90 | 9.76 | +0.86 |
| 60–70 | 885 | 19.53 | 19.20 | −0.33 | 40 | 9.27 | 11.82 | −1.55 |

Two patterns stand out. First, both coverage and fidelity errors rise with
age for RR/SR and SS, reflecting the real-world age imbalance (fewer older
subjects) that the loss reweighting only partially offsets. Second, the
RS fidelity failure is heavily concentrated in the **10–30 age range**
(+12.70 and +9.67 yrs) rather than spread evenly — a real-trained model is
far more wrong about *young* synthetic faces than older ones, and is
actually slightly *better* than SS at 40–50 and comparable at 60–70. That
40–50 crossover is on n=75 synthetic images per fold and is consistent with
regression-to-the-training-mean (the real training pool's age distribution
centers well above the synthetic pool's), not a genuine fidelity advantage —
it should not be read as "real training helps for middle-aged synthetic
faces" without a larger held-out set.

#### Skin-tone error

| Condition | light | tan | dark | Largest significant gap (Holm) |
| --- | ---: | ---: | ---: | --- |
| RR | 7.94 | 4.95 | 8.19 | tan vs dark, p=0.0026; tan vs light, p=0.0053 |
| SR | 10.70 | 9.60 | 10.66 | none survive Holm correction |
| SS | 5.65 | 4.97 | 6.38 | none survive Holm correction (tan vs dark p=0.057 uncorrected) |
| RS | 12.84 | 11.92 | 15.24 | dark vs tan, p=0.0013; dark vs light, p=0.018 |

On real data (RR), "tan" is the easiest tone and "light"/"dark" are both
harder and statistically indistinguishable from each other — likely a real
training-pool composition effect rather than a skin-tone difficulty effect
per se. Training on synthetic data instead (SR) removes that significant
tone gap on the same real test set, even though overall accuracy is worse —
synthetic pretraining does not reproduce the real-data tone disparity. On the
synthetic test set, the pattern re-emerges only for the real-trained model
(RS): "dark" is significantly worse than both other tones, a fidelity-side
effect not present when the synthetic-trained model evaluates itself (SS).

#### Uncertainty calibration

The regression head is trained with Gaussian NLL and emits `(mu, log_var)`
per image ([README.md:56-59](README.md#L56-L59)); MAE/RMSE score only `mu`
and say nothing about whether the predicted `sigma = sqrt(exp(log_var))` is
trustworthy. Calibration was computed post hoc from the raw per-sample
predictions each fold already saves (`test_predictions_raw_ddp.npz`:
`targets`, `pred_mean`, `pred_log_var`), pooling all 5 folds per condition.
Script and outputs: `runs/q2_no11k_comparison/calibration_{per_fold,pooled}.csv`.

A calibrated Gaussian predictive distribution has standardized residual
`z = (target − mu) / sigma` with mean 0 and std 1, and its nominal x%
prediction interval should empirically contain the true age x% of the time
(PICP).

| Condition | mean σ / RMSE | z mean | z std | Mean \|PICP−nominal\| (5 levels) | 95%-nominal empirical coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| RR | 0.54 | +0.61 | 1.72 | 0.148 | 79.2% |
| SS | 0.96 | −0.04 | 0.94 | 0.034 | 96.3% |
| SR | 0.70 | +0.26 | 1.40 | 0.127 | 84.4% |
| RS | 0.60 | −0.95 | 1.42 | 0.316 | 70.8% |

- **SS is the only condition that is close to well-calibrated**: predicted
  sigma tracks RMSE almost 1:1, z-std ≈0.94, negligible bias, and PICP close
  to nominal at every level (50/68/80/90/95%). This holds fold-by-fold
  (z-std 0.86–0.97 across all 5 folds — no single fold is driving it).
- **RR — the real self-domain reference — is itself markedly overconfident**:
  predicted sigma is roughly half the actual RMSE, and the nominal-95%
  interval only covers 79% of true ages, consistently across all 5 folds
  (z-std 1.43–2.06). So sigma miscalibration on real hands is not an
  artefact of the synthetic/real comparison — it's present even with no
  domain shift at all, and looks like a property of training on the
  real age distribution (which is noisier/harder) with this loss setup.
- **SR (coverage) is, if anything, less overconfident than RR** on the same
  real test set (sigma/RMSE 0.70 vs 0.54, 95% PICP 84% vs 79%) — the
  synthetic-trained model's uncertainty is somewhat better calibrated than
  the real-trained model's, even though its point predictions are worse.
- **RS (fidelity) is the worst on every calibration axis**, and adds a
  qualitative failure mode beyond the MAE/RMSE gap already reported: it has
  both the worst interval coverage (95%-nominal → 70.8% empirical) and a
  large, consistent negative bias (z-mean = −0.95, i.e. the real-trained
  model systematically predicts *older* ages than the true synthetic age,
  by close to one full predicted sigma) — z-mean is negative in all 5 folds
  (−0.82 to −1.15). RS isn't just noisier than SS, it's directionally
  biased toward overestimating age on synthetic faces.

PICP for SS/RS is estimated from only 150 test images per fold, so its
per-fold percentages are noisier than RR/SR's (2,289 images per fold);
treat exact SS/RS PICP values as approximate, the direction/size of the gap
vs. RR/SR is the reliable part.

#### Open items before this section is used for the paper

- **Single seed.** All of the above, including the calibration numbers, is
  seed 42 only; the protocol calls for seed-to-seed variation and none has
  been run yet. Treat the significance tests above as within-seed,
  across-fold only.
- **Architecture sensitivity (Stage 3) not started.** All results are
  EfficientNetV2-S only.
- **Small per-bin n above age 30.** The 30–70 age-bin counts (n=40–1,215)
  make the per-bin deltas directionally informative but not independently
  significance-tested; treat them as descriptive, not confirmatory.

