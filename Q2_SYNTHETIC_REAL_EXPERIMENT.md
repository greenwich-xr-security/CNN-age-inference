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

**Superseded configuration.** The real conditions below resolved to
`handrgbd` only, rather than the intended full real source list. Retain them
only as pipeline-validation results; do not use them for Question 2 analysis.
| Condition | Slurm job | Started (BST) | Run directory | Fold | Status | MAE | Adult-gate AUC |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| RR | 1050650 | 2026-07-23 16:54:57 | `runs/q2_main_rr_v2s_seed42` | 0 | Completed | 6.1499 | 0.8989 |
| RR | 1050650 | 2026-07-23 16:54:57 | `runs/q2_main_rr_v2s_seed42` | 1 | Completed | 6.0986 | 0.9078 |
| RR | 1050650 | 2026-07-23 16:54:57 | `runs/q2_main_rr_v2s_seed42` | 2 | Completed | 7.3629 | 0.8947 |
| RR | 1050650 | 2026-07-23 16:54:57 | `runs/q2_main_rr_v2s_seed42` | 3 | Completed | 6.5572 | 0.8916 |
| RR | 1050650 | 2026-07-23 16:54:57 | `runs/q2_main_rr_v2s_seed42` | 4 | Completed | 6.9744 | 0.8848 |
| SS | 1050651 | 2026-07-23 16:54:58 | `runs/q2_main_ss_v2s_seed42` | 0 | Completed | 6.6306 | 0.8868 |
| SS | 1050651 | 2026-07-23 16:54:58 | `runs/q2_main_ss_v2s_seed42` | 1 | Completed | 5.7070 | 0.8721 |
| SS | 1050651 | 2026-07-23 16:54:58 | `runs/q2_main_ss_v2s_seed42` | 2 | Completed | 5.4228 | 0.8823 |
| SS | 1050651 | 2026-07-23 16:54:58 | `runs/q2_main_ss_v2s_seed42` | 3 | Completed | 7.5555 | 0.8772 |
| SS | 1050651 | 2026-07-23 16:54:58 | `runs/q2_main_ss_v2s_seed42` | 4 | Completed | 6.2330 | 0.8836 |
| SR | 1050652 | 2026-07-23 16:54:58 | `runs/q2_main_sr_v2s_seed42` | 0 | Completed | 10.4998 | 0.7486 |
| SR | 1050652 | 2026-07-23 16:54:58 | `runs/q2_main_sr_v2s_seed42` | 1 | Completed | 11.2220 | 0.7403 |
| SR | 1050652 | 2026-07-23 16:54:58 | `runs/q2_main_sr_v2s_seed42` | 2 | Completed | 10.5238 | 0.7288 |
| SR | 1050652 | 2026-07-23 16:54:58 | `runs/q2_main_sr_v2s_seed42` | 3 | Completed | 11.2753 | 0.7424 |
| SR | 1050652 | 2026-07-23 16:54:58 | `runs/q2_main_sr_v2s_seed42` | 4 | Completed | 10.1947 | 0.7375 |
| RS | 1050653 | 2026-07-23 16:54:58 | `runs/q2_main_rs_v2s_seed42` | 0 | Completed | 11.7395 | 0.7158 |
| RS | 1050653 | 2026-07-23 16:54:58 | `runs/q2_main_rs_v2s_seed42` | 1 | Completed | 11.1295 | 0.7671 |
| RS | 1050653 | 2026-07-23 16:54:58 | `runs/q2_main_rs_v2s_seed42` | 2 | Completed | 12.9619 | 0.7360 |
| RS | 1050653 | 2026-07-23 16:54:58 | `runs/q2_main_rs_v2s_seed42` | 3 | Completed | 11.5755 | 0.7037 |
| RS | 1050653 | 2026-07-23 16:54:58 | `runs/q2_main_rs_v2s_seed42` | 4 | Completed | 10.5286 | 0.7634 |

### Corrected full-real main-matrix job and result snapshot

These replacement jobs use the actual real source list
`handrgbd archive primary prolific` wherever a condition requires real data.
They use the same backbone (`v2_s`), two GPUs, seed 42, five sequential folds,
384 x 384 input, NLL-only loss, and age-weight exponent `p=0.5` as the prior
matrix. Metrics will be entered only after a fold's held-out evaluation ends.

| Condition | Slurm job | Started (BST) | Train datasets | Evaluation datasets | Run directory | Fold | Status | MAE | Adult-gate AUC |
| --- | ---: | --- | --- | --- | --- | ---: | --- | ---: | ---: |
| RR | 1050658 | 2026-07-23 19:47:36 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 0 | Running | — | — |
| RR | 1050658 | 2026-07-23 19:47:36 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 1 | Pending | — | — |
| RR | 1050658 | 2026-07-23 19:47:36 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 2 | Pending | — | — |
| RR | 1050658 | 2026-07-23 19:47:36 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 3 | Pending | — | — |
| RR | 1050658 | 2026-07-23 19:47:36 | `handrgbd archive primary prolific` | `handrgbd archive primary prolific` | `runs/q2_fullreal_rr_v2s_seed42` | 4 | Pending | — | — |
| SS | 1050659 | 2026-07-23 19:53:58 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 0 | Running | — | — |
| SS | 1050659 | 2026-07-23 19:53:58 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 1 | Pending | — | — |
| SS | 1050659 | 2026-07-23 19:53:58 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 2 | Pending | — | — |
| SS | 1050659 | 2026-07-23 19:53:58 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 3 | Pending | — | — |
| SS | 1050659 | 2026-07-23 19:53:58 | `synthetic_dorsal` | `synthetic_dorsal` | `runs/q2_fullreal_ss_v2s_seed42` | 4 | Pending | — | — |
| SR | 1050660 | 2026-07-23 19:53:59 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 0 | Running | — | — |
| SR | 1050660 | 2026-07-23 19:53:59 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 1 | Pending | — | — |
| SR | 1050660 | 2026-07-23 19:53:59 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 2 | Pending | — | — |
| SR | 1050660 | 2026-07-23 19:53:59 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 3 | Pending | — | — |
| SR | 1050660 | 2026-07-23 19:53:59 | `synthetic_dorsal` | `handrgbd archive primary prolific` | `runs/q2_fullreal_sr_v2s_seed42` | 4 | Pending | — | — |
| RS | 1050661 | 2026-07-23 19:53:59 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 0 | Running | — | — |
| RS | 1050661 | 2026-07-23 19:53:59 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 1 | Pending | — | — |
| RS | 1050661 | 2026-07-23 19:53:59 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 2 | Pending | — | — |
| RS | 1050661 | 2026-07-23 19:53:59 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 3 | Pending | — | — |
| RS | 1050661 | 2026-07-23 19:53:59 | `handrgbd archive primary prolific` | `synthetic_dorsal` | `runs/q2_fullreal_rs_v2s_seed42` | 4 | Pending | — | — |

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

### Result (seed 42, main matrix complete, 20/20 folds)

| Condition | Folds complete | Mean MAE | Mean AUC | Ratio vs. in-distribution reference |
| --- | ---: | ---: | ---: | --- |
| RR | 5/5 | 6.63 | 0.896 | reference |
| SS | 5/5 | 6.31 | 0.880 | reference |
| SR | 5/5 | 10.74 | 0.740 | SR/RR = 1.62 |
| RS | 5/5 | 11.59 | 0.737 | RS/SS = 1.84 |

Means are unweighted averages of the five per-fold MAE/AUC values in the
main-matrix table above.

- **Reference performance.** RR reaches 6.63 MAE / 0.896 adult-gate AUC
  (5/5 folds). SS reaches 6.31 MAE / 0.880 AUC (5/5 folds), close to RR —
  the synthetic distribution is at least as learnable
  in-domain as real data.
- **Cross-distribution collapse.** Both cross conditions degrade sharply from
  their matched reference: SR reaches 10.74 MAE / 0.740 AUC; RS reaches
  11.59 MAE / 0.737 AUC. This is the "low RS and low SR" pattern above, not
  the fidelity pattern — RS is not high relative to RR.
- **Normalised asymmetry.** Compare each cross condition to its own
  in-distribution reference rather than as a raw MAE difference: SR/RR =
  1.62 versus RS/SS = 1.84. The direction favours a fidelity gap
  (real→synthetic transfer degrades proportionally more than synthetic→real).
- **MAE/AUC dissociation.** SR and RS have nearly identical adult-gate AUC
  (0.740 vs 0.737) despite the diverging MAE ratio above (1.62 vs 1.84). This
  is used diagnostically, to separate two artifacts that could otherwise
  inflate the RS/SS ratio without reflecting a real fidelity gap, from a
  genuine one: **variance compression** (SS's own test distribution is
  narrower, so the same absolute error reads as a bigger ratio) and **label
  leakage** (RS targets are conditioning values, not verified perceived ages,
  so conditioning/render mismatch adds error unrelated to the model). AUC is
  comparatively insensitive to both — it is a coarse binary boundary, not a
  scale-dependent continuous score — yet still tracks SR and RS together.
  That the extra MAE penalty on the real→synthetic side does not show up in
  AUC at all is consistent with genuine fine-grained fidelity loss, but the
  dissociation cannot fully rule out variance compression or label leakage as
  contributors on MAE and AUC alone.
- **Caveat — variance compression and label noise.** The RS/SS ratio has not
  been corrected for either artifact above. Variance compression can be
  checked directly from the per-fold score distributions; label noise needs
  bounding against the stage-1 biomarker/rater scatter (the gap between a
  synthetic image's conditioned age and its perceived age). Until both are
  bounded, the ratio should be reported as an upper bound on the fidelity
  gap, not a clean estimate of it.
- **Caveat — single seed.** All 20 folds above are seed 42 only. Fold-level
  spread is reported (fold-to-fold MAE ranges from 5.42 to 12.96 across the
  matrix), but seed-to-seed spread is not yet known; Stage 3's architecture
  repeat, or a second seed on `v2_s`, is needed before the asymmetry direction
  is treated as more than a single-run result.

### Bottom line

- **Coverage (can synthetic-trained models generalise to real hands?) — no,
  not on their own.** SR loses ~4.1 MAE and ~16 AUC points relative to RR.
- **Fidelity (can real-trained models generalise to synthetic hands?) — no,
  more so.** RS loses ~5.3 MAE and ~16 AUC points relative to SS, the larger
  of the two proportional drops (1.84x vs 1.62x).
- Both directions fail roughly together on the coarse adult-gate boundary
  (AUC ≈ 0.74 either way) but real→synthetic fails more on fine-grained age,
  so the result is a genuine two-way distribution mismatch with a mild
  fidelity-leaning asymmetry, not a synthetic-side shortcut that coverage
  alone would predict. This does not by itself say whether synthetic data is
  useful in combination with real training data — that is a different,
  untested condition (see below).
