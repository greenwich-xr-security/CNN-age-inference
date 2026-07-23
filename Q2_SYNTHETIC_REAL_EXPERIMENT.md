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

### Stage 2: main matrix

Run EfficientNetV2-S (`v2_s`) with one GPU per job, seeds 42/43/44, and five
folds. This produces 60 condition/seed/fold training runs (4 conditions x 3
seeds x 5 folds). Use independent single-GPU jobs rather than multi-GPU DDP so the
8-GPU `gpu-beast` node can run several conditions concurrently.

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
