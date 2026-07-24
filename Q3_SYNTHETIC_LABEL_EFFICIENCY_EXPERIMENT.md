# Question 3: Synthetic Data and Label Efficiency

## Objective

Determine whether synthetic dorsal-hand images reduce the amount of **real
age-labelled data** needed to reach a given real-world performance level.
Report a label-efficiency curve rather than one full-data score, and express
any benefit as an equivalent number of additional real labels.

The primary question is not whether synthetic pretraining improves the
full-data result. A valid outcome is no measurable benefit at 100% of the
real training set but a benefit at low-label regimes.

## Fixed data protocol

Use the same source definitions as Question 2:

| Category | Pipeline sources |
| --- | --- |
| Real | `handrgbd archive primary prolific` |
| Synthetic | `synthetic_dorsal` |

Keep `splits/held_out_test.json` as the fixed, real-only final test set for
every arm. No held-out subject may enter synthetic pretraining, real
fine-tuning, validation, subset selection, early stopping, or model
selection.

Use `splits/synthetic_test_images.json` only for diagnostic synthetic
evaluation; it must never replace the locked real test as the primary Q3
endpoint.

## Label-efficiency curve

Construct nested, user-disjoint real fine-tuning subsets from the real
training pool remaining after the locked test subjects have been removed.
Apply each fraction to **real users**, retaining every eligible image for a
selected user. This avoids allowing participants with many images to count as
many independent labels.

Run these real-label fractions:

```text
1%, 5%, 10%, 25%, 50%, 100%
```

For each fraction, record both the number of selected real users and the
number of retained real images. Stratify the user selection by age bins,
gender where available, source dataset, and skin-tone label where available.
The fraction-specific validation split must also be user-disjoint and drawn
only from the remaining real training pool.

Use at least three independent seeds. A seed controls the real-subset draw,
data-order/augmentation randomness, and model initialisation. Use the same
seed-specific subset for every arm, so comparisons are paired. Report the
mean, standard deviation, individual seed values, and a paired confidence
interval for each comparison.

## Pretraining and control arms

Fine-tune every arm on the same selected real subset, using the same backbone,
input size, augmentation, optimiser, early-stopping rule, loss weighting, and
maximum fine-tuning schedule. The primary backbone is `v2_s` at 384 x 384.

| ID | Pretraining before real fine-tuning | Purpose |
| --- | --- | --- |
| R0 | Standard backbone initialisation, then real fine-tuning only | Main real-only baseline |
| R1 | Random initialisation, then real fine-tuning only | From-scratch baseline; makes the complete pretraining chain explicit |
| S-age | Supervised age pretraining on `synthetic_dorsal`, then real fine-tuning | Candidate synthetic label-efficiency benefit |
| S-shuffle | Same synthetic images and schedule as S-age, but age labels are randomly permuted once per seed | Tests whether synthetic age conditioning, rather than generic image exposure, causes a gain |
| S-ssl | Self-supervised pretraining on the same synthetic images, then real fine-tuning | Separates representation benefit from synthetic age-label benefit |
| U-ssl | Self-supervised pretraining on a non-age-labelled, unrelated image corpus, matched to S-ssl in image count, updates, resolution, and augmentation budget | Controls for generic extra compute/image exposure |

The unrelated corpus must be fixed before running Q3 and must not contain the
Q3 real test subjects or SyntheticDorsalHands images. Prefer a non-hand,
non-age-labelled corpus. Record its manifest checksum, selected image count,
and licence in the run configuration.

All synthetic arms must exclude `splits/synthetic_test_images.json` from
pretraining. For S-age and S-shuffle, use precisely the same accepted
synthetic images, number of optimisation updates, and augmentation schedule.
For S-shuffle, shuffle labels across synthetic images within each seed while
preserving the marginal age-label distribution.

## Training schedule and fairness

Use a two-stage, traceable chain:

```text
initialisation -> optional pretraining arm -> real fine-tuning subset -> locked real test
```

Pretraining uses only its arm-specific data. Reset the optimiser and learning
rate schedule before real fine-tuning. Do not reinitialise model weights
between pretraining and fine-tuning. The R0 and R1 baselines skip the optional
pretraining stage.

Match pretraining arms by update count, effective batch size, image size,
backbone, and compute budget. Do not compensate for an arm's lower loss by
giving it more pretraining epochs. Fine-tuning uses the same real age-loss
balancing rule in every arm; calculate frequency weights only from that arm's
selected real fine-tuning subset.

## Primary outcomes

The primary endpoint is locked-real-test MAE. Report adult-gate AUC, RMSE,
calibration/interval coverage, per-age-bin MAE, and skin-tone-stratified MAE
as secondary outcomes.

For each arm and label fraction, report:

| Field | Requirement |
| --- | --- |
| Real labels | Selected real users and retained real images |
| Performance | Per-seed and mean +/- standard deviation MAE, RMSE, AUC |
| Gain | Paired MAE difference from R0 for the identical seed/subset |
| Provenance | Backbone, initialisation, pretraining corpus/checksum, update counts, real subset checksum, locked-split checksum |
| Fairness | Per-age-bin and skin-tone results, with sample counts |

Do not select an arm using the locked test. Select pretraining and
fine-tuning hyperparameters on validation data at 10% and 25% labels only,
freeze them, then run the complete curve.

## Equivalent real labels

Express a synthetic benefit at fraction `f` as the size of the R0 real-only
training set required to match that arm's performance:

```text
N_equiv(arm, f) = real-label count N such that MAE_R0(N) = MAE_arm(f)
label gain = N_equiv(arm, f) - N_real(f)
relative gain = N_equiv(arm, f) / N_real(f) - 1
```

Estimate `N_equiv` by monotone interpolation of the R0 MAE-versus-real-label
curve. If an arm is better than the 100% R0 result, report the gain as
`> 100% baseline range` rather than extrapolating. If it is worse than the 1%
R0 result, report no positive equivalent-label gain.

Compute the estimate separately for each seed using its paired R0 curve, then
report the mean and bootstrap confidence interval across seeds. State both the
user-level and image-level equivalent-label counts; user-level is the primary
claim.

## Interpretation rules

- **S-age improves at low fractions; S-shuffle and U-ssl do not:** evidence
  that synthetic age conditioning provides label-efficient supervision.
- **S-age and S-shuffle improve similarly:** generic synthetic image exposure,
  not age conditioning, explains the benefit.
- **S-age and S-ssl improve similarly:** representation learning, rather than
  supervised synthetic age labels, explains the benefit.
- **U-ssl improves similarly to S-ssl:** the effect is generic extra
  pretraining compute/data, not synthetic dorsal-hand specificity.
- **Only R1 benefits strongly:** synthetic pretraining mainly substitutes for
  generic initialisation, rather than improving a standard pretrained model.
- **No improvement at 100% but improvement at 1%--10%:** report a low-label
  label-efficiency benefit; this is still a positive result.

Do not claim that synthetic labels improve real age inference unless S-age
beats R0 on the locked real test with the paired, multi-seed comparison and
the relevant shuffled/SSL controls rule out the simpler explanations above.

## Minimum run matrix

The full matrix is 6 label fractions x 6 arms x 3 seeds = 108 fine-tuning
runs, plus the required pretraining runs. Begin with a pilot at 10% and 25%
labels for R0, S-age, S-shuffle, and S-ssl across three seeds. Promote to the
full matrix only after split integrity, checkpoint transfer, and locked-test
exclusion have been verified.
