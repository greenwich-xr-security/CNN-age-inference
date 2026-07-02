# Branch History: Quality Network For Age Regression

Branch: `feature/hand-quality-age-regression`  
Date: 2026-07-02

## Purpose

This branch explores a quality network for hand-image age regression.

The quality network is not intended to classify adult/minor status. Its purpose is to estimate how useful or reliable an input sample is for age regression, so low-quality samples can be ranked, inspected, or filtered before downstream age-assurance metrics are computed.

## Original Idea

The first quality network predicted several targets derived from the age-regression model:

```text
expected_abs_error
expected_uncertainty
expected_consistency
boundary_error_logit
quality_score
```

This made the model multi-task. It learned not only an overall quality score but also component signals describing why a sample may be poor.

## Original Experiment Set

These are the quality experiment folders used in the first comparison set, plus the later single-score b0 run for context.

| Label | Run folder | Samples | Corr quality | Corr abs error | MAE reject 0% | MAE reject 30% | MAE reject 50% | FPR reject 0% | FPR reject 30% | FPR reject 50% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2-S baseline | `v2_small_quality_full_wall3_b32_quality` | 7213 | 0.554 | 0.194 | 4.759 | 4.139 | 3.682 | 0.070 | 0.070 | 0.071 |
| V2-S age-error norm | `v2_small_quality_full_wall3_b32_quality_age_norm` | 7213 | 0.546 | 0.134 | 4.759 | 4.252 | 3.686 | 0.070 | 0.070 | 0.066 |
| V2-S component norm | `v2_small_quality_full_wall3_b32_quality_age_norm_components` | 7213 | -0.020 | 0.110 | 4.759 | 4.239 | 4.208 | 0.070 | 0.062 | 0.032 |
| B0 component norm | `v2_small_quality_full_wall3_b32_quality_age_norm_components_b0` | 7213 | -0.021 | 0.118 | 4.759 | 4.297 | 4.226 | 0.070 | 0.057 | 0.049 |
| B1 component norm | `v2_small_quality_full_wall3_b32_quality_age_norm_components_b1` | 7213 | -0.025 | 0.086 | 4.759 | 4.574 | 4.421 | 0.070 | 0.068 | 0.055 |
| B0 error+uncertainty single score | `v2_small_quality_full_wall3_b32_quality_error_uncertainty_b0` | 7213 | 0.102 | n/a | 4.759 | 4.592 | 4.242 | 0.070 | 0.049 | 0.030 |

Notes:

- `Corr quality` is correlation between target and predicted quality score on validation predictions.
- `Corr abs error` is only available for runs that predicted expected absolute error as a separate output.
- The single-score b0 run has a cleaner objective, but the older V2-S baseline and age-normalized runs produced stronger MAE filtering at high rejection fractions.

## Changeset Reference

The current branch changes related to the quality network are:

| File | Change | Reason |
| --- | --- | --- |
| `generate_quality_targets.py` | Removed `--consistency-scale` from the quality-score formula. `quality_score` is now based on normalized absolute error and normalized uncertainty only. | Avoid rewarding samples that are consistently bad under TTA. |
| `dataset/quality.py` | Reduced `QUALITY_TARGET_COLUMNS` to only `quality_score`. | Train the quality model on the single quality target rather than boundary/component targets. |
| `models/quality_assessment.py` | Reduced `QUALITY_OUTPUT_NAMES` to only `quality_score`, making the EfficientNet quality head single-output. | Keep the deployed quality network focused on one ranking/filtering score. |
| `train_quality_distributed.py` | Replaced the previous multi-task loss with `SmoothL1(sigmoid(output), quality_score)`. Removed component and boundary loss arguments. | Remove adult/minor boundary supervision and component-output supervision from the current single-score design. |
| `predict_quality.py` | Uses the shared `decode_outputs`, which now emits only `pred_quality_score`. | Prediction CSVs from new checkpoints are lean and contain only the quality score. |
| `evaluate_quality.py` | Made component and boundary metrics optional when old CSV columns exist. | Allow both old multi-output CSVs and new single-score CSVs to be evaluated. |
| `evaluate_quality_age_gate.py` | Removed dependency on `pred_boundary_error_prob` and `pred_expected_abs_error`; only `image_path` and `pred_quality_score` are required from quality predictions. | Age-gate filtering should use quality ranking only. |
| `QUALITY_EXPERIMENT_NOTE.md` | Added a short operational note for the running/completed b0 quality experiment. | Preserve current experiment context and next steps. |
| `history.md` | Added this branch history, experiment comparison table, rationale, and changeset reference. | Make the quality-network design history explicit for future work. |

Current branch-local quality files modified:

```text
dataset/quality.py
evaluate_quality.py
evaluate_quality_age_gate.py
generate_quality_targets.py
models/quality_assessment.py
train_quality_distributed.py
```

## Why Boundary Information Was Removed

The `boundary_error` target encoded whether the age prediction crossed the adult/minor threshold, normally age 18:

```text
boundary_error = true_adult != pred_adult
```

This is useful for age-assurance evaluation, but it is not pure image/sample quality. Including it risks teaching the quality network about the downstream decision boundary rather than about whether the image is useful for age regression. For this reason, boundary supervision was removed from the quality network path.

## Why Consistency Was Removed

Consistency came from test-time augmentation stability:

```text
consistency = std(age predictions across augmented versions of the image)
```

The concern is that a bad image can be consistently bad. For example, a completely black image may produce stable predictions under augmentation, even though it is uninformative. This means consistency can accidentally reward useless samples. For this reason, consistency was removed from the quality-score formula.

## Current Quality Target

The current quality target is based only on age-regression error and model uncertainty:

```text
penalty = normalized_abs_error + normalized_uncertainty

quality_score = 1 / (1 + penalty)
```

Where:

```text
normalized_abs_error =
  abs(age_pred_mean - true_age) / mean_abs_error_for_age_bin

normalized_uncertainty =
  predicted_std / mean_predicted_std_for_age_bin
```

Diagnostic columns such as `boundary_error` and `consistency_score` may still be written to the generated quality-target CSV for analysis, but they are not used by the quality model target or loss.

## Current Quality Network Design

The current quality network is a single-output model:

```text
pred_quality_score = sigmoid(output)
```

It is trained with:

```text
loss = SmoothL1(pred_quality_score, target_quality_score)
```

This is conceptually clean: the network outputs only the score used for ranking/filtering samples.

## Recent B0 Experiment

Run:

```text
v2_small_quality_full_wall3_b32_quality_error_uncertainty_b0
```

Source age run:

```text
v2_small_quality_full_wall3_b32
```

HPC job:

```text
1049175, quality-b0-eu2, completed on 2 GPUs
```

Summary:

```text
corr_quality = 0.10222283127698764
```

The single-output B0 quality network completed successfully. Held-out filtering did improve MAE as more low-quality samples were rejected, but the rejection was not balanced across age classes.

Held-out set:

```text
minors = 199
adults = 584
```

Rejected samples by quality threshold:

```text
reject 10%:  7 minors,  72 adults
reject 20%: 16 minors, 141 adults
reject 30%: 35 minors, 200 adults
reject 40%: 52 minors, 261 adults
reject 50%: 65 minors, 326 adults
```

This means the current model disproportionately rejects adult samples.

## Current Running HPC Job

The active rerun is:

```text
Job ID: 1049182
Job name: age-rw-quality-b0
Branch: feature/hand-quality-age-regression
Commit: 9db2de1
GPUs: 2
Log: /home/rb3434w/CNN-age-inference/logs/age-rw-quality-b0-1049182.log
```

This job retrains the age-regression model first, then trains the b0 quality network from the new out-of-fold age predictions.

Age run:

```text
runs/v2_small_quality_full_wall3_b32_age_reweight
```

Quality run:

```text
runs/v2_small_quality_full_wall3_b32_age_reweight_quality_b0
```

The key change is:

```text
AGE_REWEIGHT=1
```

Why this run exists:

The previous age source run, `v2_small_quality_full_wall3_b32`, was trained with:

```text
age_reweight_loss=False
```

Because the quality labels are generated from age-model predictions, any age-distribution bias in the source age regressor can leak into the quality target and the quality network. The single-score b0 quality run disproportionately rejected adult samples, so this rerun tests whether a reweighted age regressor produces better quality targets and a less biased quality filter.

The intended pipeline is:

```text
1. Train age regression with AGE_REWEIGHT=1
2. Generate quality targets from the reweighted age OOF predictions
3. Train b0 quality network
4. Evaluate held-out quality filtering
5. Check minor/adult rejection balance
```

Correction note:

An earlier launch, `1049181`, was cancelled because it was started while the HPC checkout was on `feature/age-assurance-severity-losses`. Since `train_distributed.py` differs between branches, that run was not a clean run for this branch. Its partial new output folders were removed, the HPC checkout was switched/pulled to `feature/hand-quality-age-regression` at commit `9db2de1`, and the corrected job `1049182` was launched.

## Interpretation

The single-score design is clean, but it may be too compressed. A single target may not provide enough learning signal for the network to discover robust visual cues of age-regression quality.

The earlier component-style model may have performed better because auxiliary targets gave the backbone richer supervision. However, boundary information should remain excluded because it is not sample quality.

## Suggested Direction

The next likely direction is a middle ground:

```text
outputs:
  expected_abs_error
  expected_uncertainty
  quality_score

excluded:
  boundary_error
  consistency

loss:
  SmoothL1(pred_abs_error, target_abs_error)
+ 0.5 * SmoothL1(pred_uncertainty, target_uncertainty)
+ SmoothL1(pred_quality_score, target_quality_score)
```

This keeps the quality network aligned with age-regression usefulness while restoring useful auxiliary supervision. It avoids the adult/minor boundary leak and avoids the black-image consistency failure mode.

## Practical Evaluation Direction

For future runs, evaluate:

```text
1. Correlation between predicted and target quality.
2. Held-out MAE after rejecting the lowest-quality 5%, 10%, ..., 50%.
3. Minor/adult rejection balance.
4. Thumbnail slices of rejected held-out samples.
```

The quality network should not be considered successful only because it improves aggregate MAE. It should also avoid systematically discarding one age group more than another unless that bias is understood and justified.
