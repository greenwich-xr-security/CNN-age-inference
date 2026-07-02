# Quality Network Experiment Note

Date: 2026-07-02

## Current Running Experiment

The current HPC experiment is:

```text
Job ID: 1049175
Job name: quality-b0-eu2
Branch: feature/hand-quality-age-regression
Commit: 5113279
Workdir: /home/rb3434w/CNN-age-inference
Run folder: runs/v2_small_quality_full_wall3_b32_quality_error_uncertainty_b0
Age source run: runs/v2_small_quality_full_wall3_b32
Model: EfficientNet-B0 quality assessor
GPUs: 2
```

The submitted wrapper is:

```text
logs/run_quality_error_uncertainty_b0_gpu2.sh
```

It runs only the quality stages:

```text
quality_targets -> quality -> quality_eval
```

## Why This Experiment Exists

The quality network should estimate whether an image/sample is useful for age regression. It should not learn adult/minor boundary behavior, because that mixes a downstream decision rule into an image-quality estimator.

We also removed TTA consistency from the quality target. A bad or uninformative image can be consistently bad under augmentation, so consistency can accidentally reward samples that are stable but useless.

The current target quality score is therefore:

```text
penalty = normalized_abs_error + normalized_uncertainty

quality_score = 1 / (1 + penalty)
```

The quality model itself is now a single-output network trained directly against `quality_score`:

```text
pred_quality_score = sigmoid(output)
loss = SmoothL1(pred_quality_score, target_quality_score)
```

## Intended Direction

Use the quality network as a ranking/filtering model for held-out samples. The main evaluation should ask:

```text
If we reject the lowest predicted-quality samples, does age-regression performance improve?
```

Primary things to inspect after the run:

```text
runs/v2_small_quality_full_wall3_b32_quality_error_uncertainty_b0/aggregate/quality_summary.csv
runs/v2_small_quality_full_wall3_b32_quality_error_uncertainty_b0/aggregate/quality_age_gate_reassessment.csv
runs/v2_small_quality_full_wall3_b32_quality_error_uncertainty_b0/quality_predictions_test.csv
```

For visual inspection, regenerate held-out thumbnail slices from the new `quality_predictions_test.csv`, using non-overlapping percentage buckets from 0.05 to 0.30.

## Important Notes

Old b0 quality checkpoints are incompatible with the new model head because they used the previous multi-output quality assessor.

The generated target CSV may still contain diagnostic columns such as `boundary_error` and `consistency_score`, but these are not used by the quality model target or loss.

