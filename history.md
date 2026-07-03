# Branch History: Quality Network For Age Regression

Branch: `feature/hand-quality-age-regression`  
Date: 2026-07-02

## Purpose

This branch explores a quality network for hand-image age regression.

The quality network is not intended to classify adult/minor status. Its purpose is to estimate how useful or reliable an input sample is for age regression, so low-quality samples can be ranked, inspected, or filtered before downstream age-assurance metrics are computed.

## Quality Network Design

This approach is inspired by Face Image Quality Assessment (FIQA), where quality is defined by how reliably a sample supports the primary task — here age regression — rather than by perceptual image quality.

The quality network is a multi-head auxiliary design. The backbone is supervised on three targets simultaneously, giving it richer learning signal than a single quality score alone:

```text
outputs:
  pred_abs_error    = softplus(out[0])   ← auxiliary: normalised absolute error
  pred_uncertainty  = softplus(out[1])   ← auxiliary: normalised predicted std
  pred_quality_score = sigmoid(out[2])   ← primary: quality ranking score
```

Loss:

```text
loss = SmoothL1(pred_abs_error,    target_abs_error)
     + 0.5 × SmoothL1(pred_uncertainty, target_pred_std)
     + SmoothL1(pred_quality_score, target_quality_score)
```

Where the targets are derived from the age-regression model's out-of-fold predictions:

```text
normalized_abs_error   = abs_error  / f_err(true_age)
normalized_uncertainty = pred_std   / f_std(true_age)

quality_score = 1 / (1 + normalized_abs_error + normalized_uncertainty)
```

`f_err(age)` and `f_std(age)` are smooth curves fitted over all OOF predictions using Gaussian kernel regression (Nadaraya-Watson, bandwidth = 3 years), replacing the previous discrete age-bin normalization. This avoids sparsity and boundary artefacts at extreme ages and handles non-monotonic error profiles without assumptions on curve shape. LOWESS and smoothing splines are alternatives with better local bias properties but were not used to avoid extra dependencies.

The auxiliary heads (`pred_abs_error`, `pred_uncertainty`) are the direct components of `quality_score`, so they provide intermediate supervision for exactly what the backbone needs to learn. Boundary error and consistency are excluded: boundary encodes the downstream decision rather than image quality; consistency can reward consistently uninformative samples.

## Current HPC Job

```text
Job ID: 1049193
Job name: age-infer-ddp
Run name: v2_small_quality_full_wall3_b32_age_reweight_correct_split
Test split: multitask_v2s_shared_test_users.json (128 users: 92 handrgbd, 36 prolific)
Model: V2-S, img_size=384, AGE_REWEIGHT=1
Pipeline stage: age only
GPUs: 8
```

Previous runs used `runs/test_users.json` which was created from a dataset including HaGRID. Since HaGRID is excluded from these runs, only 65 HandRGBD users were effective as the test set, with no Prolific holdout. This rerun uses the correct shared split.
