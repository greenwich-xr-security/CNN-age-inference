# Plan: BYOL SSL pretraining for EfficientNet-V2-S

Status: proposal, not started. DINOv2 work on this branch has been removed (see below).
No BYOL code written yet.

## Why this replaces the DINOv2 plan

The prior plan on this branch (`DINOV2_SSL_PLAN.md`, now deleted) added a pretrained
DINOv2 ViT backbone option to the SSL pretraining pipeline. That work is not usable
here: the study design needs the *same* backbone architecture across every arm of the
control ladder (baseline, synthetic-label pretraining, SSL pretraining), and the
project's best-performing, production backbone is EfficientNet-V2-S (`v2_s`) -- a CNN.
DINOv2's actual machinery (iBOT masked-patch prediction, register tokens, patch-level
distillation) requires a ViT token grid and cannot attach to a CNN's pooled conv
features, so it can't be swapped in for the CNN arms without breaking the
fixed-architecture constraint.

Removed as part of this pivot: `models/dinov2_backbone.py`, the DINOv2 branch in
`resolve_backbone_builder`/`patch_size_for` in `models/__init__.py`, and the
`--freeze-blocks` / patch-aligned crop-size rounding added to `train_dino.py` and
`train_dino_distributed.py` (that logic existed only to support the ViT backbone and
has no meaning for a CNN). All four files are back to their pre-DINOv2 state.

Note: the *existing* DINO v1 pipeline already on this branch (`models/dino.py`,
`train_dino.py`, `train_dino_distributed.py` targeting `--model v2_s`) is CNN-native
and unaffected by any of this -- DINO v1's image-level self-distillation doesn't need
patch tokens either. BYOL was chosen over reusing that DINO v1 pipeline directly for
reasons discussed separately (paper framing, no negative-pair/queue tuning, simpler
collapse-avoidance mechanism for a modest-batch 4-GPU setup); it is not a technical
necessity the way dropping DINOv2 was.

## Goal

A BYOL SSL pretraining pipeline for EfficientNet-V2-S, run identically (same recipe,
same augmentation budget, same backbone) to produce two pretrained checkpoints:

- **S-ssl**: BYOL pretraining on SyntheticDorsalHands2 images, then real-only NLL
  fine-tuning (mirrors the existing synthetic-label pretraining arm's structure, job
  `1050812` -> `1050819` in `HPC_JOB_HISTORY.md`, but replacing "train with synthetic
  age labels" with "self-supervise on synthetic images, no labels used").
- **U-ssl**: BYOL pretraining on unlabeled real images (HandRGBD + ProlificHands),
  then real-only NLL fine-tuning.

Both are compared against the existing baseline (real-only, no pretraining) and
synthetic-label-pretraining arms already produced by `train_distributed.py` /
`submit_distributed.slurm` -- those scripts are unchanged and out of scope here.

Out of scope / not defined by this plan: the exact composition of any other arms
(e.g. a shuffled-label control) referenced elsewhere in the broader study design --
this plan only covers the two BYOL arms and assumes the rest of the control ladder is
handled by the existing supervised training pipeline.

## What's reusable as-is

Nothing about `dataset/ssl.py`'s pairing logic or the CNN backbone classes is
DINO/DINOv2-specific -- these carry over to BYOL unchanged:

- `dataset/hand_metadata.py` (`get_dataset_root`, `load_combined_metadata`,
  `set_dataset_root`) and `dataset/utils.py` (`filter_metadata_ssl`).
- `dataset/ssl.py:HandSSLPairDataset` -- same-user/same-aspect pairing, unrelated to
  the SSL objective.
- `dataset/ssl_transforms.py:DinoMultiCropTransform` / `DinoAugmentationConfig` --
  BYOL's original recipe uses two full-strength global views (no multi-crop), which
  this transform already supports via `num_local_crops=0`; can stay as-is initially
  and be generalized/renamed later if multi-crop BYOL is worth trying.
- `models/backbones.py:EfficientNetBackbone` / `ConvNeXtBackbone` -- exactly the
  pooled, `embed_dim`-exposing feature extractors BYOL's online/target encoders need.
- `models/ssl_utils.py:load_dino_backbone` -- generic backbone-checkpoint loader
  (strips `module.`/`backbone.` prefixes); reusable for loading a BYOL-pretrained
  backbone into `train_distributed.py --ssl-pretrained` downstream.
- From `train_dino.py` / `train_dino_distributed.py`: `cosine_schedule` (momentum
  ramp), `update_teacher` (EMA parameter update), `forward_views` (batches
  same-resolution crops together before a single forward pass), `disable_inplace_ops`,
  `set_bn_eval` -- all algorithm-agnostic training-loop plumbing, not DINO-specific.

## What needs to be built

### Phase 1 -- BYOL heads and loss
- `models/byol.py`: `BYOLProjector` (2-layer MLP, BatchNorm+ReLU, matches the
  original BYOL paper), `BYOLPredictor` (2-layer MLP, online branch only -- this
  asymmetry plus stop-gradient is what BYOL uses instead of DINO's
  centering/temperature to avoid representation collapse), and a `BYOLNetwork`
  wrapper: online = backbone + projector + predictor, target = backbone + projector
  (no predictor), replacing `models/dino.py`'s `DINOHead`/`DinoNetwork`.
- BYOL loss: symmetrized negative cosine similarity (equivalently, normalized MSE)
  between `predictor(online_proj(view_a))` and `stopgrad(target_proj(view_b))`,
  summed over both view orderings. Replaces `DINOLoss` entirely -- no center buffer,
  no student/teacher temperature.

### Phase 2 -- training scripts
- `train_byol.py` / `train_byol_distributed.py`, adapted from the DINO scripts:
  keep the EMA momentum schedule (`cosine_schedule`, `update_teacher`) and the
  dataset/augmentation plumbing; drop `--student-temp`, `--teacher-temp*`,
  `--center-momentum`, `--out-dim`/`--bottleneck-dim` (DINO-head-specific) in favor of
  `--projector-dim`, `--predictor-hidden-dim`; default `--model` to `v2_s`.
- Note going in: BYOL is more sensitive to batch size and augmentation strength than
  DINO (no negatives to anchor against), so the momentum schedule and batch size used
  in the existing DINO runs are a starting point, not a guarantee -- expect some
  tuning before S-ssl/U-ssl runs are trustworthy.

### Phase 3 -- HPC validation
- Same shape as the DINOv2 validation run already done and reverted on this branch: a
  short, modest-scale `sbatch` run of `train_byol_distributed.py --model v2_s` on
  real data, confirming the loss trends down and nothing crashes, before committing to
  a full-length S-ssl/U-ssl schedule.

### Phase 4 -- full S-ssl / U-ssl runs and comparison
- Launch the two full BYOL pretraining runs (synthetic images for S-ssl, unlabeled
  real images for U-ssl), each followed by real-only NLL fine-tuning via the existing
  `train_distributed.py --ssl-pretrained <byol_checkpoint>` path (same mechanism the
  synthetic-label pretraining arm already uses).
- Record results in `HPC_JOB_HISTORY.md` alongside the existing baseline
  (`1050753`/`1050704`-style) and synthetic-label-pretraining (`1050812`/`1050819`)
  rows for a direct comparison.

## Open questions before starting implementation
- Target BYOL hyperparameters (projector/predictor dims, EMA momentum range, LR) --
  start from the original BYOL paper's defaults, adjusted for the smaller batch size
  this cluster's `gpu-beast` allocation realistically supports?
- Is multi-crop BYOL (extra local views, á la DINO) worth trying, or start with the
  canonical 2-global-view recipe and only add crops if the 2-view baseline underwhelms?
- Confirm the unlabeled-real-image pool for U-ssl (HandRGBD + ProlificHands, same
  `filter_metadata_ssl` selection already used by the DINO pipeline) is the intended
  set, and not a different split reserved for evaluation.
