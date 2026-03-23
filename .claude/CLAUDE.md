# CNN Age Inference

Probabilistic age estimation from dorsal hand images for **binary age-gate classification** (adult vs. minor). The model outputs a Gaussian posterior over age; the tail probability above a threshold is used as a soft adult-probability score.

**Core question:** Can hand appearance alone reliably gate access to age-restricted content?

---

## Architecture

- **Backbones:** EfficientNet-B{0-7}, EfficientNet-V2-{S,M,L}, ConvNeXt-{Tiny,Small,Base,Large,XLarge}, ViT-Tiny-384
- **Head:** 2-unit linear layer → `(mu, log_var)` Gaussian regression
- **Loss:** Gaussian NLL + MSE + MAE (weighted combination), optional intra-user spread penalty, optional embedding contrastive/variance losses
- **Evaluation:** probabilistic age-gate curves (FPR/FNR/TPR/TNR vs. tau), ROC/AUC for Case 1 (keep minors out of adult content) and Case 2 (keep adults out of child platforms)
- **Datasets:** LUICID, HandRGBD, Archive (dorsal views only, combined via `dataset/hand_metadata.py`)

---

## Repository Layout

```
dataset/
  age.py              # AgeDataset — PyTorch Dataset
  hand_metadata.py    # load_combined_metadata(), multi-source loader
  samplers.py         # GroupedBatchSampler, DistributedGroupedBatchSampler
  transforms.py       # build_transforms(img_size)
  utils.py            # filter_metadata, build_kfold_user_splits,
                      # build_held_out_test_split, save/load helpers
models/
  efficientnet_age.py
  convnext_age.py
  vit_age.py
  __init__.py         # resolve_model_builder()
metrics.py            # losses, aggregate_predictions_by_user,
                      # compute_age_gate_curves, challenge tables
displayUtils.py       # plot_roc_curve, DisplayUtils
make_test_split.py    # Step 1 — create held-out test split (run once)
make_kfold_splits.py  # Step 2 — create k-fold splits (excludes test users)
train_distributed.py  # Step 3 — distributed training (torchrun / SLURM)
aggregate_kfold.py    # Step 4 — aggregate k-fold results
evaluate_test.py      # Step 5 — evaluate best checkpoint on held-out test set
submit_distributed.slurm  # Full pipeline orchestration (SLURM)
```

---

## Evaluation Protocol

Three-way split, all splits done at **user level** to prevent data leakage:

| Split | Size | Purpose |
|---|---|---|
| Held-out test | ~15% of users | Final reported metrics — never touched during training |
| K-fold validation | ~85% / k per fold | Model selection, early stopping |
| Train | remaining ~85% × (k-1)/k | Gradient updates |

- Test split: stratified by adult/minor, created once by `make_test_split.py`, saved to `test_users.json`
- K-fold: `make_kfold_splits.py --test-users-file test_users.json` excludes test users before folding
- Training: `train_distributed.py --test-users-file` enforces exclusion at data-loading time
- Final evaluation: `evaluate_test.py` — run once after all model selection is complete

---

## Workflow

```bash
# 1. Create held-out test split (once, before any training)
python make_test_split.py \
  --out-file splits/test_users.json \
  --test-size 0.15 --seed 42

# 2. Create k-fold splits (excluding test users)
python make_kfold_splits.py \
  --k 5 --seed 42 \
  --test-users-file splits/test_users.json \
  --stratify-adult \
  --out-file splits/folds_k5.json

# 3. Train all folds (or submit via SLURM — handles steps 1-5 automatically)
torchrun train_distributed.py \
  --fold-file splits/folds_k5.json --fold-index 0 \
  --test-users-file splits/test_users.json \
  --model v2_m --img-size 480

# 4. Aggregate k-fold validation results
python aggregate_kfold.py --kfold-root runs/my_run

# 5. Evaluate on held-out test set (once, after model selection)
python evaluate_test.py \
  --checkpoint runs/my_run/fold_0/v2_m_age_regressor_ddp.pth \
  --test-users-file splits/test_users.json \
  --output-dir runs/my_run/test_eval
```

> `test_users.json` is frozen — never regenerate after training has started.
> The SLURM script (`submit_distributed.slurm`) automates all five steps.

---

## Key SLURM Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODELS` | `v2_m` | Backbone(s) to train |
| `KFOLDS` | `5` | Number of CV folds |
| `EPOCHS` | `240` | Max training epochs |
| `PATIENCE` | `10` | Early stopping patience |
| `BATCH_SIZE` | `8` | Per-GPU batch size |
| `IMG_SIZE` | `480` | Input resolution |
| `LR` | `2e-4` | Learning rate |
| `SEED` | `42` | Global random seed |
| `TEST_SPLIT_FILE` | `${OUTPUT_ROOT}/test_users.json` | Held-out test split path |
| `TEST_SPLIT_SIZE` | `0.15` | Fraction of users for test set |
| `RUN_TEST_EVAL` | `1` | Run test evaluation after training |
| `OUTPUT_ROOT` | `${PROJECT_ROOT}/runs` | Root output directory |
| `DATA_ROOT` | `/home/rb3434w/HandsDatasets` | Dataset root |

---

## Binary Age-Gate Evaluation

The model outputs `(mu, log_var)`. The adult probability is:

```
p_adult(tau) = P(age >= 18 | mu, sigma^2) = 1 - Phi((18 - mu) / sigma)
```

**Case 1 — Keep minors out of adult content:**
- Admit if `p_adult >= tau` → report FPR (minors admitted), FNR (adults rejected)

**Case 2 — Keep adults out of child platforms:**
- Admit if `p_adult < tau` → report FPR (adults admitted), FNR (minors rejected)

Both cases produce ROC curves (FPR vs. TPR) with AUC. Challenge tables report FPR/FNR broken down by age bin (10–12, 13–15, 16–17 for FPR; 18–19, 20–24, 25–29, 30–39, 40–49, 50+ for FNR).

---

## Notes

- All splits are user-level — all images from one user always go to the same split
- Dorsal images only (`aspect` column contains "dorsal"), known ages only
- Default cap: 16 samples per user, 20 users per age year
- Prediction aggregation: randomly sample `n` images per user at eval time; `n=1` is single-image, `n>1` pools predictions
