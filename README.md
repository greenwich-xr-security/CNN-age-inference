# CNN Age Inference

Training and evaluation code for probabilistic age regression from dorsal hand RGB images. Supports EfficientNet, ConvNeXt, Swin, ViT, MobileNet and ResNet backbones with an optional auxiliary normal-map reconstruction head for surface prior experiments.

---

### 1. Create a virtual environment

```bash
python3.10 -m venv .venv
.venv\Scripts\activate
```

---

### 2. Install dependencies
Install all base dependencies:

```bash
pip install -r requirements.txt
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. Model architecture

The default model is `EfficientNetAgeRegressor` (`models/efficientnet_age.py`), a thin wrapper over torchvision EfficientNet-B{0-7}. All other backbones (ConvNeXt, Swin, ViT, MobileNet, ResNet) follow the same interface.

- Backbone: pick the variant with `--model {b0..b7,cnt,cns,sw2t,...}`; the canonical image size is used automatically.
- Initialization: loads the best available pretrained weights (torchvision or timm).
- Head: replaces the backbone classifier with a 2-unit linear layer outputting `(mu, log_var)` for probabilistic regression.
- Loss: optimizes a weighted combination of Gaussian NLL, MSE and MAE (configurable via `--loss-weight-nll/mse/mae`).
- Outputs: `(mu, log_var)` — downstream utilities integrate the Gaussian tail to produce adult-gate probabilities.

### 3.1 Probabilistic age regression objective

The network is trained as a probabilistic regressor, not a point-estimate regressor.

- Predicted distribution: for each sample, the model outputs a Gaussian posterior over age with parameters `(mu, sigma^2)`.
- Network outputs: the head predicts `(mu, log_var)`, where `log_var = log(sigma^2)` for numerical stability.
- Training loss: age regression is optimized with Gaussian negative log-likelihood (NLL), so the model learns both:
  - accurate age center (`mu`)
  - calibrated uncertainty (`sigma^2`)

This is the age-specific training objective used by the regression head before binary age-gate thresholding.

---

### 4. Binary age-gate evaluation

After training all models, treat the age-inference head as an adult gate with an application-defined threshold (default 18 years). The network performs probabilistic age regression and outputs the parameters of a Gaussian age posterior `(mu, sigma^2)`; integrate the Gaussian tail above the threshold to obtain the probability of the user being an adult, noted as `p_adult(tau)`.

#### Adult gate
Use this configuration when minors must not access adult-only experiences.

Decision rule (threshold at 18):
- If `p_adult >= tau` -> user admitted
- If `p_adult < tau` -> user rejected

Report the following metrics:
- Minor Incorrectly Admitted (FPR): percentage of minors incorrectly admitted. <-- undesired risk.
- Adult Incorrectly Rejected (FNR): percentage of adults wrongly denied access.
- Adult Access Rate (TPR): percentage of adults correctly admitted. <-- desired usability.
- Minor Rejection Rate (TNR): percentage of minors correctly denied access.

#### ROC analysis
Produce one ROC curve for the adult gate with:
- x-axis: FPR (undesired risk).
- y-axis: TPR (desired usability).
- Each point reflects one threshold value `tau` (the confidence cutoff).
- Report the Area Under Curve (AUC) next to the plot to summarize the trade-off.

---

### 5. Auxiliary normal-map prior experiment (branch: `normals-auxiliary-prior`)

**Hypothesis:** forcing the shared encoder to reconstruct surface normal maps at training time — where normals encode hand geometry such as wrinkle depth, knuckle definition and vein prominence — makes it more geometrically sensitive and improves age regression from RGB alone.

#### Architecture (Option A — lightweight decoder)

```
RGB image  →  backbone.features  →  [B, C, h, w]
                    │
          ┌─────────┴──────────────────────┐
          │  avgpool + linear              │  NormalsHead (shallow)
          │  → (mean, log_var)  [age head] │  Conv1×1 → 256ch
          │                               │  Upsample ×2, Conv3×3 → 128ch
          │                               │  Upsample ×2, Conv3×3 →  64ch
          │                               │  Upsample ×2, Conv3×3 →   3ch + tanh
          │                               └→ pred_normals [B, 3, H, W]
          └────────────────────────────────
```

The decoder is intentionally shallow (~0.5 M params) so that geometric signal is forced into the shared encoder rather than reconstructed by the decoder alone. At inference time the normals head is discarded — only RGB → age is used.

#### Datasets

| Dataset | Samples | Normals GT | Role |
|---|---|---|---|
| handRGBD | 6,791 | — | age loss only |
| LUICIDHands | 1,714 | paired JPEGs in `normals/` | age loss + normals loss |

LUICIDHands is oversampled in training (default 40% of each batch) to ensure consistent geometric supervision despite the size imbalance.

The optional ProlificHands export can be included in split generation, training, and held-out evaluation with `--include-prolific` (or `INCLUDE_PROLIFIC=1` for the SLURM scripts). It is loaded from `HandsDatasets/ProlificHands/reference_prolific.csv`, uses masked RGB derivatives when present, and otherwise follows the same dorsal-image, known-age metadata filtering as the other age datasets.

#### Training loss

```
total_loss = age_loss  +  λ_normals × normals_loss
```

- `age_loss` fires on all samples (both datasets).
- `normals_loss` is a cosine-similarity loss on unit normal vectors, masked to samples that have GT normals (LUICIDHands only).
- Normal maps encoded as 8-bit JPEG with pixel → `[-1, 1]` decoding: `(pixel/255) × 2 − 1`.
- No spatial augmentations (flip/rotate) are applied to normal-paired samples to preserve normal vector directions.

#### Running the experiment

```bash
# Baseline — no normals head (run on the same seed for a fair comparison)
python train_age.py --model b4 --seed 42 --output-dir runs/b4_baseline

# Experiment — with normals auxiliary head
python train_age.py --model b4 --seed 42 --output-dir runs/b4_normals \
    --normals-aux \
    --loss-weight-normals 0.1 \
    --lucid-fraction 0.4
```

On the HPC via SLURM (both `submit.slurm` and `submit_distributed.slurm` support the same env vars):

```bash
# Baseline sweep
sbatch submit_distributed.slurm

# Normals experiment sweep (EfficientNet variants only)
NORMALS_AUX=1 MODELS="b4" sbatch submit_distributed.slurm

# Tune loss weight
NORMALS_AUX=1 LOSS_WEIGHT_NORMALS=0.05 MODELS="b4" sbatch submit_distributed.slurm
```

#### Key env vars

| Variable | Default | Meaning |
|---|---|---|
| `NORMALS_AUX` | `0` | Set to `1` to enable auxiliary normals head |
| `LOSS_WEIGHT_NORMALS` | `0.1` | λ for normals reconstruction loss |
| `LUCID_FRACTION` | `0.4` | Target fraction of each batch from LUICIDHands |

> **Note:** the normals head is currently implemented for EfficientNet backbones only. Other architectures (Swin, ConvNeXt, etc.) will ignore `--normals-aux` unless extended in `models/`.

