# CNN Age Inference

This repository contains code for training and evaluating convolutional neural networks (CNNs) for age inference from hand dorsal img data.
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

The default model is `EfficientNetAgeRegressor` (`models/efficientnet_age.py`), a thin wrapper over torchvision EfficientNet-B{0-7}.

- Backbone: pick the variant with `--model {b0..b7}`; the corresponding canonical image size is enforced via `--img-size`.
- Initialization: loads the best available torchvision weights (DEFAULT enum when present, otherwise ImageNet-pretrained).
- Head: replaces the EfficientNet classifier with a 2-unit linear layer that jointly regresses the Gaussian mean and log-variance of age.
- Loss: `train_age.py` optimizes the negative log-likelihood under that Gaussian (`gaussian_nll_loss`) so the network learns both central tendency and epistemic spread.
- Outputs: the forward pass returns `(mu, log_var)`, which downstream utilities convert into adult probabilities via Gaussian tail integration.

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

