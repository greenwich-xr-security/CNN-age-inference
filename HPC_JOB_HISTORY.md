# HPC job history

All completed runs below used EfficientNet-V2-S at 384 px, seed 42, five folds,
four GPUs, 16 CPU cores, 64 GB RAM, and batch size 16 per GPU unless noted.

## Job registry

| Job | State / elapsed | Dataset and split | Objective / change | Purpose |
| --- | --- | --- | --- | --- |
| `1050688` | Completed / 2:53:46 | Real only; original capped 15% test | Historical mixed loss | Baseline real-only run |
| `1050689` | Completed / 3:34:49 | Real + SyntheticDorsalHands; original capped 15% test | Historical mixed loss | Original synthetic-data comparison |
| `1050704` | Completed / 6:00:31 | Real only; original capped 15% test | Pure Gaussian NLL | Clean loss baseline |
| `1050705` | Completed / 2:37:17 | Real only; original capped 15% test | Pure Gaussian CRPS | Proper-score loss comparison |
| `1050710` | Completed / 4:12:21 | Real + synthetic; original capped 15% test | Pure Gaussian NLL | Clean synthetic-data comparison |
| `1050711` | Completed / 5:55:39 | Real only; original capped 15% test | NLL + embedding variance (0.8) | Test same-user latent consistency |
| `1050753` | Completed / 5:18:12 | Real only; uncapped fixed 20% test | Pure Gaussian NLL | Effect of removing both real-data caps |
| `1050754` | Completed / 3:51:07 | Real + synthetic; uncapped fixed 20% test | Pure Gaussian NLL | Synthetic comparison without real-data caps |
| `1050812` | Completed / 4:18:16 | Real + SyntheticDorsalHands2 only; uncapped fixed 20% test | Pure Gaussian NLL | Synthetic2-only comparison to `1050754` |
| `1050814` | Completed / 3:10:46 | Real + SyntheticDorsalHands2 only; uncapped fixed 20% test | Pure Gaussian NLL; ViT tiny 384; `MASTER_PORT=29513` | ViT architecture comparison to `1050812` |
| `1050819` | Completed / 00:49:50 | Real only; uncapped fixed 20% test | Pure Gaussian NLL; fold-matched SyntheticDorsalHands2 initialisation; LR `2e-5` | Test synthetic-to-real fine-tuning |
| `1050821` | Failed / 02:25:52 | Real + SyntheticDorsalHands2; uncapped fixed 20% test | Pure Gaussian NLL; ViT small 384; `MASTER_PORT=29515`; 3 GPUs | Fold 4 test evaluation failed because the fixed test manifest was absent |
| `1050832` | Completed / 01:38:57 | SyntheticDorsalHands2 only | BYOL S-SSL pretraining; EfficientNet-V2-S; 100 epochs; batch 32/GPU; 4 GPUs | BYOL self-supervised pretrain of V2-S backbone on synthetic dorsal hands |
| `1050851` | Cancelled / 00:02:50 | Real only; uncapped fixed 20% test | Pure Gaussian NLL; BYOL-pretrained init; LR `2e-4` | BYOL→supervised finetune attempt 1 — cancelled before fold 0 completed |
| `1050852` | Cancelled / 00:25:29 | Real only; uncapped fixed 20% test | Pure Gaussian NLL; BYOL-pretrained init; LR `2e-4` | BYOL→supervised finetune attempt 2 — cancelled during fold 0 (epoch 29) |
| `1050854` | Completed / 03:10:47 | Real only; uncapped fixed 20% test | Pure Gaussian NLL; BYOL-pretrained init; LR `2e-4` | BYOL→supervised finetune, full five-fold run |
| `1050867` | Cancelled / 01:03:39 | Real only; uncapped fixed 20% test | Pure Gaussian NLL; BYOL-pretrained init; frozen backbone; LR `1e-3`; 60 epochs max | Linear probe of BYOL representations — two folds only; catastrophic failure |
| `1050877` | Completed / 07:44:30 | Real only; uncapped fixed 20% test | Pure Gaussian NLL; random initialisation | Random-init control for real-only EfficientNet-V2-S |
| `1050938` | Failed / 00:00:19 | SyntheticDorsalHands2 only; fixed synthetic2 20% test | Pure Gaussian NLL; stdin Slurm script attempt | Launch-script quoting failed before Python started; replaced by `1050939` |
| `1050939` | Completed / 01:43:53 | SyntheticDorsalHands2 only; fixed synthetic2 20% test | Pure Gaussian NLL | Q2 SS internal-learnability / degeneracy cell |
| `1050940` | Completed / 00:02:30 | Real-trained `1050753` checkpoints evaluated on fixed synthetic2 20% test | Evaluation-only; age-threshold adult-gate mode | Q2 RS/TRTS distribution-match diagnostic |

Historical mixed loss: NLL 0.6 + MAE 0.9 + prediction spread 0.5 +
embedding variance 0.8 + embedding contrast 0.8. Pure objectives disable all
other listed losses. “Capped” means maximum 20 images per real user and 200
real images per integer age; synthetic images bypassed those caps by design.

## Results: original capped 15% test split

All values are unweighted means of five held-out fold results at image-level
aggregation (`n=1`). FPR and Adult FNR use each fold's best operating point
with FPR <= 5%, then average the achieved values.

| Training data | Objective | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Real only | Historical mixed (`1050688`) | 5.064 | 6.532 | 0.9679 | 4.47% | 13.11% |
| Real only | Pure NLL (`1050704`) | **4.891** | **6.391** | 0.9578 | 4.57% | 16.94% |
| Real only | NLL + EmbedVar (`1050711`) | 5.090 | 6.649 | 0.9589 | 4.57% | 14.40% |
| Real only | Pure CRPS (`1050705`) | 5.076 | 6.673 | 0.9598 | 4.29% | 14.24% |
| Real + synthetic | Historical mixed (`1050689`) | 4.636 | **6.093** | 0.9544 | 4.57% | 16.28% |
| Real + synthetic | Pure NLL (`1050710`) | **4.592** | 6.220 | **0.9632** | 4.57% | **15.08%** |

## Results: uncapped fixed 20% test split

The test users and images differ from the table above, so compare only rows in
this table with each other, not with the capped 15% test results. FPR and Adult
FNR use each fold's best operating point with FPR <= 5%, then average the
achieved values.

| Training data | Objective | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Real only | Pure NLL (`1050753`) | 5.017 | 6.706 | 0.9483 | 4.74% | 18.33% |
| Real + synthetic | Pure NLL (`1050754`) | 4.683 | 6.293 | 0.9594 | 4.78% | 17.91% |
| Real + SyntheticDorsalHands2 | Pure NLL, EfficientNet-V2-S (`1050812`) | 4.867 | 6.544 | 0.9546 | 4.78% | 18.14% |
| Real + SyntheticDorsalHands2 | Pure NLL, ViT tiny 384 (`1050814`) | 5.326 | 7.219 | 0.9366 | 4.78% | 20.22% |
| Real only (SyntheticDorsalHands2 initialisation) | Pure NLL fine-tuning, EfficientNet-V2-S (`1050819`) | **4.514** | **6.246** | **0.9641** | **4.69%** | **16.25%** |
| Real + SyntheticDorsalHands2 | Pure NLL, ViT small 384 (`1050821`; 4 folds) | 5.307 | 7.102 | 0.9397 | 4.63% | 19.58% |
| Real only | Pure NLL, random initialisation (`1050877`) | 6.490 | 8.696 | 0.8868 | 5.79% | 27.68% |
| Real only (BYOL S-SSL init) | Pure NLL fine-tuning, EfficientNet-V2-S (`1050854`) | 5.066 | 6.825 | 0.9460 | 4.69% | 19.69% |

For `1050877`, the threshold grid could not reach FPR <= 5% in folds 1, 3,
and 4. The reported operating-point row uses the best FPR <= 5% threshold for
folds 0 and 2, and the lowest-FPR available threshold for folds 1, 3, and 4.

## Fixed uncapped split manifests

The uncapped runs use 20% held-out real users and five folds over the remaining
80%, giving 64% train / 16% validation / 20% test users per fold.

| File | Used by |
| --- | --- |
| `splits/test_users_uncapped_20pct_seed42.json` | Both uncapped runs; 128 fixed held-out real users |
| `splits/folds_k5_uncapped_test20_real_seed42.json` | `1050753` real-only run |
| `splits/folds_k5_uncapped_test20_real_synthetic_seed42.json` | `1050754` real + synthetic run |
| `splits/test_users_synthetic2_20pct_seed42.json` | `1050939` Q2 SS SyntheticDorsalHands2 test split |
| `splits/folds_k5_synthetic2_seed42.json` | `1050939` Q2 SS SyntheticDorsalHands2 train/validation folds |

## Completed and failed launches

### Job `1050938` - Q2 SS SyntheticDorsalHands2 Pure NLL submission attempt

- Submission date: 2026-08-01 19:10:06 BST
- Initial scheduler state: RUNNING on `gpu-beast`; terminal scheduler state FAILED, elapsed `00:00:19`, exit code 2.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/q2-ss-syn2-nll-1050938.log`
- Purpose: first attempt to launch the Q2 SS SyntheticDorsalHands2-only pure NLL job.
- Data/configuration: intended to match `1050939`, but failed before Python started because the stdin-submitted Slurm script was malformed by shell quoting (`syntax error near unexpected token '>'`).
- Replacement: reproducible submit script committed as `submit_q2_ss_synthetic2.slurm`; replacement job is `1050939`.

### Job `1050939` - Q2 SS SyntheticDorsalHands2 Pure NLL

- Submission date: 2026-08-01 19:12:34 BST
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`, elapsed `00:00:25`; fold 0 started successfully.
- Terminal scheduler state: COMPLETED, elapsed `01:43:53`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/q2-ss-syn2-nll-1050939.log`
- Purpose: run the Q2 SS generator-validation cell: train on SyntheticDorsalHands2 and test on the held-out SyntheticDorsalHands2 split as an internal learnability / degeneracy check.
- Data/split: SyntheticDorsalHands2 only; no HandRGBD, ProlificHands, previous SyntheticDorsalHands, or LUICID samples; fixed `splits/test_users_synthetic2_20pct_seed42.json`; exact five-fold synthetic2 train/validation manifest copied from `splits/folds_k5_synthetic2_seed42.json` to the run directory.
- Model/objective: same baseline shape as `1050753` apart from the data source and split manifests: EfficientNet-V2-S at 384 px, ImageNet-pretrained default initialisation, pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled), LR `2e-4`, maximum 240 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU; `MASTER_PORT=29516`.
- Held-out result: five-fold unweighted `n=1` aggregate over 4,423 SyntheticDorsalHands2 held-out samples per fold: MAE 3.396 years, RMSE 4.436 years, adult-gate AUC 0.9824, mean FPR 4.85%, Adult FNR 6.77% at each fold's best FPR <= 5% operating point.

### Job `1050940` - Q2 RS/TRTS real-trained to SyntheticDorsalHands2 evaluation

- Submission date: 2026-08-01 19:34:35 BST
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`, elapsed `00:00:27`; fold 0 loaded successfully on CUDA and found 4,423 SyntheticDorsalHands2 held-out test samples.
- Terminal scheduler state: COMPLETED, elapsed `00:02:30`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/q2_rs_real1050753_to_synthetic2_eval_20260801`
- Log file: `/home/rb3434w/CNN-age-inference/logs/q2-rs-syn2-eval-1050940.log`
- Purpose: run the Q2 RS/TRTS generator-validation cell: evaluate the real-trained reference model on the held-out SyntheticDorsalHands2 split as a distribution-match diagnostic.
- Data/split: checkpoints from real-only job `1050753` (`runs/v2s_handrgbd_prolific_only_nll_uncapped_test20_k5_4gpu_16cpu_20260728_384/fold_*/v2_s_age_regressor_ddp.pth`) evaluated on SyntheticDorsalHands2 only, fixed `splits/test_users_synthetic2_20pct_seed42.json`; no HandRGBD, ProlificHands, previous SyntheticDorsalHands, or LUICID test samples.
- Model/objective: evaluation-only; EfficientNet-V2-S at 384 px with the fold-specific `1050753` checkpoint, embedding head dimension 128 to match training, image-level evaluation (`AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources: one `gpu-beast` node, 1 GPU, 4 CPU cores, 32 GB RAM, inference batch size 64.
- Held-out result: five-fold unweighted `n=1` aggregate over 4,423 SyntheticDorsalHands2 held-out samples per fold: MAE 11.056 years, RMSE 13.390 years, adult-gate AUC 0.5609, mean FPR 41.85%, Adult FNR 9.90%. The 5% FPR operating point was not attainable in any fold within the 10-30 year age-threshold sweep; the reported FPR/FNR use the lowest-FPR available threshold (`tau=30`) in each fold.

### Job `1050812` — Real + synthetic2 Pure NLL

- Submission date: 2026-07-29 21:27:49
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`, elapsed `00:00:29`
- Terminal scheduler state: COMPLETED, elapsed `04:18:16`, exit code 0
- Run directory: `/home/rb3434w/CNN-age-inference/runs/v2s_handrgbd_prolific_synthetic2_trainval_nll_uncapped_test20_k5_4gpu_16cpu_20260729_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/v2s-hrgbd-pro-syn2-nll-uncap20-4g16c-1050812.log`
- Purpose: compare SyntheticDorsalHands2 against the directly comparable SyntheticDorsalHands job `1050754`
- Data/configuration: HandRGBD + ProlificHands real data, SyntheticDorsalHands2 enabled with `INCLUDE_SYNTHETIC_DORSAL2=1`, previous SyntheticDorsalHands disabled with `INCLUDE_SYNTHETIC_DORSAL=0`, LUICID disabled, uncapped real samples, fixed `splits/test_users_uncapped_20pct_seed42.json`, five folds generated with the same seed/protocol as `1050754` after replacing the synthetic source
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM project baseline, batch size 16 per GPU
- Model/objective: EfficientNet-V2-S at 384 px, same active loss composition and weights as `1050754`: pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled; `NORMALS_AUX=0` so the printed normals default is inactive), image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`)
- Held-out result: five-fold unweighted `n=1` aggregate: MAE 4.867 years, RMSE 6.544 years, adult-gate AUC 0.9546, mean FPR 4.78%, Adult FNR 18.14% at each fold's best FPR <= 5% operating point.


### Job `1050814` — Real + synthetic2 Pure NLL, ViT tiny 384

- Submission date: 2026-07-29 22:01:20
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`, elapsed `00:00:09`
- Terminal scheduler state: COMPLETED, elapsed `03:10:46`, exit code 0
- Run directory: `/home/rb3434w/CNN-age-inference/runs/vit_tiny384_handrgbd_prolific_synthetic2_trainval_nll_uncapped_test20_k5_4gpu_16cpu_20260729_384_mp29513`
- Log file: `/home/rb3434w/CNN-age-inference/logs/vit-t384-syn2-nll-mp29513-1050814.log`
- Purpose: compare `vit_tiny_384` against the directly comparable EfficientNet-V2-S SyntheticDorsalHands2 job `1050812`
- Data/configuration: same as `1050812`: HandRGBD + ProlificHands real data, SyntheticDorsalHands2 enabled with `INCLUDE_SYNTHETIC_DORSAL2=1`, previous SyntheticDorsalHands disabled with `INCLUDE_SYNTHETIC_DORSAL=0`, LUICID disabled, uncapped real samples, fixed `splits/test_users_uncapped_20pct_seed42.json`, five folds generated with the same seed/protocol after replacing the synthetic source
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM project baseline, batch size 16 per GPU
- Model/objective: ViT tiny patch-16 at 384 px (`vit_tiny_384`), same active loss composition and weights as `1050812`: pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled; `NORMALS_AUX=0`), image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`)
- Rendezvous: `MASTER_PORT=29513` to avoid colliding with the concurrent EfficientNet-V2-S job `1050812` on the default port.
- Held-out result: five-fold unweighted `n=1` aggregate: MAE 5.326 years, RMSE 7.219 years, adult-gate AUC 0.9366, mean FPR 4.78%, Adult FNR 20.22% at each fold's best FPR <= 5% operating point.

### Job `1050819` — Synthetic2-to-real fine-tuning

- Submission date: 2026-07-30 10:30 (BST)
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`; fold 0 started successfully.
- Terminal scheduler state: COMPLETED, elapsed `00:49:50`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/v2s_hrgbd_real_finetuning_nll_lr2e5_uncapped_test20_k5_4gpu_16cpu_20260730_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/v2s-real-finetune-nll-1050819.log`
- Purpose: test whether SyntheticDorsalHands2 pretraining followed by real-only adaptation improves over direct real + synthetic training (`1050812`) and direct real-only training (`1050753`).
- Data/split: HandRGBD + ProlificHands only; no synthetic or LUICID samples; uncapped real data; fixed `splits/test_users_uncapped_20pct_seed42.json`; exact real-only fixed fold manifest copied from `splits/folds_k5_uncapped_test20_real_seed42.json`.
- Initialisation: fold `i` loads the matching best checkpoint from `1050812/fold_i/v2_s_age_regressor_ddp.pth`; optimizer and early-stopping state reset for an independent fine-tuning stage.
- Model/objective: EfficientNet-V2-S at 384 px; pure Gaussian NLL (`NLL=1`, all other regression, embedding, spread, and normals-auxiliary losses disabled); LR `2e-5` (10x below `1050812`); maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`).
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU; `MASTER_PORT=29514`.
- Held-out result: five-fold unweighted `n=1` aggregate: MAE 4.514 years, RMSE 6.246 years, adult-gate AUC 0.9641, mean FPR 4.69%, Adult FNR 16.25% at each fold's best FPR <= 5% operating point.

### Job `1050877` — Real-only random-init EfficientNet-V2-S

- Submission date: 2026-07-31 16:06:44
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`; fold 0 started successfully.
- Terminal scheduler state: COMPLETED, elapsed `07:44:30`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/r0_random_init_v2s_real_only`
- Log file: `/home/rb3434w/CNN-age-inference/logs/r0-random-init-1050877.log`
- Purpose: real-only random-initialisation control for the later Q2 matrix, separating the value of ImageNet/SSL initialisation from the base EfficientNet-V2-S architecture and fixed real-only split.
- Data/split: HandRGBD + ProlificHands only; no synthetic or LUICID samples; uncapped real data; fixed `splits/test_users_uncapped_20pct_seed42.json`; five-fold real-only train/validation split over the remaining users.
- Model/objective: EfficientNet-V2-S at 384 px with random initialisation; pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled); LR `2e-4`, maximum 240 epochs, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`).
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU.
- Held-out result: five-fold unweighted `n=1` aggregate over 8,120 test images: MAE 6.490 years, RMSE 8.696 years, adult-gate AUC 0.8868, mean FPR 5.79%, Adult FNR 27.68%. The 5% FPR threshold grid was not attainable in folds 1, 3, and 4, so those folds use the lowest-FPR available threshold.
### Job `1050821` - Real + SyntheticDorsalHands2 Pure NLL, ViT small 384

- Submission date: 2026-07-30 10:57 (BST)
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`; fold 0 started successfully. This replaces the cancelled, never-started four-GPU submission `1050820`, which is omitted from the registry.
- Terminal scheduler state: FAILED, elapsed `02:25:52`, exit code 1. Fold 4 training completed, but its held-out test evaluation could not start because `/home/rb3434w/CNN-age-inference/splits/test_users_uncapped_20pct_seed42.json` was absent from the HPC checkout.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/vit_small384_handrgbd_prolific_synthetic2_trainval_nll_uncapped_test20_k5_3gpu_16cpu_20260730_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/vit-small-syn2-nll-1050821.log`
- Purpose: direct ViT-Small 384 architecture comparison to ViT-Tiny 384 (`1050814`) and EfficientNet-V2-S (`1050812`).
- Data/split: same as `1050814`: HandRGBD + ProlificHands + SyntheticDorsalHands2; no previous SyntheticDorsalHands or LUICID; uncapped real samples; fixed `splits/test_users_uncapped_20pct_seed42.json`; exact fold manifest copied from `1050814`.
- Model/objective: ViT-Small patch-16 at 384 px (`vit_small_384`); pure Gaussian NLL (`NLL=1`, all other regression, embedding, spread, and normals-auxiliary losses disabled); LR `2e-4`, maximum 240 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`).
- Requested resources: one `gpu-beast` node, 3 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU (global batch 48 rather than the four-GPU run's 64); `MASTER_PORT=29515`.
- Held-out result (folds 0-3): unweighted four-fold `n=1` aggregate: MAE 5.307 years, RMSE 7.102 years, adult-gate AUC 0.9397, mean FPR 4.63%, and Adult FNR 19.58% at each fold's best FPR <= 5% operating point.

### Job `1050832` — BYOL S-SSL pretraining on SyntheticDorsalHands2

- Submission date: 2026-07-30 15:07:12 (BST)
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`.
- Terminal scheduler state: COMPLETED, elapsed `01:38:57`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/byol/s_ssl_synthetic2_v2_s`
- Log file: `/home/rb3434w/CNN-age-inference/logs/byol-s-ssl-synth2-1050832.log`
- Purpose: self-supervised BYOL pretraining of an EfficientNet-V2-S backbone on SyntheticDorsalHands2 only, to produce an SSL initialisation for subsequent supervised fine-tuning on real data without any age labels during pretraining.
- Data/configuration: SyntheticDorsalHands2 only; no age labels used; BYOL online/target network training with standard augmentation pipeline. 100 epochs, per-GPU batch size 32, 4 GPUs (global batch 128).
- Model/objective: EfficientNet-V2-S at 384 px; BYOL loss only. Final BYOL loss at epoch 100: 0.0166.
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM.
- Checkpoint saved to: `runs/byol/s_ssl_synthetic2_v2_s/byol_v2_s_pretrain_ddp.pth`; converted to supervised-head init via `convert_ssl_checkpoint.py` into `runs/byol/s_ssl_synthetic2_v2_s/init_checkpoint_root/fold_*/v2_s_age_regressor_ddp.pth`.

### Job `1050851` — BYOL→supervised finetune, attempt 1 (cancelled)

- Submission date: 2026-07-30 16:51:40 (BST)
- Terminal scheduler state: CANCELLED, elapsed `00:02:50`.
- Log file: `/home/rb3434w/CNN-age-inference/logs/byol-s-ssl-finetune-1050851.log`
- Purpose: first attempt to launch supervised fine-tuning from the BYOL checkpoint. Cancelled before fold 0 completed.

### Job `1050852` — BYOL→supervised finetune, attempt 2 (cancelled)

- Submission date: 2026-07-30 16:55:29 (BST)
- Terminal scheduler state: CANCELLED, elapsed `00:25:29`. Fold 0 ran to epoch 29 before cancellation.
- Log file: `/home/rb3434w/CNN-age-inference/logs/byol-s-ssl-finetune-1050852.log`
- Purpose: second attempt; cancelled mid-fold 0 (epoch 29). Replaced by `1050854`.

### Job `1050854` — BYOL→supervised finetune, full five-fold run

- Submission date: 2026-07-30 17:21:42 (BST)
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`; fold 0 started successfully.
- Terminal scheduler state: COMPLETED, elapsed `03:10:47`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/byol_s_ssl_synthetic2_v2s_real_finetune`
- Log file: `/home/rb3434w/CNN-age-inference/logs/byol-s-ssl-finetune-1050854.log`
- Purpose: supervised fine-tuning of the BYOL-pretrained V2-S backbone on real data only, to test whether BYOL pretraining on synthetic data provides a useful initialisation for the age regression task compared with ImageNet init (`1050753`) and supervised synthetic init (`1050819`).
- Data/split: HandRGBD + ProlificHands only; no synthetic or LUICID samples; uncapped real data; fixed `splits/test_users_uncapped_20pct_seed42.json`; exact real-only fixed fold manifest copied from `splits/folds_k5_uncapped_test20_real_seed42.json`.
- Initialisation: BYOL-pretrained checkpoint converted by `convert_ssl_checkpoint.py`; fold `i` loads `runs/byol/s_ssl_synthetic2_v2_s/init_checkpoint_root/fold_i/v2_s_age_regressor_ddp.pth`.
- Model/objective: EfficientNet-V2-S at 384 px; pure Gaussian NLL (`NLL=1`, all other regression, embedding, spread, and normals-auxiliary losses disabled); LR `2e-4`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`).
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU.
- Held-out result: five-fold unweighted `n=1` aggregate: MAE 5.066 years, RMSE 6.825 years, adult-gate AUC 0.9460, mean FPR 4.69%, Adult FNR 19.69% at each fold's best FPR <= 5% operating point.

### Job `1050867` — Linear probe of BYOL SSL representations (cancelled, incomplete)

- Submission date: 2026-07-31 13:37:17 (BST)
- Terminal scheduler state: CANCELLED, elapsed `01:03:39`, exit code non-zero. Only fold 0 and the beginning of fold 1 ran (fold 1 log ends at epoch 13 without test evaluation).
- Run directory: `/home/rb3434w/CNN-age-inference/runs/linear_probe_byol_ssl_v2s`
- Log file: `/home/rb3434w/CNN-age-inference/logs/probe-byol-ssl-1050867.log`
- Purpose: linear probe of the BYOL-pretrained backbone to assess the quality of SSL representations without any fine-tuning; backbone frozen (`freeze_backbone=True`), only the two-unit regression head trained.
- Data/split: same as `1050854`: HandRGBD + ProlificHands only; uncapped real data; fixed `splits/test_users_uncapped_20pct_seed42.json`; real-only five-fold manifest.
- Initialisation: same BYOL checkpoint as `1050854`. Backbone frozen throughout.
- Model/objective: EfficientNet-V2-S at 384 px, frozen backbone; pure Gaussian NLL; LR `1e-3` (higher, head-only), maximum 60 epochs, patience 10, image-level training/evaluation.
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU.
- Partial result (fold 0 only): MAE 24.187 years, RMSE 32.006 years, adult-gate AUC 0.1779. The linear probe failed catastrophically — BYOL representations learned from SyntheticDorsalHands2 without age labels are not linearly separable by age when the backbone is fully frozen. This confirms that SSL pretraining alone does not encode age in a linearly accessible way; the backbone must be fine-tuned end-to-end.
