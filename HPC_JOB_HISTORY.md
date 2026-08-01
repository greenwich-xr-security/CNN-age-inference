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
| `1050821` | Failed / 02:25:52 | Real + SyntheticDorsalHands2; uncapped fixed 20% test | Pure Gaussian NLL; ViT small 384; `MASTER_PORT=29515`; 3 GPUs | Scale the ViT comparison from `1050814` |
| `1050877` | Completed / 07:44:30 | Real only; uncapped fixed 20% test | Pure Gaussian NLL; random initialisation | Random-init control for real-only EfficientNet-V2-S |

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
| Real only | Pure NLL, random initialisation (`1050877`) | 6.490 | 8.696 | 0.8868 | 5.79% | 27.68% |

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

## Completed launches

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

## Failed launches

### Job `1050821` - Real + SyntheticDorsalHands2 Pure NLL, ViT small 384

- Submission date: 2026-07-30 10:57 (BST)
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`; fold 0 started successfully. This replaces the cancelled, never-started four-GPU submission `1050820`, which is omitted from the registry.
- Terminal scheduler state: FAILED, elapsed `02:25:52`, exit code 1.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/vit_small384_handrgbd_prolific_synthetic2_trainval_nll_uncapped_test20_k5_3gpu_16cpu_20260730_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/vit-small-syn2-nll-1050821.log`
- Purpose: direct ViT-Small 384 architecture comparison to ViT-Tiny 384 (`1050814`) and EfficientNet-V2-S (`1050812`).
- Data/split: same as `1050814`: HandRGBD + ProlificHands + SyntheticDorsalHands2; no previous SyntheticDorsalHands or LUICID; uncapped real samples; fixed `splits/test_users_uncapped_20pct_seed42.json`; exact fold manifest copied from `1050814`.
- Model/objective: ViT-Small patch-16 at 384 px (`vit_small_384`); pure Gaussian NLL (`NLL=1`, all other regression, embedding, spread, and normals-auxiliary losses disabled); LR `2e-4`, maximum 240 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`).
- Requested resources: one `gpu-beast` node, 3 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU (global batch 48 rather than the four-GPU run's 64); `MASTER_PORT=29515`.
- Failure: after training reached held-out evaluation, the job errored with `FileNotFoundError` for `/home/rb3434w/CNN-age-inference/splits/test_users_uncapped_20pct_seed42.json`; no held-out aggregate is recorded for this failed run.
