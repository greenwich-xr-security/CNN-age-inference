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
| `1050749` | Cancelled / 00:22:53 | Superseded uncapped real-only plan | Pure NLL | Replaced before training by fixed 20%-test split |
| `1050750` | Cancelled / 00:22:53 | Superseded uncapped real + synthetic plan | Pure NLL | Replaced before training by fixed 20%-test split |
| `1050753` | Completed / 5:18:12 | Real only; uncapped fixed 20% test | Pure Gaussian NLL | Effect of removing both real-data caps |
| `1050754` | Completed / 3:51:07 | Real + synthetic; uncapped fixed 20% test | Pure Gaussian NLL | Synthetic comparison without real-data caps |
| `1050812` | Running / initial 00:00:29 | Real + SyntheticDorsalHands2 only; uncapped fixed 20% test | Pure Gaussian NLL | Synthetic2-only comparison to `1050754` |

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

The test users and images differ from the table above, so compare only the two
rows in this table with each other, not with the capped 15% test results.

| Training data | Objective | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Real only | Pure NLL (`1050753`) | 5.017 | 6.706 | 0.9483 | 4.74% | 18.33% |
| Real + synthetic | Pure NLL (`1050754`) | **4.683** | **6.293** | **0.9594** | 4.78% | **17.91%** |

## Fixed uncapped split manifests

The uncapped runs use 20% held-out real users and five folds over the remaining
80%, giving 64% train / 16% validation / 20% test users per fold.

| File | Used by |
| --- | --- |
| `splits/test_users_uncapped_20pct_seed42.json` | Both uncapped runs; 128 fixed held-out real users |
| `splits/folds_k5_uncapped_test20_real_seed42.json` | `1050753` real-only run |
| `splits/folds_k5_uncapped_test20_real_synthetic_seed42.json` | `1050754` real + synthetic run |

## In-progress launches

### Job `1050812` — Real + synthetic2 Pure NLL

- Submission date: 2026-07-29 21:27:49
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`, elapsed `00:00:29`
- Run directory: `/home/rb3434w/CNN-age-inference/runs/v2s_handrgbd_prolific_synthetic2_trainval_nll_uncapped_test20_k5_4gpu_16cpu_20260729_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/v2s-hrgbd-pro-syn2-nll-uncap20-4g16c-1050812.log`
- Purpose: compare SyntheticDorsalHands2 against the directly comparable SyntheticDorsalHands job `1050754`
- Data/configuration: HandRGBD + ProlificHands real data, SyntheticDorsalHands2 enabled with `INCLUDE_SYNTHETIC_DORSAL2=1`, previous SyntheticDorsalHands disabled with `INCLUDE_SYNTHETIC_DORSAL=0`, LUICID disabled, uncapped real samples, fixed `splits/test_users_uncapped_20pct_seed42.json`, five folds generated with the same seed/protocol as `1050754` after replacing the synthetic source
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, 64 GB RAM project baseline, batch size 16 per GPU
- Model/objective: EfficientNet-V2-S at 384 px, same active loss composition and weights as `1050754`: pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled; `NORMALS_AUX=0` so the printed normals default is inactive), image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`)
