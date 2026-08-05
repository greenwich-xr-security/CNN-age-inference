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
| `1050941` | Completed / 00:01:03 | Synthetic-trained `1050939` fold 0 evaluated on fixed real 20% test | Evaluation-only fold-0 probe | Early SR/TSTR single-fold estimate |
| `1050942` | Completed / 00:04:33 | Synthetic-trained `1050939` checkpoints evaluated on fixed real 20% test | Evaluation-only; age-threshold adult-gate mode | Q2 SR/TSTR synthetic-to-real utility cell |
| `1050948` | Completed / 01:45:09 | Q3 R0, real-label fraction 5%, locked real test | Pure Gaussian NLL; ImageNet/default init | Q3 R0 label-efficiency sweep |
| `1050949` | Completed / 02:40:26 | Q3 R0, real-label fraction 10%, locked real test | Pure Gaussian NLL; ImageNet/default init | Q3 R0 label-efficiency sweep |
| `1050950` | Completed / 03:17:40 | Q3 R0, real-label fraction 25%, locked real test | Pure Gaussian NLL; ImageNet/default init | Q3 R0 label-efficiency sweep |
| `1050951` | Completed / 03:56:59 | Q3 R0, real-label fraction 50%, locked real test | Pure Gaussian NLL; ImageNet/default init | Q3 R0 label-efficiency sweep |
| `1050952` | Completed / 04:20:10 | Q3 R0, real-label fraction 100%, locked real test | Pure Gaussian NLL; ImageNet/default init | Q3 R0 label-efficiency sweep |
| `1050953` | Completed / 01:19:44 | Q3 R1, real-label fraction 5%, locked real test | Pure Gaussian NLL; random init | Q3 R1 label-efficiency sweep |
| `1050954` | Completed / 01:45:46 | Q3 R1, real-label fraction 10%, locked real test | Pure Gaussian NLL; random init | Q3 R1 label-efficiency sweep |
| `1050955` | Completed / 02:33:48 | Q3 R1, real-label fraction 25%, locked real test | Pure Gaussian NLL; random init | Q3 R1 label-efficiency sweep |
| `1050956` | Completed / 03:31:33 | Q3 R1, real-label fraction 50%, locked real test | Pure Gaussian NLL; random init | Q3 R1 label-efficiency sweep |
| `1050957` | Completed / 05:55:45 | Q3 R1, real-label fraction 100%, locked real test | Pure Gaussian NLL; random init | Q3 R1 label-efficiency sweep |
| `1050958` | Completed / 00:30:21 | Q3 S-age, real-label fraction 5%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939` | Q3 S-age label-efficiency sweep |
| `1050959` | Completed / 00:31:52 | Q3 S-age, real-label fraction 10%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939` | Q3 S-age label-efficiency sweep |
| `1050960` | Completed / 00:35:10 | Q3 S-age, real-label fraction 25%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939` | Q3 S-age label-efficiency sweep |
| `1050961` | Completed / 00:44:45 | Q3 S-age, real-label fraction 50%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939` | Q3 S-age label-efficiency sweep |
| `1050962` | Completed / 01:03:24 | Q3 S-age, real-label fraction 100%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939` | Q3 S-age label-efficiency sweep |
| `1050964` | Failed / 00:01:31 | Q3 S-ssl, real-label fraction 5%, locked real test | Pure Gaussian NLL; attempted BYOL init with mismatched `EMBED_DIM`; `gpu-standard`, 2 GPUs | Failed first S-ssl launch attempt |
| `1050965` | Failed / 00:01:26 | Q3 S-ssl, real-label fraction 10%, locked real test | Pure Gaussian NLL; attempted BYOL init with mismatched `EMBED_DIM`; `gpu-standard`, 2 GPUs | Failed first S-ssl launch attempt |
| `1050966` | Cancelled / 00:00:14 | Q3 S-ssl, real-label fraction 25%, locked real test | Pure Gaussian NLL; cancelled after bad S-ssl init was identified | Cancelled first S-ssl launch attempt |
| `1050967` | Cancelled / 00:00:00 | Q3 S-ssl, real-label fraction 50%, locked real test | Pure Gaussian NLL; cancelled before start after bad S-ssl init was identified | Cancelled first S-ssl launch attempt |
| `1050968` | Cancelled / 00:00:00 | Q3 S-ssl, real-label fraction 100%, locked real test | Pure Gaussian NLL; cancelled before start after bad S-ssl init was identified | Cancelled first S-ssl launch attempt |
| `1050969` | Completed / 02:10:31 | Q3 S-ssl, real-label fraction 5%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; `gpu-standard`, 2 GPUs | Q3 S-ssl label-efficiency sweep |
| `1050970` | Completed / 02:05:10 | Q3 S-ssl, real-label fraction 10%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; `gpu-standard`, 2 GPUs | Q3 S-ssl label-efficiency sweep |
| `1050971` | Completed / 02:15:22 | Q3 S-ssl, real-label fraction 25%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; `gpu-standard`, 2 GPUs | Q3 S-ssl label-efficiency sweep |
| `1050972` | Completed / 02:05:51 | Q3 S-ssl, real-label fraction 50%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; `gpu-standard`, 2 GPUs | Q3 S-ssl label-efficiency sweep |
| `1050973` | Completed / 01:57:45 | Q3 S-ssl, real-label fraction 100%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; `gpu-standard`, 2 GPUs | Q3 S-ssl label-efficiency sweep |
| `1050975` | Completed / 01:29:59 | Q3 S-age LR-control, real-label fraction 100%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939`; real fine-tune LR `2e-5`; max 120 epochs | Isolate fine-tuning LR vs `1050962` and compare recipe to `1050819` |
| `1050979` | Completed / 00:42:27 | Q3 S-age LR-control, real-label fraction 5%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939`; real fine-tune LR `2e-5`; max 120 epochs | Complete S-age LR-control label-efficiency sweep |
| `1050980` | Completed / 00:47:24 | Q3 S-age LR-control, real-label fraction 10%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939`; real fine-tune LR `2e-5`; max 120 epochs | Complete S-age LR-control label-efficiency sweep |
| `1050981` | Completed / 00:49:46 | Q3 S-age LR-control, real-label fraction 25%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939`; real fine-tune LR `2e-5`; max 120 epochs | Complete S-age LR-control label-efficiency sweep |
| `1050982` | Completed / 01:00:38 | Q3 S-age LR-control, real-label fraction 50%, locked real test | Pure Gaussian NLL; init from clean SyntheticDorsalHands2-only job `1050939`; real fine-tune LR `2e-5`; max 120 epochs | Complete S-age LR-control label-efficiency sweep |
| `1050983` | COMPLETED / 05:11:55 | Q3 S-ssl LR-control, real-label fraction 5%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; real fine-tune LR `2e-5`; max 120 epochs; `gpu-standard`, 2 GPUs | S-ssl learning-rate control label-efficiency sweep |
| `1050984` | COMPLETED / 04:05:45 | Q3 S-ssl LR-control, real-label fraction 10%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; real fine-tune LR `2e-5`; max 120 epochs; rerouted from `gpu-standard` to `gpu-beast`, 2 GPUs | S-ssl learning-rate control label-efficiency sweep |
| `1050985` | COMPLETED / 06:02:11 | Q3 S-ssl LR-control, real-label fraction 25%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; real fine-tune LR `2e-5`; max 120 epochs; rerouted from `gpu-standard` to `gpu-beast`, 2 GPUs | S-ssl learning-rate control label-efficiency sweep |
| `1050986` | COMPLETED / 08:44:19 | Q3 S-ssl LR-control, real-label fraction 50%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; real fine-tune LR `2e-5`; max 120 epochs; rerouted from `gpu-standard` to `gpu-beast`, 2 GPUs | S-ssl learning-rate control label-efficiency sweep |
| `1050987` | COMPLETED / 09:28:15 | Q3 S-ssl LR-control, real-label fraction 100%, locked real test | Pure Gaussian NLL; init from BYOL SyntheticDorsalHands2 job `1050832`; `EMBED_DIM=128`; real fine-tune LR `2e-5`; max 120 epochs; rerouted from `gpu-standard` to `gpu-beast`, 2 GPUs | S-ssl learning-rate control label-efficiency sweep |
| `1050996` | CANCELLED / 00:00:00 | Q3 S-shuffle pretraining on SyntheticDorsalHands2; fixed synthetic2 20% test | Initial 4-GPU `gpu-beast` request; cancelled before start and replaced by 2-GPU job `1051002` | Superseded S-shuffle pretraining launch |
| `1050997` | CANCELLED / 00:00:00 | Q3 S-shuffle LR-control, real-label fraction 5%, locked real test | Depended on cancelled `1050996`; replaced by `1051003` | Superseded S-shuffle launch |
| `1050998` | CANCELLED / 00:00:00 | Q3 S-shuffle LR-control, real-label fraction 10%, locked real test | Depended on cancelled `1050996`; replaced by `1051004` | Superseded S-shuffle launch |
| `1050999` | CANCELLED / 00:00:00 | Q3 S-shuffle LR-control, real-label fraction 25%, locked real test | Depended on cancelled `1050996`; replaced by `1051005` | Superseded S-shuffle launch |
| `1051000` | CANCELLED / 00:00:00 | Q3 S-shuffle LR-control, real-label fraction 50%, locked real test | Depended on cancelled `1050996`; replaced by `1051006` | Superseded S-shuffle launch |
| `1051001` | CANCELLED / 00:00:00 | Q3 S-shuffle LR-control, real-label fraction 100%, locked real test | Depended on cancelled `1050996`; replaced by `1051007` | Superseded S-shuffle launch |
| `1051002` | COMPLETED / 02:23:18 | Q3 S-shuffle pretraining on SyntheticDorsalHands2; fixed synthetic2 20% test | Pure Gaussian NLL; SyntheticDorsalHands2 age labels permuted once with seed 42; 2 GPUs; replacement for `1050996` | S-shuffle shuffled-label pretraining control |
| `1051003` | COMPLETED / 01:31:54 | Q3 S-shuffle LR-control, real-label fraction 5%, locked real test | Pure Gaussian NLL; init from `1051002`; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs | S-shuffle label-efficiency control |
| `1051004` | COMPLETED / 01:31:37 | Q3 S-shuffle LR-control, real-label fraction 10%, locked real test | Pure Gaussian NLL; init from `1051002`; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs | S-shuffle label-efficiency control |
| `1051005` | COMPLETED / 01:41:17 | Q3 S-shuffle LR-control, real-label fraction 25%, locked real test | Pure Gaussian NLL; init from `1051002`; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs | S-shuffle label-efficiency control |
| `1051006` | COMPLETED / 02:53:43 | Q3 S-shuffle LR-control, real-label fraction 50%, locked real test | Pure Gaussian NLL; init from `1051002`; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs | S-shuffle label-efficiency control |
| `1051007` | COMPLETED / 03:49:03 | Q3 S-shuffle LR-control, real-label fraction 100%, locked real test | Pure Gaussian NLL; init from `1051002`; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs | S-shuffle label-efficiency control |
| `1051077` | CANCELLED / 00:00:20 | Q3 U-ssl checkpoint conversion attempt | Launched without the intended pretraining dependency because PowerShell expanded the remote `sbatch` capture locally; cancelled and replaced by `1051084` | Superseded U-ssl launch attempt |
| `1051078` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 5%, locked real test | Depended on cancelled conversion attempt `1051077`; replaced by `1051085` | Superseded U-ssl launch attempt |
| `1051079` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 10%, locked real test | Depended on cancelled conversion attempt `1051077`; replaced by `1051086` | Superseded U-ssl launch attempt |
| `1051080` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 25%, locked real test | Depended on cancelled conversion attempt `1051077`; replaced by `1051087` | Superseded U-ssl launch attempt |
| `1051081` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 50%, locked real test | Depended on cancelled conversion attempt `1051077`; replaced by `1051088` | Superseded U-ssl launch attempt |
| `1051082` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 100%, locked real test | Depended on cancelled conversion attempt `1051077`; replaced by `1051089` | Superseded U-ssl launch attempt |
| `1051083` | CANCELLED / 00:35:13 | Q3 U-ssl BYOL pretraining on HaGRID stop_inverted + 11kHands + archive dorsal images | BYOL; EfficientNet-V2-S; 76 epochs; batch 32/GPU; 2 GPUs; excludes SyntheticDorsalHands2, HandRGBD, and ProlificHands; cancelled at user request | U-ssl generic SSL corpus/control pretraining |
| `1051084` | CANCELLED / 00:00:00 | Q3 U-ssl checkpoint conversion | Cancelled after U-ssl pretraining was stopped | U-ssl downstream preparation |
| `1051085` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 5%, locked real test | Cancelled after U-ssl pretraining was stopped | U-ssl label-efficiency control |
| `1051086` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 10%, locked real test | Cancelled after U-ssl pretraining was stopped | U-ssl label-efficiency control |
| `1051087` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 25%, locked real test | Cancelled after U-ssl pretraining was stopped | U-ssl label-efficiency control |
| `1051088` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 50%, locked real test | Cancelled after U-ssl pretraining was stopped | U-ssl label-efficiency control |
| `1051089` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 100%, locked real test | Cancelled after U-ssl pretraining was stopped | U-ssl label-efficiency control |
| `1051091` | CANCELLED / 00:59:51 | Q3 U-ssl BYOL pretraining on auto-downloaded STL10 unlabeled images | BYOL; EfficientNet-V2-S; 22 epochs; batch 32/GPU; 2 GPUs; cancelled to relaunch on 8-GPU beast node | Superseded U-ssl STL10 launch |
| `1051092` | CANCELLED / 00:00:00 | Q3 U-ssl checkpoint conversion | Cancelled after superseded 2-GPU STL10 pretraining was stopped | Superseded U-ssl downstream preparation |
| `1051093` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 5%, locked real test | Cancelled before start; replaced by `1051107` | Superseded U-ssl label-efficiency control |
| `1051094` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 10%, locked real test | Cancelled before start; replaced by `1051108` | Superseded U-ssl label-efficiency control |
| `1051095` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 25%, locked real test | Cancelled before start; replaced by `1051109` | Superseded U-ssl label-efficiency control |
| `1051096` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 50%, locked real test | Cancelled before start; replaced by `1051110` | Superseded U-ssl label-efficiency control |
| `1051097` | CANCELLED / 00:00:00 | Q3 U-ssl LR-control, real-label fraction 100%, locked real test | Cancelled before start; replaced by `1051111` | Superseded U-ssl label-efficiency control |
| `1051105` | COMPLETED / 01:09:57 | Q3 U-ssl BYOL pretraining on auto-downloaded STL10 unlabeled images | BYOL; EfficientNet-V2-S; 22 epochs; batch 8/GPU; 8 GPUs on `gm-hpc2-gpu801`; auto-downloaded STL10 to `.data/torchvision` | U-ssl generic SSL corpus/control pretraining |
| `1051106` | COMPLETED / 00:01:43 | Q3 U-ssl checkpoint conversion | Converted `1051105` BYOL checkpoint into fold-matched `EMBED_DIM=128` age-regressor init checkpoints; ran on `gm-hpc2-gpu001` | U-ssl downstream preparation |
| `1051107` | COMPLETED / 05:11:14 | Q3 U-ssl LR-control, real-label fraction 5%, locked real test | Pure Gaussian NLL; init from converted U-ssl BYOL checkpoint; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs on `gm-hpc2-gpu001` | U-ssl label-efficiency control |
| `1051108` | COMPLETED / 04:29:01 | Q3 U-ssl LR-control, real-label fraction 10%, locked real test | Pure Gaussian NLL; init from converted U-ssl BYOL checkpoint; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs on `gm-hpc2-gpu801` | U-ssl label-efficiency control |
| `1051109` | COMPLETED / 05:59:20 | Q3 U-ssl LR-control, real-label fraction 25%, locked real test | Pure Gaussian NLL; init from converted U-ssl BYOL checkpoint; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs on `gm-hpc2-gpu801` | U-ssl label-efficiency control |
| `1051110` | RUNNING on `gm-hpc2-gpu801` since 2026-08-05 14:53:38 | Q3 U-ssl LR-control, real-label fraction 50%, locked real test | Pure Gaussian NLL; init from converted U-ssl BYOL checkpoint; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs on `gm-hpc2-gpu801` | U-ssl label-efficiency control |
| `1051111` | RUNNING on `gm-hpc2-gpu801` since 2026-08-05 14:53:38 | Q3 U-ssl LR-control, real-label fraction 100%, locked real test | Pure Gaussian NLL; init from converted U-ssl BYOL checkpoint; real fine-tune LR `2e-5`; max 120 epochs; 2 GPUs on `gm-hpc2-gpu801` | U-ssl label-efficiency control |

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

## Active launches

### Jobs `1050948`-`1050962` - Q3 R0/R1/S-age label-fraction sweep

- Submission date: 2026-08-02 11:33 BST.
- Initial scheduler state from monitor at 2026-08-02 11:34:18 BST: `1050948` and `1050949` RUNNING; `1050950`-`1050962` PENDING.
- Progress update at 2026-08-02 13:59:47 BST: `1050948` COMPLETED; `1050949` RUNNING with 4/5 folds complete; `1050950` RUNNING with 0/5 folds complete; `1050951`-`1050962` PENDING.
- Progress update at 2026-08-02 14:18:40 BST: `1050948` and `1050949` COMPLETED; `1050950` RUNNING with 1/5 folds complete; `1050951` RUNNING with 0/5 folds complete; `1050952`-`1050962` PENDING.
- Progress update at 2026-08-02 16:50:54 BST: `1050950` COMPLETED; `1050951` RUNNING with 3/5 folds complete; `1050952` RUNNING with 0/5 folds complete; `1050953`-`1050962` PENDING.
- Progress update at 2026-08-02 18:15:51 BST: `1050951` COMPLETED; `1050952` RUNNING with 1/5 folds complete; `1050953` RUNNING with 0/5 folds complete; `1050954`-`1050962` PENDING.
- Progress update at 2026-08-02 21:23:31 BST: `1050952`, `1050953`, and `1050954` COMPLETED; `1050955` RUNNING with 0/5 folds complete; `1050956` RUNNING with 0/5 folds complete; `1050957`-`1050962` PENDING.
- Progress update at 2026-08-03 00:04:59 BST: `1050955` COMPLETED; `1050956` RUNNING with 4/5 folds complete; `1050957` RUNNING with 0/5 folds complete; `1050958`-`1050962` PENDING.
- Progress update at 2026-08-03 05:26:09 BST: `1050956`, `1050957`, and `1050958`-`1050962` all COMPLETED. The launched R0/R1/S-age sweep is complete.
- Monitor cron: every 10 minutes via `/home/rb3434w/CNN-age-inference/runs/q3_r0_r1_sage_seed42_monitor/run_monitor.sh`.
- Monitor outputs: `/home/rb3434w/CNN-age-inference/runs/q3_r0_r1_sage_seed42_monitor/q3_status.md` and `q3_status.json`.
- Jobs manifest: `/home/rb3434w/CNN-age-inference/runs/q3_r0_r1_sage_seed42_monitor/jobs.json`.
- Purpose: launch the first Q3 downstream label-efficiency sweep for the runnable arms R0, R1, and S-age over real-label fractions 5%, 10%, 25%, 50%, and 100%.
- Data/split: HandRGBD + ProlificHands only for downstream fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, or synthetic samples in downstream train/validation/test. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; validation users come from `splits/folds_k5_uncapped_test20_real_seed42.json`; test users come from `splits/test_users_uncapped_20pct_seed42.json`.
- Pretraining/initialisation: R0 uses default ImageNet/torchvision EfficientNet-V2-S initialisation; R1 uses `--no-imagenet-pretrained` random initialisation; S-age loads fold-matched checkpoints from clean SyntheticDorsalHands2-only supervised NLL job `1050939` at `/home/rb3434w/CNN-age-inference/runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384/fold_*/v2_s_age_regressor_ddp.pth`.
- Model/objective: EfficientNet-V2-S at 384 px; pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled), LR `2e-4`, maximum 240 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources per job: one `gpu-beast` node, 4 GPUs, 16 CPU cores, batch size 16 per GPU; unique `MASTER_PORT` values `29600`-`29614`.
- Held-out result so far: `1050948` completed the R0 5% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 29.935 years, RMSE 33.493 years, adult-gate AUC 0.0000, mean FPR 0.00%, Adult FNR 100.00%. This is a degenerate low-label baseline: at the selected operating point (`tau=10`) all folds have zero false positives but all adults are missed.
- Held-out result so far: `1050949` completed the R0 10% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 26.846 years, RMSE 30.901 years, adult-gate AUC 0.0252, mean FPR 0.96%, Adult FNR 99.83%. Fold 0 reached FPR 4.78% at `tau=16`; folds 1-4 selected `tau=10` with zero FPR and 100% Adult FNR.
- Held-out result so far: `1050950` completed the R0 25% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 19.787 years, RMSE 24.996 years, adult-gate AUC 0.3062, mean FPR 3.28%, Adult FNR 95.19%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `20.6`, `20.1`, `10.0`, `18.7`, and `18.2`.
- Held-out result so far: `1050951` completed the R0 50% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 10.442 years, RMSE 14.358 years, adult-gate AUC 0.7921, mean FPR 4.51%, Adult FNR 44.59%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `25.2`, `23.5`, `22.5`, `19.2`, and `23.3`.
- Held-out result so far: `1050952` completed the R0 100% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 4.903 years, RMSE 6.546 years, adult-gate AUC 0.9577, mean FPR 4.74%, Adult FNR 17.37%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `24.9`, `25.3`, `24.4`, `23.9`, and `24.8`.
- Held-out result so far: `1050953` completed the R1 5% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 30.177 years, RMSE 37.391 years, adult-gate AUC 0.0173, mean FPR 5.69%, Adult FNR 97.23%. The 5% FPR operating point was attainable in folds 0, 1, and 3; folds 2 and 4 use the lowest-FPR available threshold from the 10-30 year age-threshold sweep.
- Held-out result so far: `1050954` completed the R1 10% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 28.189 years, RMSE 34.244 years, adult-gate AUC 0.0001, mean FPR 3.96%, Adult FNR 98.40%. The 5% FPR operating point was attainable in folds 0, 1, and 3; folds 2 and 4 use the lowest-FPR available threshold from the 10-30 year age-threshold sweep.
- Held-out result so far: `1050955` completed the R1 25% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 27.308 years, RMSE 38.927 years, adult-gate AUC 0.0169, mean FPR 4.28%, Adult FNR 99.34%. The 5% FPR operating point was attainable in folds 0, 1, 3, and 4; fold 2 uses the lowest-FPR available threshold from the 10-30 year age-threshold sweep.
- Held-out result so far: `1050956` completed the R1 50% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 23.084 years, RMSE 31.710 years, adult-gate AUC 0.1546, mean FPR 5.01%, Adult FNR 97.94%. The 5% FPR operating point was attainable in folds 1, 2, and 4; folds 0 and 3 use the lowest-FPR available threshold from the 10-30 year age-threshold sweep.
- Held-out result so far: `1050957` completed the R1 100% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 11.684 years, RMSE 18.725 years, adult-gate AUC 0.7174, mean FPR 6.33%, Adult FNR 42.80%. The 5% FPR operating point was attainable only in fold 0; folds 1-4 use the lowest-FPR available threshold from the 10-30 year age-threshold sweep.
- Held-out result so far: `1050958` completed the S-age 5% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 6.745 years, RMSE 8.823 years, adult-gate AUC 0.8878, mean FPR 4.69%, Adult FNR 26.48%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `23.4`, `27.6`, `27.9`, `29.7`, and `21.8`.
- Held-out result so far: `1050959` completed the S-age 10% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.841 years, RMSE 8.015 years, adult-gate AUC 0.8988, mean FPR 5.60%, Adult FNR 24.74%. The 5% FPR operating point was attainable in folds 0, 1, 3, and 4; fold 2 uses the lowest-FPR available threshold from the 10-30 year age-threshold sweep.
- Held-out result so far: `1050960` completed the S-age 25% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.367 years, RMSE 7.208 years, adult-gate AUC 0.9214, mean FPR 4.78%, Adult FNR 23.58%. All folds attained an operating point with FPR <= 5%; all folds selected age threshold `22.9`, `27.3`, `26.5`, `26.4`, and `26.5`.
- Held-out result so far: `1050961` completed the S-age 50% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.091 years, RMSE 6.898 years, adult-gate AUC 0.9317, mean FPR 4.69%, Adult FNR 21.97%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `22.6`, `25.0`, `26.3`, `26.6`, and `27.8`.
- Held-out result so far: `1050962` completed the S-age 100% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.024 years, RMSE 6.811 years, adult-gate AUC 0.9399, mean FPR 4.74%, Adult FNR 20.76%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `25.8`, `22.9`, `26.6`, `25.2`, and `25.7`.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1050948` | R0 | 5% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r0_realfrac_f05_seed42_v2s_384` |
| `1050949` | R0 | 10% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r0_realfrac_f10_seed42_v2s_384` |
| `1050950` | R0 | 25% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r0_realfrac_f25_seed42_v2s_384` |
| `1050951` | R0 | 50% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r0_realfrac_f50_seed42_v2s_384` |
| `1050952` | R0 | 100% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r0_realfrac_f100_seed42_v2s_384` |
| `1050953` | R1 | 5% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r1_realfrac_f05_seed42_v2s_384` |
| `1050954` | R1 | 10% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r1_realfrac_f10_seed42_v2s_384` |
| `1050955` | R1 | 25% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r1_realfrac_f25_seed42_v2s_384` |
| `1050956` | R1 | 50% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r1_realfrac_f50_seed42_v2s_384` |
| `1050957` | R1 | 100% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_r1_realfrac_f100_seed42_v2s_384` |
| `1050958` | S-age | 5% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f05_seed42_v2s_384` |
| `1050959` | S-age | 10% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f10_seed42_v2s_384` |
| `1050960` | S-age | 25% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f25_seed42_v2s_384` |
| `1050961` | S-age | 50% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f50_seed42_v2s_384` |
| `1050962` | S-age | 100% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f100_seed42_v2s_384` |

### Jobs `1050969`-`1050973` - Q3 S-ssl label-fraction sweep on gpu-standard

- Submission date: 2026-08-02 15:24 BST.
- Initial scheduler state from `sacct` at 2026-08-02 15:26 BST: `1050969` RUNNING on `gpu-standard`; `1050970`-`1050973` PENDING.
- Progress update at 2026-08-02 18:15:51 BST: `1050969` COMPLETED; `1050970` RUNNING with 1/5 folds complete; `1050971`-`1050973` PENDING.
- Progress update at 2026-08-02 21:23:31 BST: `1050970` COMPLETED; `1050971` RUNNING with 3/5 folds complete; `1050972`-`1050973` PENDING.
- Progress update at 2026-08-03 00:04:59 BST: `1050971` and `1050972` COMPLETED; `1050973` RUNNING with 0/5 folds complete.
- Progress update at 2026-08-03 01:59:37 BST: `1050973` COMPLETED. The launched S-ssl downstream sweep is complete.
- Purpose: launch the Q3 S-ssl downstream label-efficiency sweep over real-label fractions 5%, 10%, 25%, 50%, and 100%.
- Data/split: HandRGBD + ProlificHands only for downstream fine-tuning and locked real testing. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; validation users come from `splits/folds_k5_uncapped_test20_real_seed42.json`; test users come from `splits/test_users_uncapped_20pct_seed42.json`.
- Pretraining/initialisation: fold-matched checkpoints converted from BYOL SyntheticDorsalHands2 S-ssl pretraining job `1050832`, loaded from `/home/rb3434w/CNN-age-inference/runs/byol/s_ssl_synthetic2_v2_s/init_checkpoint_root_embed128/fold_*/v2_s_age_regressor_ddp.pth`. The SSL converter was updated to produce checkpoints with `EMBED_DIM=128`, matching the Q3 model shape.
- Model/objective: EfficientNet-V2-S at 384 px; pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled), LR `2e-4`, maximum 240 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources per job: one `gpu-standard` node, 2 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU; unique `MASTER_PORT` values `29710`-`29714`. Note: this keeps the same per-GPU batch size as the Q3 `gpu-beast` runs but uses a smaller global batch because the standard node has 2 GPUs.
- Superseded launch attempt: `1050964` and `1050965` failed because the first BYOL init root had a 2-output classifier head while Q3 builds a `2 + 128` output head; `1050966`-`1050968` were cancelled after the mismatch was identified.
- Held-out result so far: `1050969` completed the S-ssl 5% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 27.840 years, RMSE 31.494 years, adult-gate AUC 0.0987, mean FPR 1.00%, Adult FNR 96.60%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `10.0`, `10.0`, `10.0`, `12.0`, and `10.0`.
- Held-out result so far: `1050970` completed the S-ssl 10% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 22.684 years, RMSE 26.889 years, adult-gate AUC 0.3474, mean FPR 2.49%, Adult FNR 88.19%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `18.2`, `10.0`, `10.0`, `14.4`, and `19.7`.
- Held-out result so far: `1050971` completed the S-ssl 25% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 8.185 years, RMSE 10.639 years, adult-gate AUC 0.9117, mean FPR 4.98%, Adult FNR 27.28%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `28.6`, `28.0`, `24.8`, `23.0`, and `29.8`.
- Held-out result so far: `1050972` completed the S-ssl 50% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.814 years, RMSE 7.659 years, adult-gate AUC 0.9259, mean FPR 4.91%, Adult FNR 22.56%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `26.8`, `25.8`, `26.6`, `25.6`, and `27.8`.
- Held-out result so far: `1050973` completed the S-ssl 100% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.444 years, RMSE 7.067 years, adult-gate AUC 0.9252, mean FPR 4.98%, Adult FNR 22.23%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `28.8`, `24.5`, `28.2`, `26.5`, and `28.4`.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1050969` | S-ssl | 5% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f05_seed42_v2s_384_embed128_2gpu_standard` |
| `1050970` | S-ssl | 10% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f10_seed42_v2s_384_embed128_2gpu_standard` |
| `1050971` | S-ssl | 25% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f25_seed42_v2s_384_embed128_2gpu_standard` |
| `1050972` | S-ssl | 50% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f50_seed42_v2s_384_embed128_2gpu_standard` |
| `1050973` | S-ssl | 100% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f100_seed42_v2s_384_embed128_2gpu_standard` |

### Job `1050975` - Q3 S-age 100% LR-control rerun

- Submission date: 2026-08-03 09:42:04 BST.
- Initial scheduler state: RUNNING on `gpu-beast`, elapsed `00:00:28`, exit code `0:0`.
- Terminal scheduler state: COMPLETED, elapsed `01:29:59`, exit code `0:0`.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f100_lr2e5_seed42_v2s_384`
- Log file: `/home/rb3434w/CNN-age-inference/logs/q3-sage-f100-lr2e5-1050975.log`
- Purpose: rerun the clean Q3 S-age 100% cell with the real fine-tuning learning rate used by historical job `1050819`, so the effect of `LR=2e-5` can be separated from the pretraining-source difference.
- Data/split: HandRGBD + ProlificHands only for downstream fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, or synthetic samples in downstream train/validation/test. Real train subset is 100% from `splits/q3_real_label_fractions_seed42.json`; validation users come from `splits/folds_k5_uncapped_test20_real_seed42.json`; test users come from `splits/test_users_uncapped_20pct_seed42.json`.
- Pretraining/initialisation: fold-matched clean SyntheticDorsalHands2-only supervised NLL checkpoints from job `1050939`, loaded from `/home/rb3434w/CNN-age-inference/runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384/fold_*/v2_s_age_regressor_ddp.pth`.
- Model/objective: EfficientNet-V2-S at 384 px; pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled), LR `2e-5`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources: one `gpu-beast` node, 4 GPUs, 16 CPU cores, batch size 16 per GPU; `MASTER_PORT=29630`.
- Held-out result: five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 4.817 years, RMSE 6.571 years, adult-gate AUC 0.9526, mean FPR 4.69%, Adult FNR 18.46%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `22.7`, `20.9`, `23.0`, `22.7`, and `25.0`.

### Jobs `1050979`-`1050982` - Q3 S-age LR-control remaining label fractions

- Submission date: 2026-08-03 11:25 BST.
- Initial scheduler state from `sacct` at 2026-08-03 11:26 BST: `1050979` RUNNING on `gpu-beast`, elapsed `00:00:23`; `1050980` RUNNING on `gpu-beast`, elapsed `00:00:22`; `1050981` and `1050982` PENDING.
- Progress update at 2026-08-03 12:36 BST: `1050979` COMPLETED, elapsed `00:42:27`; `1050980` COMPLETED, elapsed `00:47:24`; `1050981` RUNNING with 2/5 folds complete; `1050982` RUNNING with 2/5 folds complete.
- Progress update at 2026-08-03 13:15 BST: `1050981` COMPLETED, elapsed `00:49:46`; `1050982` COMPLETED, elapsed `01:00:38`. The S-age LR-control remaining-fractions sweep is complete.
- Purpose: run the remaining Q3 S-age LR-control fractions so the clean SyntheticDorsalHands2 supervised-pretraining arm has a full `LR=2e-5` downstream label-efficiency curve.
- Data/split: HandRGBD + ProlificHands only for downstream fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, or synthetic samples in downstream train/validation/test. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; validation users come from `splits/folds_k5_uncapped_test20_real_seed42.json`; test users come from `splits/test_users_uncapped_20pct_seed42.json`.
- Pretraining/initialisation: fold-matched clean SyntheticDorsalHands2-only supervised NLL checkpoints from job `1050939`, loaded from `/home/rb3434w/CNN-age-inference/runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384/fold_*/v2_s_age_regressor_ddp.pth`.
- Model/objective: EfficientNet-V2-S at 384 px; pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled), LR `2e-5`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources per job: one `gpu-beast` node, 4 GPUs, 16 CPU cores, batch size 16 per GPU; unique `MASTER_PORT` values `29631`-`29634`.
- Held-out result so far: `1050979` completed the S-age LR-control 5% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 6.152 years, RMSE 8.399 years, adult-gate AUC 0.9133, mean FPR 4.69%, Adult FNR 24.17%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `26.1`, `27.2`, `25.3`, `23.9`, and `21.3`.
- Held-out result so far: `1050980` completed the S-age LR-control 10% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.547 years, RMSE 7.591 years, adult-gate AUC 0.9223, mean FPR 4.78%, Adult FNR 23.95%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `21.1`, `22.2`, `27.5`, `20.3`, and `26.9`.
- Held-out result so far: `1050981` completed the S-age LR-control 25% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.357 years, RMSE 7.190 years, adult-gate AUC 0.9265, mean FPR 4.74%, Adult FNR 23.59%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `23.0`, `26.0`, `23.6`, `26.0`, and `27.5`.
- Held-out result so far: `1050982` completed the S-age LR-control 50% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.114 years, RMSE 6.943 years, adult-gate AUC 0.9363, mean FPR 4.78%, Adult FNR 21.91%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `23.4`, `23.8`, `26.8`, `23.9`, and `25.4`.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1050979` | S-age LR-control | 5% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f05_lr2e5_seed42_v2s_384` |
| `1050980` | S-age LR-control | 10% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f10_lr2e5_seed42_v2s_384` |
| `1050981` | S-age LR-control | 25% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f25_lr2e5_seed42_v2s_384` |
| `1050982` | S-age LR-control | 50% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sage_realfrac_f50_lr2e5_seed42_v2s_384` |

### Jobs `1050983`-`1050987` - Q3 S-ssl LR-control label-fraction sweep

- Submission date: 2026-08-03 11:29 BST.
- Initial scheduler state from `sacct` at 2026-08-03 11:30 BST: `1050983` RUNNING on `gpu-standard`, elapsed `00:00:22`; `1050984`-`1050987` PENDING.
- Reroute update at 2026-08-03 13:21 BST: pending jobs `1050984`-`1050987` were moved in place from `gpu-standard` to `gpu-beast` with their original job IDs and 2-GPU resource requests. `1050984`, `1050985`, and `1050986` started on `gpu-beast` at 13:21 BST; `1050987` remained PENDING for resources. `1050983` remains RUNNING on `gpu-standard`.
- Completion/update at 2026-08-03 17:08 BST: `1050983` completed successfully after `05:11:55`; `1050984`, `1050985`, `1050986`, and `1050987` are RUNNING on `gpu-beast`.
- Completion/update at 2026-08-03 18:30 BST: `1050984` completed successfully after `04:05:45`; `1050985`, `1050986`, and `1050987` are RUNNING on `gpu-beast`.
- Completion/update at 2026-08-03 19:23 BST: `1050985` completed successfully after `06:02:11`; `1050986` and `1050987` remain RUNNING on `gpu-beast`.
- Completion/update at 2026-08-03 22:05 BST: `1050986` completed successfully after `08:44:19`; `1050987` remains RUNNING on `gpu-beast`.
- Terminal update at 2026-08-04 02:24 BST: `1050987` completed successfully after `09:28:15`; the S-ssl LR-control sweep is complete.
- Purpose: run a downstream learning-rate control for the Q3 S-ssl arm, using the same `LR=2e-5` and maximum 120 epochs as the S-age LR-control sweep while keeping the BYOL synthetic-hand initialisation fixed.
- Data/split: HandRGBD + ProlificHands only for downstream fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, or synthetic samples in downstream train/validation/test. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; validation users come from `splits/folds_k5_uncapped_test20_real_seed42.json`; test users come from `splits/test_users_uncapped_20pct_seed42.json`.
- Pretraining/initialisation: fold-matched checkpoints converted from BYOL SyntheticDorsalHands2 S-ssl pretraining job `1050832`, loaded from `/home/rb3434w/CNN-age-inference/runs/byol/s_ssl_synthetic2_v2_s/init_checkpoint_root_embed128/fold_*/v2_s_age_regressor_ddp.pth`.
- Model/objective: EfficientNet-V2-S at 384 px with `EMBED_DIM=128`; pure Gaussian NLL (`NLL=1`, CRPS/MSE/MAE/spread/embed losses disabled), LR `2e-5`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources per job: one node, 2 GPUs, 16 CPU cores, batch size 16 per GPU; `1050983` remains on `gpu-standard`, while `1050984`-`1050987` were rerouted to `gpu-beast`; unique `MASTER_PORT` values `29720`-`29724`.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1050983` | S-ssl LR-control | 5% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f05_lr2e5_seed42_v2s_384_embed128_2gpu_standard` |
| `1050984` | S-ssl LR-control | 10% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f10_lr2e5_seed42_v2s_384_embed128_2gpu_standard` |
| `1050985` | S-ssl LR-control | 25% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f25_lr2e5_seed42_v2s_384_embed128_2gpu_standard` |
| `1050986` | S-ssl LR-control | 50% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f50_lr2e5_seed42_v2s_384_embed128_2gpu_standard` |
| `1050987` | S-ssl LR-control | 100% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sssl_realfrac_f100_lr2e5_seed42_v2s_384_embed128_2gpu_standard` |

- Held-out result so far: `1050983` completed the S-ssl LR-control 5% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 30.248 years, RMSE 33.794 years, adult-gate AUC 0.0000, mean FPR 0.00%, Adult FNR 100.00%. All folds selected threshold `30.0`, yielding zero false positives but missing all adults.
- Held-out result so far: `1050984` completed the S-ssl LR-control 10% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 29.589 years, RMSE 33.160 years, adult-gate AUC 0.0000, mean FPR 0.00%, Adult FNR 100.00%. All folds selected threshold `30.0`, yielding zero false positives but missing all adults.
- Held-out result so far: `1050985` completed the S-ssl LR-control 25% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 25.792 years, RMSE 29.625 years, adult-gate AUC 0.0000, mean FPR 0.00%, Adult FNR 100.00%. All folds selected threshold `30.0`, yielding zero false positives but missing all adults.
- Held-out result so far: `1050986` completed the S-ssl LR-control 50% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 12.377 years, RMSE 16.525 years, adult-gate AUC 0.8409, mean FPR 0.00%, Adult FNR 100.00%. All folds selected threshold `30.0`, yielding zero false positives but missing all adults.
- Held-out result: `1050987` completed the S-ssl LR-control 100% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.149 years, RMSE 6.960 years, adult-gate AUC 0.9510, mean FPR 4.69%, Adult FNR 16.69%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `25.3`, `21.7`, `22.4`, `23.1`, and `21.9`.

### Jobs `1050996`-`1051007` - Q3 S-shuffle pretraining and LR-control label-fraction sweep

- Submission date: 2026-08-03 18:48 BST.
- Initial scheduler state from `squeue`/`sacct` at 2026-08-03 18:48 BST: `1050996` PENDING on `gpu-beast` for resources; downstream jobs `1050997`-`1051001` PENDING with dependency on successful completion of `1050996`.
- Reroute update at 2026-08-03 19:22 BST: cancelled pending jobs `1050996`-`1051001` before start because `1050996` requested 4 GPUs on `gpu-beast`. Replacement pretraining job `1051002` requests 2 GPUs on `gpu-standard,gpu-beast` and started on `gpu-standard`; replacement downstream jobs `1051003`-`1051007` request 2 GPUs on `gpu-standard,gpu-beast` and depend on `afterok:1051002`.
- Completion/update at 2026-08-03 21:46 BST: `1051002` completed successfully after `02:23:18`; downstream jobs `1051003`, `1051004`, and `1051005` started, while `1051006` and `1051007` are pending for resources.
- Completion/update at 2026-08-03 23:27 BST: `1051003`, `1051004`, and `1051005` completed successfully; `1051006` and `1051007` are RUNNING on `gpu-beast`.
- Terminal update at 2026-08-04 03:07 BST: `1051006` and `1051007` completed successfully after `02:53:43` and `03:49:03`; the S-shuffle LR-control sweep is complete.
- Purpose: run the Q3 S-shuffle control, matching the clean S-age LR-control recipe while breaking the synthetic age-label signal. This tests whether any S-age benefit is due to labelled synthetic age signal rather than synthetic dorsal-hand image exposure/schedule.
- Pretraining job `1051002`: SyntheticDorsalHands2 only; fixed `splits/test_users_synthetic2_20pct_seed42.json`; five-fold split `splits/folds_k5_synthetic2_seed42.json`; EfficientNet-V2-S at 384 px, ImageNet/default init, pure Gaussian NLL, LR `2e-4`, max 240 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), `EMBED_DIM=128`; SyntheticDorsalHands2 ages shuffled once with seed 42 by `--shuffle-synthetic-dorsal2-age-labels`.
- Pretraining resources: one `gpu-standard` or `gpu-beast` node, 2 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU; `MASTER_PORT=29526`.
- Downstream jobs `1051003`-`1051007`: HandRGBD + ProlificHands only for real fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, or synthetic samples in downstream train/validation/test. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; test users from `splits/test_users_uncapped_20pct_seed42.json`.
- Downstream model/objective: fold-matched init from `/home/rb3434w/CNN-age-inference/runs/q3_s_shuffle_synthetic2_only_nll_k5_2gpu_16cpu_20260803_384/fold_*/v2_s_age_regressor_ddp.pth`, pure Gaussian NLL, LR `2e-5`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Downstream resources per job: one `gpu-standard` or `gpu-beast` node, 2 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU; unique `MASTER_PORT` values `29527`-`29531`.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1050996` | S-shuffle pretraining | n/a | CANCELLED before start; replaced by `1051002` | `/home/rb3434w/CNN-age-inference/runs/q3_s_shuffle_synthetic2_only_nll_k5_4gpu_16cpu_20260803_384` |
| `1050997` | S-shuffle LR-control | 5% | CANCELLED before start; replaced by `1051003` | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f05_lr2e5_seed42_v2s_384` |
| `1050998` | S-shuffle LR-control | 10% | CANCELLED before start; replaced by `1051004` | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f10_lr2e5_seed42_v2s_384` |
| `1050999` | S-shuffle LR-control | 25% | CANCELLED before start; replaced by `1051005` | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f25_lr2e5_seed42_v2s_384` |
| `1051000` | S-shuffle LR-control | 50% | CANCELLED before start; replaced by `1051006` | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f50_lr2e5_seed42_v2s_384` |
| `1051001` | S-shuffle LR-control | 100% | CANCELLED before start; replaced by `1051007` | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f100_lr2e5_seed42_v2s_384` |
| `1051002` | S-shuffle pretraining | n/a | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_s_shuffle_synthetic2_only_nll_k5_2gpu_16cpu_20260803_384` |
| `1051003` | S-shuffle LR-control | 5% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f05_lr2e5_seed42_v2s_384` |
| `1051004` | S-shuffle LR-control | 10% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f10_lr2e5_seed42_v2s_384` |
| `1051005` | S-shuffle LR-control | 25% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f25_lr2e5_seed42_v2s_384` |
| `1051006` | S-shuffle LR-control | 50% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f50_lr2e5_seed42_v2s_384` |
| `1051007` | S-shuffle LR-control | 100% | COMPLETED | `/home/rb3434w/CNN-age-inference/runs/q3_sshuffle_realfrac_f100_lr2e5_seed42_v2s_384` |

- Held-out result: `1051002` completed the S-shuffle shuffled-label SyntheticDorsalHands2 pretraining control. Five-fold unweighted `n=1` aggregate over 4,423 SyntheticDorsalHands2 held-out samples per fold: MAE 16.158 years, RMSE 18.758 years, adult-gate AUC 0.0000, mean FPR 100.00%, Adult FNR 0.00%. The selected threshold was `tau=10.0` in all folds; this is the expected degenerate behaviour for shuffled synthetic age labels and confirms the age-label signal was destroyed.
- Held-out result so far: `1051003` completed the S-shuffle LR-control 5% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 11.614 years, RMSE 13.844 years, adult-gate AUC 0.3867, mean FPR 0.00%, Adult FNR 100.00%. All folds selected threshold `30.0`, yielding zero false positives but missing all adults.
- Held-out result so far: `1051004` completed the S-shuffle LR-control 10% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 7.370 years, RMSE 9.716 years, adult-gate AUC 0.8353, mean FPR 0.00%, Adult FNR 100.00%. All folds selected threshold `30.0`, yielding zero false positives but missing all adults.
- Held-out result so far: `1051005` completed the S-shuffle LR-control 25% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 6.531 years, RMSE 8.668 years, adult-gate AUC 0.8790, mean FPR 0.00%, Adult FNR 100.00%. All folds selected threshold `30.0`, yielding zero false positives but missing all adults.
- Held-out result: `1051006` completed the S-shuffle LR-control 50% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.661 years, RMSE 7.525 years, adult-gate AUC 0.9189, mean FPR 4.83%, Adult FNR 23.26%. The 5% FPR operating point was attainable in folds 0-3; fold 4 used the lowest-FPR available threshold from the 10-30 year age-threshold sweep.
- Held-out result: `1051007` completed the S-shuffle LR-control 100% cell. Five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 5.367 years, RMSE 7.227 years, adult-gate AUC 0.9317, mean FPR 4.74%, Adult FNR 20.79%. All folds attained an operating point with FPR <= 5%; folds selected age thresholds `27.0`, `26.0`, `22.6`, `24.5`, and `27.7`.

### Jobs `1051083`-`1051089` - Q3 U-ssl pretraining and LR-control label-fraction sweep

- Submission date: 2026-08-05 11:58 BST.
- Initial scheduler state from `sacct` at 2026-08-05 11:58 BST: `1051083` RUNNING on `gpu-standard`, node `gm-hpc2-gpu001`, elapsed `00:00:03`; `1051084`-`1051089` PENDING on dependencies.
- Terminal update at 2026-08-05 12:34 BST: user requested U-ssl stop. `1051083` was cancelled after `00:35:13`; dependent jobs `1051084`-`1051089` were then cancelled before start.
- Superseded launch attempt: `1051077`-`1051082` were cancelled immediately after a local PowerShell expansion stripped the intended pretraining job dependency from the remote submission. They are not counted as experimental cells.
- Purpose: run the Q3 U-ssl control, matching the S-age/S-ssl LR-control downstream recipe while replacing SyntheticDorsalHands2 BYOL pretraining with an available no-age corpus that excludes both SyntheticDorsalHands2 and the locked real HandRGBD + ProlificHands evaluation sources.
- U-ssl pretraining data/configuration: HaGRID stop_inverted + 11kHands primary + archive dorsal images only, filtered through `filter_metadata_ssl`; 14,564 SSL images total (`6,931` HaGRID, `5,654` 11kHands primary, `1,979` archive). No SyntheticDorsalHands2, HandRGBD, ProlificHands, or LUICID samples are used for U-ssl pretraining.
- U-ssl pretraining model/objective: BYOL with EfficientNet-V2-S at 384 px, no age labels, 76 epochs, batch size 32 per GPU, LR `1e-4`, target momentum `0.996 -> 1.0`, no local crops, augmentation ramp fraction `0.2`, seed 42. The 76-epoch, 2-GPU schedule approximately matches the update count of the 100-epoch, 4-GPU S-ssl BYOL job `1050832` after accounting for the smaller U-ssl corpus.
- Pretraining resources: one `gpu-standard` or `gpu-beast` node, 2 GPUs, 16 CPU cores, 64 GB RAM; `MASTER_PORT=29532`.
- Pretraining run directory: `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_hagrid_primary_archive_v2_s`
- Pretraining log file: `/home/rb3434w/CNN-age-inference/logs/q3-ussl-pretrain-1051083.log`
- Conversion job `1051084`: depends on `afterok:1051083`; converts `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_hagrid_primary_archive_v2_s/byol_v2_s_pretrain_ddp.pth` into fold-matched `EMBED_DIM=128` init checkpoints under `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_hagrid_primary_archive_v2_s/init_checkpoint_root_embed128`.
- Downstream jobs `1051085`-`1051089`: depend on `afterok:1051084`; HandRGBD + ProlificHands only for real fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, or synthetic samples in downstream train/validation/test. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; test users from `splits/test_users_uncapped_20pct_seed42.json`.
- Downstream model/objective: fold-matched init from converted U-ssl BYOL checkpoints, pure Gaussian NLL, LR `2e-5`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Downstream resources per job: one `gpu-standard` or `gpu-beast` node, 2 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU; unique `MASTER_PORT` values `29533`-`29537`.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1051083` | U-ssl pretraining | n/a | CANCELLED after `00:35:13` | `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_hagrid_primary_archive_v2_s` |
| `1051084` | U-ssl checkpoint conversion | n/a | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_hagrid_primary_archive_v2_s/init_checkpoint_root_embed128` |
| `1051085` | U-ssl LR-control | 5% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f05_lr2e5_seed42_v2s_384` |
| `1051086` | U-ssl LR-control | 10% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f10_lr2e5_seed42_v2s_384` |
| `1051087` | U-ssl LR-control | 25% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f25_lr2e5_seed42_v2s_384` |
| `1051088` | U-ssl LR-control | 50% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f50_lr2e5_seed42_v2s_384` |
| `1051089` | U-ssl LR-control | 100% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f100_lr2e5_seed42_v2s_384` |

### Jobs `1051091`-`1051097` - Q3 U-ssl STL10 pretraining and LR-control label-fraction sweep

- Submission date: 2026-08-05 12:41 BST.
- Initial scheduler state from `sacct`/`squeue` at 2026-08-05 12:41 BST: `1051091` RUNNING on `gpu-standard`, node `gm-hpc2-gpu001`, elapsed `00:00:31`; `1051092`-`1051097` PENDING on dependencies.
- Terminal update at 2026-08-05 13:40 BST: cancelled `1051091` after `00:59:51` and cancelled dependent jobs `1051092`-`1051097` before start so U-ssl could relaunch with 8-GPU pretraining and a node-targeted downstream layout.
- Purpose: run the Q3 U-ssl control using an auto-downloaded generic natural-image corpus, matching the S-age/S-ssl LR-control downstream recipe while excluding SyntheticDorsalHands2 and the locked real HandRGBD + ProlificHands evaluation sources.
- U-ssl pretraining data/configuration: torchvision STL10 unlabeled split, 100,000 generic natural images, auto-downloaded to `/home/rb3434w/CNN-age-inference/.data/torchvision`. No project hand datasets or age labels are used for U-ssl pretraining.
- U-ssl pretraining model/objective: BYOL with EfficientNet-V2-S at 384 px, 22 epochs, batch size 32 per GPU, LR `1e-4`, target momentum `0.996 -> 1.0`, no local crops, augmentation ramp fraction `0.2`, seed 42. The 22-epoch, 2-GPU schedule approximately matches the update count of the 100-epoch, 4-GPU S-ssl BYOL job `1050832` after accounting for the larger STL10 corpus.
- Pretraining resources: one `gpu-standard` or `gpu-beast` node, 2 GPUs, 16 CPU cores, 64 GB RAM; `MASTER_PORT=29532`.
- Pretraining run directory: `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s`
- Pretraining log file: `/home/rb3434w/CNN-age-inference/logs/q3-ussl-pretrain-1051091.log`
- Initial log state: STL10 download from `http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz` started successfully.
- Conversion job `1051092`: depends on `afterok:1051091`; converts `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s/byol_v2_s_pretrain_ddp.pth` into fold-matched `EMBED_DIM=128` init checkpoints under `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s/init_checkpoint_root_embed128`.
- Downstream jobs `1051093`-`1051097`: depend on `afterok:1051092`; HandRGBD + ProlificHands only for real fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, STL10, or synthetic samples in downstream train/validation/test. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; test users from `splits/test_users_uncapped_20pct_seed42.json`.
- Downstream model/objective: fold-matched init from converted U-ssl BYOL checkpoints, pure Gaussian NLL, LR `2e-5`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Downstream resources per job: one `gpu-standard` or `gpu-beast` node, 2 GPUs, 16 CPU cores, 64 GB RAM, batch size 16 per GPU; unique `MASTER_PORT` values `29533`-`29537`.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1051091` | U-ssl STL10 pretraining | n/a | CANCELLED after `00:59:51` | `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s` |
| `1051092` | U-ssl checkpoint conversion | n/a | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s/init_checkpoint_root_embed128` |
| `1051093` | U-ssl LR-control | 5% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f05_lr2e5_seed42_v2s_384` |
| `1051094` | U-ssl LR-control | 10% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f10_lr2e5_seed42_v2s_384` |
| `1051095` | U-ssl LR-control | 25% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f25_lr2e5_seed42_v2s_384` |
| `1051096` | U-ssl LR-control | 50% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f50_lr2e5_seed42_v2s_384` |
| `1051097` | U-ssl LR-control | 100% | CANCELLED before start | `/home/rb3434w/CNN-age-inference/runs/q3_ussl_realfrac_f100_lr2e5_seed42_v2s_384` |

### Jobs `1051105`-`1051111` - Q3 U-ssl STL10 8-GPU pretraining and node-targeted LR-control sweep

- Submission date: 2026-08-05 13:41 BST.
- Initial scheduler state from `sacct`/`squeue` at 2026-08-05 13:42 BST: `1051105` RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`, elapsed `00:00:16`; `1051106`-`1051111` PENDING on dependencies.
- Progress update at 2026-08-05 15:04 BST: `1051105` COMPLETED after `01:09:57` on `gm-hpc2-gpu801`; `1051106` COMPLETED after `00:01:43` on `gm-hpc2-gpu001`; downstream jobs `1051107`-`1051111` all RUNNING since 2026-08-05 14:53:38. Job `1051107` is running on `gm-hpc2-gpu001`; jobs `1051108`-`1051111` are running on `gm-hpc2-gpu801`.
- Progress update at 2026-08-05 19:31 BST: `1051108` COMPLETED after `04:29:01`; `1051107`, `1051109`, `1051110`, and `1051111` remain RUNNING.
- Progress update at 2026-08-05 20:36 BST: `1051107` COMPLETED after `05:11:14`; `1051109`, `1051110`, and `1051111` remain RUNNING.
- Progress update at 2026-08-05 21:13 BST: `1051109` COMPLETED after `05:59:20`; `1051110` and `1051111` remain RUNNING.
- Purpose: replacement U-ssl control launch that maximizes HPC GPU use while keeping the same global BYOL batch size as the 2-GPU attempt.
- U-ssl pretraining data/configuration: torchvision STL10 unlabeled split, 100,000 generic natural images, auto-downloaded to `/home/rb3434w/CNN-age-inference/.data/torchvision`. No project hand datasets or age labels are used for U-ssl pretraining.
- U-ssl pretraining model/objective: BYOL with EfficientNet-V2-S at 384 px, 22 epochs, batch size 8 per GPU on 8 GPUs (global batch 64), LR `1e-4`, target momentum `0.996 -> 1.0`, no local crops, augmentation ramp fraction `0.2`, seed 42.
- Pretraining resources: one `gpu-beast` node, fixed node `gm-hpc2-gpu801`, 8 GPUs, 32 CPU cores, 128 GB RAM; `MASTER_PORT=29538`.
- Pretraining run directory: `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s_8gpu`
- Pretraining log file: `/home/rb3434w/CNN-age-inference/logs/q3-ussl-pretrain-1051105.log`
- Initial log state: loaded the already downloaded STL10 data and started `torchrun` with 8 local ranks.
- Pretraining/conversion artifact update: BYOL checkpoint `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s_8gpu/byol_v2_s_pretrain_ddp.pth` was produced and the fold-matched init root `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s_8gpu/init_checkpoint_root_embed128` exists.
- Held-out result so far: `1051107` completed the U-ssl 5% cell. Five-fold unweighted `n=1` aggregate over the locked real held-out split: MAE 30.097 years, RMSE 33.651 years, adult-gate AUC 0.0000, mean FPR 0.00%, Adult FNR 100.00%.
- Held-out result so far: `1051108` completed the U-ssl 10% cell. Five-fold unweighted `n=1` aggregate over the locked real held-out split: MAE 29.278 years, RMSE 32.865 years, adult-gate AUC 0.0000, mean FPR 0.00%, Adult FNR 100.00%.
- Held-out result so far: `1051109` completed the U-ssl 25% cell. Five-fold unweighted `n=1` aggregate over the locked real held-out split: MAE 25.733 years, RMSE 29.544 years, adult-gate AUC 0.0000, mean FPR 0.00%, Adult FNR 100.00%.
- Held-out result status: final U-ssl sweep results are not available yet because `1051110` and `1051111` are still running.
- Conversion job `1051106`: depends on `afterok:1051105`; targets `gpu-standard`, node `gm-hpc2-gpu001`, 1 GPU; converts `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s_8gpu/byol_v2_s_pretrain_ddp.pth` into fold-matched `EMBED_DIM=128` init checkpoints under `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s_8gpu/init_checkpoint_root_embed128`.
- Downstream jobs `1051107`-`1051111`: depend on `afterok:1051106`; each requests 2 GPUs. Job `1051107` targets `gpu-standard` node `gm-hpc2-gpu001`; jobs `1051108`-`1051111` target `gpu-beast` node `gm-hpc2-gpu801`, allowing the downstream sweep to use 10 GPUs total if both nodes are free.
- Downstream data: HandRGBD + ProlificHands only for real fine-tuning and locked real testing; no LUICID, HaGRID, 11kHands, archive, STL10, or synthetic samples in downstream train/validation/test. Real train subsets come from `splits/q3_real_label_fractions_seed42.json`; test users from `splits/test_users_uncapped_20pct_seed42.json`.
- Downstream model/objective: fold-matched init from converted U-ssl BYOL checkpoints, pure Gaussian NLL, LR `2e-5`, maximum 120 epochs, patience 10, image-level training/evaluation (`USER_GROUP_SIZES=1`, `AGG_SIZES=1`), age-threshold adult-gate evaluation.

| Job | Arm | Real-label fraction | Initial state | Run directory |
| --- | --- | ---: | --- | --- |
| `1051105` | U-ssl STL10 pretraining | n/a | COMPLETED after `01:09:57` on `gpu-beast` / `gm-hpc2-gpu801` | `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s_8gpu` |
| `1051106` | U-ssl checkpoint conversion | n/a | COMPLETED after `00:01:43` on `gpu-standard` / `gm-hpc2-gpu001` | `/home/rb3434w/CNN-age-inference/runs/byol/u_ssl_stl10_unlabeled_v2_s_8gpu/init_checkpoint_root_embed128` |
| `1051107` | U-ssl LR-control | 5% | COMPLETED after `05:11:14` on `gm-hpc2-gpu001` | `/home/rb3434w/CNN-age-inference/runs/q3_ussl8_realfrac_f05_lr2e5_seed42_v2s_384` |
| `1051108` | U-ssl LR-control | 10% | COMPLETED after `04:29:01` on `gm-hpc2-gpu801` | `/home/rb3434w/CNN-age-inference/runs/q3_ussl8_realfrac_f10_lr2e5_seed42_v2s_384` |
| `1051109` | U-ssl LR-control | 25% | COMPLETED after `05:59:20` on `gm-hpc2-gpu801` | `/home/rb3434w/CNN-age-inference/runs/q3_ussl8_realfrac_f25_lr2e5_seed42_v2s_384` |
| `1051110` | U-ssl LR-control | 50% | RUNNING on `gm-hpc2-gpu801` | `/home/rb3434w/CNN-age-inference/runs/q3_ussl8_realfrac_f50_lr2e5_seed42_v2s_384` |
| `1051111` | U-ssl LR-control | 100% | RUNNING on `gm-hpc2-gpu801` | `/home/rb3434w/CNN-age-inference/runs/q3_ussl8_realfrac_f100_lr2e5_seed42_v2s_384` |

## Completed and failed launches

### Job `1050942` - Q2 SR/TSTR SyntheticDorsalHands2-trained to real evaluation

- Submission date: 2026-08-01 21:34:12 BST
- Initial scheduler state: RUNNING on `gpu-beast`, node `gm-hpc2-gpu801`, elapsed `00:00:26`; fold 0 loaded successfully on CUDA and found the fixed real held-out split.
- Terminal scheduler state: COMPLETED, elapsed `00:04:33`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/q2_sr_ss1050939_to_real_eval_20260801`
- Log file: `/home/rb3434w/CNN-age-inference/logs/q2-sr-real-eval-1050942.log`
- Purpose: run the Q2 SR/TSTR generator-validation cell: evaluate the SyntheticDorsalHands2-trained SS checkpoints on the locked real held-out split as the direct synthetic-to-real utility measure.
- Data/split: checkpoints from SyntheticDorsalHands2-only SS job `1050939` (`runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384/fold_*/v2_s_age_regressor_ddp.pth`) evaluated on HandRGBD + ProlificHands only, fixed `splits/test_users_uncapped_20pct_seed42.json`; no synthetic or LUICID test samples.
- Model/objective: evaluation-only; EfficientNet-V2-S at 384 px with the fold-specific `1050939` checkpoint, embedding head dimension 128 to match training, image-level evaluation (`AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources: one `gpu-beast` node, 1 GPU, 4 CPU cores, 32 GB RAM, inference batch size 64.
- Held-out result: five-fold unweighted `n=1` aggregate over 1,624 real held-out samples per fold: MAE 7.165 years, RMSE 9.097 years, adult-gate AUC 0.8718, mean FPR 6.42%, Adult FNR 28.73%. The 5% FPR operating point was attainable in folds 0, 2, and 3; folds 1 and 4 use the lowest-FPR available threshold from the 10-30 year age-threshold sweep.

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

### Job `1050941` - Q2 SR/TSTR fold-0 probe

- Submission date: 2026-08-01 20:15:23 BST
- Terminal scheduler state: COMPLETED, elapsed `00:01:03`, exit code 0.
- Run directory: `/home/rb3434w/CNN-age-inference/runs/q2_sr_ss1050939_fold0_to_real_eval_20260801`
- Log file: `/home/rb3434w/CNN-age-inference/logs/q2-sr-fold0-real-eval-1050941.log`
- Purpose: early single-fold SR/TSTR estimate while the full SS job was still running.
- Data/split: fold 0 checkpoint from SyntheticDorsalHands2-only SS job `1050939` evaluated on HandRGBD + ProlificHands only, fixed `splits/test_users_uncapped_20pct_seed42.json`.
- Model/objective: evaluation-only; EfficientNet-V2-S at 384 px, embedding head dimension 128, image-level evaluation (`AGG_SIZES=1`), age-threshold adult-gate evaluation.
- Requested resources: one `gpu-beast` node, 1 GPU, 4 CPU cores, 32 GB RAM, inference batch size 64.
- Held-out result: single-fold `n=1` estimate over 1,624 real held-out samples: MAE 6.572 years, RMSE 8.336 years, adult-gate AUC 0.8990, FPR 4.78%, Adult FNR 28.69% at the best FPR <= 5% operating point (`tau=28.3`).

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
