# Q3 Generator-Validation Job Matrix

Scope: Real means HandRGBD + ProlificHands. Synthetic means
SyntheticDorsalHands2. The locked real test split is always
`splits/test_users_uncapped_20pct_seed42.json`.

## Design Clarification

Q3 is a `6 arms x 5 real-label fractions` downstream fine-tuning grid.

The label fractions are:

`5%, 10%, 25%, 50%, 100%`

These fractions apply only to the labelled real fine-tuning data. They do not
subsample the synthetic pretraining data. Synthetic pretraining arms use the
full accepted SyntheticDorsalHands2 corpus, fixed across fractions and arms.

If each Slurm submission runs all five folds for one arm/fraction pair, Q3
requires 30 downstream fine-tuning jobs. If each fold is submitted separately,
the same grid is 150 fold-level jobs.

## Current Status

Checked with `sacct`/`squeue` on 2026-08-06 08:25 BST. U-ssl pretraining job
`1051105`, checkpoint conversion job `1051106`, and all five U-ssl downstream
jobs `1051107`-`1051111` are complete.

Completed: R0, R1, S-age, S-ssl, S-shuffle, S-age LR-control, S-ssl
LR-control, S-shuffle LR-control, S-shuffle pretraining, U-ssl pretraining,
and U-ssl LR-control.

## Downstream Grid

| Arm | Pretraining before real fine-tune | Real fine-tune fractions | Synthetic data fractioned? | Role |
| --- | --- | --- | --- | --- |
| R0 | Standard init, no synthetic | 5%, 10%, 25%, 50%, 100% | No synthetic used | Main real-only baseline; equivalent-real-labels reference curve |
| R1 | Random init, no pretraining | 5%, 10%, 25%, 50%, 100% | No synthetic used | From-scratch baseline; isolates ImageNet/default init value |
| S-age | Supervised age pretraining on SyntheticDorsalHands2 | 5%, 10%, 25%, 50%, 100% | No; full synthetic corpus for pretraining | Candidate benefit arm |
| S-shuffle | Same synthetic images/schedule as S-age, but synthetic ages permuted once per seed | 5%, 10%, 25%, 50%, 100% | No; full synthetic corpus with shuffled labels | Removes synthetic age-label signal |
| S-ssl | Self-supervised pretraining on SyntheticDorsalHands2 | 5%, 10%, 25%, 50%, 100% | No; full synthetic corpus for SSL | Removes supervision, keeps dorsal-hand corpus |
| U-ssl | Same SSL method/update count on auto-downloaded STL10 unlabeled natural images | 5%, 10%, 25%, 50%, 100% | No SyntheticDorsalHands2 used | Removes dorsal-hand content; generic compute/corpus control |

## Interpretation Comparisons

| Comparison | What It Tests |
| --- | --- |
| S-age vs R0 | Does synthetic supervised age pretraining help over standard init? |
| S-age vs R1 | Is the gain just any pretraining/default init effect? |
| S-age vs S-ssl | Is age supervision on synthetic important, or just hand-image representation learning? |
| S-age vs S-shuffle | Is the synthetic age label signal real, or just image exposure/schedule? |
| S-ssl vs U-ssl | Is the benefit from dorsal-hand content, or generic SSL compute? |

## Existing Assets And Jobs

| Arm / asset | Current status | Existing job(s) | Notes |
| --- | --- | --- | --- |
| Real label-fraction manifest | Available | n/a | `splits/q3_real_label_fractions_seed42.json`; fractions `0.05`, `0.10`, `0.25`, `0.50`, `1.00` |
| R0 downstream sweep | Completed | `1050948`-`1050952`; reference `1050753` | Full Q3 ImageNet/default-init label-efficiency sweep is complete; `1050753` remains the earlier full-real reference |
| R1 downstream sweep | Completed | `1050953`-`1050957`; reference `1050877` | Full Q3 random-init label-efficiency sweep is complete; `1050877` remains the earlier full-real random-init reference |
| S-age pretraining | Completed | `1050939` | Clean SyntheticDorsalHands2-only supervised NLL checkpoints used as the Q3 S-age pretraining source |
| S-age downstream sweep | Completed | `1050958`-`1050962`; LR-control `1050975`, `1050979`-`1050982` | Clean Q3 S-age cells initialise from `1050939`; `1050819` remains a related historical run, not the final clean Q3 cell |
| S-shuffle pretraining | Completed | `1051002` | Shuffled-label SyntheticDorsalHands2 pretraining on 2 GPUs; replacement for cancelled 4-GPU queue `1050996`; downstream jobs `1051003`-`1051007` released |
| S-shuffle downstream sweep | Completed | `1051003`-`1051007` | Full Q3 shuffled synthetic-age LR-control sweep is complete |
| S-ssl pretraining | Completed | `1050832` | BYOL on SyntheticDorsalHands2 |
| S-ssl downstream sweep | Completed | `1050969`-`1050973`; LR-control `1050983`-`1050987` | Full Q3 BYOL synthetic-hand sweep and LR-control sweep are complete |
| U-ssl pretraining | Completed | `1051105` | BYOL on auto-downloaded STL10 unlabeled split; 100,000 generic natural images; 22 epochs on 8 GPUs with batch 8/GPU to keep global batch 64 |
| U-ssl checkpoint conversion | Completed | `1051106` | Converted the `1051105` BYOL checkpoint into fold-matched `EMBED_DIM=128` downstream init checkpoints |

## Launched Jobs

These downstream jobs use the same real-label fraction manifest,
`splits/q3_real_label_fractions_seed42.json`, and the same locked real test
split, `splits/test_users_uncapped_20pct_seed42.json`. One Slurm job runs all
five folds for one arm/fraction pair.

Monitor on HPC:
`/home/rb3434w/CNN-age-inference/runs/q3_r0_r1_sage_seed42_monitor/q3_status.md`

All jobs in the table below are completed.

| Arm | 5% | 10% | 25% | 50% | 100% |
| --- | --- | --- | --- | --- | --- |
| R0 | `1050948` | `1050949` | `1050950` | `1050951` | `1050952` |
| R1 | `1050953` | `1050954` | `1050955` | `1050956` | `1050957` |
| S-age | `1050958` | `1050959` | `1050960` | `1050961` | `1050962` |
| S-shuffle | `1051003` | `1051004` | `1051005` | `1051006` | `1051007` |
| S-ssl | `1050969` | `1050970` | `1050971` | `1050972` | `1050973` |
| U-ssl | `1051107` completed | `1051108` completed | `1051109` completed | `1051110` completed | `1051111` completed |

## LR-Control Reruns

| Arm / comparison | Real-label fraction | Job | State | Purpose |
| --- | ---: | --- | --- | --- |
| S-age, clean synthetic-only init, LR `2e-5` | 5% | `1050979` | Completed | Match `1050819` fine-tuning LR/max epochs while keeping Q3 clean `1050939` synthetic-only init |
| S-age, clean synthetic-only init, LR `2e-5` | 10% | `1050980` | Completed | Match `1050819` fine-tuning LR/max epochs while keeping Q3 clean `1050939` synthetic-only init |
| S-age, clean synthetic-only init, LR `2e-5` | 25% | `1050981` | Completed | Match `1050819` fine-tuning LR/max epochs while keeping Q3 clean `1050939` synthetic-only init |
| S-age, clean synthetic-only init, LR `2e-5` | 50% | `1050982` | Completed | Match `1050819` fine-tuning LR/max epochs while keeping Q3 clean `1050939` synthetic-only init |
| S-age, clean synthetic-only init, LR `2e-5` | 100% | `1050975` | Completed | Match `1050819` fine-tuning LR/max epochs while keeping Q3 clean `1050939` synthetic-only init |
| S-ssl, BYOL synthetic-hand init, LR `2e-5` | 5% | `1050983` | Completed | Match S-age LR-control downstream LR/max epochs while keeping Q3 S-ssl BYOL init |
| S-ssl, BYOL synthetic-hand init, LR `2e-5` | 10% | `1050984` | Completed | Match S-age LR-control downstream LR/max epochs while keeping Q3 S-ssl BYOL init |
| S-ssl, BYOL synthetic-hand init, LR `2e-5` | 25% | `1050985` | Completed | Match S-age LR-control downstream LR/max epochs while keeping Q3 S-ssl BYOL init |
| S-ssl, BYOL synthetic-hand init, LR `2e-5` | 50% | `1050986` | Completed | Match S-age LR-control downstream LR/max epochs while keeping Q3 S-ssl BYOL init |
| S-ssl, BYOL synthetic-hand init, LR `2e-5` | 100% | `1050987` | Completed | Match S-age LR-control downstream LR/max epochs while keeping Q3 S-ssl BYOL init |
| S-shuffle, shuffled synthetic-age init, LR `2e-5` | pretraining | `1051002` | Completed | SyntheticDorsalHands2-only pure NLL with age labels permuted once; 2-GPU replacement for `1050996` |
| S-shuffle, shuffled synthetic-age init, LR `2e-5` | 5% | `1051003` | Completed | Match S-age LR-control downstream recipe; tests label signal vs image exposure/schedule |
| S-shuffle, shuffled synthetic-age init, LR `2e-5` | 10% | `1051004` | Completed | Match S-age LR-control downstream recipe; tests label signal vs image exposure/schedule |
| S-shuffle, shuffled synthetic-age init, LR `2e-5` | 25% | `1051005` | Completed | Match S-age LR-control downstream recipe; tests label signal vs image exposure/schedule |
| S-shuffle, shuffled synthetic-age init, LR `2e-5` | 50% | `1051006` | Completed | Match S-age LR-control downstream recipe; tests label signal vs image exposure/schedule |
| S-shuffle, shuffled synthetic-age init, LR `2e-5` | 100% | `1051007` | Completed | Match S-age LR-control downstream recipe; tests label signal vs image exposure/schedule |
| U-ssl, BYOL init from STL10 unlabeled, LR `2e-5` | pretraining | `1051105` | Completed after `01:09:57` on `gpu-beast` / `gm-hpc2-gpu801` | BYOL on auto-downloaded STL10 unlabeled split; 22 epochs on 8 GPUs |
| U-ssl, checkpoint conversion to embed-dim 128 init root | conversion | `1051106` | Completed after `00:01:43` on `gpu-standard` / `gm-hpc2-gpu001` | Converts `1051105` BYOL checkpoint for fold-matched downstream initialisation |
| U-ssl, BYOL init from STL10 unlabeled, LR `2e-5` | 5% | `1051107` | Completed after `05:11:14` on `gm-hpc2-gpu001` | 2-GPU downstream job targeted to `gm-hpc2-gpu001` |
| U-ssl, BYOL init from STL10 unlabeled, LR `2e-5` | 10% | `1051108` | Completed after `04:29:01` on `gm-hpc2-gpu801` | 2-GPU downstream job targeted to `gm-hpc2-gpu801` |
| U-ssl, BYOL init from STL10 unlabeled, LR `2e-5` | 25% | `1051109` | Completed after `05:59:20` on `gm-hpc2-gpu801` | 2-GPU downstream job targeted to `gm-hpc2-gpu801` |
| U-ssl, BYOL init from STL10 unlabeled, LR `2e-5` | 50% | `1051110` | Completed after `09:13:56` on `gm-hpc2-gpu801` | 2-GPU downstream job targeted to `gm-hpc2-gpu801` |
| U-ssl, BYOL init from STL10 unlabeled, LR `2e-5` | 100% | `1051111` | Completed after `09:28:55` on `gm-hpc2-gpu801` | 2-GPU downstream job targeted to `gm-hpc2-gpu801` |

## Results

All values are unweighted means of five held-out fold results at image-level
aggregation (`n=1`) on the locked real test split. FPR and Adult FNR use each
fold's best operating point with FPR <= 5% when attainable, then average the
achieved values.

| Arm | Real-label fraction | Job | State | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 5% | `1050948` | Completed | 29.935 | 33.493 | 0.0000 | 0.00% | 100.00% |
| R0 | 10% | `1050949` | Completed | 26.846 | 30.901 | 0.0252 | 0.96% | 99.83% |
| R0 | 25% | `1050950` | Completed | 19.787 | 24.996 | 0.3062 | 3.28% | 95.19% |
| R0 | 50% | `1050951` | Completed | 10.442 | 14.358 | 0.7921 | 4.51% | 44.59% |
| R0 | 100% | `1050952` | Completed | 4.903 | 6.546 | 0.9577 | 4.74% | 17.37% |
| R1 | 5% | `1050953` | Completed | 30.177 | 37.391 | 0.0173 | 5.69% | 97.23% |
| R1 | 10% | `1050954` | Completed | 28.189 | 34.244 | 0.0001 | 3.96% | 98.40% |
| R1 | 25% | `1050955` | Completed | 27.308 | 38.927 | 0.0169 | 4.28% | 99.34% |
| R1 | 50% | `1050956` | Completed | 23.084 | 31.710 | 0.1546 | 5.01% | 97.94% |
| R1 | 100% | `1050957` | Completed | 11.684 | 18.725 | 0.7174 | 6.33% | 42.80% |
| S-age LR-control | 5% | `1050979` | Completed | 6.152 | 8.399 | 0.9133 | 4.69% | 24.17% |
| S-age LR-control | 10% | `1050980` | Completed | 5.547 | 7.591 | 0.9223 | 4.78% | 23.95% |
| S-age LR-control | 25% | `1050981` | Completed | 5.357 | 7.190 | 0.9265 | 4.74% | 23.59% |
| S-age LR-control | 50% | `1050982` | Completed | 5.114 | 6.943 | 0.9363 | 4.78% | 21.91% |
| S-age LR-control | 100% | `1050975` | Completed | 4.817 | 6.571 | 0.9526 | 4.69% | 18.46% |
| S-ssl LR-control | 5% | `1050983` | Completed | 30.248 | 33.794 | 0.0000 | 0.00% | 100.00% |
| S-ssl LR-control | 10% | `1050984` | Completed | 29.589 | 33.160 | 0.0000 | 0.00% | 100.00% |
| S-ssl LR-control | 25% | `1050985` | Completed | 25.792 | 29.625 | 0.0000 | 0.00% | 100.00% |
| S-ssl LR-control | 50% | `1050986` | Completed | 12.377 | 16.525 | 0.8409 | 0.00% | 100.00% |
| S-ssl LR-control | 100% | `1050987` | Completed | 5.149 | 6.960 | 0.9510 | 4.69% | 16.69% |
| S-shuffle LR-control | 5% | `1051003` | Completed | 11.614 | 13.844 | 0.3867 | 0.00% | 100.00% |
| S-shuffle LR-control | 10% | `1051004` | Completed | 7.370 | 9.716 | 0.8353 | 0.00% | 100.00% |
| S-shuffle LR-control | 25% | `1051005` | Completed | 6.531 | 8.668 | 0.8790 | 0.00% | 100.00% |
| S-shuffle LR-control | 50% | `1051006` | Completed | 5.661 | 7.525 | 0.9189 | 4.83% | 23.26% |
| S-shuffle LR-control | 100% | `1051007` | Completed | 5.367 | 7.227 | 0.9317 | 4.74% | 20.79% |
| U-ssl LR-control | 5% | `1051107` | Completed | 30.097 | 33.651 | 0.0000 | 0.00% | 100.00% |
| U-ssl LR-control | 10% | `1051108` | Completed | 29.278 | 32.865 | 0.0000 | 0.00% | 100.00% |
| U-ssl LR-control | 25% | `1051109` | Completed | 25.733 | 29.544 | 0.0000 | 0.00% | 100.00% |
| U-ssl LR-control | 50% | `1051110` | Completed | 7.936 | 10.495 | 0.9487 | 4.65% | 20.29% |
| U-ssl LR-control | 100% | `1051111` | Completed | 5.001 | 6.815 | 0.9492 | 4.74% | 18.48% |

## Remaining Work

No Q3 downstream jobs remain running.

Completed since launch: U-ssl pretraining job `1051105`, checkpoint conversion
job `1051106`, and U-ssl downstream jobs `1051107`-`1051111`. Final U-ssl
held-out downstream results are recorded above as five-fold image-level
aggregates.

## Job Accounting

| Stage | Jobs in Q3 plan | Current status | Notes |
| --- | ---: | --- | --- |
| R0 downstream sweep | 5 | Completed | One job per real-label fraction, all five folds per job |
| R1 downstream sweep | 5 | Completed | Same real subsets as R0 |
| S-age downstream sweep | 5 | Completed | Initialised from clean S-age synthetic checkpoints from `1050939` |
| S-shuffle downstream sweep | 5 | Completed | Initialised from shuffled-label synthetic pretraining job `1051002` |
| S-ssl downstream sweep | 5 | Completed | Initialised from BYOL checkpoints from `1050832` |
| U-ssl downstream sweep | 5 | Completed | `1051107`-`1051111`; all five folds per job |
| S-shuffle pretraining | 1 | Completed | Full SyntheticDorsalHands2 corpus, labels permuted once per seed |
| U-ssl pretraining | 1 | Completed | Job `1051105`; STL10 unlabeled; 22 BYOL epochs on 8 GPUs |

Planned downstream fine-tuning jobs: 30. Completed downstream jobs: 30.
Launched and running downstream jobs: 0.
