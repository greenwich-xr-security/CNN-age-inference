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
aggregation (`n=1`) on the locked real test split. Adult-gate AUC, FPR, and
Adult FNR use a 0--100 year predicted-age sweep with 1,001 thresholds. Each
fold selects its lowest Adult FNR point with FPR <= 5%, then achieved values
are averaged.

| Arm | Real-label fraction | Job | State | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 5% | `1050948` | Completed | 29.935 | 33.493 | 0.8434 | 4.78% | 46.53% |
| R0 | 10% | `1050949` | Completed | 26.846 | 30.901 | 0.7477 | 4.78% | 45.43% |
| R0 | 25% | `1050950` | Completed | 19.787 | 24.996 | 0.4965 | 4.78% | 78.63% |
| R0 | 50% | `1050951` | Completed | 10.442 | 14.358 | 0.7960 | 4.78% | 44.25% |
| R0 | 100% | `1050952` | Completed | 4.903 | 6.546 | 0.9677 | 4.78% | 17.33% |
| R1 | 5% | `1050953` | Completed | 30.177 | 37.391 | 0.3198 | 4.78% | 97.06% |
| R1 | 10% | `1050954` | Completed | 28.189 | 34.244 | 0.3589 | 4.78% | 97.77% |
| R1 | 25% | `1050955` | Completed | 27.308 | 38.927 | 0.3213 | 4.78% | 98.82% |
| R1 | 50% | `1050956` | Completed | 23.084 | 31.710 | 0.1840 | 4.78% | 97.79% |
| R1 | 100% | `1050957` | Completed | 11.684 | 18.725 | 0.7537 | 4.78% | 44.74% |
| S-age LR-control | 5% | `1050979` | Completed | 6.152 | 8.399 | 0.9232 | 4.78% | 24.03% |
| S-age LR-control | 10% | `1050980` | Completed | 5.547 | 7.591 | 0.9325 | 4.78% | 23.75% |
| S-age LR-control | 25% | `1050981` | Completed | 5.357 | 7.190 | 0.9371 | 4.78% | 23.49% |
| S-age LR-control | 50% | `1050982` | Completed | 5.114 | 6.943 | 0.9466 | 4.78% | 21.79% |
| S-age LR-control | 100% | `1050975` | Completed | 4.817 | 6.571 | 0.9589 | 4.78% | 18.30% |
| S-ssl LR-control | 5% | `1050983` | Completed | 30.248 | 33.794 | 0.8107 | 4.78% | 59.17% |
| S-ssl LR-control | 10% | `1050984` | Completed | 29.589 | 33.160 | 0.8992 | 4.78% | 38.30% |
| S-ssl LR-control | 25% | `1050985` | Completed | 25.792 | 29.625 | 0.9349 | 4.78% | 27.88% |
| S-ssl LR-control | 50% | `1050986` | Completed | 12.377 | 16.525 | 0.8420 | 4.78% | 47.58% |
| S-ssl LR-control | 100% | `1050987` | Completed | 5.149 | 6.960 | 0.9614 | 4.78% | 16.56% |
| S-shuffle LR-control | 5% | `1051003` | Completed | 11.614 | 13.844 | 0.6071 | 4.78% | 72.17% |
| S-shuffle LR-control | 10% | `1051004` | Completed | 7.370 | 9.716 | 0.8818 | 4.78% | 39.02% |
| S-shuffle LR-control | 25% | `1051005` | Completed | 6.531 | 8.668 | 0.9150 | 4.78% | 32.17% |
| S-shuffle LR-control | 50% | `1051006` | Completed | 5.661 | 7.525 | 0.9406 | 4.78% | 23.29% |
| S-shuffle LR-control | 100% | `1051007` | Completed | 5.367 | 7.227 | 0.9483 | 4.78% | 20.64% |
| U-ssl LR-control | 5% | `1051107` | Completed | 30.097 | 33.651 | 0.8158 | 4.78% | 66.70% |
| U-ssl LR-control | 10% | `1051108` | Completed | 29.278 | 32.865 | 0.8938 | 4.78% | 42.11% |
| U-ssl LR-control | 25% | `1051109` | Completed | 25.733 | 29.544 | 0.9130 | 4.78% | 32.02% |
| U-ssl LR-control | 50% | `1051110` | Completed | 7.936 | 10.495 | 0.9559 | 4.78% | 19.80% |
| U-ssl LR-control | 100% | `1051111` | Completed | 5.001 | 6.815 | 0.9630 | 4.78% | 18.43% |

## Fold Variability: 0--100-Year AUC Sweep

Computed from the five held-out folds at image-level aggregation (`n=1`). The
MAE variability is unchanged; Adult-gate AUC is recalculated from the 0--100
year predicted-age sweep.

| Arm | Real-label fraction | Job | Folds | MAE mean ± std | Adult-gate AUC mean ± std |
| --- | ---: | --- | ---: | ---: | ---: |
| R0 | 5% | `1050948` | 5 | 29.935 ± 0.383 | 0.8434 ± 0.0444 |
| R0 | 10% | `1050949` | 5 | 26.846 ± 3.651 | 0.7477 ± 0.3417 |
| R0 | 25% | `1050950` | 5 | 19.787 ± 4.574 | 0.4965 ± 0.2704 |
| R0 | 50% | `1050951` | 5 | 10.442 ± 5.117 | 0.7960 ± 0.2640 |
| R0 | 100% | `1050952` | 5 | 4.903 ± 0.106 | 0.9677 ± 0.0027 |
| R1 | 5% | `1050953` | 5 | 30.177 ± 1.772 | 0.3198 ± 0.0697 |
| R1 | 10% | `1050954` | 5 | 28.189 ± 1.438 | 0.3589 ± 0.1647 |
| R1 | 25% | `1050955` | 5 | 27.308 ± 3.842 | 0.3213 ± 0.1376 |
| R1 | 50% | `1050956` | 5 | 23.084 ± 2.012 | 0.1840 ± 0.0314 |
| R1 | 100% | `1050957` | 5 | 11.684 ± 10.777 | 0.7537 ± 0.3397 |
| S-age LR-control | 5% | `1050979` | 5 | 6.152 ± 0.412 | 0.9232 ± 0.0053 |
| S-age LR-control | 10% | `1050980` | 5 | 5.547 ± 0.257 | 0.9325 ± 0.0108 |
| S-age LR-control | 25% | `1050981` | 5 | 5.357 ± 0.162 | 0.9371 ± 0.0105 |
| S-age LR-control | 50% | `1050982` | 5 | 5.114 ± 0.099 | 0.9466 ± 0.0056 |
| S-age LR-control | 100% | `1050975` | 5 | 4.817 ± 0.200 | 0.9589 ± 0.0061 |
| S-ssl LR-control | 5% | `1050983` | 5 | 30.248 ± 0.158 | 0.8107 ± 0.0395 |
| S-ssl LR-control | 10% | `1050984` | 5 | 29.589 ± 0.375 | 0.8992 ± 0.0298 |
| S-ssl LR-control | 25% | `1050985` | 5 | 25.792 ± 0.498 | 0.9349 ± 0.0105 |
| S-ssl LR-control | 50% | `1050986` | 5 | 12.377 ± 3.579 | 0.8420 ± 0.1556 |
| S-ssl LR-control | 100% | `1050987` | 5 | 5.149 ± 0.567 | 0.9614 ± 0.0036 |
| S-shuffle LR-control | 5% | `1051003` | 5 | 11.614 ± 3.113 | 0.6071 ± 0.2961 |
| S-shuffle LR-control | 10% | `1051004` | 5 | 7.370 ± 0.354 | 0.8818 ± 0.0256 |
| S-shuffle LR-control | 25% | `1051005` | 5 | 6.531 ± 0.445 | 0.9150 ± 0.0191 |
| S-shuffle LR-control | 50% | `1051006` | 5 | 5.661 ± 0.438 | 0.9406 ± 0.0125 |
| S-shuffle LR-control | 100% | `1051007` | 5 | 5.367 ± 0.341 | 0.9483 ± 0.0108 |
| U-ssl LR-control | 5% | `1051107` | 5 | 30.097 ± 0.166 | 0.8158 ± 0.0507 |
| U-ssl LR-control | 10% | `1051108` | 5 | 29.278 ± 0.267 | 0.8938 ± 0.0254 |
| U-ssl LR-control | 25% | `1051109` | 5 | 25.733 ± 0.606 | 0.9130 ± 0.0147 |
| U-ssl LR-control | 50% | `1051110` | 5 | 7.936 ± 3.468 | 0.9559 ± 0.0101 |
| U-ssl LR-control | 100% | `1051111` | 5 | 5.001 ± 0.172 | 0.9630 ± 0.0045 |

## Fold Variability (Original 10--30-Year AUC Sweep)

Computed from each cell's five `fold_*/test_summary_ddp.csv` files using
`python tools/q3_fold_variance.py --root runs` before the 0--100 AUC
recalculation. The standard deviation is the sample standard deviation across
held-out folds at image-level aggregation (`n=1`).

| Arm | Real-label fraction | Job | Folds | MAE mean ± std | Adult-gate AUC mean ± std |
| --- | ---: | --- | ---: | ---: | ---: |
| R0 | 5% | `1050948` | 5 | 29.935 ± 0.383 | 0.0000 ± 0.0000 |
| R0 | 10% | `1050949` | 5 | 26.846 ± 3.651 | 0.0252 ± 0.0563 |
| R0 | 25% | `1050950` | 5 | 19.787 ± 4.574 | 0.3063 ± 0.2057 |
| R0 | 50% | `1050951` | 5 | 10.442 ± 5.117 | 0.7921 ± 0.2644 |
| R0 | 100% | `1050952` | 5 | 4.903 ± 0.106 | 0.9577 ± 0.0069 |
| R1 | 5% | `1050953` | 5 | 30.177 ± 1.772 | 0.0173 ± 0.0377 |
| R1 | 10% | `1050954` | 5 | 28.189 ± 1.438 | 0.0001 ± 0.0001 |
| R1 | 25% | `1050955` | 5 | 27.308 ± 3.842 | 0.0169 ± 0.0377 |
| R1 | 50% | `1050956` | 5 | 23.084 ± 2.012 | 0.1545 ± 0.0502 |
| R1 | 100% | `1050957` | 5 | 11.684 ± 10.777 | 0.7174 ± 0.3553 |
| S-age LR-control | 5% | `1050979` | 5 | 6.152 ± 0.412 | 0.9133 ± 0.0089 |
| S-age LR-control | 10% | `1050980` | 5 | 5.547 ± 0.257 | 0.9223 ± 0.0152 |
| S-age LR-control | 25% | `1050981` | 5 | 5.357 ± 0.162 | 0.9265 ± 0.0127 |
| S-age LR-control | 50% | `1050982` | 5 | 5.114 ± 0.099 | 0.9363 ± 0.0076 |
| S-age LR-control | 100% | `1050975` | 5 | 4.817 ± 0.200 | 0.9526 ± 0.0085 |
| S-ssl LR-control | 5% | `1050983` | 5 | 30.248 ± 0.158 | 0.0000 ± 0.0000 |
| S-ssl LR-control | 10% | `1050984` | 5 | 29.589 ± 0.375 | 0.0000 ± 0.0000 |
| S-ssl LR-control | 25% | `1050985` | 5 | 25.792 ± 0.498 | 0.0000 ± 0.0000 |
| S-ssl LR-control | 50% | `1050986` | 5 | 12.377 ± 3.579 | 0.8409 ± 0.1563 |
| S-ssl LR-control | 100% | `1050987` | 5 | 5.149 ± 0.567 | 0.9511 ± 0.0107 |
| S-shuffle LR-control | 5% | `1051003` | 5 | 11.614 ± 3.113 | 0.3867 ± 0.3612 |
| S-shuffle LR-control | 10% | `1051004` | 5 | 7.370 ± 0.354 | 0.8353 ± 0.0355 |
| S-shuffle LR-control | 25% | `1051005` | 5 | 6.531 ± 0.445 | 0.8790 ± 0.0282 |
| S-shuffle LR-control | 50% | `1051006` | 5 | 5.661 ± 0.438 | 0.9189 ± 0.0204 |
| S-shuffle LR-control | 100% | `1051007` | 5 | 5.367 ± 0.341 | 0.9317 ± 0.0178 |
| U-ssl LR-control | 5% | `1051107` | 5 | 30.097 ± 0.166 | 0.0000 ± 0.0000 |
| U-ssl LR-control | 10% | `1051108` | 5 | 29.278 ± 0.267 | 0.0000 ± 0.0000 |
| U-ssl LR-control | 25% | `1051109` | 5 | 25.733 ± 0.606 | 0.0000 ± 0.0000 |
| U-ssl LR-control | 50% | `1051110` | 5 | 7.936 ± 3.468 | 0.9487 ± 0.0070 |
| U-ssl LR-control | 100% | `1051111` | 5 | 5.001 ± 0.172 | 0.9492 ± 0.0040 |

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
