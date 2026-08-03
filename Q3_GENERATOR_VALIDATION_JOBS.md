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

## Downstream Grid

| Arm | Pretraining before real fine-tune | Real fine-tune fractions | Synthetic data fractioned? | Role |
| --- | --- | --- | --- | --- |
| R0 | Standard init, no synthetic | 5%, 10%, 25%, 50%, 100% | No synthetic used | Main real-only baseline; equivalent-real-labels reference curve |
| R1 | Random init, no pretraining | 5%, 10%, 25%, 50%, 100% | No synthetic used | From-scratch baseline; isolates ImageNet/default init value |
| S-age | Supervised age pretraining on SyntheticDorsalHands2 | 5%, 10%, 25%, 50%, 100% | No; full synthetic corpus for pretraining | Candidate benefit arm |
| S-shuffle | Same synthetic images/schedule as S-age, but synthetic ages permuted once per seed | 5%, 10%, 25%, 50%, 100% | No; full synthetic corpus with shuffled labels | Removes synthetic age-label signal |
| S-ssl | Self-supervised pretraining on SyntheticDorsalHands2 | 5%, 10%, 25%, 50%, 100% | No; full synthetic corpus for SSL | Removes supervision, keeps dorsal-hand corpus |
| U-ssl | Same SSL method/update count on unrelated non-hand, non-age corpus | 5%, 10%, 25%, 50%, 100% | No SyntheticDorsalHands2 used | Removes dorsal-hand content; generic compute/corpus control |

## Existing Assets And Jobs

| Arm / asset | Current status | Existing job(s) | Notes |
| --- | --- | --- | --- |
| Real label-fraction manifest | Done | n/a | `splits/q3_real_label_fractions_seed42.json`; fractions `0.05`, `0.10`, `0.25`, `0.50`, `1.00` |
| R0 100% | Done | `1050753` | Full-real ImageNet/default-init baseline only; remaining fractions not run |
| R1 100% | Done | `1050877` | Full-real random-init baseline only; remaining fractions not run |
| S-age pretraining | Available | `1050939` | Clean SyntheticDorsalHands2-only supervised NLL checkpoints; suitable pretraining source |
| S-age 100% fine-tune | Needs clean Q3 rerun | `1050819` is related | `1050819` used SyntheticDorsalHands2 initialisation, but not from the clean `1050939` SS source, so do not treat it as the final Q3 S-age cell without caveat |
| S-shuffle pretraining | Implementation ready, not launched | none | Use `SHUFFLE_SYNTHETIC_DORSAL2_AGE_LABELS=1`; shuffled-age synthetic pretraining is still needed before downstream fractions |
| S-ssl pretraining | Done | `1050832` | BYOL on SyntheticDorsalHands2 |
| S-ssl 100% fine-tune | Done | `1050854` | Full-real fine-tune from BYOL; remaining fractions not run |
| U-ssl pretraining | Missing | none | Need unrelated non-hand, non-age corpus and compute-matched SSL setup |

## Launched Jobs

These downstream jobs use the same real-label fraction manifest,
`splits/q3_real_label_fractions_seed42.json`, and the same locked real test
split, `splits/test_users_uncapped_20pct_seed42.json`. One Slurm job runs all
five folds for one arm/fraction pair.

Monitor on HPC:
`/home/rb3434w/CNN-age-inference/runs/q3_r0_r1_sage_seed42_monitor/q3_status.md`

| Arm | 5% | 10% | 25% | 50% | 100% |
| --- | --- | --- | --- | --- | --- |
| R0 | `1050948` | `1050949` | `1050950` | `1050951` | `1050952` |
| R1 | `1050953` | `1050954` | `1050955` | `1050956` | `1050957` |
| S-age | `1050958` | `1050959` | `1050960` | `1050961` | `1050962` |
| S-shuffle | Needs pretraining | Needs pretraining | Needs pretraining | Needs pretraining | Needs pretraining |
| S-ssl | `1050969` | `1050970` | `1050971` | `1050972` | `1050973` |
| U-ssl | Not implemented | Not implemented | Not implemented | Not implemented | Not implemented |

## LR-Control Reruns

| Arm / comparison | Real-label fraction | Job | State | Purpose |
| --- | ---: | --- | --- | --- |
| S-age, clean synthetic-only init, LR `2e-5` | 100% | `1050975` | Completed | Match `1050819` fine-tuning LR/max epochs while keeping Q3 clean `1050939` synthetic-only init |

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
| S-age | 5% | `1050958` | Completed | 6.745 | 8.823 | 0.8878 | 4.69% | 26.48% |
| S-age | 10% | `1050959` | Completed | 5.841 | 8.015 | 0.8988 | 5.60% | 24.74% |
| S-age | 25% | `1050960` | Completed | 5.367 | 7.208 | 0.9214 | 4.78% | 23.58% |
| S-age | 50% | `1050961` | Completed | 5.091 | 6.898 | 0.9317 | 4.69% | 21.97% |
| S-age | 100% | `1050962` | Completed | 5.024 | 6.811 | 0.9399 | 4.74% | 20.76% |
| S-ssl | 5% | `1050969` | Completed | 27.840 | 31.494 | 0.0987 | 1.00% | 96.60% |
| S-ssl | 10% | `1050970` | Completed | 22.684 | 26.889 | 0.3474 | 2.49% | 88.19% |
| S-ssl | 25% | `1050971` | Completed | 8.185 | 10.639 | 0.9117 | 4.98% | 27.28% |
| S-ssl | 50% | `1050972` | Completed | 5.814 | 7.659 | 0.9259 | 4.91% | 22.56% |
| S-ssl | 100% | `1050973` | Completed | 5.444 | 7.067 | 0.9252 | 4.98% | 22.23% |
| S-age LR-control | 100% | `1050975` | Completed | 4.817 | 6.571 | 0.9526 | 4.69% | 18.46% |

## Planned Job Accounting

| Stage | Jobs needed | Notes |
| --- | ---: | --- |
| R0 downstream sweep | 5 | One job per real-label fraction, all five folds per job |
| R1 downstream sweep | 5 | Same real subsets as R0 |
| S-age downstream sweep | 5 | Initialise from clean S-age synthetic checkpoints, preferably `1050939` |
| S-shuffle downstream sweep | 5 | Requires S-shuffle pretraining first; downstream code path is the same as S-age |
| S-ssl downstream sweep | 5 | Launched on `gpu-standard`, 2 GPUs/job; initialise from BYOL checkpoints from `1050832` |
| U-ssl downstream sweep | 5 | Requires U-ssl pretraining first |
| S-shuffle pretraining | 1 | Full SyntheticDorsalHands2 corpus, labels permuted once per seed |
| U-ssl pretraining | 1 | Same SSL method/update count as S-ssl, unrelated corpus |

Total planned downstream fine-tuning jobs: 30, assuming one Slurm job runs all
five folds for one arm/fraction pair.
