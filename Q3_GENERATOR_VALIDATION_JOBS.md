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
| S-shuffle pretraining | Missing | none | Need shuffled-age synthetic pretraining job before downstream fractions |
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
| S-shuffle | Not implemented | Not implemented | Not implemented | Not implemented | Not implemented |
| S-ssl | Not launched yet | Not launched yet | Not launched yet | Not launched yet | Not launched yet |
| U-ssl | Not implemented | Not implemented | Not implemented | Not implemented | Not implemented |

## Results

All values are unweighted means of five held-out fold results at image-level
aggregation (`n=1`) on the locked real test split. FPR and Adult FNR use each
fold's best operating point with FPR <= 5% when attainable, then average the
achieved values.

| Arm | Real-label fraction | Job | State | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| R0 | 5% | `1050948` | Completed | 29.935 | 33.493 | 0.0000 | 0.00% | 100.00% |

## Planned Job Accounting

| Stage | Jobs needed | Notes |
| --- | ---: | --- |
| R0 downstream sweep | 5 | One job per real-label fraction, all five folds per job |
| R1 downstream sweep | 5 | Same real subsets as R0 |
| S-age downstream sweep | 5 | Initialise from clean S-age synthetic checkpoints, preferably `1050939` |
| S-shuffle downstream sweep | 5 | Requires S-shuffle pretraining first |
| S-ssl downstream sweep | 5 | Initialise from BYOL checkpoints from `1050832` |
| U-ssl downstream sweep | 5 | Requires U-ssl pretraining first |
| S-shuffle pretraining | 1 | Full SyntheticDorsalHands2 corpus, labels permuted once per seed |
| U-ssl pretraining | 1 | Same SSL method/update count as S-ssl, unrelated corpus |

Total planned downstream fine-tuning jobs: 30, assuming one Slurm job runs all
five folds for one arm/fraction pair.
