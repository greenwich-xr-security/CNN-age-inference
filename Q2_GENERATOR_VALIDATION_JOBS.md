# Q2 Generator-Validation Job Matrix

Scope: Real means HandRGBD + ProlificHands. Synthetic means
SyntheticDorsalHands2. All cells should report the same endpoint, with MAE as
the primary metric and adult-gate AUC optional, so they compare directly.

| Cell | Train on | Test on | Name | Status | Associated job |
| --- | --- | --- | --- | --- | --- |
| RR | Real | Real locked split | Reference | Done | `1050753` |
| SS | Synthetic | held-out SyntheticDorsalHands2 split | Internal learnability / degeneracy check | Done | `1050939` |
| SR | Synthetic | Real locked split | TSTR | Done | `1050942`; uses SS checkpoints from `1050939` |
| RS | Real | held-out SyntheticDorsalHands2 split | TRTS | Done | `1050940`; uses RR checkpoints from `1050753` |

## Results: Q2 generator-validation cells

All values are unweighted means of five held-out fold results at image-level
aggregation (`n=1`). Adult-gate AUC, FPR, and Adult FNR use a 0--100 year
predicted-age sweep with 1,001 thresholds. Each fold selects its lowest Adult
FNR point with FPR <= 5%, then achieved values are averaged.

| Cell | Train on | Test on | Objective / job | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| RR | Real | Real locked split | Pure NLL (`1050753`) | 5.017 | 6.706 | 0.9633 | 4.78% | 18.28% |
| SS | Synthetic | held-out SyntheticDorsalHands2 split | Pure NLL (`1050939`) | 3.396 | 4.436 | 0.9848 | 4.93% | 6.69% |
| SR | Synthetic | Real locked split | Eval-only TSTR (`1050942`, checkpoints from `1050939`) | 7.165 | 9.097 | 0.9025 | 4.78% | 31.17% |
| RS | Real | held-out SyntheticDorsalHands2 split | Eval-only TRTS (`1050940`, checkpoints from `1050753`) | 11.056 | 13.390 | 0.8957 | 4.93% | 34.28% |

## Canonical run directories

Job IDs alone are not enough to locate these results: several similarly named
directories under `runs/` hold different experiments over different test
sources. The four cells above are backed by exactly these directories, verified
by recomputing MAE from `fold_*/test_predictions_n1_ddp.npz`.

| Cell | Job | Run directory | n (test) | Recomputed MAE |
| --- | --- | --- | ---: | ---: |
| RR | `1050753` | `v2s_handrgbd_prolific_only_nll_uncapped_test20_k5_4gpu_16cpu_20260728_384` | 1,624 | 5.017 |
| SS | `1050939` | `q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384` | 4,423 | 3.396 |
| SR | `1050942` | `q2_sr_ss1050939_to_real_eval_20260801` | 1,624 | 7.165 |
| RS | `1050940` | `q2_rs_real1050753_to_synthetic2_eval_20260801` | 4,423 | 11.056 |

Do **not** use `runs/q2_fullreal_{rr,ss,sr,rs}_v2s_seed42` for these cells.
Those directories evaluate a different 150-subject test source (n=150
synthetic, n=2,384 real, with real ages extending to 75) and yield MAE
10.32/5.81/11.53/13.74 respectively, none of which match the reported table.

Notes:

- `1050753` is the directly comparable real-only pure NLL reference run.
- `1050939` is the Q2 SS SyntheticDorsalHands2-only pure NLL job.
- SR/TSTR and RS/TRTS do not require new model training if the required
  checkpoints are available; they are evaluation jobs over the opposite locked
  test source. RS/TRTS completed as `1050940`; SR/TSTR completed as `1050942`
  from the completed SS checkpoints from `1050939`.
- `1050941` was a single-fold SR/TSTR probe from SS fold 0 to the real held-out
  split. It is not counted as the full SR cell.
- `1050938` was a failed Slurm submission attempt before Python started and is
  not counted as an experimental cell.

## Age-conditioning fidelity

Computed with `summarize_q2_age_monotonicity.py`, which reports global per-fold
Spearman/Pearson correlation, the OLS slope of predicted on conditioned age,
and a decade-level calibration curve. Every correlation is reported beside an
approximate attenuation ceiling (`sigma_true / hypot(sigma_true, sigma_resid)`),
without which the values are easy to misread.

| Cell | Reader | Images | n | OLS slope | Spearman rho | Ceiling |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RR | real-trained | real | 1,624 | 0.809 +/- 0.054 | 0.898 +/- 0.005 | 0.916 |
| SS | synth-trained | synthetic | 4,423 | 0.945 +/- 0.015 | 0.972 +/- 0.001 | 0.973 |
| RS | real-trained | synthetic | 4,423 | 0.360 +/- 0.037 | 0.842 +/- 0.018 | 0.825 |

Interpretation:

- The synthetic images encode age at close to full scale: SS recovers
  conditioned age with slope 0.945 and rho 0.972, essentially at the ceiling.
- The RS compression (slope 0.360) is a property of the reader, not of the
  images. The same images give slope 0.945 to a synthetic-trained model, and
  the same real-trained model gives slope 0.809 on real images. The TRTS error
  is therefore a domain gap, not a failure to render the conditioned age.
- SS per-decade deviation stays within 1.6 yr over ages 10-69 and reaches
  -4.1 yr at conditioned ages 70-79, above the generator's real training
  support (67). This bounds the cost of extrapolated age conditioning.
- RS rho slightly exceeds its ceiling because the ceiling folds systematic
  compression into the residual term, which makes it conservative whenever the
  slope departs from 1.

Superseded: the per-decade Spearman diagnostic in
`summarize_q2_conditioning_fidelity.py` is not a valid measure here. Slicing
into decades leaves ~10 yr of conditioned-age spread against an RS error of
~12-13 yr, attenuating the within-bin correlation to a ceiling near 0.23
regardless of generator fidelity. It was additionally run against
`q2_fullreal_rs_v2s_seed42`, which is not the RS cell. Its output CSV has been
deleted; the script is retained only as a record of the rejected approach.
