# Q2 Generator-Validation Job Matrix

Scope: Real means HandRGBD + ProlificHands. Synthetic means
SyntheticDorsalHands2. All cells should report the same endpoint, with MAE as
the primary metric and adult-gate AUC optional, so they compare directly.

| Cell | Train on | Test on | Name | Status | Associated job |
| --- | --- | --- | --- | --- | --- |
| RR | Real | Real locked split | Reference | Done | `1050753` |
| SS | Synthetic | held-out SyntheticDorsalHands2 split | Internal learnability / degeneracy check | Done | `1050939` |
| SR | Synthetic | Real locked split | TSTR | Running | `1050942`; uses SS checkpoints from `1050939` |
| RS | Real | held-out SyntheticDorsalHands2 split | TRTS | Done | `1050940`; uses RR checkpoints from `1050753` |

## Results: Q2 generator-validation cells

All values are unweighted means of five held-out fold results at image-level
aggregation (`n=1`). FPR and Adult FNR use each fold's best operating point
with FPR <= 5% when attainable, then average the achieved values.

| Cell | Train on | Test on | Objective / job | MAE (years) | RMSE (years) | Adult-gate AUC | Mean FPR | Adult FNR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| RR | Real | Real locked split | Pure NLL (`1050753`) | 5.017 | 6.706 | 0.9483 | 4.74% | 18.33% |
| SS | Synthetic | held-out SyntheticDorsalHands2 split | Pure NLL (`1050939`) | 3.396 | 4.436 | 0.9824 | 4.85% | 6.77% |
| RS | Real | held-out SyntheticDorsalHands2 split | Eval-only TRTS (`1050940`, checkpoints from `1050753`) | 11.056 | 13.390 | 0.5609 | 41.85%* | 9.90%* |

*For `1050940`, the 5% FPR operating point was not attainable in any fold
within the 10-30 year age-threshold sweep. The reported FPR/FNR use the
lowest-FPR available threshold (`tau=30`) in each fold.

Notes:

- `1050753` is the directly comparable real-only pure NLL reference run.
- `1050939` is the Q2 SS SyntheticDorsalHands2-only pure NLL job.
- SR/TSTR and RS/TRTS do not require new model training if the required
  checkpoints are available; they are evaluation jobs over the opposite locked
  test source. RS/TRTS completed as `1050940`; SR/TSTR is running as `1050942`
  from the completed SS checkpoints from `1050939`.
- `1050941` was a single-fold SR/TSTR probe from SS fold 0 to the real held-out
  split. It is not counted as the full SR cell.
- `1050938` was a failed Slurm submission attempt before Python started and is
  not counted as an experimental cell.
