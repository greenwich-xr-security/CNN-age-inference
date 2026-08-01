# Q2 Generator-Validation Job Matrix

Scope: Real means HandRGBD + ProlificHands. Synthetic means
SyntheticDorsalHands2. All cells should report the same endpoint, with MAE as
the primary metric and adult-gate AUC optional, so they compare directly.

| Cell | Train on | Test on | Name | Status | Associated job |
| --- | --- | --- | --- | --- | --- |
| RR | Real | Real locked split | Reference | Done | `1050753` |
| SS | Synthetic | held-out SyntheticDorsalHands2 split | Internal learnability / degeneracy check | Running | `1050939` |
| SR | Synthetic | Real locked split | TSTR | Not run yet | no eval job yet; use SS checkpoints from `1050939` |
| RS | Real | held-out SyntheticDorsalHands2 split | TRTS | Running | `1050940`; uses RR checkpoints from `1050753` |

Notes:

- `1050753` is the directly comparable real-only pure NLL reference run.
- `1050939` is the Q2 SS SyntheticDorsalHands2-only pure NLL job.
- SR/TSTR and RS/TRTS do not require new model training if the required
  checkpoints are available; they are evaluation jobs over the opposite locked
  test source. RS/TRTS is running as `1050940`; SR/TSTR waits for SS `1050939`
  to finish all five folds.
- `1050938` was a failed Slurm submission attempt before Python started and is
  not counted as an experimental cell.
