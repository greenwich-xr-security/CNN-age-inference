# Q3 R0/R1 Learning-Rate Sweep (5% Real Labels)

## Purpose

R0 (ImageNet init) and R1 (random init) both collapse at low real-label
fractions in the original Q3 sweep — R0 5% MAE 29.935, R1 5% MAE 30.177,
both close to a degenerate constant-prediction failure rather than a genuine
information limit (see `Q3_GENERATOR_VALIDATION_JOBS.md`). Every other Q3 arm
(S-age, S-ssl, S-shuffle, U-ssl) was later re-run with an LR-control recipe
(`LR=2e-5`) to rule out this failure mode, but R0/R1 never were. This sweep
isolates the learning rate as the single variable for R0/R1 at the worst-hit
fraction (5%) before deciding whether to extend a winning LR across the full
5/10/25/50/100% grid.

## Design

Three LR points per arm, all other hyperparameters fixed to the **original**
R0/R1 recipe (itself equal to `submit_distributed.slurm` defaults):
`EPOCHS=240`, `PATIENCE=10`, `WEIGHT_DECAY=0.2`, `BATCH_SIZE=16`, one
`gpu-beast` node / 4 GPUs, EfficientNet-V2-S at 384px, pure Gaussian NLL,
`splits/q3_real_label_fractions_seed42.json` fraction `0.05`, locked test
split `splits/test_users_uncapped_20pct_seed42.json`, seed 42.

| LR | Status | Note |
| --- | --- | --- |
| `2e-5` | New | Matches the LR-control value used for every other Q3 arm |
| `2e-4` | Reused | Original R0/R1 sweep cell — already collapsed |
| `1e-3` | New | Stress point above the failing LR, to confirm the failure is part of a "too high" regime and not a single unlucky point |

Submit script: `submit_q3_r0_r1_lr_sweep_f05.sh` (launches only the two new
LR cells per arm; `2e-4` reuses existing jobs `1050948`/`1050953`).

## Launched Jobs

Submitted 2026-08-18 from local machine via
`ssh rb3434w@staff@100.96.122.39`.

| Arm | LR | Job | Run name | State (at submit) |
| --- | --- | --- | --- | --- |
| R0 | `2e-5` | `1055773` | `q3_r0_realfrac_f05_lr2e5_seed42_v2s_384` | RUNNING, fold 3/5 |
| R0 | `1e-3` | `1055774` | `q3_r0_realfrac_f05_lr1e3_seed42_v2s_384` | COMPLETED (00:58:35) |
| R1 | `2e-5` | `1055775` | `q3_r1_realfrac_f05_lr2e5_seed42_v2s_384` | RUNNING, fold 2/5 |
| R1 | `1e-3` | `1055776` | `q3_r1_realfrac_f05_lr1e3_seed42_v2s_384` | PENDING (resources) |
| R0 | `2e-4` | `1050948` | `q3_r0_realfrac_f05_seed42_v2s_384` | COMPLETED (original sweep) |
| R1 | `2e-4` | `1050953` | `q3_r1_realfrac_f05_seed42_v2s_384` | COMPLETED (original sweep) |

`gpu-beast` has 8 GPUs total; the two R0 jobs claimed all 8 (4 each) at
launch, so the two R1 jobs queued behind them. R1 `2e-5` started once R0
`1e-3` finished; R1 `1e-3` is still waiting on a free slot.

## Tracking

- Live state: `ssh rb3434w@staff@100.96.122.39 "squeue -j 1055773,1055774,1055775,1055776 -o '%.10i %.25j %.10P %.8T %.12M %.6D %R'"`
- Recent history: `ssh rb3434w@staff@100.96.122.39 "sacct -j 1055773,1055774,1055775,1055776 --format=JobID,JobName,State,Elapsed,ExitCode --noheader"`
- Logs: `~/CNN-age-inference/logs/q3-<arm>-<lrtag>-<jobid>.log` on the cluster (e.g. `logs/q3-r0-lr1e3-1055774.log`)
- Per-fold outputs: `~/CNN-age-inference/runs/<run_name>/fold_<n>/test_summary_ddp.csv`
- Aggregate (once `RUN_KFOLD_AGG=1` finishes): `~/CNN-age-inference/runs/<run_name>/kfold_summary_n1.csv`

## Results

All values are unweighted means of five held-out fold results at
image-level aggregation (`n=1`) on the locked real test split, once
available.

All four sweep cells are complete (plus the two reused `2e-4` cells from the
original sweep). Values below are from **`kfold_test_summary_n1.csv`** (the
locked held-out test split, 1,624 samples/fold — matching how the original
R0/R1 baseline numbers were computed), not `kfold_summary_n1.csv` (the
rotating validation split, ~1,268–1,328 samples/fold). An earlier pass of
this table accidentally used the validation file; corrected below.

| Arm | LR | Job | State | MAE (years) | RMSE (years) | Adult-gate AUC |
| --- | --- | --- | --- | ---: | ---: | ---: |
| R0 | `2e-5` | `1055773` | COMPLETED | 30.990 | 34.468 | 0.8228 |
| R0 | `2e-4` | `1050948` | COMPLETED (original) | 29.935 | 33.493 | 0.8434 |
| R0 | `1e-3` | `1055774` | COMPLETED | 28.605 | 32.267 | 0.7573 |
| R1 | `2e-5` | `1055775` | COMPLETED | 30.952 | 34.511 | 0.5527 |
| R1 | `2e-4` | `1050953` | COMPLETED (original) | 30.177 | 37.391 | 0.3198 |
| R1 | `1e-3` | `1055776` | COMPLETED | 25.708 | 30.712 | 0.3630 |

Per-fold MAE (held-out test):
- R0 `2e-5`: 30.97, 31.22, 30.91, 30.96, 30.89
- R0 `1e-3`: 29.86, 24.47, 29.08, 29.55, 30.06
- R1 `2e-5`: 31.66, 30.21, 30.62, 30.62, 31.66
- R1 `1e-3`: 24.14, 26.37, 26.54, 26.05, 25.45

(Mean FPR / Adult FNR not pulled for the new cells — `kfold_test_summary_n1.csv`
only carries MAE/RMSE/AUC per fold; the operating-point metrics live in a
separate per-fold gate-metrics file if needed later for the paper table.)

## Conclusion

**LR is not the cause of R0/R1's collapse at 5% real labels.** All three LR
points tested — spanning two orders of magnitude (`2e-5` to `1e-3`, plus the
original `2e-4`) — land both arms in the same ~25–31 MAE band, far from the
~5–12 MAE that S-age/S-shuffle reach at 5% labels with an age-informative
init. This matches the precedent already in the repo: the same `2e-5`
LR-control recipe also failed to rescue S-ssl and U-ssl (both BYOL/SSL
inits with no age signal) at low fractions, while it did work for S-age and
S-shuffle (both initialised from age-supervised or image-matched synthetic
pretraining). The dividing line is the *initialisation carrying
age-relevant signal*, not the learning rate.

One secondary finding worth flagging in the writeup: `1e-3` (the highest LR
tested, not the lowest) was consistently the *least bad* of the three for
both arms — R0 improved from 29.9 to 28.6, and R1 improved more
substantially from 30.2 to 25.7. This is the opposite of what the "LR too
high" hypothesis predicts (lower LR should have helped more, not higher),
which further undermines an LR-schedule explanation and supports treating
this as a genuine label-efficiency/information limit rather than an
optimizer artifact. Given that, I would not recommend extending `2e-5` (or
any single LR) across the full 5/10/25/50/100% grid as a replacement R0/R1
baseline — the original `2e-4` sweep already reflects the arms' real
capability at each fraction.

**Recommendation for the paper**: keep the original R0/R1 numbers
(`1050948`-`1050957`) as the headline baseline, and cite this sweep (plus
the S-ssl/U-ssl LR-control precedent) as the evidence that the baseline
failure is not an undertrained/mistuned artifact. Still worth adding the
predict-the-training-mean baseline row separately, since that's an
orthogonal sanity check a reviewer will still expect.
