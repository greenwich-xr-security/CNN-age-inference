# Repository guidance

## HPC job recordkeeping

Whenever you launch an HPC/Slurm training job, update `HPC_JOB_HISTORY.md` in
the same task with its job ID, run directory, purpose, data/configuration
differences, requested resources, submission date, and initial scheduler state.

Whenever an HPC/Slurm job completes, fails, or is cancelled, update the same
entry promptly with its terminal state and elapsed time. For completed training
jobs, also record the evaluated held-out results and any directly comparable
aggregate, while clearly stating the fold count and aggregation level.

For adult-gate updates, report MAE alongside mean FPR and Adult FNR at the
specified FPR operating point. Do not report Adult TPR as well: at the same
operating point it is redundant (`TPR = 1 - FNR`).
