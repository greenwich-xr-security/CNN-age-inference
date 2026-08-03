# Repository guidance

## HPC command safety

When operating the HPC over SSH, use plain, human-readable commands. Do not use
base64-encoded command delivery, decoded shell execution, or patterns such as
`base64 -d ... | bash`, `bash -c "$(base64 -d ...)"`, or equivalent wrappers.
If quoting becomes awkward, prefer a checked-in script, a temporary script with
clear text content, or a simpler inline SSH command.

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
