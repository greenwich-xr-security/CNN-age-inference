# SKILLS.md

This file defines repo-local reusable workflows for coding agents. It is focused on operating this repository on the remote HPC cluster.

## `hpc`

Use this skill when the user asks to inspect or operate on the remote HPC cluster for this repository, including:

- syncing `~/CNN-age-inference`
- checking queued, running, or recent Slurm jobs
- reading the latest or a specific Slurm log
- submitting or cancelling distributed training jobs
- submitting Optuna workers
- inspecting remote run artifacts under `runs/`
- running direct remote Python commands for this repository

### Connection And Repo Paths

- SSH target: `ssh rb3434w@staff@100.96.122.39`
- Project root: `~/CNN-age-inference`
- Dataset root: `/home/rb3434w/HandsDatasets`
- Partition: `gpu-beast`
- Primary submit script: `~/CNN-age-inference/submit_distributed.slurm`
- Optuna submit script: `~/CNN-age-inference/submit_optuna.slurm`
- Slurm logs: `~/CNN-age-inference/logs/age-infer-ddp-<jobid>.log`
- Distributed outputs: `~/CNN-age-inference/runs/<run_name>/`

### Operating Rules

- Use a single `ssh` command with the remote command passed inline. Do not start an interactive SSH session.
- `cd ~/CNN-age-inference` before running repo commands on the cluster.
- Do not assume the remote checkout matches the local workspace. If the user wants newly edited code to run remotely, sync the branch first.
- For direct remote Python commands, activate the `xr` conda environment first with the same activation path used by `submit_distributed.slurm`.
- `sbatch submit_distributed.slurm` already handles its own environment activation and module loading.
- When reading logs, summarize fold, epoch progress, train loss, validation loss, MAE, AUC, checkpoint saves, and any CUDA, OOM, NCCL, or Python traceback failures.
- Flag when a job is still in queue, failed, timed out, or finished successfully.

### Common Commands

- Sync code:
  `ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && git status --short && git pull --ff-only"`
- Check running or queued jobs:
  `ssh rb3434w@staff@100.96.122.39 "squeue -u rb3434w"`
- Check recent jobs:
  `ssh rb3434w@staff@100.96.122.39 "sacct -u rb3434w --format=JobID,JobName,State,Elapsed,Start,End -X | tail -20"`
- Tail the latest distributed log:
  `ssh rb3434w@staff@100.96.122.39 "ls -t ~/CNN-age-inference/logs/age-infer-ddp-*.log 2>/dev/null | head -1 | xargs -r tail -100"`
- Tail a specific distributed log:
  `ssh rb3434w@staff@100.96.122.39 "tail -100 ~/CNN-age-inference/logs/age-infer-ddp-<jobid>.log"`
- Submit the default distributed run:
  `ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && sbatch submit_distributed.slurm"`
- Submit a ViT tiny run:
  `ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && MODELS=vit_tiny_384 RUN_NAME=vit_tiny_384_384_ddp IMG_SIZE=384 sbatch submit_distributed.slurm"`
- Submit an Optuna worker:
  `ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && MODEL=vit_tiny_384 STUDY_NAME=age_vit_tiny TRIALS_PER_JOB=6 sbatch submit_optuna.slurm"`
- Cancel a job:
  `ssh rb3434w@staff@100.96.122.39 "scancel <jobid>"`
- Check run roots and recent outputs:
  `ssh rb3434w@staff@100.96.122.39 "du -sh ~/CNN-age-inference/runs 2>/dev/null && find ~/CNN-age-inference/runs -maxdepth 2 -type d | tail -30"`
- Inspect k-fold aggregate outputs for one run:
  `ssh rb3434w@staff@100.96.122.39 "ls ~/CNN-age-inference/runs/<run_name>/kfold_summary_n*.csv ~/CNN-age-inference/runs/<run_name>/roc_case*_kfold_n*.png 2>/dev/null"`
- Inspect per-fold test outputs:
  `ssh rb3434w@staff@100.96.122.39 "ls ~/CNN-age-inference/runs/<run_name>/fold_<n>/test_summary_ddp.csv ~/CNN-age-inference/runs/<run_name>/fold_<n>/test_roc_case*.png 2>/dev/null"`

### Conda Activation For Direct Remote Commands

Use this pattern before running remote Python commands directly:

```bash
ssh rb3434w@staff@100.96.122.39 "source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh && conda activate xr && cd ~/CNN-age-inference && <your command>"
```

### `submit_distributed.slurm` Knobs

These environment variables are the primary control surface for remote training:

- `MODELS`: backbone list. Valid names include `b0`-`b7`, `v2_s`, `v2_m`, `v2_l`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `convnext_xlarge`, `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`, `swin_tiny`, `swin_small`, `swin_base`, `swin_large`, `swin_v2_tiny`, `swin_v2_small`, `swin_v2_base`, `swin_v2_large`, `vit_tiny_384`. Aliases from `models/__init__.py` are also accepted, including `vtt` and `age_vit`.
- `RUN_NAME`: output folder name under `runs/`.
- `KFOLDS`: number of folds. Default is `5`.
- `BATCH_SIZE`: per-process batch size.
- `USER_GROUP_SIZES`: samples per user in each training batch.
- `IMG_SIZE`: optional image-size override. Leave aligned with the chosen model unless the user explicitly wants an override.
- `EPOCHS`, `PATIENCE`, `LR`, `WEIGHT_DECAY`, `SEED`, `NUM_WORKERS`: standard training knobs.
- `AGG_SIZES` and `AGG_SEED`: evaluation aggregation settings passed to `train_distributed.py`.
- `USE_MASKS`: set `1` to enable mask application during loading.
- `AGE_REWEIGHT`, `AGE_WEIGHT_EPS`, `AGE_WEIGHT_POWER`, `AGE_WEIGHT_MIN`, `AGE_WEIGHT_MAX`: inverse-frequency age-loss weighting controls.
- `AGE_OVERSAMPLE`, `AGE_OVERSAMPLE_TARGET`, `AGE_OVERSAMPLE_MAX_MULT`: age oversampling controls.
- `TEST_SPLIT_FILE`, `TEST_SPLIT_SIZE`: held-out user split configuration.
- `FOLD_OVERWRITE`: regenerate the fold file when set to `1`.
- `RUN_KFOLD_AGG`: run `aggregate_kfold.py` after all folds when set to `1`.
- `OUTPUT_ROOT`, `PROJECT_ROOT`, `DATA_ROOT`, `TORCH_HOME`: path overrides when needed.

### Output Layout

- Fold training outputs land in `~/CNN-age-inference/runs/<run_name>/fold_<n>/`.
- Each fold writes `<model_key>_age_regressor_ddp.pth`, `history_distributed.log`, validation ROC plots, metrics CSVs, prediction dumps, challenge CSVs, and `test_summary_ddp.csv`.
- The submit script runs `aggregate_kfold.py` with `--output-dir` set to the run root, so aggregate files are written directly into `~/CNN-age-inference/runs/<run_name>/`, not an `aggregate/` subfolder.
- The held-out test split file defaults to `~/CNN-age-inference/runs/test_users.json`.

### Reporting Expectations

When summarizing HPC work for the user:

- report the exact job ID
- report whether the job is `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, or `TIMEOUT`
- include the absolute remote log path you inspected
- mention the model, run name, fold index, and any non-default env vars used at submission time
- call out missing outputs, partial folds, or stale run directories explicitly
