# HPC — Connect and operate on the remote cluster

You have SSH access to the HPC cluster. Use the Bash tool to run SSH commands.

**Connection:**
```
ssh rb3434w@staff@100.96.122.39
```

**Project location on HPC:**
```
~/CNN-age-inference
```

---

## How to handle the user's request

When the user invokes `/hpc`, read their request and perform the appropriate action below via SSH. Always use a single `ssh` command with the remote commands passed inline (not interactive sessions).

### Sync code
```bash
ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && git pull"
```

### Check running / queued jobs
```bash
ssh rb3434w@staff@100.96.122.39 "squeue -u rb3434w"
```

### Check all recent jobs (including finished)
```bash
ssh rb3434w@staff@100.96.122.39 "sacct -u rb3434w --format=JobID,JobName,State,Elapsed,Start,End -X | tail -20"
```

### View the latest SLURM log and report results
```bash
ssh rb3434w@staff@100.96.122.39 "ls -t ~/CNN-age-inference/logs/age-infer-ddp-*.log 2>/dev/null | head -1 | xargs tail -100"
```
Parse and summarise any metrics (MAE, RMSE, val_loss, epoch, AUC, errors) found in the log output.

### View a specific log
```bash
ssh rb3434w@staff@100.96.122.39 "tail -100 ~/CNN-age-inference/logs/age-infer-ddp-<jobid>.log"
```

### Submit a training job (default: EfficientNet-V2-M, 5 folds, 480px)
```bash
ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && sbatch submit_distributed.slurm"
```

Key env vars for the SLURM job:
| Variable | Example | Description |
|---|---|---|
| `MODELS` | `v2_m` | Backbone — b0-b7, v2_{s,m,l}, convnext_{tiny,small,...}, vit_tiny_384 |
| `KFOLDS` | `5` | Number of CV folds (set to 1 to disable k-fold) |
| `EPOCHS` | `240` | Max training epochs |
| `PATIENCE` | `10` | Early stopping patience |
| `BATCH_SIZE` | `8` | Per-GPU batch size |
| `IMG_SIZE` | `480` | Input resolution |
| `LR` | `2e-4` | Learning rate |
| `SEED` | `42` | Random seed |
| `TEST_SPLIT_FILE` | `${OUTPUT_ROOT}/test_users.json` | Held-out test split (auto-created if missing) |
| `TEST_SPLIT_SIZE` | `0.15` | Fraction of users for held-out test set |
| `RUN_TEST_EVAL` | `1` | Run evaluate_test.py after training |
| `OUTPUT_ROOT` | `~/CNN-age-inference/runs` | Root output directory |
| `DATA_ROOT` | `~/HandsDatasets` | Dataset root |

Example — custom backbone and size:
```bash
ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && \
  MODELS=v2_l IMG_SIZE=600 EPOCHS=120 PATIENCE=15 \
  sbatch submit_distributed.slurm"
```

### Cancel a job
```bash
ssh rb3434w@staff@100.96.122.39 "scancel <jobid>"
```

### Check disk / output artifacts
```bash
ssh rb3434w@staff@100.96.122.39 "du -sh ~/CNN-age-inference/runs/ && ls ~/CNN-age-inference/runs/ | tail -20"
```

### Check test evaluation results
```bash
ssh rb3434w@staff@100.96.122.39 "cat ~/CNN-age-inference/runs/<run_name>/test_eval/test_summary.csv"
```

### Check k-fold aggregation results
```bash
ssh rb3434w@staff@100.96.122.39 "ls ~/CNN-age-inference/runs/<run_name>/aggregate/"
```

---

## Reporting results from logs

When reading logs, look for and summarise:
- Epoch number and total epochs
- `train_mae`, `val_mae`, `train_rmse`, `val_rmse`
- `val_loss` (used for early stopping / best model selection)
- AUC values from age-gate evaluation
- Fold index (fold N/K)
- Any CUDA errors, OOM events, or Python tracebacks
- Job completion status (COMPLETED / FAILED / TIMEOUT)

Present findings in a concise table or bullet list. Flag any errors prominently.

---

## Notes
- Partition: `gpu-beast`, 8 GPUs per node
- Log files: `~/CNN-age-inference/logs/age-infer-ddp-<jobid>.log`
- Outputs: `~/CNN-age-inference/runs/<run_name>/`
- Best model checkpoint per fold: `fold_<n>/<model_key>_age_regressor_ddp.pth`
- Held-out test split: `~/CNN-age-inference/runs/test_users.json` (auto-created by SLURM, never delete)
- Dataset root: `~/HandsDatasets`
- Python environment: `.venv` inside project root (activated automatically by the SLURM script)

## Running Python commands directly over SSH

The SLURM script activates `.venv` automatically. For direct interactive commands:

```bash
ssh rb3434w@staff@100.96.122.39 "cd ~/CNN-age-inference && source .venv/bin/activate && <your command>"
```

> Note: `sbatch` commands do **not** need the venv activated on the login node — the SLURM script handles it.
