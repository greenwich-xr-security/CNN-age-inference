# Age-Reweighted EfficientNet-V2-S HPC Jobs

Date: 2026-07-02

This note records the rerun of the two EfficientNet-V2-S age-regression experiments with inverse-frequency age-loss weighting enabled. The rerun was requested because the previous multitasking experiments did not enable age reweighting (`resolved_age_reweight_loss=0`), meaning rare age groups contributed less to the optimization objective than common age groups.

## Purpose

The goal is to repeat the two existing EfficientNet-V2-S comparisons while correcting for age imbalance in the training loss:

1. NLL-only probabilistic age regression.
2. NLL plus age-assurance BCE and unified severity loss.

Both jobs use the same shared held-out test split as the previous runs so the new results remain comparable between the two rerun conditions.

## Final Submitted Jobs

| Job ID | Run name | Purpose | Status at submission check |
| --- | --- | --- | --- |
| `1049170` | `v2s_nll_age_assurance_age_reweight_shared20_r3` | NLL age regression plus age-assurance BCE and severity losses, with age-reweighted training loss | Running fold 0 |
| `1049171` | `v2s_nll_only_age_reweight_shared20_r3` | NLL-only probabilistic age regression, with age-reweighted training loss | Running fold 0 |

Remote output root:

```text
/home/rb3434w/CNN-age-inference/runs/multitasking
```

Expected final run folders:

```text
/home/rb3434w/CNN-age-inference/runs/multitasking/v2s_nll_age_assurance_age_reweight_shared20_r3
/home/rb3434w/CNN-age-inference/runs/multitasking/v2s_nll_only_age_reweight_shared20_r3
```

Local copy target after completion:

```text
C:\Users\Staff\OneDrive - University of Greenwich\slurm_runs\CNN age inference\multitasking
```

## Shared Configuration

Both final jobs were launched with:

```text
MODEL=v2_s
MODELS=v2_s
IMG_SIZE=384
KFOLDS=5
SEED=42
FOLD_SEED=42
BATCH_SIZE=64
NUM_WORKERS=4
MAX_SAMPLES_PER_USER=20
INCLUDE_HANDRGBD=1
INCLUDE_PROLIFIC=1
INCLUDE_HAGRID=0
INCLUDE_PRIMARY=0
INCLUDE_ARCHIVE=0
INCLUDE_LUCID=0
TEST_SPLIT_FILE=/home/rb3434w/CNN-age-inference/runs/multitask_v2s_shared_test_users.json
RUN_TEST_EVAL=1
RUN_KFOLD_AGG=1
FOLD_OVERWRITE=1
LOSS_WEIGHT_NLL=1.0
LOSS_WEIGHT_MSE=0.0
LOSS_WEIGHT_MAE=0.0
LOSS_WEIGHT_SPREAD=0.0
LOSS_WEIGHT_NORMALS=0.0
LOSS_WEIGHT_EMBED_VAR=0.0
LOSS_WEIGHT_EMBED_CONTRAST=0.0
NORMALS_AUX=0
NORMALS_PRIVILEGED=0
EVAL_AGE_GATE_MODE=age_threshold
```

SLURM resources:

```text
#SBATCH --gres=gpu:1
--cpus-per-task=8
--mem=64G
```

The script confirmed:

```text
Model: EfficientNet-V2_S
Image size: 384
Per-rank batch size: 64
World size: 1
```

## Age Reweighting

Age reweighting was enabled for both jobs:

```text
AGE_REWEIGHT=1
AGE_WEIGHT_EPS=1.0
AGE_WEIGHT_POWER=1.0
AGE_WEIGHT_MIN=0.25
AGE_WEIGHT_MAX=4.0
```

The log confirmed the resolved weights:

```text
[loss] Age reweighting enabled (min=0.423, max=4.000).
```

The implemented weighting uses rounded integer ages:

```text
w(a) = (count(round(a)) + eps)^(-power)
```

Weights are normalized to mean 1 and clipped to the configured range. This reduces domination by common ages while preventing very rare ages from receiving unstable, excessively large weights.

## Condition-Specific Losses

Job `1049170` uses:

```text
LOSS_WEIGHT_AGE_ASSURANCE_BCE=0.25
LOSS_WEIGHT_AGE_ASSURANCE_SEVERITY=0.50
AGE_ASSURANCE_THRESHOLD=18.0
AGE_ASSURANCE_SEVERITY_RADIUS=5.0
```

Job `1049171` uses:

```text
LOSS_WEIGHT_AGE_ASSURANCE_BCE=0.0
LOSS_WEIGHT_AGE_ASSURANCE_SEVERITY=0.0
AGE_ASSURANCE_THRESHOLD=18.0
AGE_ASSURANCE_SEVERITY_RADIUS=5.0
```

## Canceled Attempts

The following jobs were intentionally canceled and their partial output folders removed:

| Job IDs | Reason |
| --- | --- |
| `1049166`, `1049167` | Submitted with `MODEL_NAME=v2_s`, but the SLURM script uses `MODEL`/`MODELS`, so they launched the default `v2_m`. |
| `1049168`, `1049169` | Correct model, but both jobs inherited the same default `MASTER_PORT=29500` while running concurrently on the same node. They were canceled and resubmitted with distinct ports. |

The final jobs use distinct rendezvous ports:

```text
1049170: MASTER_PORT=29511
1049171: MASTER_PORT=29512
```

## Cleanup Performed Before Rerun

The previous non-age-reweighted results were deleted locally and on the HPC before launching the new runs:

```text
v2s_nll_age_assurance_shared20_r3
v2s_nll_only_shared20_r3
```

Partial folders from the canceled age-reweighted attempts were also removed before final submission.
