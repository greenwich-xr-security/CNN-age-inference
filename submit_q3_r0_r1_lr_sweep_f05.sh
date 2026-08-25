#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
LABEL_FRACTION_FILE="${LABEL_FRACTION_FILE:-${PROJECT_ROOT}/splits/q3_real_label_fractions_seed42.json}"
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-${PROJECT_ROOT}/splits/test_users_uncapped_20pct_seed42.json}"
FRACTION="0.05"
TAG="f05"

# Q3 R0/R1 LR sweep at 5% real labels. Isolates LR only: every other knob
# matches the original R0/R1 sweep (1050948, 1050953), which itself equals
# submit_distributed.slurm's defaults (EPOCHS=240, PATIENCE=10,
# WEIGHT_DECAY=0.2, one gpu-beast node/4 GPUs, batch 16/GPU). The existing
# LR=2e-4 cells (1050948 R0, 1050953 R1) are reused as the third sweep point;
# this script only launches the new LR=2e-5 and LR=1e-3 cells.

declare -a ARMS=("r0" "r1")
declare -A NO_IMAGENET=([r0]=0 [r1]=1)
declare -a LRS=("2e-5" "1e-3")
declare -a LR_TAGS=("lr2e5" "lr1e3")
declare -a PORTS=("29750" "29751" "29752" "29753")

IDX=0
for ARM in "${ARMS[@]}"; do
  for LR_IDX in "${!LRS[@]}"; do
    LR="${LRS[$LR_IDX]}"
    LR_TAG="${LR_TAGS[$LR_IDX]}"
    PORT="${PORTS[$IDX]}"
    RUN_NAME="q3_${ARM}_realfrac_${TAG}_${LR_TAG}_seed42_v2s_384"
    JOB_NAME="q3-${ARM}-${LR_TAG}"

    JOB_ID="$(
      sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --partition=gpu-beast \
        --gres=gpu:4 \
        --cpus-per-task=16 \
        --mem=64G \
        --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",RUN_NAME="${RUN_NAME}",LABEL_FRACTION_FILE="${LABEL_FRACTION_FILE}",LABEL_FRACTION="${FRACTION}",TEST_SPLIT_FILE="${TEST_SPLIT_FILE}",MODELS=v2_s,BATCH_SIZE=16,LR="${LR}",IMG_SIZE=384,SEED=42,NO_IMAGENET_PRETRAINED="${NO_IMAGENET[$ARM]}",MAX_SAMPLES_PER_USER=0,MAX_SAMPLES_PER_AGE_BIN=0,LOSS_WEIGHT_NLL=1,LOSS_WEIGHT_CRPS=0,LOSS_WEIGHT_MSE=0,LOSS_WEIGHT_MAE=0,LOSS_WEIGHT_SPREAD=0,LOSS_WEIGHT_EMBED_VAR=0,LOSS_WEIGHT_EMBED_CONTRAST=0,USER_GROUP_SIZES=1,AGG_SIZES=1,AGG_SEED=42,EMBED_DIM=128,INCLUDE_HANDRGBD=1,INCLUDE_PROLIFIC=1,INCLUDE_HAGRID=0,INCLUDE_SYNTHETIC_DORSAL=0,INCLUDE_SYNTHETIC_DORSAL2=0,INCLUDE_PRIMARY=0,INCLUDE_ARCHIVE=0,INCLUDE_LUCID=0,RUN_KFOLD_AGG=1,MASTER_PORT="${PORT}" \
        "${PROJECT_ROOT}/submit_distributed.slurm"
    )"
    echo "${ARM} ${LR} ${JOB_ID} ${RUN_NAME}"
    IDX=$((IDX + 1))
  done
done
