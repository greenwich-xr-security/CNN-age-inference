#!/bin/bash
set -euo pipefail

cd "${HOME}/CNN-age-inference"

PROJECT_ROOT="${HOME}/CNN-age-inference"
DATA_ROOT="/home/rb3434w/HandsDatasets"
REAL_FOLD_FILE="${PROJECT_ROOT}/splits/folds_k5_uncapped_test20_real_seed42.json"
SYN2_FOLD_FILE="${PROJECT_ROOT}/splits/folds_k5_uncapped_test20_real_synthetic_seed42.json"
TEST_SPLIT_FILE="${PROJECT_ROOT}/splits/test_users_uncapped_20pct_seed42.json"
SEED_VALUE=46

submit_train() {
  local arm="$1"
  local run_name="$2"
  local fold_file="$3"
  local include_syn2="$4"
  local port="$5"
  local job_name="minpair-s${SEED_VALUE}-${arm}"

  sbatch --parsable \
    --job-name="${job_name}" \
    --partition=gpu-beast \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --cpus-per-task=16 \
    --mem=64G \
    --output="logs/${job_name}-%j.log" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATA_ROOT="${DATA_ROOT}",RUN_NAME="${run_name}",MODELS=v2_s,SEED="${SEED_VALUE}",FOLD_SEED=42,IMG_SIZE=384,BATCH_SIZE=16,LR=2e-4,WEIGHT_DECAY=0.2,EPOCHS=240,PATIENCE=10,NUM_WORKERS=2,KFOLDS=5,FOLD_FILE_OVERRIDE="${fold_file}",TEST_SPLIT_FILE="${TEST_SPLIT_FILE}",INCLUDE_HANDRGBD=1,INCLUDE_PROLIFIC=1,INCLUDE_LUCID=0,INCLUDE_HAGRID=0,INCLUDE_PRIMARY=0,INCLUDE_ARCHIVE=0,INCLUDE_SYNTHETIC_DORSAL=0,INCLUDE_SYNTHETIC_DORSAL2="${include_syn2}",MAX_SAMPLES_PER_USER=0,MAX_SAMPLES_PER_AGE_BIN=0,USER_GROUP_SIZES=1,AGG_SIZES=1,AGG_SEED=42,EMBED_DIM=128,LOSS_WEIGHT_NLL=1,LOSS_WEIGHT_CRPS=0,LOSS_WEIGHT_MSE=0,LOSS_WEIGHT_MAE=0,LOSS_WEIGHT_SPREAD=0,LOSS_WEIGHT_EMBED_VAR=0,LOSS_WEIGHT_EMBED_CONTRAST=0,NORMALS_AUX=0,EVAL_AGE_GATE_MODE=age_threshold,AGE_GATE_THRESHOLD_MIN=10,AGE_GATE_THRESHOLD_MAX=30,MASTER_PORT="${port}",RUN_KFOLD_AGG=1 \
    submit_distributed.slurm
}

submit_finetune() {
  local arm="$1"
  local run_name="$2"
  local init_root="$3"
  local dependency="$4"
  local port="$5"
  local job_name="minpair-s${SEED_VALUE}-${arm}-ft"

  sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --job-name="${job_name}" \
    --partition=gpu-beast \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --cpus-per-task=16 \
    --mem=64G \
    --output="logs/${job_name}-%j.log" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATA_ROOT="${DATA_ROOT}",RUN_NAME="${run_name}",INIT_CHECKPOINT_ROOT="${init_root}",MODELS=v2_s,SEED="${SEED_VALUE}",FOLD_SEED=42,IMG_SIZE=384,BATCH_SIZE=16,LR=2e-5,WEIGHT_DECAY=0.2,EPOCHS=120,PATIENCE=10,NUM_WORKERS=2,KFOLDS=5,FOLD_FILE_OVERRIDE="${REAL_FOLD_FILE}",TEST_SPLIT_FILE="${TEST_SPLIT_FILE}",INCLUDE_HANDRGBD=1,INCLUDE_PROLIFIC=1,INCLUDE_LUCID=0,INCLUDE_HAGRID=0,INCLUDE_PRIMARY=0,INCLUDE_ARCHIVE=0,INCLUDE_SYNTHETIC_DORSAL=0,INCLUDE_SYNTHETIC_DORSAL2=0,MAX_SAMPLES_PER_USER=0,MAX_SAMPLES_PER_AGE_BIN=0,USER_GROUP_SIZES=1,AGG_SIZES=1,AGG_SEED=42,EMBED_DIM=128,LOSS_WEIGHT_NLL=1,LOSS_WEIGHT_CRPS=0,LOSS_WEIGHT_MSE=0,LOSS_WEIGHT_MAE=0,LOSS_WEIGHT_SPREAD=0,LOSS_WEIGHT_EMBED_VAR=0,LOSS_WEIGHT_EMBED_CONTRAST=0,NORMALS_AUX=0,EVAL_AGE_GATE_MODE=age_threshold,AGE_GATE_THRESHOLD_MIN=10,AGE_GATE_THRESHOLD_MAX=30,MASTER_PORT="${port}",RUN_KFOLD_AGG=1 \
    submit_distributed.slurm
}

real_run="minpair_seed46_real_nll_v2s_384"
syn_run="minpair_seed46_realsyn2_nll_v2s_384"
real_ft_run="minpair_seed46_real_from_real_lr2e5_v2s_384"
syn_ft_run="minpair_seed46_real_from_syn2_lr2e5_v2s_384"

real_job="$(submit_train real "${real_run}" "${REAL_FOLD_FILE}" 0 29760)"
syn_job="$(submit_train syn2 "${syn_run}" "${SYN2_FOLD_FILE}" 1 29761)"
real_ft_job="$(submit_finetune real "${real_ft_run}" "${PROJECT_ROOT}/runs/${real_run}" "${real_job}" 29762)"
syn_ft_job="$(submit_finetune syn2 "${syn_ft_run}" "${PROJECT_ROOT}/runs/${syn_run}" "${syn_job}" 29763)"

echo "seed=46 real_train=${real_job} syn2_train=${syn_job} real_ft=${real_ft_job} syn2_ft=${syn_ft_job}"
