#!/bin/bash
set -euo pipefail

cd "${HOME}/CNN-age-inference"

PROJECT_ROOT="${HOME}/CNN-age-inference"
DATA_ROOT="/home/rb3434w/HandsDatasets"
REAL_FOLD_FILE="${PROJECT_ROOT}/splits/folds_k5_uncapped_test20_real_seed42.json"
SYN2_FOLD_FILE="${PROJECT_ROOT}/splits/folds_k5_uncapped_test20_real_synthetic_seed42.json"
TEST_SPLIT_FILE="${PROJECT_ROOT}/splits/test_users_uncapped_20pct_seed42.json"

submit_train() {
  local seed_value="$1"
  local arm="$2"
  local run_name="$3"
  local fold_file="$4"
  local include_syn2="$5"
  local port="$6"
  local job_name="minpair-s${seed_value}-${arm}"

  sbatch --parsable \
    --job-name="${job_name}" \
    --partition=gpu-beast \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --cpus-per-task=16 \
    --mem=64G \
    --output="logs/${job_name}-%j.log" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATA_ROOT="${DATA_ROOT}",RUN_NAME="${run_name}",MODELS=v2_s,SEED="${seed_value}",FOLD_SEED=42,IMG_SIZE=384,BATCH_SIZE=16,LR=2e-4,WEIGHT_DECAY=0.2,EPOCHS=240,PATIENCE=10,NUM_WORKERS=2,KFOLDS=5,FOLD_FILE_OVERRIDE="${fold_file}",TEST_SPLIT_FILE="${TEST_SPLIT_FILE}",INCLUDE_HANDRGBD=1,INCLUDE_PROLIFIC=1,INCLUDE_LUCID=0,INCLUDE_HAGRID=0,INCLUDE_PRIMARY=0,INCLUDE_ARCHIVE=0,INCLUDE_SYNTHETIC_DORSAL=0,INCLUDE_SYNTHETIC_DORSAL2="${include_syn2}",MAX_SAMPLES_PER_USER=0,MAX_SAMPLES_PER_AGE_BIN=0,USER_GROUP_SIZES=1,AGG_SIZES=1,AGG_SEED=42,EMBED_DIM=128,LOSS_WEIGHT_NLL=1,LOSS_WEIGHT_CRPS=0,LOSS_WEIGHT_MSE=0,LOSS_WEIGHT_MAE=0,LOSS_WEIGHT_SPREAD=0,LOSS_WEIGHT_EMBED_VAR=0,LOSS_WEIGHT_EMBED_CONTRAST=0,NORMALS_AUX=0,EVAL_AGE_GATE_MODE=age_threshold,AGE_GATE_THRESHOLD_MIN=10,AGE_GATE_THRESHOLD_MAX=30,MASTER_PORT="${port}",RUN_KFOLD_AGG=1 \
    submit_distributed.slurm
}

submit_finetune() {
  local seed_value="$1"
  local arm="$2"
  local run_name="$3"
  local init_root="$4"
  local dependency="$5"
  local port="$6"
  local job_name="minpair-s${seed_value}-${arm}-ft"

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
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATA_ROOT="${DATA_ROOT}",RUN_NAME="${run_name}",INIT_CHECKPOINT_ROOT="${init_root}",MODELS=v2_s,SEED="${seed_value}",FOLD_SEED=42,IMG_SIZE=384,BATCH_SIZE=16,LR=2e-5,WEIGHT_DECAY=0.2,EPOCHS=120,PATIENCE=10,NUM_WORKERS=2,KFOLDS=5,FOLD_FILE_OVERRIDE="${REAL_FOLD_FILE}",TEST_SPLIT_FILE="${TEST_SPLIT_FILE}",INCLUDE_HANDRGBD=1,INCLUDE_PROLIFIC=1,INCLUDE_LUCID=0,INCLUDE_HAGRID=0,INCLUDE_PRIMARY=0,INCLUDE_ARCHIVE=0,INCLUDE_SYNTHETIC_DORSAL=0,INCLUDE_SYNTHETIC_DORSAL2=0,MAX_SAMPLES_PER_USER=0,MAX_SAMPLES_PER_AGE_BIN=0,USER_GROUP_SIZES=1,AGG_SIZES=1,AGG_SEED=42,EMBED_DIM=128,LOSS_WEIGHT_NLL=1,LOSS_WEIGHT_CRPS=0,LOSS_WEIGHT_MSE=0,LOSS_WEIGHT_MAE=0,LOSS_WEIGHT_SPREAD=0,LOSS_WEIGHT_EMBED_VAR=0,LOSS_WEIGHT_EMBED_CONTRAST=0,NORMALS_AUX=0,EVAL_AGE_GATE_MODE=age_threshold,AGE_GATE_THRESHOLD_MIN=10,AGE_GATE_THRESHOLD_MAX=30,MASTER_PORT="${port}",RUN_KFOLD_AGG=1 \
    submit_distributed.slurm
}

submit_seed() {
  local seed_value="$1"
  local base_port="$2"

  local real_run="minpair_seed${seed_value}_real_nll_v2s_384"
  local syn_run="minpair_seed${seed_value}_realsyn2_nll_v2s_384"
  local real_ft_run="minpair_seed${seed_value}_real_from_real_lr2e5_v2s_384"
  local syn_ft_run="minpair_seed${seed_value}_real_from_syn2_lr2e5_v2s_384"

  local real_job
  local syn_job
  local real_ft_job
  local syn_ft_job

  real_job="$(submit_train "${seed_value}" real "${real_run}" "${REAL_FOLD_FILE}" 0 "${base_port}")"
  syn_job="$(submit_train "${seed_value}" syn2 "${syn_run}" "${SYN2_FOLD_FILE}" 1 "$((base_port + 1))")"
  real_ft_job="$(submit_finetune "${seed_value}" real "${real_ft_run}" "${PROJECT_ROOT}/runs/${real_run}" "${real_job}" "$((base_port + 2))")"
  syn_ft_job="$(submit_finetune "${seed_value}" syn2 "${syn_ft_run}" "${PROJECT_ROOT}/runs/${syn_run}" "${syn_job}" "$((base_port + 3))")"

  echo "seed=${seed_value} real_train=${real_job} syn2_train=${syn_job} real_ft=${real_ft_job} syn2_ft=${syn_ft_job}"
}

submit_seed 47 29770
submit_seed 48 29780
