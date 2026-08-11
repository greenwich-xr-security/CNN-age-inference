#!/bin/bash
set -euo pipefail

cd "${HOME}/CNN-age-inference"

PROJECT_ROOT="${HOME}/CNN-age-inference"
DATA_ROOT="/home/rb3434w/HandsDatasets"
SYN2_FOLD_FILE="${PROJECT_ROOT}/splits/folds_k5_synthetic2_seed42.json"
SYN2_TEST_SPLIT_FILE="${PROJECT_ROOT}/splits/test_users_synthetic2_20pct_seed42.json"
REAL_FOLD_FILE="${PROJECT_ROOT}/splits/folds_k5_uncapped_test20_real_seed42.json"
REAL_TEST_SPLIT_FILE="${PROJECT_ROOT}/splits/test_users_uncapped_20pct_seed42.json"

submit_sage_checkpoint() {
  local seed_value="$1"
  local run_name="$2"
  local port="$3"
  local job_name="q4-sage-s${seed_value}-ckpt"

  sbatch --parsable \
    --job-name="${job_name}" \
    --partition=gpu-beast \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --cpus-per-task=16 \
    --mem=64G \
    --output="logs/${job_name}-%j.log" \
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATA_ROOT="${DATA_ROOT}",RUN_NAME="${run_name}",SEED_VALUE="${seed_value}",MODEL_NAME=v2_s,SOURCE_FOLD_FILE="${SYN2_FOLD_FILE}",TEST_SPLIT_FILE="${SYN2_TEST_SPLIT_FILE}",MASTER_PORT="${port}" \
    submit_q4_sage_checkpoint.slurm
}

submit_sage_finetune() {
  local seed_value="$1"
  local run_name="$2"
  local init_root="$3"
  local dependency="$4"
  local port="$5"
  local job_name="q4-sage-s${seed_value}-ft"

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
    --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",DATA_ROOT="${DATA_ROOT}",RUN_NAME="${run_name}",INIT_CHECKPOINT_ROOT="${init_root}",MODELS=v2_s,SEED="${seed_value}",FOLD_SEED=42,IMG_SIZE=384,BATCH_SIZE=16,LR=2e-5,WEIGHT_DECAY=0.2,EPOCHS=120,PATIENCE=10,NUM_WORKERS=2,KFOLDS=5,FOLD_FILE_OVERRIDE="${REAL_FOLD_FILE}",TEST_SPLIT_FILE="${REAL_TEST_SPLIT_FILE}",INCLUDE_HANDRGBD=1,INCLUDE_PROLIFIC=1,INCLUDE_LUCID=0,INCLUDE_HAGRID=0,INCLUDE_PRIMARY=0,INCLUDE_ARCHIVE=0,INCLUDE_SYNTHETIC_DORSAL=0,INCLUDE_SYNTHETIC_DORSAL2=0,MAX_SAMPLES_PER_USER=0,MAX_SAMPLES_PER_AGE_BIN=0,USER_GROUP_SIZES=1,AGG_SIZES=1,AGG_SEED=42,EMBED_DIM=128,LOSS_WEIGHT_NLL=1,LOSS_WEIGHT_CRPS=0,LOSS_WEIGHT_MSE=0,LOSS_WEIGHT_MAE=0,LOSS_WEIGHT_SPREAD=0,LOSS_WEIGHT_EMBED_VAR=0,LOSS_WEIGHT_EMBED_CONTRAST=0,NORMALS_AUX=0,EVAL_AGE_GATE_MODE=age_threshold,AGE_GATE_THRESHOLD_MIN=10,AGE_GATE_THRESHOLD_MAX=30,MASTER_PORT="${port}",RUN_KFOLD_AGG=1 \
    submit_distributed.slurm
}

submit_seed() {
  local seed_value="$1"
  local base_port="$2"
  local ckpt_run="q4_sage_seed${seed_value}_synthetic2_only_nll_v2s_384"
  local ft_run="q4_sage_seed${seed_value}_real_ft_lr2e5_v2s_384"
  local ckpt_job
  local ft_job

  ckpt_job="$(submit_sage_checkpoint "${seed_value}" "${ckpt_run}" "${base_port}")"
  ft_job="$(submit_sage_finetune "${seed_value}" "${ft_run}" "${PROJECT_ROOT}/runs/${ckpt_run}" "${ckpt_job}" "$((base_port + 1))")"

  echo "seed=${seed_value} sage_checkpoint=${ckpt_job} sage_finetune=${ft_job}"
}

submit_seed 43 29830
submit_seed 44 29840
submit_seed 45 29850
submit_seed 46 29860
submit_seed 47 29870
submit_seed 48 29880
