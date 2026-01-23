#!/bin/bash
#SBATCH --job-name=age-infer-ddp          # Descriptive job name
#SBATCH --nodes=1                         # Single node on gpu-beast
#SBATCH --ntasks-per-node=1               # torchrun will launch local ranks
#SBATCH --partition=gpu-beast             # GPU partition
#SBATCH --cpus-per-task=16                # CPU cores available to each task
#SBATCH --output=logs/%x-%j.log           # Stdout/stderr log file
#SBATCH --mail-type=ALL                   # Email notifications
#SBATCH --mail-user=rb3434w@gre.ac.uk     # TODO: change to preferred email

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"

mkdir -p "${OUTPUT_ROOT}" logs

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# Slurm rendezvous configuration (single node -> hostname works)
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${SLURM_JOB_NUM_NODES:-1}
# gpu-beast does not export SLURM_GPUS_ON_NODE by default; assume 8 unless overridden.
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE_OVERRIDE:-8}}}

export MASTER_ADDR MASTER_PORT

SEEDS="${SEEDS:-${SEED:-32}}"              # space-separated seeds (or a single seed) 32 33 34 35 52 53 54 55 120 123 124 125 200 203 204 205
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LR="${LR:-3e-4}"
EPOCHS="${EPOCHS:-240}"
LOSS_WEIGHT_NLL="${LOSS_WEIGHT_NLL:-0.6}"
LOSS_WEIGHT_MSE="${LOSS_WEIGHT_MSE:-0.9}"
LOSS_WEIGHT_MAE="${LOSS_WEIGHT_MAE:-0.4}"
LOSS_WEIGHT_SPREAD="${LOSS_WEIGHT_SPREAD:-0.3}"

USER_GROUP_SIZES="${USER_GROUP_SIZES:-2}"
PATIENCE="${PATIENCE:-30}"
MODELS="${MODELS:-${MODEL:-b2}}" # choices: b0-b7, v2_{s,m,l}, convnext_{tiny,small,base,large,xlarge}, aliases: cnt,cns,cnb,cnl,cnx
AGG_SIZES="${AGG_SIZES:-${EVAL_AGG_SIZES:-1 2 3 4}}"   # space-separated list, e.g., "1 2 3"
AGG_SEED="${AGG_SEED:-${EVAL_AGG_SEED:-${SEED}}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
# K-fold settings (set KFOLDS > 1 to enable).
KFOLDS="${KFOLDS:-5}"
FOLD_SEED="${FOLD_SEED:-${SEED}}"
RUN_KFOLD_AGG="${RUN_KFOLD_AGG:-1}"
FOLD_OVERWRITE="${FOLD_OVERWRITE:-0}"

for USER_GROUP_SIZE in ${USER_GROUP_SIZES}; do
  for MODEL_NAME in ${MODELS}; do
    for SEED_VAL in ${SEEDS}; do
      AGG_SEED_EFF="${AGG_SEED:-${SEED_VAL}}"
      FOLDER_NAME="${MODEL_NAME}_bs${BATCH_SIZE}_seed${SEED_VAL}_nll${LOSS_WEIGHT_NLL}_mse${LOSS_WEIGHT_MSE}_mae${LOSS_WEIGHT_MAE}_spread${LOSS_WEIGHT_SPREAD}_ug${USER_GROUP_SIZE}"
      if [[ -n "${RUN_SUFFIX}" ]]; then
        FOLDER_NAME="${FOLDER_NAME}_${RUN_SUFFIX}"
      fi
      FOLD_FILE="${OUTPUT_ROOT}/${FOLDER_NAME}/folds_k${KFOLDS}.json"

      if [[ "${KFOLDS}" -gt 1 ]]; then
        if [[ ! -f "${FOLD_FILE}" || "${FOLD_OVERWRITE}" -gt 0 ]]; then
          python "${PROJECT_ROOT}/make_kfold_splits.py" \
            --data-root "${DATA_ROOT}" \
            --k "${KFOLDS}" \
            --seed "${FOLD_SEED}" \
            --out-file "${FOLD_FILE}" \
            $( [[ "${FOLD_OVERWRITE}" -gt 0 ]] && echo "--overwrite" )
        fi
        for ((FOLD_IDX=0; FOLD_IDX<KFOLDS; FOLD_IDX++)); do
          RUN_OUT="${OUTPUT_ROOT}/${FOLDER_NAME}/fold_${FOLD_IDX}"
          echo ">>> Starting distributed age inference training for ${MODEL_NAME} seed=${SEED_VAL} fold=${FOLD_IDX} (output: ${RUN_OUT})"
          torchrun \
            --nnodes="${NNODES}" \
            --nproc_per_node="${GPUS_PER_NODE}" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            "${PROJECT_ROOT}/train_distributed.py" \
            --data-root "${DATA_ROOT}" \
            --output-dir "${RUN_OUT}" \
            --model "${MODEL_NAME}" \
            --batch-size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --lr "${LR}" \
            --num-workers "${NUM_WORKERS}" \
            --seed "${SEED_VAL}" \
            --patience "${PATIENCE}" \
            --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
            --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
            --loss-weight-mae "${LOSS_WEIGHT_MAE}" \
            --loss-weight-spread "${LOSS_WEIGHT_SPREAD}" \
            --user-group-size "${USER_GROUP_SIZE}" \
            --eval-aggregation-sizes ${AGG_SIZES} \
            --eval-aggregation-seed "${AGG_SEED_EFF}" \
            --fold-file "${FOLD_FILE}" \
            --fold-index "${FOLD_IDX}" \
            --dist-backend nccl
          echo ">>> Training for ${MODEL_NAME} seed=${SEED_VAL} fold=${FOLD_IDX} complete. Artifacts in ${RUN_OUT}"
        done

        if [[ "${RUN_KFOLD_AGG}" -gt 0 ]]; then
          AGG_OUT="${OUTPUT_ROOT}/${FOLDER_NAME}"
          python "${PROJECT_ROOT}/aggregate_kfold.py" \
            --kfold-root "${AGG_OUT}" \
            --output-dir "${AGG_OUT}" \
            --group-sizes ${AGG_SIZES}
        fi
      else
        RUN_OUT="${OUTPUT_ROOT}/${FOLDER_NAME}"
        echo ">>> Starting distributed age inference training for ${MODEL_NAME} seed=${SEED_VAL} (output: ${RUN_OUT})"
        torchrun \
          --nnodes="${NNODES}" \
          --nproc_per_node="${GPUS_PER_NODE}" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
          "${PROJECT_ROOT}/train_distributed.py" \
          --data-root "${DATA_ROOT}" \
          --output-dir "${RUN_OUT}" \
          --model "${MODEL_NAME}" \
          --batch-size "${BATCH_SIZE}" \
          --epochs "${EPOCHS}" \
          --lr "${LR}" \
          --num-workers "${NUM_WORKERS}" \
          --seed "${SEED_VAL}" \
          --patience "${PATIENCE}" \
          --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
          --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
          --loss-weight-mae "${LOSS_WEIGHT_MAE}" \
          --loss-weight-spread "${LOSS_WEIGHT_SPREAD}" \
          --user-group-size "${USER_GROUP_SIZE}" \
          --eval-aggregation-sizes ${AGG_SIZES} \
          --eval-aggregation-seed "${AGG_SEED_EFF}" \
          --dist-backend nccl
        echo ">>> Training for ${MODEL_NAME} seed=${SEED_VAL} complete. Artifacts in ${RUN_OUT}"
      fi
    done
  done
done
