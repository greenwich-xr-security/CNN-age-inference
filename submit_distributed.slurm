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

SEEDS="${SEEDS:-${SEED:-32 53}}"              # space-separated seeds (or a single seed) 32 33 34 35 52 53 54 55 120 123 124 125 200 203 204 205
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
DINO_EPOCHS="${DINO_EPOCHS:-200}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-${BATCH_SIZE}}"
DINO_PAIR_SAME_HAND_PROB="${DINO_PAIR_SAME_HAND_PROB:-0.5}"
DINO_NUM_LOCAL_CROPS="${DINO_NUM_LOCAL_CROPS:-4}"
DINO_AUG_RAMP_FRACTION="${DINO_AUG_RAMP_FRACTION:-0.2}"
DINO_NUM_WORKERS="${DINO_NUM_WORKERS:-8}"
DINO_OUTPUT_ROOT="${DINO_OUTPUT_ROOT:-${OUTPUT_ROOT}/dino}"
LOSS_WEIGHT_NLL="${LOSS_WEIGHT_NLL:-0.65}"
LOSS_WEIGHT_MSE="${LOSS_WEIGHT_MSE:-0.15}"
LOSS_WEIGHT_MAE="${LOSS_WEIGHT_MAE:-0.15}"
LOSS_WEIGHT_SPREAD="${LOSS_WEIGHT_SPREAD:-0.15}"
USER_GROUP_SIZES="2 4 8"
PATIENCE="${PATIENCE:-60}"
MODELS="${MODELS:-${MODEL:-b2}}" # choices: b0-b7, v2_{s,m,l}, convnext_{tiny,small,base,large,xlarge}, aliases: cnt,cns,cnb,cnl,cnx
STRATIFICATION="${STRATIFICATION:-no}"  # choices: no, minorAdults, bins
AGG_SIZES="${AGG_SIZES:-${EVAL_AGG_SIZES:- 1 2 3 4}}"   # space-separated list, e.g., "1 2 3"
AGG_SEED="${AGG_SEED:-${EVAL_AGG_SEED:-${SEED}}}"

for USER_GROUP_SIZE in ${USER_GROUP_SIZES}; do
  for MODEL_NAME in ${MODELS}; do
    for SEED_VAL in ${SEEDS}; do
      AGG_SEED_EFF="${AGG_SEED:-${SEED_VAL}}"
      BASELINE_NAME="${MODEL_NAME}_bs${BATCH_SIZE}_seed${SEED_VAL}_nll${LOSS_WEIGHT_NLL}_mse${LOSS_WEIGHT_MSE}_mae${LOSS_WEIGHT_MAE}_spread${LOSS_WEIGHT_SPREAD}_ug${USER_GROUP_SIZE}_strat${STRATIFICATION}_baseline"
      BASELINE_OUT="${OUTPUT_ROOT}/${BASELINE_NAME}"
      echo ">>> Starting BASELINE training for ${MODEL_NAME} seed=${SEED_VAL} -> ${BASELINE_OUT}"
      torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${PROJECT_ROOT}/train_distributed.py" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${BASELINE_OUT}" \
        --model "${MODEL_NAME}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs 240 \
        --lr 3e-4 \
        --num-workers 8 \
        --seed "${SEED_VAL}" \
        --patience "${PATIENCE}" \
        --stratification "${STRATIFICATION}" \
        --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
        --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
        --loss-weight-mae "${LOSS_WEIGHT_MAE}" \
        --loss-weight-spread "${LOSS_WEIGHT_SPREAD}" \
        --user-group-size "${USER_GROUP_SIZE}" \
        --eval-aggregation-sizes ${AGG_SIZES} \
        --eval-aggregation-seed "${AGG_SEED_EFF}" \
        --dist-backend nccl
      echo ">>> BASELINE training for ${MODEL_NAME} seed=${SEED_VAL} complete. Artifacts in ${BASELINE_OUT}"

      DINO_OUT="${DINO_OUTPUT_ROOT}/${MODEL_NAME}_seed${SEED_VAL}_ssl60"
      DINO_CKPT="${DINO_OUT}/dino_${MODEL_NAME}_pretrain_ddp.pth"
      if [[ ! -f "${DINO_CKPT}" ]]; then
        echo ">>> Starting distributed DINO pretraining (60 epochs) for ${MODEL_NAME} seed=${SEED_VAL} (output: ${DINO_OUT})"
        torchrun \
          --nnodes="${NNODES}" \
          --nproc_per_node="${GPUS_PER_NODE}" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
          "${PROJECT_ROOT}/train_dino_distributed.py" \
          --data-root "${DATA_ROOT}" \
          --output-dir "${DINO_OUT}" \
          --model "${MODEL_NAME}" \
          --batch-size "${DINO_BATCH_SIZE}" \
          --epochs 60 \
          --num-workers "${DINO_NUM_WORKERS}" \
          --seed "${SEED_VAL}" \
          --pair-same-hand-prob "${DINO_PAIR_SAME_HAND_PROB}" \
          --num-local-crops "${DINO_NUM_LOCAL_CROPS}" \
          --aug-ramp-fraction "${DINO_AUG_RAMP_FRACTION}" \
          --dist-backend nccl
        echo ">>> DINO pretraining for ${MODEL_NAME} seed=${SEED_VAL} complete."
      else
        echo ">>> Found existing DINO checkpoint for ${MODEL_NAME} seed=${SEED_VAL}: ${DINO_CKPT}"
      fi

      SSL_NAME="${MODEL_NAME}_bs${BATCH_SIZE}_seed${SEED_VAL}_nll${LOSS_WEIGHT_NLL}_mse${LOSS_WEIGHT_MSE}_mae${LOSS_WEIGHT_MAE}_spread${LOSS_WEIGHT_SPREAD}_ug${USER_GROUP_SIZE}_strat${STRATIFICATION}_ssl60"
      SSL_OUT="${OUTPUT_ROOT}/${SSL_NAME}"
      echo ">>> Starting SSL-60 training for ${MODEL_NAME} seed=${SEED_VAL} -> ${SSL_OUT}"
      torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${PROJECT_ROOT}/train_distributed.py" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${SSL_OUT}" \
        --model "${MODEL_NAME}" \
        --ssl-pretrained "${DINO_CKPT}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs 240 \
        --lr 3e-4 \
        --num-workers 8 \
        --seed "${SEED_VAL}" \
        --patience "${PATIENCE}" \
        --stratification "${STRATIFICATION}" \
        --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
        --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
        --loss-weight-mae "${LOSS_WEIGHT_MAE}" \
        --loss-weight-spread "${LOSS_WEIGHT_SPREAD}" \
        --user-group-size "${USER_GROUP_SIZE}" \
        --eval-aggregation-sizes ${AGG_SIZES} \
        --eval-aggregation-seed "${AGG_SEED_EFF}" \
        --dist-backend nccl
      echo ">>> SSL-60 training for ${MODEL_NAME} seed=${SEED_VAL} complete. Artifacts in ${SSL_OUT}"
    done
  done
done
