#!/bin/bash
#SBATCH --job-name=byol-s-ssl-synth2      # Descriptive job name
#SBATCH --nodes=1                         # Single node on gpu-beast
#SBATCH --ntasks-per-node=1               # torchrun will launch local ranks
#SBATCH --partition=gpu-beast             # GPU partition
#SBATCH --gres=gpu:4                      # Request 4 GPUs, matching the v2_s supervised-run convention
#SBATCH --cpus-per-task=32                # CPU cores available to each task
#SBATCH --output=logs/%x-%j.log           # Stdout/stderr log file
#SBATCH --mail-type=END,FAIL              # Email on completion or failure
#SBATCH --mail-user=rb3434w@gre.ac.uk

# S-ssl: BYOL SSL pretraining of EfficientNet-V2-S on SyntheticDorsalHands2 only
# (no real images, no age labels). This is the self-supervised counterpart to
# job 1050819's synthetic-label pretraining -- same backbone, same synthetic
# data source, same downstream real-only fine-tuning to follow as a separate
# job once this checkpoint is produced and inspected.

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"

MODEL="${MODEL:-v2_s}"
BYOL_EPOCHS="${BYOL_EPOCHS:-100}"
BYOL_BATCH_SIZE="${BYOL_BATCH_SIZE:-32}"
BYOL_NUM_LOCAL_CROPS="${BYOL_NUM_LOCAL_CROPS:-0}"
BYOL_TARGET_MOMENTUM="${BYOL_TARGET_MOMENTUM:-0.996}"
BYOL_TARGET_MOMENTUM_END="${BYOL_TARGET_MOMENTUM_END:-1.0}"
BYOL_NUM_WORKERS="${BYOL_NUM_WORKERS:-8}"
BYOL_AUG_RAMP_FRACTION="${BYOL_AUG_RAMP_FRACTION:-0.2}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

RUN_NAME="${RUN_NAME:-s_ssl_synthetic2_${MODEL}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/byol/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}" logs "${TORCH_HOME}"
export TORCH_HOME
export PYTHONUNBUFFERED=1

# Activate conda xr environment
source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

extract_gpu_count() {
  local raw="${1:-}"
  if [[ "${raw}" =~ ([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29530}
NNODES=${SLURM_JOB_NUM_NODES:-1}
if ! GPUS_PER_NODE="$(extract_gpu_count "${SLURM_GPUS_ON_NODE:-}")"; then
  if ! GPUS_PER_NODE="$(extract_gpu_count "${SLURM_GPUS_PER_NODE:-}")"; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a _CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES}"
      GPUS_PER_NODE="${#_CUDA_IDS[@]}"
    else
      GPUS_PER_NODE="${GPUS_PER_NODE_OVERRIDE:-4}"
    fi
  fi
fi

export MASTER_ADDR MASTER_PORT

echo ">>> S-ssl: BYOL pretraining on SyntheticDorsalHands2 only"
echo ">>> Model: ${MODEL} | Epochs: ${BYOL_EPOCHS} | Per-GPU batch size: ${BYOL_BATCH_SIZE} | GPUs: ${GPUS_PER_NODE}"
echo ">>> Output dir: ${OUTPUT_DIR}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${PROJECT_ROOT}/train_byol_distributed.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --include-synthetic-dorsal2 \
  --batch-size "${BYOL_BATCH_SIZE}" \
  --epochs "${BYOL_EPOCHS}" \
  --num-local-crops "${BYOL_NUM_LOCAL_CROPS}" \
  --target-momentum "${BYOL_TARGET_MOMENTUM}" \
  --target-momentum-end "${BYOL_TARGET_MOMENTUM_END}" \
  --num-workers "${BYOL_NUM_WORKERS}" \
  --aug-ramp-fraction "${BYOL_AUG_RAMP_FRACTION}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --dist-backend nccl

echo ">>> S-ssl BYOL pretraining complete. Artifacts in ${OUTPUT_DIR}"
