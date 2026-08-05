#!/bin/bash
#SBATCH --job-name=q3-ussl-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-beast
#SBATCH --nodelist=gm-hpc2-gpu801
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rb3434w@gre.ac.uk

# U-ssl: BYOL SSL pretraining on auto-downloaded STL10 unlabeled images. This
# generic natural-image corpus excludes hands/age labels by design and keeps the
# U-ssl control independent of SyntheticDorsalHands2 and the locked real test
# sources.

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"

MODEL="${MODEL:-v2_s}"
BYOL_EPOCHS="${BYOL_EPOCHS:-22}"
BYOL_BATCH_SIZE="${BYOL_BATCH_SIZE:-8}"
BYOL_NUM_LOCAL_CROPS="${BYOL_NUM_LOCAL_CROPS:-0}"
BYOL_TARGET_MOMENTUM="${BYOL_TARGET_MOMENTUM:-0.996}"
BYOL_TARGET_MOMENTUM_END="${BYOL_TARGET_MOMENTUM_END:-1.0}"
BYOL_NUM_WORKERS="${BYOL_NUM_WORKERS:-8}"
BYOL_AUG_RAMP_FRACTION="${BYOL_AUG_RAMP_FRACTION:-0.2}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

RUN_NAME="${RUN_NAME:-u_ssl_stl10_unlabeled_${MODEL}_8gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/byol/${RUN_NAME}}"
TORCHVISION_ROOT="${TORCHVISION_ROOT:-${PROJECT_ROOT}/.data/torchvision}"

mkdir -p "${OUTPUT_DIR}" logs "${TORCH_HOME}"
export TORCH_HOME
export PYTHONUNBUFFERED=1

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
MASTER_PORT=${MASTER_PORT:-29538}
NNODES=${SLURM_JOB_NUM_NODES:-1}
if ! GPUS_PER_NODE="$(extract_gpu_count "${SLURM_GPUS_ON_NODE:-}")"; then
  if ! GPUS_PER_NODE="$(extract_gpu_count "${SLURM_GPUS_PER_NODE:-}")"; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a _CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES}"
      GPUS_PER_NODE="${#_CUDA_IDS[@]}"
    else
      GPUS_PER_NODE="${GPUS_PER_NODE_OVERRIDE:-8}"
    fi
  fi
fi

export MASTER_ADDR MASTER_PORT

echo ">>> U-ssl: BYOL pretraining on auto-downloaded STL10 unlabeled images"
echo ">>> Model: ${MODEL} | Epochs: ${BYOL_EPOCHS} | Per-GPU batch size: ${BYOL_BATCH_SIZE} | GPUs: ${GPUS_PER_NODE}"
echo ">>> Output dir: ${OUTPUT_DIR}"
echo ">>> Torchvision data root: ${TORCHVISION_ROOT}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${PROJECT_ROOT}/train_byol_distributed.py" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --torchvision-dataset stl10_unlabeled \
  --torchvision-root "${TORCHVISION_ROOT}" \
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

echo ">>> U-ssl BYOL pretraining complete. Artifacts in ${OUTPUT_DIR}"
