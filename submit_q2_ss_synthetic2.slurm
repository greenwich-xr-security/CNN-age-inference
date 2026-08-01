#!/bin/bash
#SBATCH --job-name=q2-ss-syn2-nll
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-beast
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/q2-ss-syn2-nll-%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"
TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
RUN_NAME="${RUN_NAME:-q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-${PROJECT_ROOT}/splits/test_users_synthetic2_20pct_seed42.json}"
SOURCE_FOLD_FILE="${SOURCE_FOLD_FILE:-${PROJECT_ROOT}/splits/folds_k5_synthetic2_seed42.json}"
FOLD_FILE="${FOLD_FILE:-${RUN_DIR}/folds_k5.json}"
MODEL_NAME="${MODEL_NAME:-v2_s}"
MASTER_PORT="${MASTER_PORT:-29516}"

mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs" "${TORCH_HOME}"
cp "${SOURCE_FOLD_FILE}" "${FOLD_FILE}"
export TORCH_HOME PYTHONUNBUFFERED=1 MASTER_PORT

source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

MASTER_ADDR="${MASTER_ADDR:-$(hostname -s)}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export MASTER_ADDR

cd "${PROJECT_ROOT}"
echo "Q2 SS SyntheticDorsalHands2-only pure NLL"
echo "Run directory: ${RUN_DIR}"
echo "Test split: ${TEST_SPLIT_FILE}"
echo "Fold file: ${FOLD_FILE}"
echo "Resources: ${GPUS_PER_NODE} GPUs, ${SLURM_CPUS_PER_TASK:-16} CPUs"

for FOLD_IDX in 0 1 2 3 4; do
  RUN_OUT="${RUN_DIR}/fold_${FOLD_IDX}"
  echo "Starting fold ${FOLD_IDX} at $(date -Is); output: ${RUN_OUT}"
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${PROJECT_ROOT}/train_distributed.py" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_OUT}" \
    --model "${MODEL_NAME}" \
    --batch-size 16 \
    --epochs 240 \
    --lr 2e-4 \
    --weight-decay 0.2 \
    --img-size 384 \
    --num-workers 2 \
    --seed 42 \
    --patience 10 \
    --max-samples-per-user 0 \
    --max-samples-per-age-bin 0 \
    --loss-weight-nll 1 \
    --loss-weight-crps 0 \
    --loss-weight-mse 0 \
    --loss-weight-mae 0 \
    --loss-weight-spread 0 \
    --user-group-size 1 \
    --eval-aggregation-sizes 1 \
    --eval-aggregation-seed 42 \
    --embed-dim 128 \
    --loss-weight-embed-var 0 \
    --loss-weight-embed-contrast 0 \
    --embed-age-slack 5.0 \
    --embed-contrast-margin 1.0 \
    --embed-contrast-age-thresh 5.0 \
    --include-synthetic-dorsal2 \
    --fold-file "${FOLD_FILE}" \
    --fold-index "${FOLD_IDX}" \
    --test-users-file "${TEST_SPLIT_FILE}" \
    --eval-age-gate-mode age_threshold \
    --age-gate-threshold-min 10 \
    --age-gate-threshold-max 30 \
    --dist-backend nccl
  echo "Fold ${FOLD_IDX} complete at $(date -Is). Artifacts in ${RUN_OUT}"
done

echo "Aggregating folds at $(date -Is)"
python "${PROJECT_ROOT}/aggregate_kfold.py" \
  --kfold-root "${RUN_DIR}" \
  --output-dir "${RUN_DIR}" \
  --group-sizes 1 \
  --eval-age-gate-mode age_threshold \
  --age-gate-threshold-min 10 \
  --age-gate-threshold-max 30

echo "Q2 SS job complete at $(date -Is)"
