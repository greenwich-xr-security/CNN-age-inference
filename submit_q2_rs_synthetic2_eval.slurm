#!/bin/bash
#SBATCH --job-name=q2-rs-syn2-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-beast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/q2-rs-syn2-eval-%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
RR_RUN_ROOT="${RR_RUN_ROOT:-${PROJECT_ROOT}/runs/v2s_handrgbd_prolific_only_nll_uncapped_test20_k5_4gpu_16cpu_20260728_384}"
RUN_NAME="${RUN_NAME:-q2_rs_real1050753_to_synthetic2_eval_20260801}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-${PROJECT_ROOT}/splits/test_users_synthetic2_20pct_seed42.json}"

mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1

source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

cd "${PROJECT_ROOT}"
echo "Q2 RS/TRTS evaluation: Real-trained 1050753 checkpoints -> SyntheticDorsalHands2 held-out test"
echo "RR checkpoint root: ${RR_RUN_ROOT}"
echo "Run directory: ${RUN_DIR}"
echo "Test split: ${TEST_SPLIT_FILE}"

for FOLD_IDX in 0 1 2 3 4; do
  CHECKPOINT="${RR_RUN_ROOT}/fold_${FOLD_IDX}/v2_s_age_regressor_ddp.pth"
  OUT_DIR="${RUN_DIR}/fold_${FOLD_IDX}"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi
  echo "Starting RS eval fold ${FOLD_IDX} at $(date -Is); checkpoint: ${CHECKPOINT}"
  python "${PROJECT_ROOT}/evaluate_test.py" \
    --checkpoint "${CHECKPOINT}" \
    --test-users-file "${TEST_SPLIT_FILE}" \
    --model v2_s \
    --img-size 384 \
    --embed-dim 128 \
    --data-root "${DATA_ROOT}" \
    --max-samples-per-user 0 \
    --max-samples-per-age-bin 0 \
    --include-synthetic-dorsal2 \
    --batch-size 64 \
    --num-workers 4 \
    --eval-aggregation-sizes 1 \
    --eval-aggregation-seed 42 \
    --eval-age-gate-mode age_threshold \
    --age-gate-threshold-min 10 \
    --age-gate-threshold-max 30 \
    --ddp-output-names \
    --output-dir "${OUT_DIR}"
  echo "Finished RS eval fold ${FOLD_IDX} at $(date -Is)"
done

echo "Aggregating RS folds at $(date -Is)"
python "${PROJECT_ROOT}/aggregate_kfold.py" \
  --kfold-root "${RUN_DIR}" \
  --output-dir "${RUN_DIR}" \
  --group-sizes 1 \
  --eval-age-gate-mode age_threshold \
  --age-gate-threshold-min 10 \
  --age-gate-threshold-max 30 \
  --skip-val

echo "Q2 RS/TRTS evaluation complete at $(date -Is)"
