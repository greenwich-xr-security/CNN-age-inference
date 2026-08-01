#!/bin/bash
#SBATCH --job-name=q2-sr-real-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-beast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/q2-sr-real-eval-%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
SS_RUN_ROOT="${SS_RUN_ROOT:-${PROJECT_ROOT}/runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384}"
RUN_NAME="${RUN_NAME:-q2_sr_ss1050939_to_real_eval_20260801}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-${PROJECT_ROOT}/splits/test_users_uncapped_20pct_seed42.json}"

mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1

source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

cd "${PROJECT_ROOT}"
echo "Q2 SR/TSTR evaluation: SyntheticDorsalHands2-trained 1050939 checkpoints -> real held-out test"
echo "SS checkpoint root: ${SS_RUN_ROOT}"
echo "Run directory: ${RUN_DIR}"
echo "Test split: ${TEST_SPLIT_FILE}"

for FOLD_IDX in 0 1 2 3 4; do
  CHECKPOINT="${SS_RUN_ROOT}/fold_${FOLD_IDX}/v2_s_age_regressor_ddp.pth"
  OUT_DIR="${RUN_DIR}/fold_${FOLD_IDX}"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi
  echo "Starting SR eval fold ${FOLD_IDX} at $(date -Is); checkpoint: ${CHECKPOINT}"
  python "${PROJECT_ROOT}/evaluate_test.py" \
    --checkpoint "${CHECKPOINT}" \
    --test-users-file "${TEST_SPLIT_FILE}" \
    --model v2_s \
    --img-size 384 \
    --embed-dim 128 \
    --data-root "${DATA_ROOT}" \
    --max-samples-per-user 0 \
    --max-samples-per-age-bin 0 \
    --include-handrgbd \
    --include-prolific \
    --batch-size 64 \
    --num-workers 4 \
    --eval-aggregation-sizes 1 \
    --eval-aggregation-seed 42 \
    --eval-age-gate-mode age_threshold \
    --age-gate-threshold-min 10 \
    --age-gate-threshold-max 30 \
    --ddp-output-names \
    --output-dir "${OUT_DIR}"
  echo "Finished SR eval fold ${FOLD_IDX} at $(date -Is)"
done

echo "Aggregating SR folds at $(date -Is)"
python "${PROJECT_ROOT}/aggregate_kfold.py" \
  --kfold-root "${RUN_DIR}" \
  --output-dir "${RUN_DIR}" \
  --group-sizes 1 \
  --eval-age-gate-mode age_threshold \
  --age-gate-threshold-min 10 \
  --age-gate-threshold-max 30 \
  --skip-val

echo "Q2 SR/TSTR evaluation complete at $(date -Is)"
