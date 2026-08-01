#!/bin/bash
#SBATCH --job-name=q2-sr-f0-real-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-beast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/q2-sr-fold0-real-eval-%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
SS_CHECKPOINT="${SS_CHECKPOINT:-${PROJECT_ROOT}/runs/q2_ss_synthetic2_only_nll_k5_4gpu_16cpu_20260801_384/fold_0/v2_s_age_regressor_ddp.pth}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/q2_sr_ss1050939_fold0_to_real_eval_20260801}"
OUT_DIR="${RUN_DIR}/fold_0"
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-${PROJECT_ROOT}/splits/test_users_uncapped_20pct_seed42.json}"

mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1

source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

cd "${PROJECT_ROOT}"
echo "Q2 SR/TSTR fold-0 probe: SS 1050939 fold_0 checkpoint -> real held-out test"
echo "Checkpoint: ${SS_CHECKPOINT}"
echo "Run directory: ${RUN_DIR}"
echo "Test split: ${TEST_SPLIT_FILE}"

if [[ ! -f "${SS_CHECKPOINT}" ]]; then
  echo "ERROR: missing checkpoint: ${SS_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${TEST_SPLIT_FILE}" ]]; then
  echo "ERROR: missing test split: ${TEST_SPLIT_FILE}" >&2
  exit 1
fi

python "${PROJECT_ROOT}/evaluate_test.py" \
  --checkpoint "${SS_CHECKPOINT}" \
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

echo "Q2 SR/TSTR fold-0 probe complete at $(date -Is)"
