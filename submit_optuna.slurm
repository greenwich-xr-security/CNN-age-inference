#!/bin/bash
#SBATCH --job-name=age-infer-optuna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-beast
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
BASE_OUTPUT="${BASE_OUTPUT:-${PROJECT_ROOT}/runs/optuna}"
STUDY_NAME="${STUDY_NAME:-age_infer_trans}"
STORAGE="${STORAGE:-sqlite:///${BASE_OUTPUT}/optuna.db}"

MODEL="${MODEL:-b2}"
SEED="${SEED:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
USER_GROUP_SIZE="${USER_GROUP_SIZE:-2}"
EPOCHS="${EPOCHS:-240}"
NUM_WORKERS="${NUM_WORKERS:-8}"
KFOLDS="${KFOLDS:-5}"
AGG_SIZE="${AGG_SIZE:-2}"
AUC_CASE="${AUC_CASE:-mean}"
TRIALS_PER_JOB="${TRIALS_PER_JOB:-6}"

TUNE_LR="${TUNE_LR:-0}"
LR="${LR:-3e-4}"
LR_MIN="${LR_MIN:-1e-4}"
LR_MAX="${LR_MAX:-1e-3}"

TUNE_AUG_LEVELS="${TUNE_AUG_LEVELS:-1}"
AUG_PHOTOMETRIC_LEVEL_MIN="${AUG_PHOTOMETRIC_LEVEL_MIN:-0.0}"
AUG_PHOTOMETRIC_LEVEL_MAX="${AUG_PHOTOMETRIC_LEVEL_MAX:-1.0}"
AUG_FOCUS_LEVEL_MIN="${AUG_FOCUS_LEVEL_MIN:-0.0}"
AUG_FOCUS_LEVEL_MAX="${AUG_FOCUS_LEVEL_MAX:-1.0}"
AUG_OCCLUSION_LEVEL_MIN="${AUG_OCCLUSION_LEVEL_MIN:-0.0}"
AUG_OCCLUSION_LEVEL_MAX="${AUG_OCCLUSION_LEVEL_MAX:-1.0}"

NLL_MIN="${NLL_MIN:-0.6}"
NLL_MAX="${NLL_MAX:-0.6}"
MSE_MIN="${MSE_MIN:-0.9}"
MSE_MAX="${MSE_MAX:-0.9}"
MAE_MIN="${MAE_MIN:-0.4}"
MAE_MAX="${MAE_MAX:-0.4}"
SPREAD_MIN="${SPREAD_MIN:-0.3}"
SPREAD_MAX="${SPREAD_MAX:-0.3}"

mkdir -p "${BASE_OUTPUT}" logs

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

CMD=(
  python "${PROJECT_ROOT}/optuna_worker.py"
  --study-name "${STUDY_NAME}"
  --storage "${STORAGE}"
  --n-trials "${TRIALS_PER_JOB}"
  --base-output "${BASE_OUTPUT}"
  --submit-script "${PROJECT_ROOT}/submit_distributed.slurm"
  --model "${MODEL}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --user-group-size "${USER_GROUP_SIZE}"
  --epochs "${EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --kfolds "${KFOLDS}"
  --agg-size "${AGG_SIZE}"
  --auc-case "${AUC_CASE}"
  --lr "${LR}"
  --lr-min "${LR_MIN}"
  --lr-max "${LR_MAX}"
  --nll-min "${NLL_MIN}"
  --nll-max "${NLL_MAX}"
  --mse-min "${MSE_MIN}"
  --mse-max "${MSE_MAX}"
  --mae-min "${MAE_MIN}"
  --mae-max "${MAE_MAX}"
  --spread-min "${SPREAD_MIN}"
  --spread-max "${SPREAD_MAX}"
  --aug-photometric-level-min "${AUG_PHOTOMETRIC_LEVEL_MIN}"
  --aug-photometric-level-max "${AUG_PHOTOMETRIC_LEVEL_MAX}"
  --aug-focus-level-min "${AUG_FOCUS_LEVEL_MIN}"
  --aug-focus-level-max "${AUG_FOCUS_LEVEL_MAX}"
  --aug-occlusion-level-min "${AUG_OCCLUSION_LEVEL_MIN}"
  --aug-occlusion-level-max "${AUG_OCCLUSION_LEVEL_MAX}"
)

if [[ "${TUNE_LR}" -gt 0 ]]; then
  CMD+=(--tune-lr)
fi
if [[ "${TUNE_AUG_LEVELS}" -gt 0 ]]; then
  CMD+=(--tune-aug-levels)
fi
echo ">>> Starting Optuna worker: study=${STUDY_NAME}, trials=${TRIALS_PER_JOB}, storage=${STORAGE}"
"${CMD[@]}"
