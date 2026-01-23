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
IMG_SIZE="${IMG_SIZE:-}"
TUNE_IMG_SIZE="${TUNE_IMG_SIZE:-0}"
IMG_SIZE_MIN="${IMG_SIZE_MIN:-224}"
IMG_SIZE_MAX="${IMG_SIZE_MAX:-600}"
IMG_SIZE_STEP="${IMG_SIZE_STEP:-1}"
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
TUNE_WEIGHT_DECAY="${TUNE_WEIGHT_DECAY:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WEIGHT_DECAY_MIN="${WEIGHT_DECAY_MIN:-0.0}"
WEIGHT_DECAY_MAX="${WEIGHT_DECAY_MAX:-0.1}"

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
  --weight-decay "${WEIGHT_DECAY}"
  --weight-decay-min "${WEIGHT_DECAY_MIN}"
  --weight-decay-max "${WEIGHT_DECAY_MAX}"
  --img-size-min "${IMG_SIZE_MIN}"
  --img-size-max "${IMG_SIZE_MAX}"
  --img-size-step "${IMG_SIZE_STEP}"
  --nll-min "${NLL_MIN}"
  --nll-max "${NLL_MAX}"
  --mse-min "${MSE_MIN}"
  --mse-max "${MSE_MAX}"
  --mae-min "${MAE_MIN}"
  --mae-max "${MAE_MAX}"
  --spread-min "${SPREAD_MIN}"
  --spread-max "${SPREAD_MAX}"
)

if [[ -n "${IMG_SIZE}" ]]; then
  CMD+=(--img-size "${IMG_SIZE}")
fi
if [[ "${TUNE_IMG_SIZE}" -gt 0 ]]; then
  CMD+=(--tune-img-size)
fi
if [[ "${TUNE_LR}" -gt 0 ]]; then
  CMD+=(--tune-lr)
fi
if [[ "${TUNE_WEIGHT_DECAY}" -gt 0 ]]; then
  CMD+=(--tune-weight-decay)
fi
echo ">>> Starting Optuna worker: study=${STUDY_NAME}, trials=${TRIALS_PER_JOB}, storage=${STORAGE}"
"${CMD[@]}"
