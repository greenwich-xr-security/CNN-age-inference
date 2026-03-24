#!/bin/bash
#SBATCH --job-name=age-infer-ddp          # Descriptive job name
#SBATCH --nodes=1                         # Single node on gpu-beast
#SBATCH --ntasks-per-node=1               # torchrun will launch local ranks
#SBATCH --partition=gpu-beast             # GPU partition
#SBATCH --gres=gpu:6                      # Request 6 GPUs on the node
#SBATCH --cpus-per-task=16                # CPU cores available to each task
#SBATCH --output=logs/%x-%j.log           # Stdout/stderr log file
#SBATCH --mail-type=FAIL                  # Email on failure only
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs}"
TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"

mkdir -p "${OUTPUT_ROOT}" logs "${TORCH_HOME}"
export TORCH_HOME
export PYTHONUNBUFFERED=1

# Activate conda xr environment
source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

# Robust GPU count detection (mirrors DepthHandAgeBench approach)
extract_gpu_count() {
  local raw="${1:-}"
  if [[ "${raw}" =~ ([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29500}
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

BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.2}"
# Leave empty to use the canonical input size for the selected model.
IMG_SIZE="${IMG_SIZE:-}"
EPOCHS="${EPOCHS:-240}"

# Losses
LOSS_WEIGHT_NLL="${LOSS_WEIGHT_NLL:-1}"
LOSS_WEIGHT_MSE="${LOSS_WEIGHT_MSE:-0}"
LOSS_WEIGHT_MAE="${LOSS_WEIGHT_MAE:-0}"
LOSS_WEIGHT_SPREAD="${LOSS_WEIGHT_SPREAD:-0}"
LOSS_WEIGHT_EMBED_VAR="${LOSS_WEIGHT_EMBED_VAR:-0}"
LOSS_WEIGHT_EMBED_CONTRAST="${LOSS_WEIGHT_EMBED_CONTRAST:-0}"

USER_GROUP_SIZES="${USER_GROUP_SIZES:-1}"
PATIENCE="${PATIENCE:-10}"
MODELS="${MODELS:-${MODEL:-v2_m}}" # choices: b0-b7, v2_{s,m,l}, convnext_{tiny,small,base,large,xlarge}, resnet50, swin_{tiny,small,base,large,v2_tiny,v2_tiny_384,v2_tiny_512,v2_small,v2_base,v2_base_384,v2_large}, vit_{tiny_384,small_384}; aliases: cnt,cns,cnb,cnl,cnx,rn50,sw2t,sw2t384,sw2t512,sw2s,sw2b,sw2b384,sw2l,vtt,vts,age_vit,age_vit_small
AGG_SIZES="${AGG_SIZES:-${EVAL_AGG_SIZES:-1}}"   # space-separated list, e.g., "1 2 3"
AGG_SEED="${AGG_SEED:-${EVAL_AGG_SEED:-${SEED}}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Optional override; when empty the script derives a model-specific run name.
RUN_NAME="${RUN_NAME:-}"
# Age imbalance controls
AGE_REWEIGHT="${AGE_REWEIGHT:-1}"                     # enabled by default for HandRGBD-only age balancing
AGE_WEIGHT_EPS="${AGE_WEIGHT_EPS:-1.0}"
AGE_WEIGHT_POWER="${AGE_WEIGHT_POWER:-1.0}"
AGE_WEIGHT_MIN="${AGE_WEIGHT_MIN:-0.25}"
AGE_WEIGHT_MAX="${AGE_WEIGHT_MAX:-4.0}"
AGE_OVERSAMPLE="${AGE_OVERSAMPLE:-0}"                 # set >0 to oversample rare ages in train split
AGE_OVERSAMPLE_TARGET="${AGE_OVERSAMPLE_TARGET:-}"    # optional explicit target per age (int)
AGE_OVERSAMPLE_MAX_MULT="${AGE_OVERSAMPLE_MAX_MULT:-3.0}"
# Embedding head / losses (default disabled)
EMBED_DIM="${EMBED_DIM:-128}"
EMBED_AGE_SLACK="${EMBED_AGE_SLACK:-5.0}"
EMBED_CONTRAST_MARGIN="${EMBED_CONTRAST_MARGIN:-1.0}"
EMBED_CONTRAST_AGE_THRESH="${EMBED_CONTRAST_AGE_THRESH:-5.0}"
USE_MASKS="${USE_MASKS:-0}"                        # set >0 to apply dataset masks (black background)
NO_HAGRID="${NO_HAGRID:-0}"                       # set >0 to exclude HaGRIDv2 from split generation and training
# K-fold settings (set KFOLDS > 1 to enable).
KFOLDS="${KFOLDS:-5}"
FOLD_SEED="${FOLD_SEED:-${SEED}}"
RUN_KFOLD_AGG="${RUN_KFOLD_AGG:-1}"
FOLD_OVERWRITE="${FOLD_OVERWRITE:-0}"
# Held-out test split (created once before any training).
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-${OUTPUT_ROOT}/test_users.json}"
TEST_SPLIT_SIZE="${TEST_SPLIT_SIZE:-0.15}"
SPLIT_STRATIFY_MODE="${SPLIT_STRATIFY_MODE:-age_bins}"  # age_bins | age_bins_skin_color | adult | none
RUN_TEST_EVAL="${RUN_TEST_EVAL:-1}"

# ── Create held-out test split (once, before any training) ───────────────────
if [[ ! -f "${TEST_SPLIT_FILE}" ]]; then
  echo ">>> Creating held-out test split (size=${TEST_SPLIT_SIZE}) → ${TEST_SPLIT_FILE}"
  python "${PROJECT_ROOT}/make_test_split.py" \
    --data-root "${DATA_ROOT}" \
    --test-size "${TEST_SPLIT_SIZE}" \
    --seed "${SEED}" \
    --out-file "${TEST_SPLIT_FILE}" \
    --stratify-mode "${SPLIT_STRATIFY_MODE}" \
    $( [[ "${NO_HAGRID}" -gt 0 ]] && echo "--no-hagrid" )
else
  echo ">>> Using existing test split: ${TEST_SPLIT_FILE}"
fi

for USER_GROUP_SIZE in ${USER_GROUP_SIZES}; do
  for MODEL_NAME in ${MODELS}; do
    for SEED_VAL in ${SEED}; do
      AGG_SEED_EFF="${AGG_SEED:-${SEED_VAL}}"
      DEFAULT_RUN_NAME="${MODEL_NAME}"
      if [[ -n "${IMG_SIZE}" ]]; then
        DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_${IMG_SIZE}"
      fi
      DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_ddp"
      FOLDER_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
      FOLD_FILE="${OUTPUT_ROOT}/${FOLDER_NAME}/folds_k${KFOLDS}.json"

      if [[ "${KFOLDS}" -gt 1 ]]; then
        if [[ ! -f "${FOLD_FILE}" || "${FOLD_OVERWRITE}" -gt 0 ]]; then
          python "${PROJECT_ROOT}/make_kfold_splits.py" \
            --data-root "${DATA_ROOT}" \
            --k "${KFOLDS}" \
            --seed "${FOLD_SEED}" \
            --out-file "${FOLD_FILE}" \
            --test-users-file "${TEST_SPLIT_FILE}" \
            --stratify-mode "${SPLIT_STRATIFY_MODE}" \
            $( [[ "${NO_HAGRID}" -gt 0 ]] && echo "--no-hagrid" ) \
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
            --weight-decay "${WEIGHT_DECAY}" \
            $( [[ -n "${IMG_SIZE}" ]] && echo "--img-size ${IMG_SIZE}" ) \
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
            --embed-dim "${EMBED_DIM}" \
            --loss-weight-embed-var "${LOSS_WEIGHT_EMBED_VAR}" \
            --loss-weight-embed-contrast "${LOSS_WEIGHT_EMBED_CONTRAST}" \
            --embed-age-slack "${EMBED_AGE_SLACK}" \
            --embed-contrast-margin "${EMBED_CONTRAST_MARGIN}" \
            --embed-contrast-age-thresh "${EMBED_CONTRAST_AGE_THRESH}" \
            $( [[ "${USE_MASKS}" -gt 0 ]] && echo "--use-masks" ) \
            $( [[ "${NO_HAGRID}" -gt 0 ]] && echo "--no-hagrid" ) \
            $( [[ "${AGE_REWEIGHT}" -gt 0 ]] && echo "--age-reweight-loss --age-weight-eps ${AGE_WEIGHT_EPS} --age-weight-power ${AGE_WEIGHT_POWER} --age-weight-min ${AGE_WEIGHT_MIN} --age-weight-max ${AGE_WEIGHT_MAX}" ) \
            $( [[ "${AGE_OVERSAMPLE}" -gt 0 ]] && echo "--age-oversample --age-oversample-max-multiplier ${AGE_OVERSAMPLE_MAX_MULT}" ) \
            $( [[ -n "${AGE_OVERSAMPLE_TARGET}" ]] && echo "--age-oversample-target ${AGE_OVERSAMPLE_TARGET}" ) \
            --fold-file "${FOLD_FILE}" \
            --fold-index "${FOLD_IDX}" \
            --test-users-file "${TEST_SPLIT_FILE}" \
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

        # Test-set evaluation now runs inside each fold's train_distributed.py
        # (via --test-users-file) — no separate evaluate_test.py step needed.
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
          --weight-decay "${WEIGHT_DECAY}" \
          $( [[ -n "${IMG_SIZE}" ]] && echo "--img-size ${IMG_SIZE}" ) \
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
          --embed-dim "${EMBED_DIM}" \
          --loss-weight-embed-var "${LOSS_WEIGHT_EMBED_VAR}" \
          --loss-weight-embed-contrast "${LOSS_WEIGHT_EMBED_CONTRAST}" \
          --embed-age-slack "${EMBED_AGE_SLACK}" \
          --embed-contrast-margin "${EMBED_CONTRAST_MARGIN}" \
          --embed-contrast-age-thresh "${EMBED_CONTRAST_AGE_THRESH}" \
          $( [[ "${USE_MASKS}" -gt 0 ]] && echo "--use-masks" ) \
          $( [[ "${NO_HAGRID}" -gt 0 ]] && echo "--no-hagrid" ) \
          $( [[ "${AGE_REWEIGHT}" -gt 0 ]] && echo "--age-reweight-loss --age-weight-eps ${AGE_WEIGHT_EPS} --age-weight-power ${AGE_WEIGHT_POWER} --age-weight-min ${AGE_WEIGHT_MIN} --age-weight-max ${AGE_WEIGHT_MAX}" ) \
          $( [[ "${AGE_OVERSAMPLE}" -gt 0 ]] && echo "--age-oversample --age-oversample-max-multiplier ${AGE_OVERSAMPLE_MAX_MULT}" ) \
          $( [[ -n "${AGE_OVERSAMPLE_TARGET}" ]] && echo "--age-oversample-target ${AGE_OVERSAMPLE_TARGET}" ) \
          --test-users-file "${TEST_SPLIT_FILE}" \
          --dist-backend nccl
        echo ">>> Training for ${MODEL_NAME} seed=${SEED_VAL} complete. Artifacts in ${RUN_OUT}"
      fi
    done
  done
done
