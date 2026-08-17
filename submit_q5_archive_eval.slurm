#!/bin/bash
#SBATCH --job-name=q5-archive-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/q5-archive-eval-%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
DATA_ROOT="${DATA_ROOT:-/home/rb3434w/HandsDatasets}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/q5_archive_eval_20260813}"
TEST_SPLIT_FILE="${TEST_SPLIT_FILE:-${PROJECT_ROOT}/splits/archive_users_all.json}"
MODEL="${MODEL:-v2_s}"
IMG_SIZE="${IMG_SIZE:-384}"
EMBED_DIM="${EMBED_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_AGG_SEED="${EVAL_AGG_SEED:-42}"

mkdir -p "${RUN_ROOT}" "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1

source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

cd "${PROJECT_ROOT}"

echo "Q5 archive inference"
echo "Started: $(date -Is)"
echo "Partition: gpu-standard"
echo "Project root: ${PROJECT_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Archive split: ${TEST_SPLIT_FILE}"

if [[ ! -f "${TEST_SPLIT_FILE}" ]]; then
  echo "Archive split missing; creating ${TEST_SPLIT_FILE}"
  python "${PROJECT_ROOT}/make_archive_eval_split.py" \
    --data-root "${DATA_ROOT}" \
    --out-file "${TEST_SPLIT_FILE}" \
    --max-samples-per-user 0 \
    --max-samples-per-age-bin 0 \
    --overwrite
fi

eval_q5_run() {
  local arm="$1"
  local seed="$2"
  local source_run="$3"
  local dest_run="${RUN_ROOT}/${arm}_seed${seed}"

  echo
  echo "Evaluating arm=${arm} seed=${seed}"
  echo "Checkpoint run: ${source_run}"
  echo "Output run: ${dest_run}"
  mkdir -p "${dest_run}"

  for fold_idx in 0 1 2 3 4; do
    local checkpoint="${PROJECT_ROOT}/runs/${source_run}/fold_${fold_idx}/${MODEL}_age_regressor_ddp.pth"
    local out_dir="${dest_run}/fold_${fold_idx}"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "ERROR: missing checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    echo "Starting ${arm} seed ${seed} fold ${fold_idx} at $(date -Is)"
    python "${PROJECT_ROOT}/evaluate_test.py" \
      --checkpoint "${checkpoint}" \
      --test-users-file "${TEST_SPLIT_FILE}" \
      --model "${MODEL}" \
      --img-size "${IMG_SIZE}" \
      --embed-dim "${EMBED_DIM}" \
      --no-imagenet-pretrained \
      --data-root "${DATA_ROOT}" \
      --max-samples-per-user 0 \
      --max-samples-per-age-bin 0 \
      --include-archive \
      --use-masks \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --eval-aggregation-sizes 1 \
      --eval-aggregation-seed "${EVAL_AGG_SEED}" \
      --eval-age-gate-mode age_threshold \
      --age-gate-threshold-min 10 \
      --age-gate-threshold-max 30 \
      --ddp-output-names \
      --output-dir "${out_dir}"
    echo "Finished ${arm} seed ${seed} fold ${fold_idx} at $(date -Is)"
  done

  echo "Aggregating ${arm} seed ${seed} at $(date -Is)"
  python "${PROJECT_ROOT}/aggregate_kfold.py" \
    --kfold-root "${dest_run}" \
    --output-dir "${dest_run}" \
    --group-sizes 1 \
    --eval-age-gate-mode age_threshold \
    --age-gate-threshold-min 10 \
    --age-gate-threshold-max 30 \
    --skip-val
}

eval_q5_run "R0" 42 "q3_r0_from1050753_real_finetune_lr2e5_seed42_v2s_384"
eval_q5_run "Pooled" 42 "v2s_hrgbd_real_finetuning_nll_lr2e5_uncapped_test20_k5_4gpu_16cpu_20260730_384"

for seed in 43 44 45 46 47 48; do
  eval_q5_run "R0" "${seed}" "minpair_seed${seed}_real_from_real_lr2e5_v2s_384"
  eval_q5_run "Pooled" "${seed}" "minpair_seed${seed}_real_from_syn2_lr2e5_v2s_384"
done

echo
echo "Summarising Q5 archive evaluation at $(date -Is)"
python "${PROJECT_ROOT}/summarize_q5_archive_eval.py" \
  --run-root "${RUN_ROOT}" \
  --out-md "${PROJECT_ROOT}/Q5_ARCHIVE_EVAL_RESULTS.md"

echo "Q5 archive inference complete at $(date -Is)"
