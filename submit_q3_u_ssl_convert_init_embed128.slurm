#!/bin/bash
#SBATCH --job-name=q3-ussl-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-standard,gpu-beast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rb3434w@gre.ac.uk

set -euo pipefail

module load www-proxy
module load CUDA

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/CNN-age-inference}"
SSL_CHECKPOINT="${SSL_CHECKPOINT:-${PROJECT_ROOT}/runs/byol/u_ssl_stl10_unlabeled_v2_s/byol_v2_s_pretrain_ddp.pth}"
INIT_CHECKPOINT_ROOT="${INIT_CHECKPOINT_ROOT:-${PROJECT_ROOT}/runs/byol/u_ssl_stl10_unlabeled_v2_s/init_checkpoint_root_embed128}"
MODEL="${MODEL:-v2_s}"
EMBED_DIM="${EMBED_DIM:-128}"

source /opt/software/eb/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate xr

if [[ ! -f "${SSL_CHECKPOINT}" ]]; then
  echo "ERROR: SSL checkpoint not found: ${SSL_CHECKPOINT}" >&2
  exit 1
fi

for FOLD_IDX in 0 1 2 3 4; do
  OUT_DIR="${INIT_CHECKPOINT_ROOT}/fold_${FOLD_IDX}"
  mkdir -p "${OUT_DIR}"
  python "${PROJECT_ROOT}/convert_ssl_checkpoint.py" \
    "${SSL_CHECKPOINT}" \
    "${OUT_DIR}/${MODEL}_age_regressor_ddp.pth" \
    --model "${MODEL}" \
    --embed-dim "${EMBED_DIM}"
done

echo ">>> Converted U-ssl BYOL checkpoint to ${INIT_CHECKPOINT_ROOT}"
