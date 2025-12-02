#!/bin/bash
#SBATCH --job-name=xr-security-train    # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --partition=gpu-beast           # GPU partition (gpu-standard or gpu-beast)
# SBATCH --gres=gpu:2                    # Request 2 GPUs (adjust as needed)
#SBATCH --output=%x-%j.log              # Output file jobname-jobid.log
#SBATCH --mail-type=ALL                 # Send email for queue,start,error,completed etc...
#SBATCH --mail-user=rb3434w@gre.ac.uk   # Update to your email address

set -euo pipefail

module load www-proxy    # Load www-proxy for internet access (seemed required)
module load CUDA         # Load CUDA module

MODELS="${MODELS:-${MODEL:-b0 b1 b2 b3 b4 b5 b6}}" # choices: b0-b7, v2_{s,m,l}, convnext_{tiny,small,base,large,xlarge}, aliases: cnt,cns,cnb,cnl,cnx
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LOSS_WEIGHT_NLL="${LOSS_WEIGHT_NLL:-0.9}"
LOSS_WEIGHT_MSE="${LOSS_WEIGHT_MSE:-0.05}"
LOSS_WEIGHT_MAE="${LOSS_WEIGHT_MAE:-0.05}"
PATIENCE="${PATIENCE:-60}"
STRATIFICATION="${STRATIFICATION:-no}"  # choices: no, minorAdults, bins
AGG_SIZES="${AGG_SIZES:-${EVAL_AGG_SIZES:-1}}"   # space-separated list, e.g., "1 2 3"
AGG_SEED="${AGG_SEED:-${EVAL_AGG_SEED:-${SEED}}}"
AGG_LABEL=${AGG_SIZES// /-}

for MODEL_NAME in ${MODELS}; do
    FOLDER_NAME="${MODEL_NAME}_bs${BATCH_SIZE}_seed${SEED}_agg${AGG_LABEL}_nll${LOSS_WEIGHT_NLL}_mse${LOSS_WEIGHT_MSE}_mae${LOSS_WEIGHT_MAE}_strat${STRATIFICATION}"
    OUTPUT_DIR="/home/rb3434w/CNN-age-inference/runs/${FOLDER_NAME}"

    echo ">>> Starting training for model ${MODEL_NAME} (output: ${OUTPUT_DIR})"
    python train_age.py \
        --data-root "/home/rb3434w/HandsDatasets" \
        --output-dir "${OUTPUT_DIR}" \
        --model "${MODEL_NAME}" \
        --batch-size "${BATCH_SIZE}" \
        --epochs 40 \
        --seed "${SEED}" \
        --lr 0.0003 \
        --patience "${PATIENCE}" \
        --stratification "${STRATIFICATION}" \
        --loss-weight-nll "${LOSS_WEIGHT_NLL}" \
        --loss-weight-mse "${LOSS_WEIGHT_MSE}" \
        --loss-weight-mae "${LOSS_WEIGHT_MAE}" \
        --eval-aggregation-sizes ${AGG_SIZES} \
        --eval-aggregation-seed "${AGG_SEED}"
    echo ">>> Finished training for model ${MODEL_NAME}"
done
