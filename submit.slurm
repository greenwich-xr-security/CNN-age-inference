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
DINO_EPOCHS="${DINO_EPOCHS:-100}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-${BATCH_SIZE}}"
DINO_PAIR_SAME_HAND_PROB="${DINO_PAIR_SAME_HAND_PROB:-0.5}"
DINO_NUM_LOCAL_CROPS="${DINO_NUM_LOCAL_CROPS:-4}"
DINO_AUG_RAMP_FRACTION="${DINO_AUG_RAMP_FRACTION:-0.2}"
DINO_OUTPUT_ROOT="${DINO_OUTPUT_ROOT:-/home/rb3434w/CNN-age-inference/runs/dino}"
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
    DINO_OUT="${DINO_OUTPUT_ROOT}/${MODEL_NAME}_seed${SEED}"
    DINO_CKPT="${DINO_OUT}/dino_${MODEL_NAME}_pretrain.pth"

    if [[ ! -f "${DINO_CKPT}" ]]; then
        echo ">>> Starting DINO pretraining for model ${MODEL_NAME} (output: ${DINO_OUT})"
        python train_dino.py \
            --data-root "/home/rb3434w/HandsDatasets" \
            --output-dir "${DINO_OUT}" \
            --model "${MODEL_NAME}" \
            --batch-size "${DINO_BATCH_SIZE}" \
            --epochs "${DINO_EPOCHS}" \
            --seed "${SEED}" \
            --pair-same-hand-prob "${DINO_PAIR_SAME_HAND_PROB}" \
            --num-local-crops "${DINO_NUM_LOCAL_CROPS}" \
            --aug-ramp-fraction "${DINO_AUG_RAMP_FRACTION}"
        echo ">>> Finished DINO pretraining for model ${MODEL_NAME}"
    else
        echo ">>> Found existing DINO checkpoint for model ${MODEL_NAME}: ${DINO_CKPT}"
    fi

    echo ">>> Starting training for model ${MODEL_NAME} (output: ${OUTPUT_DIR})"
    python train_age.py \
        --data-root "/home/rb3434w/HandsDatasets" \
        --output-dir "${OUTPUT_DIR}" \
        --model "${MODEL_NAME}" \
        --ssl-pretrained "${DINO_CKPT}" \
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
