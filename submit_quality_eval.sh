#!/bin/bash
# Submit quality evaluation for an existing trained quality run.
# Runs evaluate_quality.py (val), predict_quality.py (test), evaluate_quality_age_gate.py.
#
# Usage:
#   ./submit_quality_eval.sh <RUN_NAME> <QUALITY_RUN_NAME> [QUALITY_MODEL]
#
# Example:
#   ./submit_quality_eval.sh \
#     v2_small_quality_full_wall3_b32_age_reweight_correct_split \
#     v2_small_quality_full_wall3_b32_age_reweight_correct_split_quality_b0_d3 \
#     b0

set -euo pipefail

RUN_NAME="${1:?Usage: $0 <RUN_NAME> <QUALITY_RUN_NAME> [QUALITY_MODEL]}"
QUALITY_RUN_NAME="${2:?Usage: $0 <RUN_NAME> <QUALITY_RUN_NAME> [QUALITY_MODEL]}"
QUALITY_MODEL="${3:-b0}"

echo "Submitting quality_eval for:"
echo "  RUN_NAME:         ${RUN_NAME}"
echo "  QUALITY_RUN_NAME: ${QUALITY_RUN_NAME}"
echo "  QUALITY_MODEL:    ${QUALITY_MODEL}"

RUN_NAME="${RUN_NAME}" \
QUALITY_RUN_NAME="${QUALITY_RUN_NAME}" \
QUALITY_MODEL="${QUALITY_MODEL}" \
PIPELINE_STAGE=quality_eval \
sbatch --gres=gpu:2 submit_distributed.slurm
