#!/usr/bin/env bash
# run_evaluate.sh — evaluate saved checkpoints for all configs
set -euo pipefail

# Example usage:
#   bash scripts/run_evaluate.sh output/20240101_120000 configs/ace05_bn2bc.json

CHECKPOINT_DIR="${1:-output/best}"
CONFIG="${2:-configs/ace05_bn2bc.json}"
CHECKPOINT="${CHECKPOINT_DIR}/best_model.pt"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Evaluating ${CHECKPOINT} with config ${CONFIG}"
python evaluate.py -c "${CONFIG}" --checkpoint "${CHECKPOINT}"
