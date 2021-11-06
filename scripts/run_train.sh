#!/usr/bin/env bash
# run_train.sh — train DAA for all ACE-05 domain-adaptation pairs
set -euo pipefail

CONFIG_DIR="configs"
LOG_ROOT="output"

CONFIGS=(
    "${CONFIG_DIR}/ace05_bn2bc.json"
    "${CONFIG_DIR}/ace05_bn2cts.json"
    "${CONFIG_DIR}/ace05_bn2wl.json"
    "${CONFIG_DIR}/ace05_bn2un.json"
    "${CONFIG_DIR}/timebank2litbank.json"
    "${CONFIG_DIR}/litbank2timebank.json"
)

for cfg in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Training with config: ${cfg}"
    echo "========================================"
    python train.py -c "${cfg}"
done
