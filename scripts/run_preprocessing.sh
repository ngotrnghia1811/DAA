#!/usr/bin/env bash
# run_preprocessing.sh — convert raw datasets to DAA JSON-lines format
set -euo pipefail

ACE05_INPUT="${1:-/path/to/ace05/English}"
TIMEBANK_INPUT="${2:-/path/to/timebank}"
LITBANK_INPUT="${3:-/path/to/litbank/events}"

echo "=== Preprocessing ACE-05 ==="
python preprocessing/process_ace05.py \
    --input_dir "${ACE05_INPUT}" \
    --output_dir data/ace05

echo "=== Preprocessing TimeBank ==="
python preprocessing/process_binary.py \
    --dataset timebank \
    --input_dir "${TIMEBANK_INPUT}" \
    --output_dir data/timebank

echo "=== Preprocessing LitBank ==="
python preprocessing/process_binary.py \
    --dataset litbank \
    --input_dir "${LITBANK_INPUT}" \
    --output_dir data/litbank

echo "Done."
