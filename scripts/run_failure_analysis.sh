#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAFNET_ROOT="${NAFNET_ROOT:-$ROOT_DIR/third_party/NAFNet}"
INPUT_PATH="${GOPRO_INPUT:-$NAFNET_ROOT/datasets/GoPro/test/input.lmdb}"
GT_PATH="${GOPRO_GT:-$NAFNET_ROOT/datasets/GoPro/test/target.lmdb}"
PRED_PATH="${BASELINE_PRED:-$NAFNET_ROOT/results/NAFNet-GoPro-width64-test/visualization/gopro-test}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -m src.failure_mining \
  --input "$INPUT_PATH" \
  --gt "$GT_PATH" \
  --pred "$PRED_PATH" \
  --output-dir "$ROOT_DIR/results" \
  --config "$ROOT_DIR/configs/failure_analysis.yaml"
