#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAFNET_ROOT="${NAFNET_ROOT:-$ROOT_DIR/third_party/NAFNet}"
INPUT_PATH="${GOPRO_INPUT:-$NAFNET_ROOT/datasets/GoPro/test/input.lmdb}"
GT_PATH="${GOPRO_GT:-$NAFNET_ROOT/datasets/GoPro/test/target.lmdb}"
CHECKPOINT="${NAFNET_CHECKPOINT:-$NAFNET_ROOT/experiments/pretrained_models/NAFNet-GoPro-width64.pth}"
OPT_FILE="${NAFNET_OPT:-$NAFNET_ROOT/options/test/GoPro/NAFNet-width64.yml}"
BASELINE_PRED="${BASELINE_PRED:-$NAFNET_ROOT/results/NAFNet-GoPro-width64-test/visualization/gopro-test}"
BASELINE_CSV="${BASELINE_CSV:-$ROOT_DIR/results/csv/baseline_per_image_metrics.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE="$ROOT_DIR/logs/tta_eval.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT_DIR/logs"

cd "$ROOT_DIR"

"$PYTHON_BIN" -m src.eval_tta \
  --nafnet-root "$NAFNET_ROOT" \
  --config "$OPT_FILE" \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT_PATH" \
  --gt "$GT_PATH" \
  --output-dir "$ROOT_DIR/results/tta/images" \
  --csv "$ROOT_DIR/results/csv/tta_per_image_metrics.csv" \
  --summary-csv "$ROOT_DIR/results/csv/before_after_summary.csv" \
  --baseline-csv "$BASELINE_CSV" \
  --baseline-pred "$BASELINE_PRED" 2>&1 | tee "$LOG_FILE"
