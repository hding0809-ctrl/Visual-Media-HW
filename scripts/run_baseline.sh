#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAFNET_ROOT="${NAFNET_ROOT:-$ROOT_DIR/third_party/NAFNet}"
INPUT_LMDB="${GOPRO_INPUT_LMDB:-$NAFNET_ROOT/datasets/GoPro/test/input.lmdb}"
GT_LMDB="${GOPRO_GT_LMDB:-$NAFNET_ROOT/datasets/GoPro/test/target.lmdb}"
CHECKPOINT="${NAFNET_CHECKPOINT:-$NAFNET_ROOT/experiments/pretrained_models/NAFNet-GoPro-width64.pth}"
OPT_FILE="${NAFNET_OPT:-$NAFNET_ROOT/options/test/GoPro/NAFNet-width64.yml}"
LOG_FILE="$ROOT_DIR/logs/baseline_eval.log"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$ROOT_DIR/logs"

if [[ ! -d "$NAFNET_ROOT" ]]; then
  echo "ERROR: NAFNet repository not found: $NAFNET_ROOT" >&2
  echo "Clone official NAFNet to third_party/NAFNet first." >&2
  exit 1
fi

if [[ ! -d "$INPUT_LMDB" ]]; then
  echo "ERROR: GoPro input LMDB not found: $INPUT_LMDB" >&2
  exit 1
fi

if [[ ! -d "$GT_LMDB" ]]; then
  echo "ERROR: GoPro target LMDB not found: $GT_LMDB" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$OPT_FILE" ]]; then
  echo "ERROR: test config not found: $OPT_FILE" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$NAFNET_ROOT:${PYTHONPATH:-}"

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Writing log to $LOG_FILE"

cd "$NAFNET_ROOT"

"$PYTHON_BIN" -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-4321}" \
  --use_env \
  basicsr/test.py \
  -opt ./options/test/GoPro/NAFNet-width64.yml \
  --launcher pytorch 2>&1 | tee "$LOG_FILE"
