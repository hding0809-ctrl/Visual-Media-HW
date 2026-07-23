#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NAFNET_ROOT="${NAFNET_ROOT:-$ROOT_DIR/third_party/NAFNet}"
ZIP_PATH="${GOPRO_ZIP:-$ROOT_DIR/downloads/GOPRO_Large.zip}"
RAW_DIR="${GOPRO_RAW_DIR:-$ROOT_DIR/data/gopro_raw}"

cd "$ROOT_DIR"

"$PYTHON_BIN" -m src.prepare_gopro_lmdb \
  --zip "$ZIP_PATH" \
  --raw-dir "$RAW_DIR" \
  --nafnet-root "$NAFNET_ROOT"

