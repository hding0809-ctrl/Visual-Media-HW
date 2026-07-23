#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p \
  configs \
  scripts \
  src \
  third_party \
  data \
  checkpoints \
  results/baseline \
  results/tta/images \
  results/csv \
  results/figures \
  results/failure_cases/heavy_blur \
  results/failure_cases/high_saturation_or_dark \
  results/failure_cases/high_texture \
  results/failure_cases/tta_improved_examples \
  logs \
  report/figures

echo "Prepared project directories under $ROOT_DIR"

