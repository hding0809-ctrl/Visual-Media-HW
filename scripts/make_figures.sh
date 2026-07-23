#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT_DIR"

"$PYTHON_BIN" -m src.visualization \
  --summary-csv results/csv/failure_group_summary.csv \
  --baseline-csv results/csv/baseline_per_image_metrics.csv \
  --before-after-csv results/csv/before_after_summary.csv \
  --output-dir results/figures
