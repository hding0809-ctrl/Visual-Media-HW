#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"

if [[ "${1:-}" != "--create" ]]; then
  cat <<'EOF'
This script is intentionally conservative.

It does not create or modify environments unless you pass --create.

Recommended options:

1. Use an existing compatible PyTorch/CUDA environment, then run:
   pip install -r requirements_extra.txt

2. Create the project environment:
   bash scripts/setup_env.sh --create

Official NAFNet target stack:
   Python 3.9.x, PyTorch 1.11.0, CUDA 11.3
EOF
  exit 0
fi

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
  echo "ERROR: conda command was not found: $CONDA_BIN" >&2
  echo "Set CONDA_BIN=/path/to/conda or create the environment manually." >&2
  exit 1
fi

"$CONDA_BIN" env create -f "$ROOT_DIR/environment.yml"

cat <<'EOF'

Environment created.

Next:
  conda activate nafnet-vm
  cd third_party/NAFNet
  pip install -r requirements.txt
  python setup.py develop --no_cuda_ext
EOF

