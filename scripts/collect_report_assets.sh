#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p report/figures

if compgen -G "results/figures/*.png" > /dev/null; then
  cp -p results/figures/*.png report/figures/
fi

if compgen -G "results/failure_cases/*/*.png" > /dev/null; then
  for group_dir in results/failure_cases/*; do
    [[ -d "$group_dir" ]] || continue
    group="$(basename "$group_dir")"
    find "$group_dir" -maxdepth 1 -type f -name "*.png" | sort | while read -r file; do
      base="$(basename "$file")"
      cp -p "$file" "report/figures/${group}_${base}"
    done
  done
fi

echo "Collected report figures in $ROOT_DIR/report/figures"
