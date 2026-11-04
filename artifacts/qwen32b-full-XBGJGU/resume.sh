#!/usr/bin/env bash
set -euo pipefail
FULL_RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/SSR"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

"$HOME/miniconda3/envs/ssr/bin/python" -u \
  "$FULL_RUN_DIR/run_full.py" all \
  --run-dir "$FULL_RUN_DIR" \
  2>&1 | tee -a "$FULL_RUN_DIR/full.log"
