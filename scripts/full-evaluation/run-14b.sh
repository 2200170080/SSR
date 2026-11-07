#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY="$HOME/miniconda3/envs/ssr/bin/python"
[[ -x "$PY" ]] || { echo "找不到解释器：$PY" >&2; exit 1; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "$PY" -u "$HERE/launcher.py" 14 "$@"
