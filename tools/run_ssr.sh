#!/bin/bash
# run_ssr.sh -- SSR pipeline launcher that does NOT depend on `conda activate`.
# Usage: bash ~/SSR/tools/run_ssr.sh [check|smoke1|smoke20|smoke200|full]
# Every run clears artifacts/llm_cache.jsonl first; prefix SSR_KEEP_CACHE=1 to resume instead.
# All output goes to both the terminal (tmux pane) and logs/<tag>-<MMDD-HHMM>.log
set -u

PY="$HOME/miniconda3/envs/ssr/bin/python"
ROOT="$HOME/SSR"
MODEL="$ROOT/models/bge-m3"
CODE="src/run_pipeline.py"

[ -x "$PY" ] || { echo "ERROR: python not found: $PY"; exit 1; }
[ -d "$ROOT" ] || { echo "ERROR: run root not found: $ROOT"; exit 1; }
[ -f "$ROOT/$CODE" ] || { echo "ERROR: pipeline not found: $ROOT/$CODE"; exit 1; }

MODE="${1:-full}"

# FAISS/OpenMP: this node has 128 logical cpus and is shared (load ~150).
# Letting OpenMP grab all 128 threads makes flat-IP search ~14x SLOWER (89 s vs 6.2 s per batch).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

case "$MODE" in
  check)    ARGS=() ;;
  smoke1)   ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --sample-size 1 --concurrency 1) ;;
  smoke20)  ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --sample-size 20 --concurrency 8) ;;
  smoke200) ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --sample-size 200 --concurrency 8 --random-seed 7) ;;
  full)     ARGS=(--run-text-retrieval --run-graph-retrieval --run-graph-qa --concurrency 8) ;;
  *) echo "usage: bash ~/SSR/tools/run_ssr.sh [check|smoke1|smoke20|smoke200|full]"; exit 2 ;;
esac

echo "[run_ssr] mode=$MODE"
echo "[run_ssr] python=$PY  OMP_NUM_THREADS=$OMP_NUM_THREADS"
"$PY" -c 'import sys,torch,transformers;print("[run_ssr] python",sys.version.split()[0],"torch",torch.__version__,"cuda",torch.cuda.is_available(),"tf",transformers.__version__)'

if [ "$MODE" = "check" ]; then
  "$PY" -c 'import faiss;print("[run_ssr] faiss.omp_get_max_threads()=",faiss.omp_get_max_threads())'
  echo "[run_ssr] check OK (launcher works without conda activate)"
  exit 0
fi

cd "$ROOT" || { echo "ERROR: cd $ROOT failed"; exit 1; }
export PYTHONUNBUFFERED=1
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL_NAME="qwen14b"

# 每次实验前自动清缓存；断点续跑时 SSR_KEEP_CACHE=1 可跳过
if [ "${SSR_KEEP_CACHE:-0}" = "1" ]; then
  echo "[run_ssr] keep llm_cache.jsonl (resume mode: SSR_KEEP_CACHE=1)"
else
  rm -f "$ROOT/artifacts/llm_cache.jsonl"
  echo "[run_ssr] cleared artifacts/llm_cache.jsonl (旧缓存会让 Step 2 假加速)"
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${MODE}-$(date +%m%d-%H%M).log"
echo "[run_ssr] start=$(date '+%F %T')  log=$LOG"
"$PY" "$CODE" "${ARGS[@]}" --text-retriever-model-name "$MODEL" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[run_ssr] end=$(date '+%F %T')  exit=$RC"
exit "$RC"
