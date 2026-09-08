#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ROOT="$ROOT/src/prism_gao/results/concurrency_20260907"
exec 7>/tmp/prism_max_storage.lock
flock -w 1200 7
exec 9>/tmp/prism_max_gpu2.lock
flock -n 9
if nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "GPU 2 is occupied" >&2
  exit 3
fi
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/panzihang/venvs/vllm-stable/bin/python -m prism_gao.benchmark_concurrency \
  --run-root "$RUN_ROOT" --backend buffered --phase ready --workers threads \
  --samples-per-task 4 --concurrency 1 2 4 --rounds 2 \
  --output "$RUN_ROOT/replay_buffered_warm.jsonl"
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/panzihang/venvs/vllm-stable/bin/python -m pytest -q \
  -o cache_dir="$RUN_ROOT/pytest_cache" src/prism_gao/tests \
  > "$RUN_ROOT/tests.txt" 2>&1
sha256sum -c "$RUN_ROOT/original_source_before.sha256" > "$RUN_ROOT/original_source_verified.txt"
echo VALIDATION_COMPLETE
