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
for backend in buffered direct; do
  CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/panzihang/venvs/vllm-stable/bin/python -m prism_gao.benchmark_concurrency \
    --run-root "$RUN_ROOT" --backend "$backend" --phase ready --workers processes \
    --samples-per-task 4 --concurrency 4 --rounds 2 \
    --output "$RUN_ROOT/process_${backend}.jsonl"
done
echo PROCESS_REPLAY_COMPLETE
