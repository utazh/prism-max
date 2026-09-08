#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ROOT="${RUN_ROOT:-$ROOT/src/prism_gao/results/concurrency_20260907}"
PYTHON=/home/panzihang/venvs/vllm-stable/bin/python
exec 7>/tmp/prism_max_storage.lock
flock -n 7
exec 9>/tmp/prism_max_gpu2.lock
flock -n 9
if nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "GPU 2 is occupied" >&2
  exit 3
fi
{
  date -Is
  git status --porcelain
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv,noheader
  findmnt -T /home/panzihang/contiguous_fuxian_ssd/prism_ultra_payload_g32_v1
  cat /sys/block/sda/queue/logical_block_size
} > "$RUN_ROOT/replay_environment.txt"
sha256sum src/contiguous_fuxian/*.py > "$RUN_ROOT/original_source_before.sha256"
for backend in buffered direct; do
  CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" -m prism_gao.benchmark_concurrency \
    --run-root "$RUN_ROOT" --backend "$backend" --phase ready \
    --samples-per-task 4 --concurrency 1 2 4 --rounds 2 \
    --output "$RUN_ROOT/replay_${backend}.jsonl"
done
sha256sum -c "$RUN_ROOT/original_source_before.sha256" > "$RUN_ROOT/original_source_verified.txt"
echo REPLAY_COMPLETE
