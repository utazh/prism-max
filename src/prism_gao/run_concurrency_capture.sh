#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ROOT="${RUN_ROOT:-$ROOT/src/prism_gao/results/concurrency_20260907}"
mkdir -p "$RUN_ROOT/traces"
for task in sst2 subj trec rte; do
  GPU=2 REQUIRE_IDLE_RESERVE=false TASK="$task" BUDGET_TAG=050 \
  METHOD=promixed SELECTOR_BACKEND=k4 SAMPLES_PER_TASK=8 SAMPLE_OFFSET=16 \
  WARMUP_PASSES=1 WARMUP_SAMPLES_PER_TASK=2 SETTLE_SECONDS=0 \
  RUN_ROOT="$RUN_ROOT/capture" RUN_NAME=capture_fp16 \
  PRISM_GAO_CAPTURE_TRACE="$RUN_ROOT/traces/$task.jsonl" \
  PRISM_GAO_SELECTOR_MODE=exact PRISM_GAO_SELECTOR_RESIDENT=true \
  PRISM_GAO_FIXED_PERIOD=0 PRISM_GAO_PERIOD_POLICY=profiled \
  PRISM_GAO_GLOBAL_KEEP_RATIO=0.50 \
  PRISM_GAO_REUSE_PROFILE="$ROOT/src/prism_gao/results/reuse_profile_v4.json" \
  PRISM_GAO_PRECISION_MODE=fp16 PRISM_GAO_HOST_ASYNC_PREFETCH=true \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash src/prism_gao/run_stage_c.sh
done
echo "TRACE_CAPTURE_COMPLETE"
