#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval_strict}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
AS_KV_DIR="${AS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_as_c64_plain_v1}"
BUILD_ROOT="${BUILD_ROOT:-$ROOT/results/as_pcache_build}"
PLAN="${PLAN:-$ROOT/configs/qwen25_k005_impress.json}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
MARKER="$AS_KV_DIR/.contiguous_fuxian_complete"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

validate_marker() {
  "$PYTHON" - "$MARKER" <<'PYMARKER'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "schema_version": 3,
    "method": "attentionstore_as_baselines",
    "chunk_size": 64,
    "selector_kv_head_ids": [0, 1, 2, 3],
    "online_selection": True,
    "physical_layout": "plain-logical-token-order",
    "impress_reorder_sha256": None,
    "registered_store_tasks": ["sst2", "subj", "trec", "rte"],
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"{path}: {key}={payload.get(key)!r}, expected {value!r}")
PYMARKER
}

[[ -x "$PYTHON" ]] || fail "Python runtime is missing: $PYTHON"
[[ -f "$PLAN" ]] || fail "shape-only chunk-64 plan is missing: $PLAN"
[[ -f "$BUNDLE_DIR/metadata.json" ]] || fail "strict bundle is missing"
[[ -f "$FLEXGEN_ROOT/my_pcache_fast.py" ]] || fail "HyperInfer Pcache is missing"
for task in sst2 subj trec rte; do
  [[ -f "$STORE_ROOT/$task/metadata.json" ]] || fail "full KV store is missing task $task"
done

if [[ -f "$MARKER" ]]; then
  validate_marker
  echo "Validated existing plain chunk-64 AttentionStore Pcache: $AS_KV_DIR"
  exit 0
fi

if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  fail "GPU $GPU has an active compute process"
fi

mkdir -p "$AS_KV_DIR" "$BUILD_ROOT"
exec 7>"/tmp/prism_max_storage.lock"
flock -n 7 || fail "another Prism-Max process owns the shared-storage lock"
exec 9>"/tmp/prism_max_gpu${GPU}.lock"
flock -n 9 || fail "another Prism-Max process owns GPU lock $GPU"

attempt="$BUILD_ROOT/build_$(date +%Y%m%d_%H%M%S)_$$"
resume_args=()
if find "$AS_KV_DIR" -mindepth 1 -type f -print -quit | grep -q .; then
  resume_args=(--resume-flexgen-kv)
fi

CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$ROOT/src" \
timeout --kill-after=60s "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" \
  -m contiguous_fuxian.flexgen_qwen_reprefill \
  --model-path "$MODEL_PATH" \
  --bundle-dir "$BUNDLE_DIR" \
  --store-root "$STORE_ROOT" \
  --plan "$PLAN" \
  --output-dir "$attempt" \
  --flexgen-root "$FLEXGEN_ROOT" \
  --flexgen-kv-dir "$AS_KV_DIR" \
  --tasks sst2 \
  --store-tasks sst2,subj,trec,rte \
  --samples-per-task 1 \
  --max-tokens 1 \
  --accuracy-scoring label_continuation_loglikelihood \
  --device cuda \
  --dtype bfloat16 \
  --gpu-cache-mb 0 \
  --cpu-cache-mb 0 \
  --cache-type LRU \
  --prefetch-time-budget 10000 \
  --online-selection \
  --period-size 8 \
  --subperiod-size 4 \
  --expected-keep-ratio 0.05 \
  --warmup-passes 0 \
  --probe-query-heads 0,7,14,21 \
  --selector-kv-head-ids 0,1,2,3 \
  --as-baseline-mode as_lru \
  --no-impress-async-prefetch \
  --no-defer-cache-score-updates \
  "${resume_args[@]}"

[[ -s "$attempt/summary.json" ]] || fail "Pcache build run did not finish"
[[ $(find "$AS_KV_DIR" -mindepth 1 -type f | wc -l) -gt 0 ]] || fail "Pcache build produced no files"

"$PYTHON" - "$MARKER" <<'PYMARKER'
import json
import os
import pathlib
import tempfile
import sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 3,
    "method": "attentionstore_as_baselines",
    "chunk_size": 64,
    "selector_kv_head_ids": [0, 1, 2, 3],
    "online_selection": True,
    "physical_layout": "plain-logical-token-order",
    "impress_reorder_sha256": None,
    "registered_store_tasks": ["sst2", "subj", "trec", "rte"],
    "build_semantics": "fresh Qwen FP16 K/V payload; no IMPRESS physical reorder",
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".as-layout-", dir=path.parent, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PYMARKER
validate_marker
echo "Completed plain chunk-64 AttentionStore Pcache: $AS_KV_DIR"
