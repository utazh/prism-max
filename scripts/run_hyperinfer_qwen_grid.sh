#!/usr/bin/env bash
set -euo pipefail

# Official HyperInfer fast-Pcache backend with the Qwen2.5 adapter.
ROOT="${ROOT:-/home/panzihang/src/contiguous_fuxian}"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-3}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles}"
STORE_ROOT="${STORE_ROOT:-/data1/contiguous_fuxian_sparse_kv/qwen25_7b_paper_seed42}"
HYPERINFER_REPO="${HYPERINFER_REPO:-/home/panzihang/src/impress_fuxian/HyperInfer}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$HYPERINFER_REPO/h2o_flexgen/flexgen}"

TASKS="${TASKS:-sst2,subj,trec,rte}"
TASK_SET_ID="${TASK_SET_ID:-paper4}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
MAX_TOKENS="${MAX_TOKENS:-4}"
RATIO_SPECS="${RATIO_SPECS:-005:0.05 010:0.10 025:0.25 050:0.50}"
PLAN_DIR="${PLAN_DIR:-$ROOT/results/hyperinfer_qwen_plans/${TASK_SET_ID}_paper}"
GRID_ID="${GRID_ID:-$(date +%Y%m%d_%H%M%S)}"
GRID_DIR="${GRID_DIR:-$ROOT/results/flexgen_qwen_grid/$GRID_ID}"

ONLINE_SELECTION="${ONLINE_SELECTION:-0}"
PROBE_QUERY_HEADS="${PROBE_QUERY_HEADS:-0,1,2}"
IMPRESS_SELECTOR_KV_HEAD_IDS="${IMPRESS_SELECTOR_KV_HEAD_IDS:-0}"
SIMILARITY_ALPHA="${SIMILARITY_ALPHA:-0.6}"
IMPRESS_ASYNC_PREFETCH="${IMPRESS_ASYNC_PREFETCH:-0}"
IMPRESS_REORDER_MANIFEST="${IMPRESS_REORDER_MANIFEST-$ROOT/results/impress_reorder/qwen25_7b_paper4_disjoint_history32_35_v2.json}"
IMPRESS_REQUIRE_REORDER="${IMPRESS_REQUIRE_REORDER:-1}"
if [[ "$ONLINE_SELECTION" == "1" ]]; then
  default_contig_kv_dir="/data1/contiguous_fuxian_hyperinfer_kv/${TASK_SET_ID}_online_contig_c16"
  if [[ -n "$IMPRESS_REORDER_MANIFEST" ]]; then
    default_impress_kv_dir="/data1/contiguous_fuxian_hyperinfer_kv/${TASK_SET_ID}_online_impress_c64_gqa_unique_reordered_disjoint"
  else
    default_impress_kv_dir="/data1/contiguous_fuxian_hyperinfer_kv/${TASK_SET_ID}_online_impress_c64_gqa_unique"
  fi
else
  default_contig_kv_dir="/data1/contiguous_fuxian_hyperinfer_kv/${TASK_SET_ID}_contig_c16"
  default_impress_kv_dir="/data1/contiguous_fuxian_hyperinfer_kv/${TASK_SET_ID}_impress_c64"
fi
CONTIG_KV_DIR="${CONTIG_KV_DIR:-$default_contig_kv_dir}"
IMPRESS_KV_DIR="${IMPRESS_KV_DIR:-$default_impress_kv_dir}"
CACHE_TYPE="${CACHE_TYPE:-LRU}"
GPU_CACHE_MB="${GPU_CACHE_MB:-0}"
CPU_CACHE_MB="${CPU_CACHE_MB:-0}"
PREFETCH_TIME_BUDGET="${PREFETCH_TIME_BUDGET:-10000}"
WARMUP_PASSES="${WARMUP_PASSES:-0}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3600}"
GENERATE_PLANS="${GENERATE_PLANS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "$PLAN_DIR" "$GRID_DIR"

if [[ "$IMPRESS_ASYNC_PREFETCH" != "0" && "$IMPRESS_ASYNC_PREFETCH" != "1" ]]; then
  echo "IMPRESS_ASYNC_PREFETCH must be 0 or 1" >&2
  exit 2
fi
if [[ "$IMPRESS_REQUIRE_REORDER" != "0" && "$IMPRESS_REQUIRE_REORDER" != "1" ]]; then
  echo "IMPRESS_REQUIRE_REORDER must be 0 or 1" >&2
  exit 2
fi
if [[ "$ONLINE_SELECTION" == "1" && "$IMPRESS_REQUIRE_REORDER" == "1" ]]; then
  [[ -s "$IMPRESS_REORDER_MANIFEST" ]] || {
    echo "Paper IMPRESS requires a reorder manifest: $IMPRESS_REORDER_MANIFEST" >&2
    exit 2
  }
fi
if [[ -n "$IMPRESS_REORDER_MANIFEST" ]]; then
  [[ -s "$IMPRESS_REORDER_MANIFEST" ]] || {
    echo "IMPRESS reorder manifest does not exist: $IMPRESS_REORDER_MANIFEST" >&2
    exit 2
  }
  IMPRESS_REORDER_SHA256="$(sha256sum "$IMPRESS_REORDER_MANIFEST" | awk '{print $1}')"
else
  IMPRESS_REORDER_SHA256=""
fi

is_zero() {
  awk -v value="$1" 'BEGIN { exit !((value + 0) == 0) }'
}

validate_store_layout() {
  local marker="$1"
  local method="$2"
  local chunk_size="$3"
  local selector_heads="$4"
  local reorder_sha256="${5:-}"
  [[ "$ONLINE_SELECTION" == "1" ]] || return 0
  [[ -s "$marker" ]] || {
    echo "Online Pcache marker is missing layout metadata: $marker" >&2
    return 1
  }
  "$PYTHON" - "$marker" "$method" "$chunk_size" "$selector_heads" "$reorder_sha256" <<'PY'
import json
import sys

path, method, chunk_size, selector_heads, reorder_sha256 = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
expected_heads = [int(item) for item in selector_heads.split(",")]
assert int(payload["schema_version"]) in {1, 2}
assert payload["method"] == method
assert int(payload["chunk_size"]) == int(chunk_size)
assert payload["selector_kv_head_ids"] == expected_heads
assert payload["online_selection"] is True
if reorder_sha256:
    assert int(payload["schema_version"]) >= 2
    assert payload.get("impress_reorder_sha256") == reorder_sha256
else:
    assert payload.get("impress_reorder_sha256") in {None, ""}
PY
}

write_store_layout() {
  local marker="$1"
  local method="$2"
  local chunk_size="$3"
  local selector_heads="$4"
  local reorder_sha256="${5:-}"
  "$PYTHON" - "$marker" "$method" "$chunk_size" "$selector_heads" \
    "$ONLINE_SELECTION" "$reorder_sha256" <<'PY'
import json
import os
import sys
import tempfile

path, method, chunk_size, selector_heads, online, reorder_sha256 = sys.argv[1:]
payload = {
    "schema_version": 2 if reorder_sha256 else 1,
    "method": method,
    "chunk_size": int(chunk_size),
    "selector_kv_head_ids": [int(item) for item in selector_heads.split(",")],
    "online_selection": online == "1",
    "impress_reorder_sha256": reorder_sha256 or None,
}
directory = os.path.dirname(path)
fd, temporary = tempfile.mkstemp(prefix=".layout-", dir=directory, text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

is_completed_run() {
  local summary="$1"
  local expected_method="$2"
  local expected_ratio="$3"
  local expected_async="$4"
  local expected_reorder_sha256="$5"
  [[ -f "$summary" ]] || return 1
  "$PYTHON" - "$summary" "$expected_method" "$expected_ratio" \
    "$MODEL_DTYPE" "$TASKS" "$SAMPLES_PER_TASK" "$ONLINE_SELECTION" \
    "$expected_async" "$expected_reorder_sha256" <<'PY'
import json
import math
import sys

path, method, ratio, dtype, tasks, samples_per_task, online_selection, expected_async, reorder_sha256 = sys.argv[1:]
summary = json.load(open(path, encoding="utf-8"))
runtime = summary["runtime"]
expected_samples = len([task for task in tasks.split(",") if task]) * int(samples_per_task)
assert runtime["method"] == method
assert math.isclose(float(runtime["keep_ratio"]), float(ratio), rel_tol=0.0, abs_tol=1e-12)
assert runtime["model_compute_dtype"] == dtype
assert bool(runtime.get("online_selection", False)) == (online_selection == "1")
assert bool(runtime.get("impress_async_inter_layer_prefetch", False)) == (expected_async == "1")
assert bool(runtime.get("impress_reorder_enabled", False)) == bool(reorder_sha256)
assert (runtime.get("impress_reorder_sha256") or "") == reorder_sha256
assert int(summary["overall"]["samples"]) == expected_samples
PY
}

if [[ ! -f "$FLEXGEN_ROOT/my_pcache_fast.py" ]]; then
  echo "HyperInfer fast Pcache was not found under $FLEXGEN_ROOT" >&2
  exit 2
fi

if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU $GPU has an active compute process; refusing to overlap runs" >&2
  exit 2
fi

declare -a report_args=()
for spec in $RATIO_SPECS; do
  tag="${spec%%:*}"
  ratio="${spec#*:}"
  if [[ -z "$tag" || "$tag" == "$ratio" ]]; then
    echo "Ratio spec must be TAG:FRACTION, got $spec" >&2
    exit 2
  fi

  contig_plan="$PLAN_DIR/k${tag}_contig.json"
  impress_plan="$PLAN_DIR/k${tag}_impress.json"
  if [[ "$GENERATE_PLANS" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH="$ROOT/src" "$PYTHON" \
      -m contiguous_fuxian.paper_plan_generator \
      --model-path "$MODEL_PATH" \
      --bundle-dir "$BUNDLE_DIR" \
      --tasks "$TASKS" \
      --samples-per-task "$SAMPLES_PER_TASK" \
      --keep-ratio "$ratio" \
      --contiguous-output "$contig_plan" \
      --impress-output "$impress_plan" \
      --contiguous-chunk-size 16 \
      --impress-chunk-size 64 \
      --period-size 8 \
      --subperiod-size 4 \
      --probe-heads 0,1,2 \
      --similarity-alpha 0.6 \
      --int8-layers 0 \
      --max-prompt-tokens 8192 \
      --device cuda \
      --allow-gpu \
      --dtype "$MODEL_DTYPE" \
      > "$GRID_DIR/k${tag}_plan.log" 2>&1
  fi

  for method in contig impress; do
    plan_var="${method}_plan"
    plan="${!plan_var}"
    if [[ "$method" == "contig" ]]; then
      kv_dir="$CONTIG_KV_DIR"
      selector_kv_head_ids="0,1,2,3"
      store_chunk_size="16"
      expected_async="0"
      reorder_sha256=""
    else
      kv_dir="$IMPRESS_KV_DIR"
      selector_kv_head_ids="$IMPRESS_SELECTOR_KV_HEAD_IDS"
      store_chunk_size="64"
      expected_async="$IMPRESS_ASYNC_PREFETCH"
      reorder_sha256="$IMPRESS_REORDER_SHA256"
    fi
    online_args=()
    if [[ "$ONLINE_SELECTION" == "1" ]]; then
      online_args=(
        --online-selection
        --probe-query-heads "$PROBE_QUERY_HEADS"
        --selector-kv-head-ids "$selector_kv_head_ids"
        --similarity-alpha "$SIMILARITY_ALPHA"
      )
    fi
    if [[ "$method" == "impress" ]]; then
      if [[ "$IMPRESS_ASYNC_PREFETCH" == "1" ]]; then
        online_args+=(--impress-async-prefetch)
      fi
      if [[ -n "$IMPRESS_REORDER_MANIFEST" ]]; then
        online_args+=(--impress-reorder-manifest "$IMPRESS_REORDER_MANIFEST")
      fi
    fi
    expected_method="$method"
    if [[ "$method" == "contig" ]]; then
      expected_method="contigkv"
    fi
    run_dir="$GRID_DIR/k${tag}_${method}"
    if [[ "$SKIP_COMPLETED" == "1" ]] && \
      is_completed_run "$run_dir/summary.json" "$expected_method" "$ratio" \
        "$expected_async" "$reorder_sha256"; then
      echo "Skipping completed $method run at keep ratio $ratio"
      continue
    fi

    persistence_args=()
    complete_marker="$kv_dir/.contiguous_fuxian_complete"
    store_has_files=0
    if [[ -d "$kv_dir" ]] && find "$kv_dir" -mindepth 1 -print -quit | grep -q .; then
      store_has_files=1
    fi
    if [[ -f "$complete_marker" ]]; then
      validate_store_layout \
        "$complete_marker" "$expected_method" "$store_chunk_size" \
        "$selector_kv_head_ids" "$reorder_sha256"
      persistence_args=(--reuse-flexgen-kv)
    elif [[ "$store_has_files" == "1" ]]; then
      persistence_args=(--resume-flexgen-kv)
    fi

    if [[ ! -f "$complete_marker" ]] && { ! is_zero "$GPU_CACHE_MB" || ! is_zero "$CPU_CACHE_MB"; }; then
      prebuild_args=()
      if [[ "$store_has_files" == "1" ]]; then
        prebuild_args=(--resume-flexgen-kv)
      fi
      prebuild_dir="$GRID_DIR/_prebuild_k${tag}_${method}"
      CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT/src" \
        timeout "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" -m contiguous_fuxian.flexgen_qwen_reprefill \
        --model-path "$MODEL_PATH" \
        --bundle-dir "$BUNDLE_DIR" \
        --store-root "$STORE_ROOT" \
        --plan "$plan" \
        --output-dir "$prebuild_dir" \
        --flexgen-root "$FLEXGEN_ROOT" \
        --flexgen-kv-dir "$kv_dir" \
        --tasks "$TASKS" \
        --samples-per-task 1 \
        --max-tokens 1 \
        --device cuda \
        --dtype "$MODEL_DTYPE" \
        --gpu-cache-mb 0 \
        --cpu-cache-mb 0 \
        --cache-type LRU \
        --prefetch-time-budget "$PREFETCH_TIME_BUDGET" \
        --period-size 8 \
        --subperiod-size 4 \
        --expected-keep-ratio "$ratio" \
        --warmup-passes 0 \
        "${online_args[@]}" \
        "${prebuild_args[@]}" \
        > "$GRID_DIR/_prebuild_k${tag}_${method}.log" 2>&1
      write_store_layout \
        "$complete_marker" "$expected_method" "$store_chunk_size" \
        "$selector_kv_head_ids" "$reorder_sha256"
      persistence_args=(--reuse-flexgen-kv)
    fi

    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT/src" \
      timeout "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" -m contiguous_fuxian.flexgen_qwen_reprefill \
      --model-path "$MODEL_PATH" \
      --bundle-dir "$BUNDLE_DIR" \
      --store-root "$STORE_ROOT" \
      --plan "$plan" \
      --output-dir "$run_dir" \
      --flexgen-root "$FLEXGEN_ROOT" \
      --flexgen-kv-dir "$kv_dir" \
      --tasks "$TASKS" \
      --samples-per-task "$SAMPLES_PER_TASK" \
      --max-tokens "$MAX_TOKENS" \
      --device cuda \
      --dtype "$MODEL_DTYPE" \
      --gpu-cache-mb "$GPU_CACHE_MB" \
      --cpu-cache-mb "$CPU_CACHE_MB" \
      --cache-type "$CACHE_TYPE" \
      --prefetch-time-budget "$PREFETCH_TIME_BUDGET" \
      --period-size 8 \
      --subperiod-size 4 \
      --expected-keep-ratio "$ratio" \
      --warmup-passes "$WARMUP_PASSES" \
      "${online_args[@]}" \
      "${persistence_args[@]}" \
      > "$GRID_DIR/k${tag}_${method}.log" 2>&1
    if is_zero "$GPU_CACHE_MB" && is_zero "$CPU_CACHE_MB"; then
      write_store_layout \
        "$complete_marker" "$expected_method" "$store_chunk_size" \
        "$selector_kv_head_ids" "$reorder_sha256"
    fi
  done

  comparison="$GRID_DIR/k${tag}_comparison.json"
  PYTHONPATH="$ROOT/src" "$PYTHON" -m contiguous_fuxian.sparse_qwen_compare \
    --contiguous-run "$GRID_DIR/k${tag}_contig" \
    --impress-run "$GRID_DIR/k${tag}_impress" \
    --output "$comparison" \
    > "$GRID_DIR/k${tag}_comparison.log" 2>&1
  report_args+=(--comparison "$ratio=$comparison")
done

PYTHONPATH="$ROOT/src" "$PYTHON" -m contiguous_fuxian.paper_grid_report \
  "${report_args[@]}" \
  --output-json "$GRID_DIR/summary.json" \
  --output-markdown "$GRID_DIR/summary.md"

git -C "$HYPERINFER_REPO" rev-parse HEAD > "$GRID_DIR/hyperinfer_commit.txt"
sha256sum \
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py" \
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py" \
  "$ROOT/src/contiguous_fuxian/impress_reorder.py" \
  "$ROOT/src/contiguous_fuxian/paper_plan_generator.py" \
  "$ROOT/scripts/run_hyperinfer_qwen_grid.sh" \
  > "$GRID_DIR/reproduction_source_sha256.txt"
input_files=("$BUNDLE_DIR/metadata.json")
for task in ${TASKS//,/ }; do
  input_files+=("$BUNDLE_DIR/$task.jsonl")
done
if [[ -n "$IMPRESS_REORDER_MANIFEST" ]]; then
  input_files+=("$IMPRESS_REORDER_MANIFEST")
fi
sha256sum "${input_files[@]}" > "$GRID_DIR/reproduction_input_sha256.txt"
echo "$GRID_DIR"
