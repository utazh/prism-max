#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/panzihang/src/prism_fig9_11_20260803}"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-/usr/bin/python3}"
GPU="${GPU:-3}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/fig9_11_qwen7_128_20260803}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-/home/panzihang/src/prism_phase_ab_20260730/data/paper_task_bundles_32}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
CONTIG_KV_DIR="${CONTIG_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
OURS_KV_DIR="${OURS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c16_head0_blockselect_v1}"
IMPRESS_KV_DIR="${IMPRESS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c64_gqa_unique_reordered_disjoint_v33}"
IMPRESS_REORDER="${IMPRESS_REORDER:-/home/panzihang/src/contiguous_fuxian/results/impress_reorder/qwen25_7b_paper4_disjoint_history32_35_v2.json}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
MANIFEST="$ROOT/configs/fig9_11_manifest.json"
SETTLE_SECONDS="${SETTLE_SECONDS:-10}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-43200}"

for required in \
  "$PYTHON" \
  "$ANALYSIS_PYTHON" \
  "$BUNDLE_DIR/metadata.json" \
  "$CONTIG_KV_DIR/.contiguous_fuxian_complete" \
  "$OURS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_REORDER" \
  "$FLEXGEN_ROOT/my_pcache_fast.py" \
  "$MANIFEST"; do
  [[ -f "$required" ]] || {
    echo "Required file is missing: $required" >&2
    exit 2
  }
done

mkdir -p "$RUN_ROOT/environment"
exec 9>"/tmp/prism_paper_matrix_gpu${GPU}.lock"
flock -n 9 || {
  echo "Another PRISM matrix run owns GPU lock $GPU" >&2
  exit 3
}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src"

guard_gpu() {
  local active
  active="$(
    nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader |
      tr -d '[:space:]'
  )"
  [[ -z "$active" ]] || {
    echo "GPU $GPU has active compute process $active; refusing to overlap" >&2
    exit 3
  }
}

snapshot_environment() {
  local label="$1"
  {
    date -Is
    nvidia-smi -i "$GPU" \
      --query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu \
      --format=csv,noheader
    awk '$3 == "sda" {print}' /proc/diskstats
  } >"$RUN_ROOT/environment/$label.txt"
}

run_variant() {
  local label="$1"
  local family="$2"
  local budget_tag="$3"
  local keep_ratio="$4"
  local profile="$5"
  local plan="$ROOT/configs/qwen25_k${budget_tag}_${family}.json"
  local output="$RUN_ROOT/$label"
  local kv_dir
  local method_args=()
  local profile_args=()

  if [[ -f "$output/summary.json" ]]; then
    echo "Skipping completed $label"
    return
  fi
  if [[ -d "$output" ]]; then
    mv "$output" "$output.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  if [[ "$family" == "contigkv" ]]; then
    kv_dir="$CONTIG_KV_DIR"
    method_args=(
      --probe-query-heads 0,1,2
      --selector-kv-head-ids 0,1,2,3
      --similarity-alpha 0.6
    )
  elif [[ "$family" == "impress" ]]; then
    kv_dir="$IMPRESS_KV_DIR"
    method_args=(
      --probe-query-heads 0,1,2
      --selector-kv-head-ids 0
      --similarity-alpha 0.6
      --impress-reorder-manifest "$IMPRESS_REORDER"
      --no-impress-async-prefetch
    )
  elif [[ "$family" == "ours" ]]; then
    kv_dir="$OURS_KV_DIR"
    method_args=(
      --probe-query-heads 0,1,2
      --selector-kv-head-ids 0
      --similarity-alpha 1.0
      --impress-selection-block-size 16
      --impress-async-prefetch
      --impress-period-prefetch-size 4
      --impress-period-prefetch-budget-scale 0.25
      --no-impress-priority-prefetch
      --no-impress-deferred-compute-timing
      --no-impress-rolling-period-prefetch
      --impress-value-ordered-prefetch
      --impress-value-prefetch-budget-scale 0.9
      --exact-layer-block-budget
    )
    profile_args=(--layer-budget-profile "$profile")
  else
    echo "Unknown family: $family" >&2
    exit 2
  fi

  [[ -f "$plan" ]] || {
    echo "Plan is missing: $plan" >&2
    exit 2
  }
  guard_gpu
  snapshot_environment "$label.before"
  echo "[$(date -Is)] starting $label"
  timeout "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" \
    -m contiguous_fuxian.flexgen_qwen_reprefill \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --store-root "$STORE_ROOT" \
    --plan "$plan" \
    --output-dir "$output" \
    --flexgen-root "$FLEXGEN_ROOT" \
    --flexgen-kv-dir "$kv_dir" \
    --tasks sst2,subj,trec,rte \
    --samples-per-task 32 \
    --max-tokens 4 \
    --device cuda \
    --dtype bfloat16 \
    --gpu-cache-mb 55 \
    --cpu-cache-mb 131 \
    --cache-type CKLFU \
    --prefetch-time-budget 10000 \
    --online-selection \
    --reuse-flexgen-kv \
    --period-size 8 \
    --subperiod-size 4 \
    --expected-keep-ratio "$keep_ratio" \
    --warmup-passes 1 \
    "${method_args[@]}" \
    "${profile_args[@]}" \
    >"$RUN_ROOT/$label.log" 2>&1
  snapshot_environment "$label.after"
  echo "[$(date -Is)] completed $label"
  sleep "$SETTLE_SECONDS"
}

guard_gpu
snapshot_environment matrix.start
"$PYTHON" -m unittest \
  tests.test_flexgen_pcache \
  tests.test_flexgen_qwen_reprefill \
  tests.test_layer_budget \
  tests.test_prism_period_prefetch \
  tests.test_priority_executor \
  tests.test_paper_matrix_analysis \
  -v \
  >"$RUN_ROOT/unit_tests.log" 2>&1

# Rotated order limits one method from always benefiting from an early or late run.
run_variant k005_impress impress 005 0.05 -
run_variant k005_contiguouskv contigkv 005 0.05 -
run_variant k005_ours ours 005 0.05 "$ROOT/configs/layer_budget_k005_sensitivity.json"

run_variant k010_contiguouskv contigkv 010 0.10 -
run_variant k010_ours ours 010 0.10 "$ROOT/configs/layer_budget_k010_sensitivity.json"
run_variant k010_impress impress 010 0.10 -

run_variant k025_contiguouskv contigkv 025 0.25 -
run_variant k025_ours ours 025 0.25 "$ROOT/configs/layer_budget_k025_scaled.json"
run_variant k025_impress impress 025 0.25 -

run_variant k050_ours ours 050 0.50 "$ROOT/configs/layer_budget_k050_sensitivity.json"
run_variant k050_impress impress 050 0.50 -
run_variant k050_contiguouskv contigkv 050 0.50 -

"$ANALYSIS_PYTHON" "$ROOT/scripts/analyze_paper_matrix.py" \
  --run-root "$RUN_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$RUN_ROOT/report" \
  >"$RUN_ROOT/analysis.log" 2>&1

sha256sum \
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py" \
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py" \
  "$ROOT/vendor/flexgen/my_pcache_fast.py" \
  "$ROOT/scripts/analyze_paper_matrix.py" \
  "$ROOT/scripts/run_paper_matrix.sh" \
  "$ROOT/configs/"*.json \
  "$BUNDLE_DIR/metadata.json" \
  "$CONTIG_KV_DIR/.contiguous_fuxian_complete" \
  "$OURS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_REORDER" \
  >"$RUN_ROOT/source_and_input_sha256.txt"

snapshot_environment matrix.end
cat "$RUN_ROOT/report/fig9_11_report.md"
