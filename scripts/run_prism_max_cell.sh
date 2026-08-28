#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-0}"
RESERVE_GPU="${RESERVE_GPU:-3}"
REQUIRE_IDLE_RESERVE="${REQUIRE_IDLE_RESERVE:-true}"
TASK="${TASK:-trec}"
BUDGET_TAG="${BUDGET_TAG:-010}"
METHOD="${METHOD:-promixed}"
SELECTOR_BACKEND="${SELECTOR_BACKEND:-k4}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-128}"
WARMUP_PASSES="${WARMUP_PASSES:-1}"
WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-32}"
DEFER_CACHE_SCORE_UPDATES="${DEFER_CACHE_SCORE_UPDATES:-false}"
PROMIXED_ADAPTIVE_COVERAGE="${PROMIXED_ADAPTIVE_COVERAGE:-false}"
PROMIXED_COVERAGE_FRACTION="${PROMIXED_COVERAGE_FRACTION:-0.5}"
PROMIXED_MARGIN_REFERENCE="${PROMIXED_MARGIN_REFERENCE:-0.05}"
PROMIXED_AGREEMENT_WEIGHT="${PROMIXED_AGREEMENT_WEIGHT:-0.75}"
PROMIXED_SENSITIVITY_WEIGHT="${PROMIXED_SENSITIVITY_WEIGHT:-0.1}"
PROMIXED_P1_THRESHOLD="${PROMIXED_P1_THRESHOLD:-0.90}"
PROMIXED_P2_THRESHOLD="${PROMIXED_P2_THRESHOLD:-0.82}"
PROMIXED_P4_THRESHOLD="${PROMIXED_P4_THRESHOLD:-0.68}"
PROMIXED_UTILITY_MAX_WEIGHT="${PROMIXED_UTILITY_MAX_WEIGHT:-0.55}"
PROMIXED_UTILITY_MEAN_WEIGHT="${PROMIXED_UTILITY_MEAN_WEIGHT:-0.35}"
PROMIXED_UTILITY_VOTE_WEIGHT="${PROMIXED_UTILITY_VOTE_WEIGHT:-0.10}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-86400}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/prism_max_screen_20260824}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
GPU_CACHE_MB="${GPU_CACHE_MB:-55}"
CPU_CACHE_MB="${CPU_CACHE_MB:-131}"

MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
IMPRESS_KV_DIR="${IMPRESS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c64_gqa_unique_reordered_disjoint_v33}"
IMPRESS_REORDER="${IMPRESS_REORDER:-/home/panzihang/src/contiguous_fuxian/results/impress_reorder/qwen25_7b_paper4_disjoint_history32_35_v2.json}"
AS_KV_DIR="${AS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_as_c64_plain_v1}"
SELECTOR_INDEX="${SELECTOR_INDEX:-$ROOT/assets/selector_index_k4_g32}"

gpu_has_compute_process() {
  nvidia-smi -i "$1" \
    --query-compute-apps=pid \
    --format=csv,noheader 2>/dev/null |
    grep -Eq '[0-9]'
}

guard_gpus() {
  case "$REQUIRE_IDLE_RESERVE" in
    true|false) ;;
    *)
      echo "REQUIRE_IDLE_RESERVE must be true or false" >&2
      exit 2
      ;;
  esac
  if [[ "$REQUIRE_IDLE_RESERVE" == "true" && "$GPU" == "$RESERVE_GPU" ]]; then
    echo "GPU and RESERVE_GPU must differ" >&2
    exit 2
  fi
  if gpu_has_compute_process "$GPU"; then
    echo "Refusing to start: experiment GPU $GPU is occupied" >&2
    exit 3
  fi
  if [[ "$REQUIRE_IDLE_RESERVE" == "true" ]] &&
     gpu_has_compute_process "$RESERVE_GPU"; then
    echo "Refusing to start: reserve GPU $RESERVE_GPU is occupied" >&2
    exit 3
  fi
}

budget_ratio() {
  case "$1" in
    005) echo 0.05 ;;
    010) echo 0.10 ;;
    025) echo 0.25 ;;
    050) echo 0.50 ;;
    *) echo "Unsupported budget tag: $1" >&2; return 2 ;;
  esac
}

budget_profile() {
  case "$1" in
    005) echo "$ROOT/configs/layer_budget_k005_sensitivity.json" ;;
    010) echo "$ROOT/configs/layer_budget_k010_sensitivity.json" ;;
    025) echo "$ROOT/configs/layer_budget_k025_scaled.json" ;;
    050) echo "$ROOT/configs/layer_budget_k050_sensitivity.json" ;;
    *) echo "Unsupported budget tag: $1" >&2; return 2 ;;
  esac
}

json_string_field() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
field = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload.get(field)
if not isinstance(value, str) or not value:
    raise SystemExit(f"{path} is missing non-empty string field {field!r}")
print(value)
PY
}

validate_as_completion_marker() {
  "$PYTHON" - "$1" <<'PY'
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
problems = []
for field, expected_value in expected.items():
    if field not in payload:
        problems.append(f"missing {field!r}")
    elif payload[field] != expected_value:
        problems.append(
            f"{field}={payload[field]!r}, expected {expected_value!r}"
        )
if problems:
    raise SystemExit(f"invalid AS completion marker {path}: " + "; ".join(problems))
PY
}

case "$SELECTOR_BACKEND" in
  fp16) index_args=() ;;
  k4)
    index_args=(--selector-index-dir "$SELECTOR_INDEX")
    [[ -f "$SELECTOR_INDEX/manifest.json" ]] || {
      echo "K4 selector index is missing: $SELECTOR_INDEX" >&2
      exit 2
    }
    ;;
  *)
    echo "SELECTOR_BACKEND must be fp16 or k4" >&2
    exit 2
    ;;
esac

case "$DEFER_CACHE_SCORE_UPDATES" in
  true)
    cache_score_args=(--defer-cache-score-updates)
    score_tag="defer"
    ;;
  false)
    cache_score_args=(--no-defer-cache-score-updates)
    score_tag="nodefer"
    ;;
  *)
    echo "DEFER_CACHE_SCORE_UPDATES must be true or false" >&2
    exit 2
    ;;
esac

case "$PROMIXED_ADAPTIVE_COVERAGE" in
  true)
    adaptive_coverage_args=(--promixed-adaptive-coverage)
    coverage_tag="_adaptive"
    ;;
  false)
    adaptive_coverage_args=(--no-promixed-adaptive-coverage)
    coverage_tag=""
    ;;
  *)
    echo "PROMIXED_ADAPTIVE_COVERAGE must be true or false" >&2
    exit 2
    ;;
esac

if [[ "$BUDGET_TAG" == "full" ]]; then
  if [[ "$METHOD" != "as_lru" ]]; then
    echo "BUDGET_TAG=full is only valid for budget-independent AS+LRU" >&2
    exit 2
  fi
  # AS+LRU retains all K/V. The 5% plan is shape-only metadata for the runtime.
  KEEP_RATIO="0.05"
  PROFILE=""
else
  KEEP_RATIO="$(budget_ratio "$BUDGET_TAG")"
  PROFILE="$(budget_profile "$BUDGET_TAG")"
fi
method_args=()
profile_args=()
selected_reorder_manifest=""
CACHE_TYPE="CKLFU"
AS_BASELINE_MODE="none"
BUDGET_SEMANTICS="selected-kv-retention"
AS_FULL_KEY_RATIO="not-applicable"
AS_VALUE_KEEP_RATIO="not-applicable"
AS_LOGICAL_TOTAL_KV_RATIO="not-applicable"

case "$METHOD" in
  contigkv)
    PLAN="$ROOT/configs/qwen25_k${BUDGET_TAG}_contigkv.json"
    method_args=(
      --probe-query-heads 0,1,2
      --selector-kv-head-ids 0,1,2,3
      --similarity-alpha 0.6
    )
    ;;
  impress)
    if [[ "$SELECTOR_BACKEND" != "fp16" ]]; then
      echo "Canonical IMPRESS requires SELECTOR_BACKEND=fp16; its physical reorder is incompatible with K4 selector indexes" >&2
      exit 2
    fi
    PLAN="$ROOT/configs/qwen25_k${BUDGET_TAG}_impress.json"
    KV_DIR="$IMPRESS_KV_DIR"
    selected_reorder_manifest="$IMPRESS_REORDER"
    method_args=(
      --probe-query-heads 0,1,2
      --selector-kv-head-ids 0
      --similarity-alpha 0.6
      --impress-reorder-manifest "$selected_reorder_manifest"
      --no-impress-async-prefetch
    )
    ;;
  promixed)
    PLAN="$ROOT/configs/qwen25_k${BUDGET_TAG}_ours.json"
    profile_args=(--layer-budget-profile "$PROFILE")
    method_args=(
      --probe-query-heads 0,7,14,21
      --selector-kv-head-ids 0,1,2,3
      --similarity-alpha 0.6
      --impress-selection-block-size 16
      --impress-selection-period-size 8
      --promixed-gqa-selection
      --promixed-coverage-fraction "$PROMIXED_COVERAGE_FRACTION"
      --promixed-margin-reference "$PROMIXED_MARGIN_REFERENCE"
      --promixed-agreement-weight "$PROMIXED_AGREEMENT_WEIGHT"
      --promixed-sensitivity-weight "$PROMIXED_SENSITIVITY_WEIGHT"
      --promixed-p1-threshold "$PROMIXED_P1_THRESHOLD"
      --promixed-p2-threshold "$PROMIXED_P2_THRESHOLD"
      --promixed-p4-threshold "$PROMIXED_P4_THRESHOLD"
      --promixed-utility-max-weight "$PROMIXED_UTILITY_MAX_WEIGHT"
      --promixed-utility-mean-weight "$PROMIXED_UTILITY_MEAN_WEIGHT"
      --promixed-utility-vote-weight "$PROMIXED_UTILITY_VOTE_WEIGHT"
      "${adaptive_coverage_args[@]}"
      --impress-async-prefetch
      --impress-period-prefetch-size 1
      --no-impress-priority-prefetch
      --no-impress-deferred-compute-timing
      --no-impress-rolling-period-prefetch
      --impress-value-ordered-prefetch
      --impress-value-prefetch-budget-scale 0.9
      --exact-layer-block-budget
      --impress-known-period-prefetch
    )
    ;;
  as_lru)
    if [[ "$SELECTOR_BACKEND" != "fp16" ]]; then
      echo "AS+LRU requires SELECTOR_BACKEND=fp16; selector indexes are not part of the original baseline" >&2
      exit 2
    fi
    if [[ "$BUDGET_TAG" != "full" && "$BUDGET_TAG" != "005" ]]; then
      echo "AS+LRU is budget-independent; use BUDGET_TAG=full (or 005 as a compatibility label)" >&2
      exit 2
    fi
    # Shape-only plan: configure_as_full_retention() ignores its 5% selection.
    PLAN="$ROOT/configs/qwen25_k005_impress.json"
    KEEP_RATIO="0.05"
    KV_DIR="$AS_KV_DIR"
    CACHE_TYPE="LRU"
    AS_BASELINE_MODE="as_lru"
    BUDGET_SEMANTICS="full-kv-budget-independent"
    AS_FULL_KEY_RATIO="1.0"
    AS_VALUE_KEEP_RATIO="1.0"
    AS_LOGICAL_TOTAL_KV_RATIO="1.0"
    method_args=(
      --as-baseline-mode "$AS_BASELINE_MODE"
      --probe-query-heads 0,7,14,21
      --selector-kv-head-ids 0,1,2,3
    )
    ;;
  as_h2o_lru)
    if [[ "$SELECTOR_BACKEND" != "fp16" ]]; then
      echo "AS+H2O+LRU requires SELECTOR_BACKEND=fp16; selector indexes are not part of the original baseline" >&2
      exit 2
    fi
    PLAN="$ROOT/configs/qwen25_k${BUDGET_TAG}_impress.json"
    KV_DIR="$AS_KV_DIR"
    CACHE_TYPE="LRU"
    AS_BASELINE_MODE="as_h2o_lru"
    BUDGET_SEMANTICS="full-keys-selected-values"
    AS_FULL_KEY_RATIO="1.0"
    AS_VALUE_KEEP_RATIO="$KEEP_RATIO"
    case "$BUDGET_TAG" in
      005) AS_LOGICAL_TOTAL_KV_RATIO="0.525" ;;
      010) AS_LOGICAL_TOTAL_KV_RATIO="0.55" ;;
      025) AS_LOGICAL_TOTAL_KV_RATIO="0.625" ;;
      050) AS_LOGICAL_TOTAL_KV_RATIO="0.75" ;;
    esac
    method_args=(
      --as-baseline-mode "$AS_BASELINE_MODE"
      --probe-query-heads 0,7,14,21
      --selector-kv-head-ids 0,1,2,3
    )
    ;;
  *)
    echo "METHOD must be contigkv, impress, promixed, as_lru, or as_h2o_lru" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-k${BUDGET_TAG}_${METHOD}_${SELECTOR_BACKEND}_${score_tag}${coverage_tag}}"
OUTPUT="$RUN_ROOT/$TASK/$RUN_NAME"
STAGING_OUTPUT="${OUTPUT}.partial.$$"
LOG="$RUN_ROOT/$TASK/$RUN_NAME.log"

[[ -f "$PLAN" ]] || {
  echo "Plan is missing: $PLAN" >&2
  exit 2
}
if [[ "$METHOD" == "promixed" && ! -f "$PROFILE" ]]; then
  echo "Layer-budget profile is missing: $PROFILE" >&2
  exit 2
fi
BUNDLE_METADATA="$BUNDLE_DIR/metadata.json"
[[ -d "$BUNDLE_DIR" ]] || {
  echo "Bundle directory is missing: $BUNDLE_DIR" >&2
  exit 2
}
[[ -f "$BUNDLE_METADATA" ]] || {
  echo "Bundle metadata is missing: $BUNDLE_METADATA" >&2
  exit 2
}
STORE_TASK_METADATA="$STORE_ROOT/$TASK/metadata.json"
[[ -f "$STORE_TASK_METADATA" ]] || {
  echo "Store task metadata is missing: $STORE_TASK_METADATA" >&2
  exit 2
}
KV_COMPLETE_MARKER="$KV_DIR/.contiguous_fuxian_complete"
[[ -f "$KV_COMPLETE_MARKER" ]] || {
  echo "Pcache completion marker is missing: $KV_COMPLETE_MARKER" >&2
  exit 2
}
if [[ "$AS_BASELINE_MODE" != "none" ]] &&
   ! validate_as_completion_marker "$KV_COMPLETE_MARKER"; then
  echo "AS baselines require the dedicated plain chunk-64 Pcache: $KV_COMPLETE_MARKER" >&2
  exit 2
fi
if [[ -n "$selected_reorder_manifest" && ! -f "$selected_reorder_manifest" ]]; then
  echo "IMPRESS reorder manifest is missing: $selected_reorder_manifest" >&2
  exit 2
fi
PLAN_SHA256="$(sha256sum "$PLAN" | awk '{print $1}')"
BUNDLE_METADATA_SHA256="$(sha256sum "$BUNDLE_METADATA" | awk '{print $1}')"
STORE_TASK_METADATA_SHA256="$(sha256sum "$STORE_TASK_METADATA" | awk '{print $1}')"
KV_COMPLETE_SHA256="$(sha256sum "$KV_COMPLETE_MARKER" | awk '{print $1}')"
if [[ -n "$selected_reorder_manifest" ]]; then
  REORDER_SHA256="$(sha256sum "$selected_reorder_manifest" | awk '{print $1}')"
  if ! KV_DECLARED_REORDER_SHA256="$(
    json_string_field "$KV_COMPLETE_MARKER" "impress_reorder_sha256"
  )"; then
    echo "IMPRESS completion marker has invalid reorder provenance: $KV_COMPLETE_MARKER" >&2
    exit 2
  fi
  if [[ "$KV_DECLARED_REORDER_SHA256" != "$REORDER_SHA256" ]]; then
    echo "IMPRESS reorder SHA mismatch: completion marker declares $KV_DECLARED_REORDER_SHA256 but manifest is $REORDER_SHA256" >&2
    exit 2
  fi
else
  REORDER_SHA256="none"
  KV_DECLARED_REORDER_SHA256="none"
fi
[[ ! -e "$OUTPUT" ]] || {
  echo "Output already exists: $OUTPUT" >&2
  exit 4
}

mkdir -p "$RUN_ROOT/$TASK" "$RUN_ROOT/environment"

# Serialize all prism_max cells that share the SSD-backed Pcache directory.
exec 7>"/tmp/prism_max_storage.lock"
flock -n 7 || {
  echo "Another prism_max run owns the shared-storage lock" >&2
  exit 3
}
exec 9>"/tmp/prism_max_gpu${GPU}.lock"
flock -n 9 || {
  echo "Another prism_max run owns GPU lock $GPU" >&2
  exit 3
}
if [[ "$REQUIRE_IDLE_RESERVE" == "true" ]]; then
  exec 8>"/tmp/prism_max_gpu${RESERVE_GPU}.lock"
  flock -n 8 || {
    echo "Another prism_max run owns reserve GPU lock $RESERVE_GPU" >&2
    exit 3
  }
fi

guard_gpus
sleep "$SETTLE_SECONDS"
guard_gpus

{
  date -Is
  printf 'source_commit=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
  printf 'source_status=%s\n' "$(git -C "$ROOT" status --porcelain | tr '\n' ';')"
  printf 'method=%s\nselector_backend=%s\nscore_mode=%s\n' \
    "$METHOD" "$SELECTOR_BACKEND" "$score_tag"
  printf 'as_baseline_mode=%s\ncache_type=%s\nbudget_semantics=%s\n' \
    "$AS_BASELINE_MODE" "$CACHE_TYPE" "$BUDGET_SEMANTICS"
  printf 'as_full_key_ratio=%s\nas_value_keep_ratio=%s\nas_logical_total_kv_ratio=%s\n' \
    "$AS_FULL_KEY_RATIO" "$AS_VALUE_KEEP_RATIO" "$AS_LOGICAL_TOTAL_KV_RATIO"
  printf 'plan=%s\nplan_sha256=%s\n' "$PLAN" "$PLAN_SHA256"
  printf 'bundle_dir=%s\nbundle_metadata=%s\nbundle_metadata_sha256=%s\n' \
    "$BUNDLE_DIR" "$BUNDLE_METADATA" "$BUNDLE_METADATA_SHA256"
  printf 'store_root=%s\nstore_task_metadata=%s\nstore_task_metadata_sha256=%s\n' \
    "$STORE_ROOT" "$STORE_TASK_METADATA" "$STORE_TASK_METADATA_SHA256"
  printf 'kv_dir=%s\nkv_complete_marker=%s\nkv_complete_sha256=%s\n' \
    "$KV_DIR" "$KV_COMPLETE_MARKER" "$KV_COMPLETE_SHA256"
  printf 'impress_reorder_manifest=%s\nimpress_reorder_sha256=%s\nkv_complete_impress_reorder_sha256=%s\n' \
    "${selected_reorder_manifest:-none}" "$REORDER_SHA256" \
    "$KV_DECLARED_REORDER_SHA256"
  printf 'adaptive_coverage=%s\nsamples=%s\nwarmup_passes=%s\n' \
    "$PROMIXED_ADAPTIVE_COVERAGE" "$SAMPLES_PER_TASK" "$WARMUP_PASSES"
  printf 'warmup_samples=%s\ngpu_cache_mb=%s\ncpu_cache_mb=%s\n' \
    "$WARMUP_SAMPLES_PER_TASK" "$GPU_CACHE_MB" "$CPU_CACHE_MB"
  findmnt -T "$KV_DIR" -o SOURCE,TARGET,FSTYPE,OPTIONS
} >"$RUN_ROOT/environment/${TASK}.${RUN_NAME}.protocol.txt"

{
  date -Is
  nvidia-smi -i "$GPU" \
    --query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
  awk '$3 == "sda" {print}' /proc/diskstats
} >"$RUN_ROOT/environment/${TASK}.${RUN_NAME}.before.txt"

echo "[$(date -Is)] $METHOD/$SELECTOR_BACKEND $TASK k$BUDGET_TAG on GPU $GPU; deferred=$DEFER_CACHE_SCORE_UPDATES adaptive_coverage=$PROMIXED_ADAPTIVE_COVERAGE" |
  tee "$LOG"
echo "Pcache CPU budget is fixed at ${CPU_CACHE_MB} MiB; selector-index memory is additive and reported separately." |
  tee -a "$LOG"

guard_gpus
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
  --output-dir "$STAGING_OUTPUT" \
  --flexgen-root "$FLEXGEN_ROOT" \
  --flexgen-kv-dir "$KV_DIR" \
  --tasks "$TASK" \
  --store-tasks sst2,subj,trec,rte \
  --samples-per-task "$SAMPLES_PER_TASK" \
  --max-tokens 1 \
  --accuracy-scoring label_continuation_loglikelihood \
  --device cuda \
  --dtype bfloat16 \
  --gpu-cache-mb "$GPU_CACHE_MB" \
  --cpu-cache-mb "$CPU_CACHE_MB" \
  --cache-type "$CACHE_TYPE" \
  --prefetch-time-budget 10000 \
  --online-selection \
  --reuse-flexgen-kv \
  --period-size 8 \
  --subperiod-size 4 \
  --expected-keep-ratio "$KEEP_RATIO" \
  --warmup-passes "$WARMUP_PASSES" \
  --warmup-samples-per-task "$WARMUP_SAMPLES_PER_TASK" \
  "${method_args[@]}" \
  "${profile_args[@]}" \
  "${cache_score_args[@]}" \
  "${index_args[@]}" \
  >>"$LOG" 2>&1

if [[ ! -s "$STAGING_OUTPUT/summary.json" ||
      ! -s "$STAGING_OUTPUT/scored_records.jsonl" ]]; then
  echo "Run did not produce a complete result in $STAGING_OUTPUT" >&2
  exit 5
fi
mv "$STAGING_OUTPUT" "$OUTPUT"

{
  date -Is
  nvidia-smi -i "$GPU" \
    --query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
  awk '$3 == "sda" {print}' /proc/diskstats
} >"$RUN_ROOT/environment/${TASK}.${RUN_NAME}.after.txt"

echo "[$(date -Is)] completed $OUTPUT" | tee -a "$LOG"
