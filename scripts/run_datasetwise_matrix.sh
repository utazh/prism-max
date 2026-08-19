#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/panzihang/src/prism_datasetwise_logit_20260804}"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-3}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/qwen7_datasetwise_logit_20260804}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
CONTIG_KV_DIR="${CONTIG_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
OURS_KV_DIR="${OURS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c16_head0_blockselect_v1}"
IMPRESS_KV_DIR="${IMPRESS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c64_gqa_unique_reordered_disjoint_v33}"
IMPRESS_REORDER="${IMPRESS_REORDER:-/home/panzihang/src/contiguous_fuxian/results/impress_reorder/qwen25_7b_paper4_disjoint_history32_35_v2.json}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
TASKS="${TASKS:-sst2 subj trec rte}"
BUDGETS="${BUDGETS:-005 010 025 050}"
FAMILIES="${FAMILIES:-impress contigkv ours}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1000000}"
WARMUP_PASSES="${WARMUP_PASSES:-1}"
WARMUP_SAMPLES_PER_TASK="${WARMUP_SAMPLES_PER_TASK:-32}"
OURS_SELECTION_PERIOD_SIZE="${OURS_SELECTION_PERIOD_SIZE:-1}"
OURS_KNOWN_PERIOD_PREFETCH="${OURS_KNOWN_PERIOD_PREFETCH:-false}"
ACCURACY_SCORING="${ACCURACY_SCORING:-label_continuation_loglikelihood}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-86400}"

for required in \
  "$PYTHON" \
  "$BUNDLE_DIR/metadata.json" \
  "$CONTIG_KV_DIR/.contiguous_fuxian_complete" \
  "$OURS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_REORDER" \
  "$FLEXGEN_ROOT/my_pcache_fast.py"; do
  [[ -f "$required" ]] || {
    echo "Required file is missing: $required" >&2
    exit 2
  }
done

mkdir -p "$RUN_ROOT/environment"
exec 9>"/tmp/prism_datasetwise_gpu${GPU}.lock"
flock -n 9 || {
  echo "Another dataset-wise run owns GPU lock $GPU" >&2
  exit 3
}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

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

validate_bundle() {
  "$PYTHON" - "$BUNDLE_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
if metadata.get("evaluation_mode") != "each dataset is an independent workload":
    raise SystemExit("the bundle is not marked for independent dataset evaluation")
if metadata.get("pooled_headline_metrics_allowed") is not False:
    raise SystemExit("the bundle does not forbid pooled headline metrics")
for task in ("sst2", "subj", "trec", "rte"):
    expected = int(metadata["evaluation_requests_by_task"][task])
    rows = [line for line in (root / f"{task}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != expected:
        raise SystemExit(f"{task} has {len(rows)} rows; expected {expected}")
PY
}

run_variant() {
  local task="$1"
  local budget_tag="$2"
  local family="$3"
  local keep_ratio profile plan output kv_dir
  local method_args=()
  local profile_args=()

  keep_ratio="$(budget_ratio "$budget_tag")"
  profile="$(budget_profile "$budget_tag")"
  plan="$ROOT/configs/qwen25_k${budget_tag}_${family}.json"
  output="$RUN_ROOT/$task/k${budget_tag}_${family}"

  if [[ -f "$output/summary.json" ]]; then
    echo "Skipping completed $task k$budget_tag $family"
    return
  fi
  if [[ -d "$output" ]]; then
    mv "$output" "$output.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi

  case "$family" in
    contigkv)
      kv_dir="$CONTIG_KV_DIR"
      method_args=(
        --probe-query-heads 0,1,2
        --selector-kv-head-ids 0,1,2,3
        --similarity-alpha 0.6
      )
      ;;
    impress)
      kv_dir="$IMPRESS_KV_DIR"
      method_args=(
        --probe-query-heads 0,1,2
        --selector-kv-head-ids 0
        --similarity-alpha 0.6
        --impress-reorder-manifest "$IMPRESS_REORDER"
        --no-impress-async-prefetch
      )
      ;;
    ours)
      kv_dir="$OURS_KV_DIR"
      method_args=(
        --probe-query-heads 0,1,2
        --selector-kv-head-ids 0
        --similarity-alpha 1.0
        --impress-selection-block-size 16
        --impress-selection-period-size "$OURS_SELECTION_PERIOD_SIZE"
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
      if [[ "$OURS_KNOWN_PERIOD_PREFETCH" == "true" ]]; then
        method_args+=(--impress-known-period-prefetch)
      elif [[ "$OURS_KNOWN_PERIOD_PREFETCH" != "false" ]]; then
        echo "OURS_KNOWN_PERIOD_PREFETCH must be true or false" >&2
        exit 2
      fi
      profile_args=(--layer-budget-profile "$profile")
      ;;
    *)
      echo "Unknown family: $family" >&2
      exit 2
      ;;
  esac

  [[ -f "$plan" ]] || {
    echo "Plan is missing: $plan" >&2
    exit 2
  }
  guard_gpu
  mkdir -p "$(dirname "$output")"
  snapshot_environment "${task}.k${budget_tag}_${family}.before"
  echo "[$(date -Is)] starting $task k$budget_tag $family"
  timeout "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" \
    -m contiguous_fuxian.flexgen_qwen_reprefill \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --store-root "$STORE_ROOT" \
    --plan "$plan" \
    --output-dir "$output" \
    --flexgen-root "$FLEXGEN_ROOT" \
    --flexgen-kv-dir "$kv_dir" \
    --tasks "$task" \
    --store-tasks sst2,subj,trec,rte \
    --samples-per-task "$SAMPLES_PER_TASK" \
    --max-tokens 1 \
    --accuracy-scoring "$ACCURACY_SCORING" \
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
    --warmup-passes "$WARMUP_PASSES" \
    --warmup-samples-per-task "$WARMUP_SAMPLES_PER_TASK" \
    "${method_args[@]}" \
    "${profile_args[@]}" \
    >"$RUN_ROOT/$task/k${budget_tag}_${family}.log" 2>&1
  snapshot_environment "${task}.k${budget_tag}_${family}.after"
  echo "[$(date -Is)] completed $task k$budget_tag $family"
  sleep "$SETTLE_SECONDS"
}

guard_gpu
validate_bundle
snapshot_environment matrix.start
"$PYTHON" -m unittest \
  tests.test_paper_client \
  tests.test_flexgen_pcache \
  tests.test_flexgen_qwen_reprefill \
  tests.test_layer_budget \
  tests.test_prism_period_prefetch \
  -q >"$RUN_ROOT/unit_tests.log" 2>&1

for task in $TASKS; do
  for budget in $BUDGETS; do
    for family in $FAMILIES; do
      run_variant "$task" "$budget" "$family"
    done
  done
done

"$PYTHON" "$ROOT/scripts/analyze_datasetwise_matrix.py" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/datasetwise_report.md"

sha256sum \
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py" \
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py" \
  "$ROOT/src/contiguous_fuxian/paper_client.py" \
  "$ROOT/scripts/build_datasetwise_bundles.py" \
  "$ROOT/scripts/run_datasetwise_matrix.sh" \
  "$ROOT/scripts/analyze_datasetwise_matrix.py" \
  "$BUNDLE_DIR/metadata.json" \
  "$BUNDLE_DIR/"*.jsonl \
  >"$RUN_ROOT/source_and_input_sha256.txt"

snapshot_environment matrix.end
cat "$RUN_ROOT/datasetwise_report.md"
