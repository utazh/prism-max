#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/panzihang/src/prism_phase_ab_20260730}"
OFFICIAL_REPO="${OFFICIAL_REPO:-/home/panzihang/src/hyperinfer_fuxian_20260720/HyperInfer}"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
GPU="${GPU:-0}"

MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_32}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
PLAN="${PLAN:-$ROOT/configs/qwen25_k050_hyperinfer_block16.json}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c16_head0_blockselect_v1}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/prism_phase_ab_20260730}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-/home/panzihang/src/hyperinfer_fuxian_20260720/results/qwen25_k050_ssd_hybrid_block16_alpha1_128_v1/k050_hyperinfer_block16/summary.json}"

FLEXGEN_ROOT="$OFFICIAL_REPO/h2o_flexgen/flexgen"
CALIBRATION_DIR="$RUN_ROOT/calibration"
PROFILE="$CALIBRATION_DIR/layer_budget_profile.json"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
SIMILARITY_ALPHA="${SIMILARITY_ALPHA:-1.0}"
RUN_FORMAL="${RUN_FORMAL:-0}"
MAX_ACCURACY_DROP="${MAX_ACCURACY_DROP:-0.0}"
MAX_PHASE_AB_SLOWDOWN_PERCENT="${MAX_PHASE_AB_SLOWDOWN_PERCENT:-0.0}"

require_file() {
  [[ -f "$1" ]] || {
    echo "Required file is missing: $1" >&2
    exit 2
  }
}

guard_gpu() {
  local active
  active="$(
    nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader |
      tr -d '[:space:]'
  )"
  if [[ -n "$active" ]]; then
    echo "GPU $GPU has an active compute process ($active); refusing to overlap" >&2
    exit 3
  fi
}

require_file "$PYTHON"
require_file "$PLAN"
require_file "$BUNDLE_DIR/metadata.json"
require_file "$FLEXGEN_ROOT/my_pcache_fast.py"
require_file "$KV_DIR/.contiguous_fuxian_complete"
require_file "$BASELINE_SUMMARY"
mkdir -p "$RUN_ROOT"

exec 9>"/tmp/prism_phase_ab_gpu${GPU}.lock"
if ! flock -n 9; then
  echo "Another PRISM phase A/B pipeline owns GPU lock $GPU" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/src"

if [[ ! -f "$PROFILE" ]]; then
  guard_gpu
  mkdir -p "$CALIBRATION_DIR"
  timeout "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" \
    "$ROOT/scripts/calibrate_layer_budget.py" \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --store-root "$STORE_ROOT" \
    --plan "$PLAN" \
    --output-dir "$CALIBRATION_DIR" \
    --flexgen-root "$FLEXGEN_ROOT" \
    --flexgen-kv-dir "$KV_DIR" \
    --tasks sst2,subj,trec,rte \
    --samples-per-task 1 \
    --device cuda \
    --dtype bfloat16 \
    --target-mean-ratio 0.50 \
    --perturbed-ratio 0.25 \
    --profile-delta 0.25 \
    --extreme-fraction 0.25 \
    --selection-block-size 16 \
    --period-size 8 \
    --subperiod-size 4 \
    --probe-query-heads 0,1,2 \
    --selector-kv-head-ids 0 \
    --similarity-alpha "$SIMILARITY_ALPHA" \
    --async-prefetch \
    --gpu-cache-mb 55 \
    --cpu-cache-mb 131 \
    --cache-type LRU \
    --prefetch-time-budget 10000 \
    >"$CALIBRATION_DIR/calibration.log" 2>&1
fi
require_file "$PROFILE"

run_variant() {
  local label="$1"
  local samples="$2"
  local warmup="$3"
  local predictive_period="$4"
  local profile_path="$5"
  local run_dir="$RUN_ROOT/$label"
  local log_path="$RUN_ROOT/$label.log"
  local profile_args=()

  if [[ -f "$run_dir/summary.json" ]]; then
    echo "Skipping completed $label"
    return
  fi
  if [[ -n "$profile_path" ]]; then
    profile_args=(--layer-budget-profile "$profile_path")
  fi
  guard_gpu
  timeout "${RUN_TIMEOUT_SECONDS}s" "$PYTHON" \
    -m contiguous_fuxian.flexgen_qwen_reprefill \
    --model-path "$MODEL_PATH" \
    --bundle-dir "$BUNDLE_DIR" \
    --store-root "$STORE_ROOT" \
    --plan "$PLAN" \
    --output-dir "$run_dir" \
    --flexgen-root "$FLEXGEN_ROOT" \
    --flexgen-kv-dir "$KV_DIR" \
    --tasks sst2,subj,trec,rte \
    --samples-per-task "$samples" \
    --max-tokens 4 \
    --device cuda \
    --dtype bfloat16 \
    --gpu-cache-mb 55 \
    --cpu-cache-mb 131 \
    --cache-type CKLFU \
    --prefetch-time-budget 10000 \
    --online-selection \
    --probe-query-heads 0,1,2 \
    --selector-kv-head-ids 0 \
    --similarity-alpha "$SIMILARITY_ALPHA" \
    --impress-selection-block-size 16 \
    --impress-async-prefetch \
    --impress-period-prefetch-size "$predictive_period" \
    --reuse-flexgen-kv \
    --period-size 8 \
    --subperiod-size 4 \
    --expected-keep-ratio 0.50 \
    --warmup-passes "$warmup" \
    "${profile_args[@]}" \
    >"$log_path" 2>&1
}

# Small matched screen: uniform HyperInfer, phase A, and phase A+B at P=4/P=8.
run_variant "pilot_uniform_p1_16" 4 0 1 ""
run_variant "pilot_phase_a_p1_16" 4 0 1 "$PROFILE"
run_variant "pilot_phase_ab_p4_16" 4 0 4 "$PROFILE"
run_variant "pilot_phase_ab_p8_16" 4 0 8 "$PROFILE"

"$PYTHON" - "$RUN_ROOT" "$MAX_ACCURACY_DROP" "$MAX_PHASE_AB_SLOWDOWN_PERCENT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
max_accuracy_drop = float(sys.argv[2])
max_phase_ab_slowdown = float(sys.argv[3])
labels = ("pilot_phase_a_p1_16", "pilot_phase_ab_p4_16", "pilot_phase_ab_p8_16")

def records(label):
    path = root / label / "scored_records.jsonl"
    return {
        row["uid"]: row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    }

reference = records(labels[0])
uniform_summary = json.loads(
    (root / "pilot_uniform_p1_16" / "summary.json").read_text(encoding="utf-8")
)
uniform_accuracy = uniform_summary["overall"]["accuracy"]
phase_a_summary = json.loads(
    (root / "pilot_phase_a_p1_16" / "summary.json").read_text(encoding="utf-8")
)
phase_a_accuracy = phase_a_summary["overall"]["accuracy"]
phase_a_ttft = phase_a_summary["overall"]["mean_ttft_ms"]
phase_a_passed = phase_a_accuracy >= uniform_accuracy - max_accuracy_drop
rows = []
valid_periods = []
for label, period in zip(labels, (1, 4, 8)):
    summary = json.loads((root / label / "summary.json").read_text(encoding="utf-8"))
    current = records(label)
    prediction_mismatches = sum(
        current[uid]["prediction"] != reference[uid]["prediction"] for uid in reference
    )
    selection_mismatches = sum(
        current[uid]["layer_token_selection_sha256"]
        != reference[uid]["layer_token_selection_sha256"]
        for uid in reference
    )
    row = {
        "label": label,
        "period": period,
        "accuracy": summary["overall"]["accuracy"],
        "mean_ttft_ms": summary["overall"]["mean_ttft_ms"],
        "prediction_mismatches_vs_phase_a": prediction_mismatches,
        "selection_mismatches_vs_phase_a": selection_mismatches,
    }
    row["ttft_change_vs_phase_a_percent"] = (
        (row["mean_ttft_ms"] / phase_a_ttft) - 1.0
    ) * 100.0
    rows.append(row)
    if (
        period > 1
        and prediction_mismatches == 0
        and selection_mismatches == 0
        and row["ttft_change_vs_phase_a_percent"] <= max_phase_ab_slowdown
    ):
        valid_periods.append(row)

best = (
    min(valid_periods, key=lambda row: (row["mean_ttft_ms"], row["period"]))
    if phase_a_passed and valid_periods
    else None
)
payload = {
    "uniform_accuracy": uniform_accuracy,
    "phase_a_accuracy": phase_a_accuracy,
    "phase_a_quality_gate_passed": phase_a_passed,
    "max_accuracy_drop": max_accuracy_drop,
    "max_phase_ab_slowdown_percent": max_phase_ab_slowdown,
    "variants": rows,
    "selected_period": best["period"] if best else None,
    "selected_label": best["label"] if best else None,
}
(root / "pilot_comparison.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
selected_path = root / "selected_period.txt"
if best:
    selected_path.write_text(str(best["period"]) + "\n", encoding="utf-8")
elif selected_path.exists():
    selected_path.unlink()
PY

if [[ "$RUN_FORMAL" != "1" ]]; then
  echo "Pilot completed; formal runs require RUN_FORMAL=1 after reviewing pilot_comparison.json"
  exit 0
fi
if [[ ! -f "$RUN_ROOT/selected_period.txt" ]]; then
  echo "Pilot quality gate failed; refusing to launch formal runs" >&2
  exit 4
fi

BEST_PERIOD="$(tr -d '[:space:]' <"$RUN_ROOT/selected_period.txt")"
run_variant "formal_phase_a_p1_128" 32 1 1 "$PROFILE"
run_variant "formal_phase_ab_p${BEST_PERIOD}_128" 32 1 "$BEST_PERIOD" "$PROFILE"

"$PYTHON" - "$BASELINE_SUMMARY" "$RUN_ROOT" "$BEST_PERIOD" <<'PY'
import json
import pathlib
import statistics
import sys

baseline_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
period = int(sys.argv[3])
paths = {
    "baseline_hyperinfer_block16": baseline_path,
    "phase_a_layer_budget": root / "formal_phase_a_p1_128" / "summary.json",
    "phase_ab_layer_budget_period": root / f"formal_phase_ab_p{period}_128" / "summary.json",
}
baseline_ttft = json.loads(baseline_path.read_text(encoding="utf-8"))["overall"][
    "mean_ttft_ms"
]
table = []
for label, path in paths.items():
    summary = json.loads(path.read_text(encoding="utf-8"))
    overall = summary["overall"]
    table.append(
        {
            "method": label,
            "samples": overall["samples"],
            "accuracy": overall["accuracy"],
            "mean_ttft_ms": overall["mean_ttft_ms"],
            "p95_ttft_ms": overall["p95_ttft_ms"],
            "ttft_change_vs_baseline_percent": (
                (overall["mean_ttft_ms"] / baseline_ttft) - 1.0
            )
            * 100.0,
            "effective_mean_keep_ratio": overall.get("mean_effective_keep_ratio"),
            "runtime_variant": summary["runtime"]["runtime_variant"],
        }
    )

def load_records(path):
    return {
        row["uid"]: row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    }

a = load_records(root / "formal_phase_a_p1_128" / "scored_records.jsonl")
ab = load_records(
    root / f"formal_phase_ab_p{period}_128" / "scored_records.jsonl"
)
integrity = {
    "prediction_mismatches_phase_a_vs_ab": sum(
        a[uid]["prediction"] != ab[uid]["prediction"] for uid in a
    ),
    "selection_mismatches_phase_a_vs_ab": sum(
        a[uid]["layer_token_selection_sha256"]
        != ab[uid]["layer_token_selection_sha256"]
        for uid in a
    ),
    "mean_period_prefetch_jobs": statistics.fmean(
        float(row["impress_period_prefetch_jobs"]) for row in ab.values()
    ),
    "mean_period_prefetch_tokens": statistics.fmean(
        float(row["impress_period_prefetch_tokens"]) for row in ab.values()
    ),
}
payload = {
    "selected_predictive_period": period,
    "methods": table,
    "integrity": integrity,
}
(root / "formal_comparison.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY

sha256sum \
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py" \
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py" \
  "$ROOT/src/contiguous_fuxian/layer_budget.py" \
  "$ROOT/scripts/calibrate_layer_budget.py" \
  "$ROOT/scripts/run_prism_phase_ab.sh" \
  "$PLAN" \
  "$PROFILE" \
  "$FLEXGEN_ROOT/my_pcache_fast.py" \
  "$BUNDLE_DIR/metadata.json" \
  "$KV_DIR/.contiguous_fuxian_complete" \
  >"$RUN_ROOT/source_and_input_sha256.txt"

printf '%s\n' "$RUN_ROOT"
