#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
CELL_RUNNER="${CELL_RUNNER:-$ROOT/scripts/run_prism_max_cell.sh}"
ANALYZER="${ANALYZER:-$ROOT/scripts/analyze_five_method_grid.py}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/prism_max_five_method_strict_response_warm1_nodefer_r1_20260828}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval_strict}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
IMPRESS_KV_DIR="${IMPRESS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c64_gqa_unique_reordered_disjoint_v33}"
IMPRESS_REORDER="${IMPRESS_REORDER:-/home/panzihang/src/contiguous_fuxian/results/impress_reorder/qwen25_7b_paper4_disjoint_history32_35_v2.json}"
AS_KV_DIR="${AS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_as_c64_plain_v1}"
SELECTOR_INDEX="${SELECTOR_INDEX:-$ROOT/assets/selector_index_k4_g32}"
GPU="${GPU:-0}"
RESERVE_GPU="${RESERVE_GPU:-3}"
REQUIRE_IDLE_RESERVE="${REQUIRE_IDLE_RESERVE:-true}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
GRID_LOCK_PATH="${GRID_LOCK_PATH:-/tmp/prism_max_five_method_strict_r1.lock}"

TASKS=(sst2 subj trec rte)
BUDGETS=(005 010 025 050)
METHOD_ORDERS=(
  "impress contigkv promixed as_h2o_lru"
  "contigkv promixed as_h2o_lru impress"
  "promixed as_h2o_lru impress contigkv"
  "as_h2o_lru impress contigkv promixed"
)
AS_INSERT_OFFSETS=(0 5 10 16)
declare -A EXPECTED_COUNTS=(
  [sst2]=867
  [subj]=998
  [trec]=495
  [rte]=272
)

SCHEDULE_MANIFEST="$RUN_ROOT/schedule_manifest.json"
EXECUTION_TSV="$RUN_ROOT/execution_order.tsv"
ANALYSIS_STEM="$RUN_ROOT/prism_max_five_method_strict_response_warm1_nodefer_r1"
DRIVER_LOG="$RUN_ROOT/grid_driver.log"
FINGERPRINT="$RUN_ROOT/environment/source_and_input_sha256.txt"
SOURCE_COMMIT_FILE="$RUN_ROOT/environment/source_commit.txt"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

case "$REQUIRE_IDLE_RESERVE" in
  true|false) ;;
  *) die "REQUIRE_IDLE_RESERVE must be true or false" ;;
esac
[[ "$GPU" != "$RESERVE_GPU" || "$REQUIRE_IDLE_RESERVE" == "false" ]] ||
  die "GPU and RESERVE_GPU must differ when an idle reserve is required"
[[ "$WAIT_SECONDS" =~ ^[0-9]+$ && "$WAIT_SECONDS" -gt 0 ]] ||
  die "WAIT_SECONDS must be a positive integer"
[[ "$RUN_TIMEOUT_SECONDS" =~ ^[0-9]+$ && "$RUN_TIMEOUT_SECONDS" -gt 0 ]] ||
  die "RUN_TIMEOUT_SECONDS must be a positive integer"
[[ "$RUN_ROOT" == /* ]] || die "RUN_ROOT must be absolute"

for required in \
  "$PYTHON" \
  "$CELL_RUNNER" \
  "$ANALYZER" \
  "$BUNDLE_DIR/metadata.json" \
  "$KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_REORDER" \
  "$AS_KV_DIR/.contiguous_fuxian_complete" \
  "$SELECTOR_INDEX/manifest.json"; do
  [[ -f "$required" ]] || die "required file is missing: $required"
done

mkdir -p "$RUN_ROOT/environment" "$RUN_ROOT/validation" "$RUN_ROOT/incomplete"
exec 6>"$GRID_LOCK_PATH"
flock -n 6 || die "another five-method launcher owns $GRID_LOCK_PATH"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "[$(date -Is)] five-method strict response-ready grid launcher starts"

assert_tracked_source_clean() {
  git -C "$ROOT" diff --quiet -- ||
    die "tracked worktree changes detected; commit or remove them before running"
  git -C "$ROOT" diff --cached --quiet -- ||
    die "staged changes detected; commit or remove them before running"
}

git -C "$ROOT" ls-files --error-unmatch -- \
  scripts/run_prism_max_five_method_grid.sh \
  scripts/run_prism_max_cell.sh \
  scripts/analyze_five_method_grid.py \
  src/contiguous_fuxian/as_baselines.py \
  src/contiguous_fuxian/flexgen_pcache.py \
  src/contiguous_fuxian/flexgen_qwen_reprefill.py >/dev/null ||
  die "five-method runtime sources must be committed before running"

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
assert_tracked_source_clean
if [[ -f "$SOURCE_COMMIT_FILE" ]]; then
  [[ "$(<"$SOURCE_COMMIT_FILE")" == "$SOURCE_COMMIT" ]] ||
    die "RUN_ROOT is frozen to a different source commit"
else
  printf '%s\n' "$SOURCE_COMMIT" >"$SOURCE_COMMIT_FILE"
fi

"$PYTHON" - "$BUNDLE_DIR" "$AS_KV_DIR/.contiguous_fuxian_complete" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
as_marker_path = pathlib.Path(sys.argv[2])
expected = {"sst2": 867, "subj": 998, "trec": 495, "rte": 272}
metadata_path = root / "metadata.json"
metadata_bytes = metadata_path.read_bytes()
metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
if metadata_sha256 != "bf70ce022f771b59bcc664d321b0787a5c8e1d50f761e6833d95c1c9160abdb5":
    raise SystemExit(f"strict metadata SHA changed: {metadata_sha256}")
metadata = json.loads(metadata_bytes.decode("utf-8"))
if metadata.get("evaluation_mode") != "each dataset is an independent workload":
    raise SystemExit("strict bundle has an invalid evaluation_mode")
if metadata.get("pooled_headline_metrics_allowed") is not False:
    raise SystemExit("strict bundle does not forbid pooled headline metrics")
if metadata.get("evaluation_requests_by_task") != expected:
    raise SystemExit("strict bundle counts do not match the frozen grid counts")
strict_filter = metadata.get("strict_eval_filter")
if not isinstance(strict_filter, dict):
    raise SystemExit("bundle is not marked with strict_eval_filter provenance")
declared_hashes = strict_filter.get("output_task_jsonl_sha256")
if not isinstance(declared_hashes, dict):
    raise SystemExit("strict bundle lacks output JSONL fingerprints")
for task, count in expected.items():
    task_path = root / f"{task}.jsonl"
    task_bytes = task_path.read_bytes()
    actual_sha256 = hashlib.sha256(task_bytes).hexdigest()
    if declared_hashes.get(task) != actual_sha256:
        raise SystemExit(f"{task} JSONL SHA differs from strict metadata")
    records = [
        json.loads(line)
        for line in task_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    uids = [row.get("uid") for row in records]
    if len(records) != count or len(set(uids)) != count:
        raise SystemExit(f"{task} strict bundle is not {count} unique rows")
    if any(row.get("task") != task for row in records):
        raise SystemExit(f"{task} strict bundle contains another task")
    if f"{task}-0" in set(uids):
        raise SystemExit(f"{task}-0 calibration UID remains in strict bundle")

as_marker = json.loads(as_marker_path.read_text(encoding="utf-8"))
expected_marker = {
    "schema_version": 3,
    "method": "attentionstore_as_baselines",
    "chunk_size": 64,
    "selector_kv_head_ids": [0, 1, 2, 3],
    "online_selection": True,
    "physical_layout": "plain-logical-token-order",
    "impress_reorder_sha256": None,
    "registered_store_tasks": ["sst2", "subj", "trec", "rte"],
}
for key, expected_value in expected_marker.items():
    if as_marker.get(key) != expected_value:
        raise SystemExit(
            f"invalid plain AS marker {as_marker_path}: "
            f"{key}={as_marker.get(key)!r}, expected {expected_value!r}"
        )
PY

"$PYTHON" - "$RUN_ROOT" "$SCHEDULE_MANIFEST" "$EXECUTION_TSV" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2])
execution_path = pathlib.Path(sys.argv[3])
tasks = ("sst2", "subj", "trec", "rte")
budgets = ("005", "010", "025", "050")
orders = (
    ("impress", "contigkv", "promixed", "as_h2o_lru"),
    ("contigkv", "promixed", "as_h2o_lru", "impress"),
    ("promixed", "as_h2o_lru", "impress", "contigkv"),
    ("as_h2o_lru", "impress", "contigkv", "promixed"),
)
as_insert_offsets = (0, 5, 10, 16)
runs = []
cell_index = 0
for task_index, task in enumerate(tasks):
    task_runs = []
    for budget in budgets:
        for method in orders[cell_index % len(orders)]:
            selector = "k4" if method == "promixed" else "fp16"
            name = (
                f"k{budget}_{method}_{selector}_nodefer_warm1_response_r1"
            )
            task_runs.append(
                {
                    "task": task,
                    "budget": budget,
                    "method": method,
                    "path": str(run_root / task / name),
                }
            )
        cell_index += 1
    as_name = "full_as_lru_fp16_nodefer_warm1_response_r1"
    task_runs.insert(
        as_insert_offsets[task_index],
        {
            "task": task,
            "budget": "full",
            "method": "as_lru",
            "path": str(run_root / task / as_name),
        },
    )
    runs.extend(task_runs)
payload = {
    "schema_version": 1,
    "purpose": (
        "Strict one-repeat five-method re-prefill grid; response-ready primary, "
        "one 32-request warm-up, nodefer; AS+LRU full-KV projected over budgets."
    ),
    "runs": runs,
}
if cell_index != 16:
    raise SystemExit(f"internal schedule error: expected 16 cells, got {cell_index}")
if len(runs) != 68:
    raise SystemExit(f"internal schedule error: expected 68 runs, got {len(runs)}")
identities = {(row["task"], row["budget"], row["method"]) for row in runs}
if len(identities) != 68:
    raise SystemExit("internal schedule error: duplicate execution identity")
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit(f"existing schedule manifest differs: {manifest_path}")
else:
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
tsv = "".join(
    f"{row['task']}\t{row['budget']}\t{row['method']}\t{row['path']}\n"
    for row in runs
)
if execution_path.exists():
    if execution_path.read_text(encoding="utf-8") != tsv:
        raise SystemExit(f"existing execution order differs: {execution_path}")
else:
    execution_path.write_text(tsv, encoding="utf-8")
PY

fingerprint_inputs=(
  "$ROOT/scripts/run_prism_max_five_method_grid.sh"
  "$CELL_RUNNER"
  "$ANALYZER"
  "$ROOT/src/contiguous_fuxian/as_baselines.py"
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py"
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py"
  "$ROOT/src/contiguous_fuxian/paper_client.py"
  "$BUNDLE_DIR/metadata.json"
  "$BUNDLE_DIR/sst2.jsonl"
  "$BUNDLE_DIR/subj.jsonl"
  "$BUNDLE_DIR/trec.jsonl"
  "$BUNDLE_DIR/rte.jsonl"
  "$KV_DIR/.contiguous_fuxian_complete"
  "$IMPRESS_KV_DIR/.contiguous_fuxian_complete"
  "$AS_KV_DIR/.contiguous_fuxian_complete"
  "$IMPRESS_REORDER"
)
for task in "${TASKS[@]}"; do
  fingerprint_inputs+=("$STORE_ROOT/$task/metadata.json")
done
for budget in "${BUDGETS[@]}"; do
  fingerprint_inputs+=(
    "$ROOT/configs/qwen25_k${budget}_contigkv.json"
    "$ROOT/configs/qwen25_k${budget}_impress.json"
    "$ROOT/configs/qwen25_k${budget}_ours.json"
    "$(case "$budget" in
        005) echo "$ROOT/configs/layer_budget_k005_sensitivity.json" ;;
        010) echo "$ROOT/configs/layer_budget_k010_sensitivity.json" ;;
        025) echo "$ROOT/configs/layer_budget_k025_scaled.json" ;;
        050) echo "$ROOT/configs/layer_budget_k050_sensitivity.json" ;;
      esac)"
  )
done
while IFS= read -r -d '' index_file; do
  fingerprint_inputs+=("$index_file")
done < <(find "$SELECTOR_INDEX" -type f -print0 | sort -z)
for model_file in \
  "$MODEL_PATH/config.json" \
  "$MODEL_PATH/generation_config.json" \
  "$MODEL_PATH/tokenizer_config.json" \
  "$MODEL_PATH/tokenizer.json"; do
  [[ ! -f "$model_file" ]] || fingerprint_inputs+=("$model_file")
done
for input in "${fingerprint_inputs[@]}"; do
  [[ -f "$input" ]] || die "fingerprinted input is missing: $input"
done

fingerprint_candidate="$RUN_ROOT/environment/.source_and_input_sha256.candidate.$$"
sha256sum "${fingerprint_inputs[@]}" >"$fingerprint_candidate"
if [[ -f "$FINGERPRINT" ]]; then
  if ! cmp -s "$FINGERPRINT" "$fingerprint_candidate"; then
    mismatch="${FINGERPRINT}.mismatch.$(date +%Y%m%d_%H%M%S).$$"
    mv "$fingerprint_candidate" "$mismatch"
    die "source/input fingerprints changed; candidate preserved at $mismatch"
  fi
  rm -f "$fingerprint_candidate"
else
  mv "$fingerprint_candidate" "$FINGERPRINT"
fi

assert_source_frozen() {
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] ||
    die "source commit changed during the grid"
  assert_tracked_source_clean
  sha256sum --status --check "$FINGERPRINT" ||
    die "a frozen source or input file changed during the grid"
}

snapshot_path="$RUN_ROOT/environment/grid.start.txt"
if [[ -e "$snapshot_path" ]]; then
  snapshot_path="$RUN_ROOT/environment/grid.resume.$(date +%Y%m%d_%H%M%S).$$.txt"
fi
{
  date -Is
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'run_root=%s\nbundle_dir=%s\n' "$RUN_ROOT" "$BUNDLE_DIR"
  printf 'gpu=%s\nreserve_gpu=%s\nrequire_idle_reserve=%s\n' \
    "$GPU" "$RESERVE_GPU" "$REQUIRE_IDLE_RESERVE"
  printf 'primary_latency=response_ready_ms\n'
  printf 'accuracy_scoring=label_continuation_loglikelihood\n'
  printf 'warmup_passes=1\nwarmup_samples_per_task=32\n'
  printf 'score_mode=nodefer\nrepeats=1\nexecutions=68\nexpanded_cells=80\n'
  printf 'baseline_selector_backend=fp16\npromixed_selector_backend=k4\n'
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
} >"$snapshot_path"

validate_output() {
  local output="$1" task="$2" budget="$3" method="$4" expected="$5"

  "$PYTHON" - "$output" "$BUNDLE_DIR/$task.jsonl" \
    "$task" "$budget" "$method" "$expected" \
    "$SELECTOR_INDEX" "$IMPRESS_REORDER" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
bundle_path = pathlib.Path(sys.argv[2])
task, budget, method = sys.argv[3], sys.argv[4], sys.argv[5]
expected = int(sys.argv[6])
selector_index = pathlib.Path(sys.argv[7]).resolve()
impress_reorder = pathlib.Path(sys.argv[8]).resolve()
summary_path = output / "summary.json"
records_path = output / "scored_records.jsonl"
if not summary_path.is_file() or not records_path.is_file():
    raise SystemExit(f"missing completed output files in {output}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = [
    json.loads(line)
    for line in records_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
uids = [row.get("uid") for row in rows]
if len(rows) != expected or len(set(uids)) != expected:
    raise SystemExit(
        f"{output} has {len(rows)} rows, expected {expected} unique rows"
    )
bundle_uids = {
    json.loads(line)["uid"]
    for line in bundle_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
if set(uids) != bundle_uids:
    raise SystemExit(f"{output} UID set differs from strict {task} bundle")
if any(row.get("task") != task for row in rows):
    raise SystemExit(f"{output} contains records outside task {task}")
if f"{task}-0" in set(uids):
    raise SystemExit(f"{output} contains excluded calibration UID {task}-0")
for row in rows:
    if type(row.get("correct")) is not bool:
        raise SystemExit(f"{output} has a non-boolean correctness value")
    for field in ("logits_ready_ms", "response_ready_ms"):
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise SystemExit(f"{output} has invalid {field}")
    if float(row["response_ready_ms"]) < float(row["logits_ready_ms"]):
        raise SystemExit(f"{output} has response-ready before logits-ready")

tasks = summary.get("tasks")
if not isinstance(tasks, dict) or set(tasks) != {task}:
    raise SystemExit(f"{output} is not an independent {task} run")
if summary.get("overall", {}).get("samples") != expected:
    raise SystemExit(f"{output} overall sample count does not match {expected}")
if tasks[task].get("samples") != expected:
    raise SystemExit(f"{output} task sample count does not match {expected}")
runtime = summary.get("runtime")
if not isinstance(runtime, dict):
    raise SystemExit(f"{output} is missing runtime metadata")

common = {
    "warmup_passes": 1,
    "warmup_requests": 32,
    "warmup_samples_per_task": 32,
    "defer_cache_score_updates": False,
    "accuracy_scoring": "label_continuation_loglikelihood",
    "generation_max_tokens": 1,
    "model_compute_dtype": "bfloat16",
    "pcache_storage_dtype": "float16",
    "online_selection": True,
    "registered_store_tasks": ["sst2", "subj", "trec", "rte"],
    "response_ready_metric_valid_for_first_token": True,
    "response_ready_excludes_accuracy_scoring": True,
    "evaluation_ready_includes_accuracy_scoring": True,
}
for key, value in common.items():
    if runtime.get(key) != value:
        raise SystemExit(
            f"{output} runtime.{key}={runtime.get(key)!r}, expected {value!r}"
        )

ratio = None if budget == "full" else int(budget) / 100.0
expected_runtime_method = "impress" if method == "promixed" else method
expected_plan_method = (
    "contigkv" if method == "contigkv" else "impress"
)
if runtime.get("method") != expected_runtime_method:
    raise SystemExit(f"{output} has wrong runtime method")
if runtime.get("plan_method") != expected_plan_method:
    raise SystemExit(f"{output} has wrong plan method")
expected_baseline_mode = method if method in {"as_lru", "as_h2o_lru"} else "none"
if runtime.get("as_baseline_mode") != expected_baseline_mode:
    raise SystemExit(f"{output} has wrong AS baseline mode")

if method == "as_lru":
    numeric = {
        "keep_ratio": 1.0,
        "configured_plan_keep_ratio": 0.05,
        "actual_key_keep_ratio": 1.0,
        "actual_value_keep_ratio": 1.0,
        "actual_total_logical_kv_ratio": 1.0,
    }
    exact = {
        "budget_semantics": "full-kv-budget-independent",
        "cache_type": "LRU",
        "cache_update_in_ttft": False,
        "chunk_size": 64,
        "selector_kv_head_ids": [0, 1, 2, 3],
    }
elif method == "as_h2o_lru":
    numeric = {
        "keep_ratio": ratio,
        "configured_plan_keep_ratio": ratio,
        "actual_key_keep_ratio": 1.0,
        "actual_value_keep_ratio": ratio,
        "actual_total_logical_kv_ratio": (1.0 + ratio) / 2.0,
    }
    exact = {
        "budget_semantics": "full-key-plus-budgeted-values",
        "cache_type": "LRU",
        "cache_update_in_ttft": False,
        "chunk_size": 64,
        "selector_kv_head_ids": [0, 1, 2, 3],
        "selector_score_reduction": "gqa-group-sum-per-kv-head-topk",
    }
elif method == "impress":
    numeric = {"keep_ratio": ratio}
    exact = {
        "budget_semantics": "matched-sparse-kv-retention",
        "cache_type": "CKLFU",
        "cache_update_in_ttft": True,
        "chunk_size": 64,
        "selector_kv_head_ids": [0],
        "promixed_gqa_selection": False,
    }
elif method == "contigkv":
    numeric = {"keep_ratio": ratio}
    exact = {
        "budget_semantics": "matched-sparse-kv-retention",
        "cache_type": "CKLFU",
        "cache_update_in_ttft": True,
        "chunk_size": 16,
        "selector_kv_head_ids": [0, 1, 2, 3],
        "promixed_gqa_selection": False,
    }
else:
    numeric = {"keep_ratio": ratio}
    exact = {
        "budget_semantics": "matched-sparse-kv-retention",
        "cache_type": "CKLFU",
        "cache_update_in_ttft": True,
        "chunk_size": 16,
        "selector_kv_head_ids": [0, 1, 2, 3],
        "promixed_gqa_selection": True,
        "exact_layer_block_budget": True,
    }
for key, value in exact.items():
    if runtime.get(key) != value:
        raise SystemExit(
            f"{output} runtime.{key}={runtime.get(key)!r}, expected {value!r}"
        )
for key, value in numeric.items():
    try:
        matches = math.isclose(
            float(runtime.get(key)), float(value), rel_tol=1e-9, abs_tol=1e-9
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise SystemExit(
            f"{output} runtime.{key}={runtime.get(key)!r}, expected {value!r}"
        )

uses_k4 = method == "promixed"
if uses_k4:
    manifest = selector_index / "manifest.json"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    selector_dir = pathlib.Path(runtime.get("selector_index_dir", "")).resolve()
    if selector_dir != selector_index:
        raise SystemExit(f"{output} used the wrong K4 selector index")
    if runtime.get("selector_index_bits") != 4:
        raise SystemExit(f"{output} did not use a K4 selector index")
    if runtime.get("selector_index_group_size") != 32:
        raise SystemExit(f"{output} used the wrong K4 group size")
    if runtime.get("selector_index_manifest_sha256") != manifest_sha:
        raise SystemExit(f"{output} K4 manifest fingerprint differs")
    if runtime.get("selector_index_preloaded_tasks") != [task]:
        raise SystemExit(f"{output} did not preload exactly the active task index")
    if not (
        type(runtime.get("selector_index_preloaded_bytes")) in (int, float)
        and runtime["selector_index_preloaded_bytes"] > 0
    ):
        raise SystemExit(f"{output} reports no preloaded K4 bytes")
    policy = runtime.get("promixed_policy")
    if not isinstance(policy, dict) or policy.get("adaptive_coverage") is not False:
        raise SystemExit(f"{output} enables adaptive ProMixed coverage")
else:
    nullable = (
        "selector_index_dir",
        "selector_index_bits",
        "selector_index_group_size",
        "selector_index_manifest_sha256",
    )
    for key in nullable:
        if runtime.get(key) is not None:
            raise SystemExit(f"{output} unexpectedly sets runtime.{key}")
    if runtime.get("selector_index_preloaded_tasks") != []:
        raise SystemExit(f"{output} unexpectedly preloaded selector tasks")
    if runtime.get("selector_index_preloaded_bytes") not in (0, 0.0):
        raise SystemExit(f"{output} unexpectedly preloaded selector bytes")

if method == "impress":
    reorder_path = pathlib.Path(runtime.get("impress_reorder_manifest", "")).resolve()
    if runtime.get("impress_reorder_enabled") is not True:
        raise SystemExit(f"{output} disabled canonical IMPRESS reordering")
    if reorder_path != impress_reorder:
        raise SystemExit(f"{output} used a different IMPRESS reorder manifest")
    if runtime.get("impress_reorder_sha256") != hashlib.sha256(
        impress_reorder.read_bytes()
    ).hexdigest():
        raise SystemExit(f"{output} IMPRESS reorder fingerprint differs")
else:
    if runtime.get("impress_reorder_enabled") is not False:
        raise SystemExit(f"{output} unexpectedly enabled IMPRESS reordering")
    if runtime.get("impress_reorder_manifest") is not None:
        raise SystemExit(f"{output} unexpectedly reports a reorder manifest")
PY
}

archive_stale_attempts() {
  local task="$1" run_name="$2"
  local output="$RUN_ROOT/$task/$run_name"
  local protocol_prefix="$RUN_ROOT/environment/${task}.${run_name}"
  local -a artifacts=()
  local artifact archive

  shopt -s nullglob
  artifacts+=("${output}.partial."*)
  shopt -u nullglob
  for artifact in \
    "$RUN_ROOT/$task/$run_name.log" \
    "${protocol_prefix}.protocol.txt" \
    "${protocol_prefix}.before.txt" \
    "${protocol_prefix}.after.txt"; do
    [[ ! -e "$artifact" ]] || artifacts+=("$artifact")
  done
  (( ${#artifacts[@]} == 0 )) && return
  archive="$RUN_ROOT/incomplete/${task}.${run_name}.$(date +%Y%m%d_%H%M%S).$$"
  mkdir -p "$archive"
  mv -- "${artifacts[@]}" "$archive/"
  echo "[$(date -Is)] archived stale attempt artifacts at $archive"
}

run_one() {
  local task="$1" budget="$2" method="$3" manifest_output="$4"
  local expected="${EXPECTED_COUNTS[$task]}"
  local selector="fp16"
  local run_name output status

  [[ "$method" != "promixed" ]] || selector="k4"
  if [[ "$budget" == "full" ]]; then
    run_name="full_as_lru_fp16_nodefer_warm1_response_r1"
  else
    run_name="k${budget}_${method}_${selector}_nodefer_warm1_response_r1"
  fi
  output="$RUN_ROOT/$task/$run_name"
  [[ "$output" == "$manifest_output" ]] ||
    die "manifest path and computed output differ for $task/$budget/$method"

  if [[ -e "$output" ]]; then
    if validate_output "$output" "$task" "$budget" "$method" "$expected"; then
      echo "[$(date -Is)] skip validated completed $task k$budget $method"
      return
    fi
    die "existing output is incomplete or invalid and will not be overwritten: $output"
  fi

  while true; do
    archive_stale_attempts "$task" "$run_name"
    assert_source_frozen
    echo "[$(date -Is)] launch $task k$budget $method as $run_name"
    set +e
    PYTHON="$PYTHON" GPU="$GPU" RESERVE_GPU="$RESERVE_GPU" \
    REQUIRE_IDLE_RESERVE="$REQUIRE_IDLE_RESERVE" TASK="$task" \
    BUDGET_TAG="$budget" METHOD="$method" SELECTOR_BACKEND="$selector" \
    SELECTOR_INDEX="$SELECTOR_INDEX" \
    SAMPLES_PER_TASK=1000000 WARMUP_PASSES=1 WARMUP_SAMPLES_PER_TASK=32 \
    DEFER_CACHE_SCORE_UPDATES=false PROMIXED_ADAPTIVE_COVERAGE=false \
    PROMIXED_COVERAGE_FRACTION=0.5 PROMIXED_MARGIN_REFERENCE=0.05 \
    PROMIXED_AGREEMENT_WEIGHT=0.75 PROMIXED_SENSITIVITY_WEIGHT=0.1 \
    PROMIXED_P1_THRESHOLD=0.90 PROMIXED_P2_THRESHOLD=0.82 \
    PROMIXED_P4_THRESHOLD=0.68 PROMIXED_UTILITY_MAX_WEIGHT=0.55 \
    PROMIXED_UTILITY_MEAN_WEIGHT=0.35 PROMIXED_UTILITY_VOTE_WEIGHT=0.10 \
    RUN_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" RUN_ROOT="$RUN_ROOT" \
    RUN_NAME="$run_name" SETTLE_SECONDS="$SETTLE_SECONDS" \
    GPU_CACHE_MB=55 CPU_CACHE_MB=131 MODEL_PATH="$MODEL_PATH" \
    BUNDLE_DIR="$BUNDLE_DIR" STORE_ROOT="$STORE_ROOT" FLEXGEN_ROOT="$FLEXGEN_ROOT" \
    KV_DIR="$KV_DIR" IMPRESS_KV_DIR="$IMPRESS_KV_DIR" \
    IMPRESS_REORDER="$IMPRESS_REORDER" AS_KV_DIR="$AS_KV_DIR" \
    "$CELL_RUNNER"
    status=$?
    set -e

    if [[ -e "$output" ]] &&
       validate_output "$output" "$task" "$budget" "$method" "$expected"; then
      echo "[$(date -Is)] accepted validated output after runner status $status"
      return
    fi
    case "$status" in
      0)
        die "runner returned success without a valid final output: $output"
        ;;
      3)
        echo "[$(date -Is)] resource guard/lock busy; retrying in ${WAIT_SECONDS}s"
        sleep "$WAIT_SECONDS"
        ;;
      4)
        die "runner reported an existing output that is not valid: $output"
        ;;
      *)
        die "runner failed for $task k$budget $method with status $status"
        ;;
    esac
  done
}

execution_count=0
while IFS=$'\t' read -r task budget method manifest_output; do
  [[ -n "$task" && -n "$budget" && -n "$method" && -n "$manifest_output" ]] ||
    die "invalid execution-order row"
  run_one "$task" "$budget" "$method" "$manifest_output"
  execution_count=$((execution_count + 1))
done <"$EXECUTION_TSV"
[[ "$execution_count" -eq 68 ]] ||
  die "internal execution count is $execution_count, expected 68"

assert_source_frozen
analysis_candidate_dir="$RUN_ROOT/.analysis_candidate.$$"
analysis_candidate="$analysis_candidate_dir/five_method"
mkdir -p "$analysis_candidate_dir"
"$PYTHON" "$ANALYZER" \
  --manifest "$SCHEDULE_MANIFEST" \
  --output "$analysis_candidate"
for suffix in json md; do
  candidate="${analysis_candidate}.${suffix}"
  destination="${ANALYSIS_STEM}.${suffix}"
  if [[ -e "$destination" ]]; then
    if ! cmp -s "$candidate" "$destination"; then
      mismatch="${destination}.mismatch.$(date +%Y%m%d_%H%M%S).$$"
      mv "$candidate" "$mismatch"
      die "existing analysis differs; candidate preserved at $mismatch"
    fi
    rm -f "$candidate"
  else
    mv "$candidate" "$destination"
  fi
done
rmdir "$analysis_candidate_dir"

if [[ ! -e "$RUN_ROOT/environment/grid.end.txt" ]]; then
  {
    date -Is
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'executions=68\nexpanded_cells=80\nrepeats=1\n'
    printf 'primary_latency=response_ready_ms\n'
    printf 'analysis_json=%s.json\n' "$ANALYSIS_STEM"
    printf 'analysis_markdown=%s.md\n' "$ANALYSIS_STEM"
  } >"$RUN_ROOT/environment/grid.end.txt"
fi
if [[ ! -e "$RUN_ROOT/grid.done" ]]; then
  date -Is >"$RUN_ROOT/grid.done"
fi
echo "[$(date -Is)] completed 68 executions / 80 projected cells"
echo "analysis: ${ANALYSIS_STEM}.{json,md}"
