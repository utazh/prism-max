#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
CELL_RUNNER="${CELL_RUNNER:-$ROOT/scripts/run_prism_max_cell.sh}"
ANALYZER="${ANALYZER:-$ROOT/scripts/analyze_prism_max_grid.py}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/prism_max_grid_fp16_nodefer_strict_r1_20260825}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval_strict}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
IMPRESS_KV_DIR="${IMPRESS_KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_impress_c64_gqa_unique_reordered_disjoint_v33}"
IMPRESS_REORDER="${IMPRESS_REORDER:-/home/panzihang/src/contiguous_fuxian/results/impress_reorder/qwen25_7b_paper4_disjoint_history32_35_v2.json}"
GPU="${GPU:-0}"
RESERVE_GPU="${RESERVE_GPU:-3}"
REQUIRE_IDLE_RESERVE="${REQUIRE_IDLE_RESERVE:-true}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
VALIDATION_BOOTSTRAP_SAMPLES="${VALIDATION_BOOTSTRAP_SAMPLES:-100}"
FINAL_BOOTSTRAP_SAMPLES="${FINAL_BOOTSTRAP_SAMPLES:-5000}"
GRID_LOCK_PATH="${GRID_LOCK_PATH:-/tmp/prism_max_grid_fp16_nodefer_strict_r1.lock}"

TASKS=(sst2 subj trec rte)
BUDGETS=(005 010 025 050)
METHOD_ORDERS=(
  "contigkv impress promixed"
  "impress promixed contigkv"
  "promixed contigkv impress"
)
declare -A EXPECTED_COUNTS=(
  [sst2]=867
  [subj]=998
  [trec]=495
  [rte]=272
)

SCHEDULE_MANIFEST="$RUN_ROOT/schedule_manifest.json"
ANALYSIS_STEM="$RUN_ROOT/prism_max_grid_fp16_nodefer_strict_r1"
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
[[ "$VALIDATION_BOOTSTRAP_SAMPLES" =~ ^[0-9]+$ &&
   "$VALIDATION_BOOTSTRAP_SAMPLES" -gt 0 ]] ||
  die "VALIDATION_BOOTSTRAP_SAMPLES must be a positive integer"
[[ "$FINAL_BOOTSTRAP_SAMPLES" =~ ^[0-9]+$ &&
   "$FINAL_BOOTSTRAP_SAMPLES" -gt 0 ]] ||
  die "FINAL_BOOTSTRAP_SAMPLES must be a positive integer"
[[ "$RUN_ROOT" == /* ]] || die "RUN_ROOT must be absolute"

for required in \
  "$PYTHON" \
  "$CELL_RUNNER" \
  "$ANALYZER" \
  "$BUNDLE_DIR/metadata.json" \
  "$KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_KV_DIR/.contiguous_fuxian_complete" \
  "$IMPRESS_REORDER"; do
  [[ -f "$required" ]] || die "required file is missing: $required"
done

mkdir -p "$RUN_ROOT/environment" "$RUN_ROOT/validation" "$RUN_ROOT/incomplete"
exec 6>"$GRID_LOCK_PATH"
flock -n 6 || die "another Prism-Max one-pass grid launcher owns $GRID_LOCK_PATH"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "[$(date -Is)] Prism-Max one-pass strict FP16/nodefer grid launcher starts"

assert_tracked_source_clean() {
  git -C "$ROOT" diff --quiet -- ||
    die "tracked worktree changes detected; commit or remove them before running"
  git -C "$ROOT" diff --cached --quiet -- ||
    die "staged changes detected; commit or remove them before running"
}

git -C "$ROOT" ls-files --error-unmatch -- \
  scripts/run_prism_max_grid_once.sh scripts/run_prism_max_cell.sh \
  scripts/analyze_prism_max_grid.py >/dev/null ||
  die "launcher, cell runner, and analyzer must be committed before running"

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
assert_tracked_source_clean
if [[ -f "$SOURCE_COMMIT_FILE" ]]; then
  [[ "$(<"$SOURCE_COMMIT_FILE")" == "$SOURCE_COMMIT" ]] ||
    die "RUN_ROOT is frozen to a different source commit"
else
  printf '%s\n' "$SOURCE_COMMIT" >"$SOURCE_COMMIT_FILE"
fi

"$PYTHON" - "$BUNDLE_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
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
    records = []
    for line_number, line in enumerate(
        task_bytes.decode("utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid JSON in {task}.jsonl:{line_number}: {error}")
    uids = [row.get("uid") for row in records]
    if len(records) != count or len(set(uids)) != count:
        raise SystemExit(f"{task} strict bundle is not {count} unique rows")
    if any(row.get("task") != task for row in records):
        raise SystemExit(f"{task} strict bundle contains another task")
    if f"{task}-0" in set(uids):
        raise SystemExit(f"{task}-0 calibration UID remains in strict bundle")
PY

"$PYTHON" - "$RUN_ROOT" "$SCHEDULE_MANIFEST" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2])
tasks = ("sst2", "subj", "trec", "rte")
budgets = ("005", "010", "025", "050")
orders = (
    ("contigkv", "impress", "promixed"),
    ("impress", "promixed", "contigkv"),
    ("promixed", "contigkv", "impress"),
)
runs = []
cell_index = 0
for task in tasks:
    for budget in budgets:
        for method in orders[cell_index % len(orders)]:
            name = f"k{budget}_{method}_fp16_nodefer_r1"
            runs.append(
                {
                    "task": task,
                    "budget": budget,
                    "method": method,
                    "repeat": "r1",
                    "path": str(run_root / task / name),
                }
            )
        cell_index += 1
payload = {
    "schema_version": 1,
    "purpose": "Single-pass strict Prism-Max FP16/nodefer re-prefill grid.",
    "runs": runs,
}
if len(runs) != 48:
    raise SystemExit(f"internal schedule error: expected 48 runs, got {len(runs)}")
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != payload:
        raise SystemExit(f"existing schedule manifest differs: {manifest_path}")
else:
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
PY

fingerprint_inputs=(
  "$ROOT/scripts/run_prism_max_grid_once.sh"
  "$CELL_RUNNER"
  "$ANALYZER"
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py"
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py"
  "$ROOT/src/contiguous_fuxian/paper_client.py"
  "$BUNDLE_DIR/metadata.json"
  "$BUNDLE_DIR/sst2.jsonl"
  "$BUNDLE_DIR/subj.jsonl"
  "$BUNDLE_DIR/trec.jsonl"
  "$BUNDLE_DIR/rte.jsonl"
  "$ROOT/configs/layer_budget_k005_sensitivity.json"
  "$ROOT/configs/layer_budget_k010_sensitivity.json"
  "$ROOT/configs/layer_budget_k025_scaled.json"
  "$ROOT/configs/layer_budget_k050_sensitivity.json"
  "$KV_DIR/.contiguous_fuxian_complete"
  "$IMPRESS_KV_DIR/.contiguous_fuxian_complete"
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
  )
done
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
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'bundle_dir=%s\n' "$BUNDLE_DIR"
  printf 'gpu=%s\nreserve_gpu=%s\nrequire_idle_reserve=%s\n' \
    "$GPU" "$RESERVE_GPU" "$REQUIRE_IDLE_RESERVE"
  printf 'selector_backend=fp16\nscore_mode=nodefer\nwarmup_passes=0\n'
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
} >"$snapshot_path"

validate_output() {
  local output="$1" task="$2" budget="$3" method="$4" expected="$5"
  local validation_dir="$RUN_ROOT/validation/.${task}.k${budget}.${method}.r1.$$"
  local validation_stem="$validation_dir/check"
  mkdir -p "$validation_dir"

  "$PYTHON" - "$output" "$BUNDLE_DIR/$task.jsonl" \
    "$task" "$budget" "$method" "$expected" <<'PY' || return
import json
import math
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
bundle_path = pathlib.Path(sys.argv[2])
task, budget, method, expected = sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
summary_path = output / "summary.json"
records_path = output / "scored_records.jsonl"
if not summary_path.is_file() or not records_path.is_file():
    raise SystemExit(f"missing completed output files in {output}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = []
for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {records_path}:{line_number}: {error}")
uids = [row.get("uid") for row in rows]
if len(rows) != expected or len(set(uids)) != expected:
    raise SystemExit(f"{output} has {len(rows)} rows, expected {expected} unique rows")
bundle_uids = {
    json.loads(line)["uid"]
    for line in bundle_path.read_text(encoding="utf-8").splitlines() if line.strip()
}
if set(uids) != bundle_uids:
    raise SystemExit(f"{output} UID set differs from strict {task} bundle")
if any(row.get("task") != task for row in rows):
    raise SystemExit(f"{output} contains records outside task {task}")
if f"{task}-0" in set(uids):
    raise SystemExit(f"{output} contains excluded calibration UID {task}-0")
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
required = {
    "warmup_passes": 0,
    "warmup_requests": 0,
    "warmup_samples_per_task": 32,
    "defer_cache_score_updates": False,
    "accuracy_scoring": "label_continuation_loglikelihood",
    "generation_max_tokens": 1,
    "model_compute_dtype": "bfloat16",
    "pcache_storage_dtype": "float16",
    "online_selection": True,
    "cache_type": "CKLFU",
    "registered_store_tasks": ["sst2", "subj", "trec", "rte"],
    "selector_index_dir": None,
    "selector_index_bits": None,
    "selector_index_group_size": None,
    "selector_index_manifest_sha256": None,
}
for key, value in required.items():
    if runtime.get(key) != value:
        raise SystemExit(f"{output} runtime.{key}={runtime.get(key)!r}, expected {value!r}")
preloaded = runtime.get("selector_index_preloaded_bytes")
if preloaded is not None and not (
    type(preloaded) in (int, float) and preloaded == 0
):
    raise SystemExit(f"{output} unexpectedly preloaded a selector index")
if method == "promixed" and runtime.get("promixed_policy", {}).get(
    "adaptive_coverage"
) is not False:
    raise SystemExit(f"{output} enables adaptive ProMixed coverage")
if not math.isclose(float(runtime.get("keep_ratio")), int(budget) / 100.0):
    raise SystemExit(f"{output} keep ratio does not match k{budget}")
PY

  if ! "$PYTHON" "$ANALYZER" \
    --run "$task" "$budget" "$method" r1 "$output" \
    --output "$validation_stem" \
    --bootstrap-samples "$VALIDATION_BOOTSTRAP_SAMPLES" \
    --seed 42 >/dev/null; then
    return 1
  fi
  rm -f "${validation_stem}.json" "${validation_stem}.md"
  rmdir "$validation_dir"
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
  local task="$1" budget="$2" method="$3"
  local expected="${EXPECTED_COUNTS[$task]}"
  local run_name="k${budget}_${method}_fp16_nodefer_r1"
  local output="$RUN_ROOT/$task/$run_name"
  local status

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
    BUDGET_TAG="$budget" METHOD="$method" SELECTOR_BACKEND=fp16 \
    SAMPLES_PER_TASK=1000000 WARMUP_PASSES=0 WARMUP_SAMPLES_PER_TASK=32 \
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
    IMPRESS_REORDER="$IMPRESS_REORDER" "$CELL_RUNNER"
    status=$?
    set -e

    if [[ -e "$output" ]] &&
       validate_output "$output" "$task" "$budget" "$method" "$expected"; then
      echo "[$(date -Is)] accepted validated final output after runner status $status"
      return
    fi

    case "$status" in
      0)
        die "runner returned success without a valid final output: $output"
        ;;
      3)
        echo "[$(date -Is)] resource guard/lock changed; retrying in ${WAIT_SECONDS}s"
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

cell_index=0
for task in "${TASKS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    read -r -a methods <<<"${METHOD_ORDERS[$((cell_index % 3))]}"
    echo "[$(date -Is)] cell $((cell_index + 1))/16: $task k$budget order=${methods[*]}"
    for method in "${methods[@]}"; do
      run_one "$task" "$budget" "$method"
    done
    cell_index=$((cell_index + 1))
  done
done
[[ "$cell_index" -eq 16 ]] || die "internal grid cell count is not 16"

assert_source_frozen
analysis_candidate_dir="$RUN_ROOT/.analysis_candidate.$$"
analysis_candidate="$analysis_candidate_dir/prism_max_grid_fp16_nodefer_strict_r1"
mkdir -p "$analysis_candidate_dir"
"$PYTHON" "$ANALYZER" \
  --manifest "$SCHEDULE_MANIFEST" \
  --output "$analysis_candidate" \
  --bootstrap-samples "$FINAL_BOOTSTRAP_SAMPLES" \
  --seed 42
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
    printf 'executions=48\nrepeats=1\n'
    printf 'analysis_json=%s.json\n' "$ANALYSIS_STEM"
    printf 'analysis_markdown=%s.md\n' "$ANALYSIS_STEM"
  } >"$RUN_ROOT/environment/grid.end.txt"
fi
if [[ ! -e "$RUN_ROOT/grid.done" ]]; then
  date -Is >"$RUN_ROOT/grid.done"
fi
echo "[$(date -Is)] completed 48-run grid; analysis: ${ANALYSIS_STEM}.{json,md}"
