#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/panzihang/venvs/vllm-stable/bin/python}"
CELL_RUNNER="${CELL_RUNNER:-$ROOT/scripts/run_prism_max_cell.sh}"
PAIR_ANALYZER="${PAIR_ANALYZER:-$ROOT/scripts/analyze_prism_max_pair.py}"
GRID_ANALYZER="${GRID_ANALYZER:-$ROOT/scripts/analyze_prism_max_grid.py}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/prism_max_k4_nodefer_strict_abba_diag_20260825}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/data/paper_task_bundles_full_eval_strict}"
MODEL_PATH="${MODEL_PATH:-/data1/llm/Qwen/Qwen2.5-7B-Instruct}"
STORE_ROOT="${STORE_ROOT:-/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42}"
FLEXGEN_ROOT="${FLEXGEN_ROOT:-$ROOT/vendor/flexgen}"
KV_DIR="${KV_DIR:-/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14}"
SELECTOR_INDEX="${SELECTOR_INDEX:-$ROOT/assets/selector_index_k4_g32}"
GPU="${GPU:-0}"
RESERVE_GPU="${RESERVE_GPU:-3}"
REQUIRE_IDLE_RESERVE="${REQUIRE_IDLE_RESERVE:-true}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
PAIR_BOOTSTRAP_SAMPLES="${PAIR_BOOTSTRAP_SAMPLES:-20000}"
GRID_BOOTSTRAP_SAMPLES="${GRID_BOOTSTRAP_SAMPLES:-5000}"
LOCK_PATH="${LOCK_PATH:-/tmp/prism_max_k4_abba_diagnostics.lock}"

CELLS=("subj 025" "subj 050" "trec 010" "trec 050")
ABBA=("contigkv r1" "promixed r1" "promixed r2" "contigkv r2")
declare -A EXPECTED_COUNTS=([subj]=998 [trec]=495)

SCHEDULE_MANIFEST="$RUN_ROOT/schedule_manifest.json"
ANALYSIS_STEM="$RUN_ROOT/prism_max_k4_nodefer_strict_abba_diagnostics"
DRIVER_LOG="$RUN_ROOT/diagnostics_driver.log"
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
  die "GPU and RESERVE_GPU must differ when reserve guarding is enabled"
[[ "$WAIT_SECONDS" =~ ^[0-9]+$ && "$WAIT_SECONDS" -gt 0 ]] ||
  die "WAIT_SECONDS must be a positive integer"
[[ "$RUN_TIMEOUT_SECONDS" =~ ^[0-9]+$ && "$RUN_TIMEOUT_SECONDS" -gt 0 ]] ||
  die "RUN_TIMEOUT_SECONDS must be a positive integer"
[[ "$RUN_ROOT" == /* ]] || die "RUN_ROOT must be absolute"

for required in \
  "$PYTHON" \
  "$CELL_RUNNER" \
  "$PAIR_ANALYZER" \
  "$GRID_ANALYZER" \
  "$BUNDLE_DIR/metadata.json" \
  "$KV_DIR/.contiguous_fuxian_complete" \
  "$SELECTOR_INDEX/manifest.json"; do
  [[ -f "$required" ]] || die "required file is missing: $required"
done

mkdir -p "$RUN_ROOT/environment" "$RUN_ROOT/validation" "$RUN_ROOT/incomplete"
exec 6>"$LOCK_PATH"
flock -n 6 || die "another K4 ABBA diagnostics launcher owns $LOCK_PATH"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "[$(date -Is)] Prism-Max K4/nodefer strict ABBA diagnostics starts"

assert_tracked_source_clean() {
  git -C "$ROOT" diff --quiet -- ||
    die "tracked worktree changes detected; commit them before running"
  git -C "$ROOT" diff --cached --quiet -- ||
    die "staged changes detected; commit them before running"
}

git -C "$ROOT" ls-files --error-unmatch -- \
  scripts/run_prism_max_k4_abba_diagnostics.sh \
  scripts/run_prism_max_cell.sh \
  scripts/analyze_prism_max_pair.py \
  scripts/analyze_prism_max_grid.py >/dev/null ||
  die "launcher, cell runner, and analyzers must be committed"

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
assert_tracked_source_clean
if [[ -f "$SOURCE_COMMIT_FILE" ]]; then
  [[ "$(<"$SOURCE_COMMIT_FILE")" == "$SOURCE_COMMIT" ]] ||
    die "RUN_ROOT is frozen to another source commit"
else
  printf '%s\n' "$SOURCE_COMMIT" >"$SOURCE_COMMIT_FILE"
fi

"$PYTHON" - "$BUNDLE_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {"subj": 998, "trec": 495}
metadata_path = root / "metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
strict_filter = metadata.get("strict_eval_filter")
if not isinstance(strict_filter, dict) or strict_filter.get("schema_version") != 1:
    raise SystemExit("bundle lacks schema-v1 strict_eval_filter provenance")
declared = strict_filter.get("output_task_jsonl_sha256")
excluded = strict_filter.get("excluded_uids_by_task")
if not isinstance(declared, dict) or not isinstance(excluded, dict):
    raise SystemExit("strict bundle lacks hashes or exclusion provenance")
for task, count in expected.items():
    path = root / f"{task}.jsonl"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != declared.get(task):
        raise SystemExit(f"{task} strict JSONL hash mismatch")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    uids = [str(row.get("uid")) for row in rows]
    if len(rows) != count or len(set(uids)) != count:
        raise SystemExit(f"{task} strict bundle count/UID mismatch")
    if excluded.get(task) != [f"{task}-0"] or f"{task}-0" in set(uids):
        raise SystemExit(f"{task} UID0 exclusion provenance is invalid")
PY

"$PYTHON" - "$RUN_ROOT" "$SCHEDULE_MANIFEST" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
path = pathlib.Path(sys.argv[2])
cells = (("subj", "025"), ("subj", "050"), ("trec", "010"), ("trec", "050"))
abba = (("contigkv", "r1"), ("promixed", "r1"),
        ("promixed", "r2"), ("contigkv", "r2"))
runs = []
for task, budget in cells:
    for method, repeat in abba:
        name = f"k{budget}_{method}_k4_nodefer_{repeat}"
        runs.append({
            "task": task,
            "budget": budget,
            "method": method,
            "repeat": repeat,
            "path": str(root / task / name),
        })
payload = {
    "schema_version": 1,
    "purpose": "Four-cell matched K4/nodefer strict ABBA re-prefill diagnostics.",
    "runs": runs,
}
if len(runs) != 16:
    raise SystemExit("internal schedule must contain exactly 16 runs")
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise SystemExit(f"existing schedule differs: {path}")
else:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
PY

fingerprint_inputs=(
  "$ROOT/scripts/run_prism_max_k4_abba_diagnostics.sh"
  "$CELL_RUNNER"
  "$PAIR_ANALYZER"
  "$GRID_ANALYZER"
  "$ROOT/src/contiguous_fuxian/flexgen_qwen_reprefill.py"
  "$ROOT/src/contiguous_fuxian/flexgen_pcache.py"
  "$ROOT/src/contiguous_fuxian/promixed.py"
  "$BUNDLE_DIR/metadata.json"
  "$BUNDLE_DIR/subj.jsonl"
  "$BUNDLE_DIR/trec.jsonl"
  "$SELECTOR_INDEX/manifest.json"
  "$KV_DIR/.contiguous_fuxian_complete"
  "$STORE_ROOT/subj/metadata.json"
  "$STORE_ROOT/trec/metadata.json"
  "$ROOT/configs/qwen25_k025_contigkv.json"
  "$ROOT/configs/qwen25_k025_ours.json"
  "$ROOT/configs/qwen25_k050_contigkv.json"
  "$ROOT/configs/qwen25_k050_ours.json"
  "$ROOT/configs/qwen25_k010_contigkv.json"
  "$ROOT/configs/qwen25_k010_ours.json"
  "$ROOT/configs/layer_budget_k025_scaled.json"
  "$ROOT/configs/layer_budget_k050_sensitivity.json"
  "$ROOT/configs/layer_budget_k010_sensitivity.json"
)
for input in "${fingerprint_inputs[@]}"; do
  [[ -f "$input" ]] || die "fingerprinted input is missing: $input"
done

fingerprint_candidate="$RUN_ROOT/environment/.source_and_input_sha256.candidate.$$"
sha256sum "${fingerprint_inputs[@]}" >"$fingerprint_candidate"
if [[ -f "$FINGERPRINT" ]]; then
  cmp -s "$FINGERPRINT" "$fingerprint_candidate" ||
    die "source/input fingerprints differ from the frozen diagnostics"
  rm -f "$fingerprint_candidate"
else
  mv "$fingerprint_candidate" "$FINGERPRINT"
fi

assert_source_frozen() {
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] ||
    die "source commit changed during diagnostics"
  assert_tracked_source_clean
  sha256sum --status --check "$FINGERPRINT" ||
    die "a frozen source or input changed during diagnostics"
}

{
  date -Is
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'selector_backend=k4\nscore_mode=nodefer\nwarmup_passes=1\n'
  printf 'warmup_samples=32\ngpu=%s\nreserve_gpu=%s\n' "$GPU" "$RESERVE_GPU"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu \
    --format=csv,noheader
} >"$RUN_ROOT/environment/diagnostics.start.txt"

validate_output() {
  local output="$1" task="$2" budget="$3" method="$4" expected="$5"
  "$PYTHON" - "$output" "$BUNDLE_DIR/$task.jsonl" \
    "$task" "$budget" "$method" "$expected" <<'PY'
import json
import math
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
bundle = pathlib.Path(sys.argv[2])
task, budget, method = sys.argv[3:6]
expected = int(sys.argv[6])
summary_path = output / "summary.json"
records_path = output / "scored_records.jsonl"
if not summary_path.is_file() or not records_path.is_file():
    raise SystemExit("missing completed output files")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in records_path.read_text(
    encoding="utf-8").splitlines() if line]
uids = [str(row.get("uid")) for row in rows]
bundle_uids = {str(json.loads(line)["uid"]) for line in bundle.read_text(
    encoding="utf-8").splitlines() if line}
if len(rows) != expected or len(set(uids)) != expected or set(uids) != bundle_uids:
    raise SystemExit("result UID set differs from strict input bundle")
if f"{task}-0" in set(uids):
    raise SystemExit("pre-excluded UID0 reappeared in result")
runtime = summary.get("runtime", {})
required = {
    "defer_cache_score_updates": False,
    "cache_update_in_ttft": True,
    "warmup_passes": 1,
    "warmup_samples_per_task": 32,
    "warmup_requests": 32,
    "online_selection": True,
    "cache_type": "CKLFU",
    "selector_index_bits": 4,
    "selector_index_group_size": 32,
    "selector_index_preloaded_tasks": [task],
}
for key, value in required.items():
    if runtime.get(key) != value:
        raise SystemExit(f"runtime.{key}={runtime.get(key)!r}, expected {value!r}")
if not math.isclose(float(runtime.get("keep_ratio")), int(budget) / 100.0):
    raise SystemExit("keep ratio differs from requested budget")
if (
    not isinstance(runtime.get("selector_index_preloaded_bytes"), int)
    or runtime["selector_index_preloaded_bytes"] <= 0
):
    raise SystemExit("K4 selector index was not resident")
if any(float(row.get("selector_disk_source_bytes", -1)) != 0 for row in rows):
    raise SystemExit("request path unexpectedly read selector data from SSD")
if summary.get("overall", {}).get("samples") != expected:
    raise SystemExit("summary sample count mismatch")
PY
}

run_one() {
  local task="$1" budget="$2" method="$3" repeat="$4"
  local expected="${EXPECTED_COUNTS[$task]}"
  local name="k${budget}_${method}_k4_nodefer_${repeat}"
  local output="$RUN_ROOT/$task/$name"
  local status

  if [[ -e "$output" ]]; then
    validate_output "$output" "$task" "$budget" "$method" "$expected" ||
      die "existing output is incomplete or invalid: $output"
    echo "[$(date -Is)] skip validated completed $task k$budget $method $repeat"
    return
  fi

  while true; do
    assert_source_frozen
    echo "[$(date -Is)] launch $task k$budget $method $repeat"
    set +e
    PYTHON="$PYTHON" GPU="$GPU" RESERVE_GPU="$RESERVE_GPU" \
      REQUIRE_IDLE_RESERVE="$REQUIRE_IDLE_RESERVE" TASK="$task" \
      BUDGET_TAG="$budget" METHOD="$method" SELECTOR_BACKEND=k4 \
      SELECTOR_INDEX="$SELECTOR_INDEX" SAMPLES_PER_TASK=1000000 \
      WARMUP_PASSES=1 WARMUP_SAMPLES_PER_TASK=32 \
      DEFER_CACHE_SCORE_UPDATES=false PROMIXED_ADAPTIVE_COVERAGE=false \
      RUN_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" RUN_ROOT="$RUN_ROOT" \
      RUN_NAME="$name" SETTLE_SECONDS="$SETTLE_SECONDS" \
      GPU_CACHE_MB=55 CPU_CACHE_MB=131 MODEL_PATH="$MODEL_PATH" \
      BUNDLE_DIR="$BUNDLE_DIR" STORE_ROOT="$STORE_ROOT" \
      FLEXGEN_ROOT="$FLEXGEN_ROOT" KV_DIR="$KV_DIR" "$CELL_RUNNER"
    status=$?
    set -e
    if [[ -e "$output" ]] &&
       validate_output "$output" "$task" "$budget" "$method" "$expected"; then
      echo "[$(date -Is)] accepted validated output after status $status"
      return
    fi
    case "$status" in
      3)
        echo "[$(date -Is)] GPU/storage guard busy; retry in ${WAIT_SECONDS}s"
        sleep "$WAIT_SECONDS"
        ;;
      *) die "runner failed for $task k$budget $method $repeat (status $status)" ;;
    esac
  done
}

publish_pair_analysis() {
  local task="$1" budget="$2" expected="${EXPECTED_COUNTS[$1]}"
  local stem="$RUN_ROOT/validation/${task}_k${budget}_k4_nodefer_abba"
  local candidate="$RUN_ROOT/validation/.${task}_k${budget}_candidate.$$"
  "$PYTHON" "$PAIR_ANALYZER" \
    --run-root "$RUN_ROOT" --task "$task" --budget "$budget" \
    --backend k4 --score-mode nodefer \
    --expected-samples "$expected" \
    --expected-warmup-passes 1 --expected-warmup-samples 32 \
    --bundle-metadata "$BUNDLE_DIR/metadata.json" \
    --bootstrap-samples "$PAIR_BOOTSTRAP_SAMPLES" \
    --output "$candidate"
  for suffix in json md; do
    if [[ -e "${stem}.${suffix}" ]]; then
      cmp -s "${candidate}.${suffix}" "${stem}.${suffix}" ||
        die "existing pair analysis differs for $task k$budget"
      rm -f "${candidate}.${suffix}"
    else
      mv "${candidate}.${suffix}" "${stem}.${suffix}"
    fi
  done
}

for cell in "${CELLS[@]}"; do
  read -r task budget <<<"$cell"
  echo "[$(date -Is)] cell $task k$budget ABBA begins"
  for item in "${ABBA[@]}"; do
    read -r method repeat <<<"$item"
    run_one "$task" "$budget" "$method" "$repeat"
  done
  publish_pair_analysis "$task" "$budget"
done

assert_source_frozen
analysis_candidate="$RUN_ROOT/.grid_analysis_candidate.$$"
"$PYTHON" "$GRID_ANALYZER" \
  --manifest "$SCHEDULE_MANIFEST" \
  --bundle-metadata "$BUNDLE_DIR/metadata.json" \
  --bootstrap-samples "$GRID_BOOTSTRAP_SAMPLES" \
  --output "$analysis_candidate"
for suffix in json md; do
  if [[ -e "${ANALYSIS_STEM}.${suffix}" ]]; then
    cmp -s "${analysis_candidate}.${suffix}" "${ANALYSIS_STEM}.${suffix}" ||
      die "existing combined diagnostics analysis differs"
    rm -f "${analysis_candidate}.${suffix}"
  else
    mv "${analysis_candidate}.${suffix}" "${ANALYSIS_STEM}.${suffix}"
  fi
done

{
  date -Is
  printf 'source_commit=%s\nexecutions=16\nrepeats_per_method=2\n' "$SOURCE_COMMIT"
  printf 'primary_latency=response_ready_ms\nphase_latency=logits_ready_ms\n'
} >"$RUN_ROOT/environment/diagnostics.end.txt"
date -Is >"$RUN_ROOT/diagnostics.done"
echo "[$(date -Is)] completed four-cell K4/nodefer strict ABBA diagnostics"
