#!/usr/bin/env bash
set -euo pipefail

# This script mirrors scripts/lm_performance.sh, but runs the fast pcache path.
# The comparison objects and dataset/model parameters are intentionally kept the
# same as the original script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${REPO_ROOT}/flex_opt_fast_pcache.py}"
weight_path="${WEIGHT_PATH:-${HOME}/opt_weights}"
hf_cache_dir="${HYPERINFER_HF_CACHE:-${HF_HOME:-}}"

runtime_args=(--path "$weight_path")
if [[ -n "$hf_cache_dir" ]]; then
  runtime_args+=(--hf-cache-dir "$hf_cache_dir")
fi

models=("opt-6.7b" "opt-13b" "opt-30b")
sele_percents=(50 25 25 25)
sim_threds=(0.5 0.3 0.3 0.3)
gpu_size="${GPU_SIZE:-10240}"
cpu_size="${CPU_SIZE:-32768}"

datasets=("copa-expand" "rte-expand" "openbookqa-small" "piqa-small")
dataset_paths=(
  "./fewshots_datasets/input/copa-expand.jsonl"
  "./fewshots_datasets/input/rte-expand.jsonl"
  "./fewshots_datasets/input/openbookqa-small.jsonl"
  "./fewshots_datasets/input/piqa-small.jsonl"
)

declare -A PAD_MUL=(
  ["copa-expand|opt-6.7b"]=57
  ["copa-expand|opt-13b"]=54
  ["copa-expand|opt-30b"]=25

  ["rte-expand|opt-6.7b"]=10
  ["rte-expand|opt-13b"]=10
  ["rte-expand|opt-30b"]=5

  ["openbookqa-small|opt-6.7b"]=54
  ["openbookqa-small|opt-13b"]=54
  ["openbookqa-small|opt-30b"]=27

  ["piqa-small|opt-6.7b"]=27
  ["piqa-small|opt-13b"]=27
  ["piqa-small|opt-30b"]=10
)

declare -A SUF_MUL=(
  ["copa-expand|opt-6.7b"]=33
  ["copa-expand|opt-13b"]=33
  ["copa-expand|opt-30b"]=1

  ["rte-expand|opt-6.7b"]=8
  ["rte-expand|opt-13b"]=8
  ["rte-expand|opt-30b"]=1

  ["openbookqa-small|opt-6.7b"]=32
  ["openbookqa-small|opt-13b"]=32
  ["openbookqa-small|opt-30b"]=15

  ["piqa-small|opt-6.7b"]=15
  ["piqa-small|opt-13b"]=15
  ["piqa-small|opt-30b"]=6
)

get_val() {
  local map_name=$1
  local key=$2
  local val

  eval 'val="${'"$map_name"'[$key]-__MISSING__}"'
  if [[ "$val" == "__MISSING__" ]]; then
    echo "missing parameter: ${map_name}[${key}]" >&2
    exit 1
  fi
  echo "$val"
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  "$@"
}

run_timed() {
  print_cmd "$@"
  time "$@"
}

for di in "${!datasets[@]}"; do
  dataset="${datasets[$di]}"
  dataset_path="${dataset_paths[$di]}"
  sele_percent="${sele_percents[$di]}"
  sim_thred="${sim_threds[$di]}"

  for model in "${models[@]}"; do
    key="${dataset}|${model}"
    padding_mul="$(get_val PAD_MUL "$key")"
    suffix_mul="$(get_val SUF_MUL "$key")"
    model_path="facebook/${model}"

    echo
    echo "===== dataset=${dataset} model=${model} padding_mul=${padding_mul} suffix_mul=${suffix_mul} ====="

    # Prepare the important-token mapping list used by selective loading.
    run_cmd "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --prefix-aware-inf --full-load --sele-inf \
      --input-path "$dataset_path" --model-type opt --logits --gen-len 1 \
      --sele-percent "$sele_percent" --no-cache --padding-mul "$padding_mul" \
      --no-prefetch --generate-mapping-list

    # Build prefix KV data. This is a preparation step, not a TTFT comparison.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --prefix-dump --input-path "$dataset_path" --model-type opt --logits \
      --gen-len 1 --padding-mul "$padding_mul" \
      --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --recompute --sep-layer false --suffix-mul "$suffix_mul"

    # AS: full KV loading.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --input-path "$dataset_path" --model-type opt --logits --gen-len 1 \
      --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --sep-layer false --prefix-aware-inf --full-load --suffix-comp \
      --cache-type LRU --suffix-mul "$suffix_mul"

    # AS + H2O + LRU: load all keys, select tokens, then load selected values.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --input-path "$dataset_path" --model-type opt --logits --gen-len 1 \
      --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp \
      --cache-type LRU --disk-type KV_Division \
      --sele-percent "$sele_percent" --suffix-mul "$suffix_mul"

    # AS + H2O + LFU.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --input-path "$dataset_path" --model-type opt --logits --gen-len 1 \
      --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --sep-layer false --prefix-aware-inf --full-only-key-load --suffix-comp \
      --cache-type LFU --disk-type KV_Division \
      --sele-percent "$sele_percent" --suffix-mul "$suffix_mul"

    # IMPRESS: selective loading with CKLFU and reorder, no async prefetch.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --input-path "$dataset_path" --model-type opt --logits --gen-len 1 \
      --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --sep-layer false --prefix-aware-inf --sele-load --suffix-comp \
      --cache-type CKLFU --sele-percent "$sele_percent" --sim-thred "$sim_thred" \
      --disk-type KV_Division --sele-load-by-percent --fill-keys-zero \
      --no-prefetch --reorder --suffix-mul "$suffix_mul"

    # IMPRESS ablation: no reorder, LRU cache.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --input-path "$dataset_path" --model-type opt --logits --gen-len 1 \
      --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --sep-layer false --prefix-aware-inf --sele-load --suffix-comp \
      --cache-type LRU --sele-percent "$sele_percent" --sim-thred "$sim_thred" \
      --disk-type KV_Division --sele-load-by-percent --fill-keys-zero \
      --no-prefetch --suffix-mul "$suffix_mul"

    # IMPRESS ablation: reorder with LRU cache.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --input-path "$dataset_path" --model-type opt --logits --gen-len 1 \
      --padding-mul "$padding_mul" --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --sep-layer false --prefix-aware-inf --sele-load --suffix-comp \
      --cache-type LRU --sele-percent "$sele_percent" --sim-thred "$sim_thred" \
      --disk-type KV_Division --sele-load-by-percent --fill-keys-zero \
      --no-prefetch --reorder --suffix-mul "$suffix_mul"

    # HyperInfer ablation: async prefetch, no reorder, LRU cache.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --prefix-aware-inf --input-path "$dataset_path" --model-type opt \
      --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" \
      --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp \
      --cache-type LRU --fill-keys-zero --disk-type KV_Division \
      --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul"

    # HyperInfer ablation: async prefetch, reorder, LRU cache.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --prefix-aware-inf --input-path "$dataset_path" --model-type opt \
      --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" \
      --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp \
      --cache-type LRU --fill-keys-zero --disk-type KV_Division \
      --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul" \
      --reorder

    # HyperInfer: async prefetch, reorder, CKLFU cache.
    run_timed "$PYTHON_BIN" "$PYTHON_SCRIPT" "${runtime_args[@]}" \
      --gpu-batch-size 1 --overlap false --model "$model_path" \
      --prefix-aware-inf --input-path "$dataset_path" --model-type opt \
      --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" \
      --sele-load-by-percent --sim-thred "$sim_thred" --suffix-comp \
      --cache-type CKLFU --fill-keys-zero --disk-type KV_Division \
      --gpu-size "$gpu_size" --cpu-size "$cpu_size" \
      --padding-mul "$padding_mul" --sep-layer false --suffix-mul "$suffix_mul" \
      --reorder
  done
done
