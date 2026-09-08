#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="./fewshots_datasets/output"
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*
WEIGHT_PATH="${WEIGHT_PATH:-${HOME}/opt_weights}"
HF_CACHE_DIR="${HYPERINFER_HF_CACHE:-${HF_HOME:-}}"
RUNTIME_ARGS=(--path "$WEIGHT_PATH")
if [[ -n "$HF_CACHE_DIR" ]]; then
    RUNTIME_ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
fi

result_prefix="fewshots_datasets/"

# 定义任务列表，这里只有"rte"
tasks=("rte")

# 定义相似性阈值数组
sim_thred=(1 0.55 0.3 0.17 0.09 0.05 0.03 0.016 0.009 0.0045 0.0027)

run_sele_load_by_percent() {
    local sele_percent="$1"
    local sim_thred="$2"

    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time "$PYTHON_BIN" flex_opt.py "${RUNTIME_ARGS[@]}" \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
            --prefix-aware-inf \
            --sele-load \
            --sele-load-by-percent \
            --sele-head 0 1 2 \
            --sele-percent "$sele_percent" \
            --sim-thred "$sim_thred" \
            --input-path "$input_file" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --no-cache \
            --fill-keys-zero
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"

        time "$PYTHON_BIN" evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "sele_load_by_percent_${sele_percent}_sim_thred${sim_thred}-prekv-ours"
    done
}

# 假设这里的sele_percent值固定为某个具体数值，你可按需修改
sele_percent="25"

for s in "${sim_thred[@]}"; do
    run_sele_load_by_percent "$sele_percent" "$s"
done

time "$PYTHON_BIN" ./scripts/extract_acc.py\
            --figure22
