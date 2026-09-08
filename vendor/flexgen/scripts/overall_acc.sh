#!/bin/bash

result_prefix="./fewshots_datasets"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${result_prefix}/output"
mkdir -p "$OUTPUT_DIR"
WEIGHT_PATH="${WEIGHT_PATH:-${HOME}/opt_weights}"
HF_CACHE_DIR="${HYPERINFER_HF_CACHE:-${HF_HOME:-}}"
RUNTIME_ARGS=(--path "$WEIGHT_PATH")
if [[ -n "$HF_CACHE_DIR" ]]; then
    RUNTIME_ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
fi

# 定义任务列表
tasks=("copa")

model=("facebook/opt-6.7b" "facebook/opt-30b")

# h20+as+lru
run_full_load_sele_inf() {
    local sele_percent="$1"

    for m in "${model[@]}"; do
        for task in "${tasks[@]}"; do
            input_file="${result_prefix}/input/${task}.jsonl"

            time "$PYTHON_BIN" flex_opt.py "${RUNTIME_ARGS[@]}" \
                --gpu-batch-size 1 \
                --overlap false \
                --model "$m" \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --logits \
                --gen-len 1 \
                --no-cache
                # --full-load-sele-inf-by-accum
        done

        for task in "${tasks[@]}"; do
            result_file="${result_prefix}/output/${task}-${m##*/}-full.jsonl"

            time "$PYTHON_BIN" evaluate_task_result.py \
                --result-file "$result_file" \
                --task-name "$task" \
                --model-type opt \
                --exp "full_load-sele_inf-percent${sele_percent}-prekv-as"
        done
    done
}
# ours
run_sele_load_by_percent() {
    local sele_percent="$1"
    local sim_thred="$2"

    for m in "${model[@]}"; do
        for task in "${tasks[@]}"; do
            input_file="${result_prefix}/input/${task}.jsonl"

            time "$PYTHON_BIN" flex_opt.py "${RUNTIME_ARGS[@]}" \
                --gpu-batch-size 1 \
                --overlap false \
                --model "$m" \
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
            result_file="${result_prefix}/output/${task}-${m##*/}-full.jsonl"

            time "$PYTHON_BIN" evaluate_task_result.py \
                --result-file "$result_file" \
                --task-name "$task" \
                --model-type opt \
                --exp "sele_load_by-percent_${sele_percent}_sim_thred${sim_thred}-prekv-ours"
        done
    done
}
# recompute
run_full_load_full_inf() {
    for m in "${model[@]}"; do
        for task in "${tasks[@]}"; do
            input_file="${result_prefix}/input/${task}.jsonl"

            time "$PYTHON_BIN" flex_opt.py "${RUNTIME_ARGS[@]}" \
                --gpu-batch-size 1 \
                --overlap false \
                --model "$m" \
                --prefix-aware-inf \
                --full-load \
                --input-path "$input_file" \
                --model-type opt \
                --logits \
                --gen-len 1 \
                --no-cache
        done

        for task in "${tasks[@]}"; do
            result_file="${result_prefix}/output/${task}-${m##*/}-full.jsonl"

            time "$PYTHON_BIN" evaluate_task_result.py \
                --result-file "$result_file" \
                --task-name "$task" \
                --model-type opt \
                --exp "full_load-full_inf-recomp"
        done
    done
}
