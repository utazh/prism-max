#!/bin/bash

# 定义文件路径前缀
result_prefix="fewshots_datasets/"

# 定义任务列表
# tasks=("piqa" "openbookqa" "winogrande")
tasks=("2wikimqa_long")
# # =====> 不压缩，原始模型精度
run_ori_acc() {
    # 运行 flex_opt.py 对于每个任务
    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
            --input-path "$input_file" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --no-cache
    done

    # 评测精度
    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "ori"
    done

}


# # =====> 存储 prefix kv (time 12m, 5m, 3m, 5m, 7m)
run_prefixkv_dump() {
    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-30b \
            --input-path "$input_file" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --prefix-dump \
            --no-cache \
            --padding-mul 1 \
            --suffix-mul 1 \
            --reorder
    done

}




# # =====> 不同层筛选一定百分比的 KV (full load, sele inf) (9m, 4m, 2m, 4m, 6m)
run_full_load_sele_inf() {
    local sele_percent="$1"

    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
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

    # for task in "${tasks[@]}"; do
    #     result_file="${result_prefix}/output_experiment/${task}-opt-6.7b-full.jsonl"

    #     time python evaluate_task_result.py \
    #         --result-file "$result_file" \
    #         --task-name "$task" \
    #         --model-type opt \
    #         --exp "full_load-sele_inf-${sele_percent}-prekv"
    # done
}

# # =====> 不同层筛选变化的百分比的 KV
run_full_load_sele_inf_by_accum() {
    local accum_percent="$1"

    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
            --prefix-aware-inf \
            --full-load \
            --sele-inf \
            --accum-percent "$accum_percent" \
            --input-path "$input_file" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --no-cache \
            --full-load-sele-inf-by-accum
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "full_load-sele_inf-by-accum-${accum_percent}-prekv"
    done
}

# motivation2
run_overlap(){
    local sele_percent="$1"

    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"
        if [[ "$task" == "openbookqa" ]]; then
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model facebook/opt-6.7b \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --logits \
                --gen-len 1 \
                --motiv2
        else
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model facebook/opt-6.7b \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --logits \
                --gen-len 1 \
                --perplexity \
                --motiv2
        fi            
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"
        if [[ "$task" == "openbookqa" ]]; then
            time python evaluate_task_result.py \
                --result-file "$result_file" \
                --task-name "$task" \
                --model-type opt \
                --exp "full_load-sele_inf-${sele_percent}-prekv" \
                --motiv2 \
                --sele-percent "$sele_percent"
        fi
    done
}
run_samekv(){
    local sele_percent="$1"

    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"
        if [[ "$task" == "openbookqa" ]]; then
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model facebook/opt-6.7b \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --logits \
                --gen-len 1 \
                --motiv2 \
                --samekv
        else
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model facebook/opt-6.7b \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --logits \
                --gen-len 1 \
                --perplexity \
                --motiv2 \
                --samekv
        fi
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"
        if [[ "$task" == "openbookqa" ]]; then
            time python evaluate_task_result.py \
                --result-file "$result_file" \
                --task-name "$task" \
                --model-type opt \
                --exp "full_load-sele_inf-${sele_percent}-prekv" \
                --samekv \
                --sele-percent "$sele_percent"
        fi
    done
}
# =====> full load and full inf
run_full_load_full_inf() {
    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
            --prefix-aware-inf \
            --full-load \
            --input-path "$input_file" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --no-cache
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "full_load-full_inf-prekv"
    done
}


# =====> design 1 by accumulative values (time 13m, 5m, 2m, 6m) (8m, 1m, 1m, 1m, 1m)
run_sele_load_by_accum() {
    local accum_percent="$1"
    local sim_thred="$2"

    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
            --prefix-aware-inf \
            --sele-load \
            --sele-head 0 1 2 \
            --accum-percent "$accum_percent" \
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

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "sele_load_by_accum_${accum_percent}_${sim_thred}-prekv-fillkzeros"
    done
}

# =====> design 1 by specified percent
run_sele_load_by_percent() {
    local sele_percent="$1"
    local sim_thred="$2"
    local model="$3"
    local short_model=$(echo "$model" | sed -E 's/.*\/opt-([0-9]+(\.[0-9]+)?b).*/\1/')
    echo "$short_model"


    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model "$model" \
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
        result_file="${result_prefix}/output_experiment/${task}-opt-${short_model}-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "sele_load_by_percent_${sele_percent}_${sim_thred}-prekv-fillkeyzero"
    done
}

# 定义一个函数来处理未知选项
handle_unknown_option() {
    echo "Unknown option: $1"
    echo "Usage: $0 [ori_acc | prefixkv_dump | full_load_sele_inf | full_load_sele_inf_by_accum | full_load_full_inf | sele_load ] ..."
    exit 1
}

# 主程序
main() {
    # 遍历所有参数
    for arg in "$@"; do
        case "$arg" in
            ori_acc)
                run_ori_acc
                ;;
            prefixkv_dump)
                run_prefixkv_dump
                ;;
            full_load_sele_inf)
                # run_full_load_sele_inf 50
                run_full_load_sele_inf 25
                # run_full_load_sele_inf 10
                # run_full_load_sele_inf 20
                ;;
            full_load_sele_inf_by_accum)
                run_full_load_sele_inf_by_accum 90
                ;;
            full_load_full_inf)
                run_full_load_full_inf
                ;;
            sele_load_by_accum)
                run_sele_load_by_accum 90 0.5
                # run_sele_load_by_accum 95 0.5
                # run_sele_load_by_accum 98 0.4
                ;;
            motiv2)
                #run_overlap 10
                #run_samekv 10
                #run_overlap 20
                #run_samekv 20
                #run_overlap 40
                #run_samekv 40
                #run_overlap 60
                #run_samekv 60
                run_overlap 98
                #run_samekv 80
                ;;
            sele_load_by_percent)
                #run_sele_load_by_percent 50 0.5
                #run_sele_load_by_percent 25 0.1
                #run_sele_load_by_percent 25 0.3
                #run_sele_load_by_percent 25 0.5
                #run_sele_load_by_percent 25 0.6
                #run_sele_load_by_percent 25 0.8
                #run_sele_load_by_percent 25 1.0
                #run_sele_load_by_percent 10 0.0188 facebook/opt-6.7b
                #run_sele_load_by_percent 10 0.014 facebook/opt-6.7b
                # run_sele_load_by_percent 25 0.3 facebook/opt-6.7b
                # run_sele_load_by_percent 50 0.5 facebook/opt-6.7b
                # run_sele_load_by_percent 25 0.3 facebook/opt-13b
                run_sele_load_by_percent 25 0.3 facebook/opt-13b
                run_sele_load_by_percent 25 0.3 facebook/opt-6.7b   
                #run_sele_load_by_percent 25 0.678 facebook/opt-13b
                # run_sele_load_by_percent 25 0.459 facebook/opt-13b
                # run_sele_load_by_percent 25 0.311 facebook/opt-13b
                # run_sele_load_by_percent 25 0.211 facebook/opt-13b
                # run_sele_load_by_percent 25 0.143 facebook/opt-13b
                # run_sele_load_by_percent 25 0.097 facebook/opt-13b
                # run_sele_load_by_percent 25 0.066 facebook/opt-13b
                # run_sele_load_by_percent 25 0.044 facebook/opt-13b
                # run_sele_load_by_percent 25 0.030 facebook/opt-13b
                # run_sele_load_by_percent 25 0.020 facebook/opt-13b
                # run_sele_load_by_percent 10 0.307 facebook/opt-6.7b
                # run_sele_load_by_percent 10 0.095 facebook/opt-6.7b
                # run_sele_load_by_percent 10 0.053 facebook/opt-6.7b
                # run_sele_load_by_percent 10 0.029 facebook/opt-6.7b
                # run_sele_load_by_percent 10 1.0 facebook/opt-6.7b
                # run_sele_load_by_percent 10 0.0162 facebook/opt-6.7b
                # run_sele_load_by_percent 10 0.009 facebook/opt-6.7b
                # run_sele_load_by_percent 10 0.005 facebook/opt-6.7b
                # run_sele_load_by_percent 10 0.00227 facebook/opt-6.7b
                #run_sele_load_by_percent 25 0.3 facebook/opt-30b
                #run_sele_load_by_percent 50 0.5 facebook/opt-30b
                #run_sele_load_by_percent 10 0.17 facebook/opt-13b
                #run_sele_load_by_percent 5 0.1 facebook/opt-13b
                #run_sele_load_by_percent 25 0.3 facebook/opt-13b
                #run_sele_load_by_percent 50 0.5 facebook/opt-13b
                # run_sele_load_by_percent 1 0.0
                # run_sele_load_by_percent 50 0.9
                # run_sele_load_by_percent 50 0.5
                # run_sele_load_by_percent 50 0.0
                # run_sele_load_by_percent 25 0.9
                # run_sele_load_by_percent 25 0.3
                # run_sele_load_by_percent 25 0.0
                # run_sele_load_by_percent 10 0.9
                # run_sele_load_by_percent 10 0.17
                # run_sele_load_by_percent 10 0.0
                # run_sele_load_by_percent 5 0.9
                # run_sele_load_by_percent 5 0.1
                # run_sele_load_by_percent 5 0.0
                # run_sele_load_by_percent 0 0.0
                # run_sele_load_by_percent 0 0.0
                # run_sele_load_by_percent 0 0.0
                ;;
            *)
                # 处理未知选项
                handle_unknown_option "$arg"
                ;;
        esac
    done
}

# 运行主程序
main "$@"