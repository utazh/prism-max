#!/bin/bash

# 定义文件路径前缀
result_prefix="fewshots_datasets/"

# 定义任务列表
#tasks=("rte" "piqa" "openbookqa" "winogrande")
#tasks=("rte" "openbookqa")
#tasks=("rte" "piqa" "openbookqa" "winogrande")
#tasks=("rte" "openbookqa")
#tasks=("copa" "squad_val")
tasks=("openbookqa" "winogrande" "copa" "piqa" "rte")
# tasks=("piqa")
#tasks=("squad_val")

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
    local padding_mul="$1"

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
            --prefix-dump \
            --no-cache \
            --padding-mul "$padding_mul"
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

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "full_load-sele_inf-${sele_percent}-prekv"
    done
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
    local model="$2"
    local short_model=$(echo "$model" | sed -E 's/.*\/opt-([0-9]+(\.[0-9]+)?b).*/\1/')
    echo "$short_model"
    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"
        if [[ "$task" != "squad_val" ]]; then
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model "$model" \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --gen-len 32 \
                --motiv2 \
                --rouge \
                --no-cache \
                # --logits
        else
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model "$model" \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --gen-len 48 \
                --motiv2 \
                --no-cache
                # --logits
                # --perplexity
        fi            
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-${short_model}-full.jsonl"
        if [[ "$task" != "squad_val" ]]; then
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
    local model="$2"
    local short_model=$(echo "$model" | sed -E 's/.*\/opt-([0-9]+(\.[0-9]+)?b).*/\1/')
    echo "$short_model"
    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"
        if [[ "$task" != "squad_val" ]]; then
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model "$model" \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --logits \
                --gen-len 1 \
                --motiv2 \
                --samekv \
                --no-cache
                # --logits
        else
            time python flex_motivation2.py \
                --gpu-batch-size 1 \
                --overlap false \
                --model "$model" \
                --prefix-aware-inf \
                --full-load \
                --sele-inf \
                --sele-percent "$sele_percent" \
                --input-path "$input_file" \
                --model-type opt \
                --gen-len 48 \
                --motiv2 \
                --samekv \
                --no-cache
                # --logits
                # --perplexity
        fi
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-${short_model}-full.jsonl"
        if [[ "$task" != "squad_val" ]]; then
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
            --logits \
            --gen-len 1 \
            --no-cache 
            # --logits
     done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output_experiment/${task}-opt-6.7b-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "full_load-full_inf-prekv"
    done
}

# =====> full load keys and sele load values
run_full_only_key_load() {
    local sele_percent="$1"

    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-6.7b \
            --prefix-aware-inf \
            --full-only-key-load \
            --sele-percent "$sele_percent" \
            --input-path "$input_file" \
            --model-type opt \
            --logits \
            --gen-len 1 \
            --disk-type KV_Division \
            # --no-cache
            
    done

    for task in "${tasks[@]}"; do
        result_file="${result_prefix}/output/${task}-opt-6.7b-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "full_only_key_load-${sele_percent}-prekv"
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


    for task in "${tasks[@]}"; do
        input_file="${result_prefix}/input/${task}.jsonl"

        time python flex_opt.py \
            --gpu-batch-size 1 \
            --overlap false \
            --model facebook/opt-30b \
            --model facebook/opt-30b \
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
        result_file="${result_prefix}/output_experiment/${task}-opt-30b-full.jsonl"
        result_file="${result_prefix}/output_experiment/${task}-opt-30b-full.jsonl"

        time python evaluate_task_result.py \
            --result-file "$result_file" \
            --task-name "$task" \
            --model-type opt \
            --exp "sele_load_by_percent_${sele_percent}_${sim_thred}-prekv-fillkeyzero"
    done
}

# =====> generate_mapping_list
run_generate_mapping_list() {
    local padding_mul="$1"
    local sele_percent="$2"

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
            --no-cache \
            --padding-mul "$padding_mul" \
            --generate-mapping-list 
            # --full-load-sele-inf-by-accum
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
                run_prefixkv_dump 57
                ;;
            full_load_sele_inf)
                run_full_load_sele_inf 1
                ;;
            full_load_full_inf)
                run_full_load_full_inf
                ;;
            full_only_key_load)
                run_full_only_key_load 50
                ;;
            sele_load_by_accum)
                run_sele_load_by_accum 90 0.5
                # run_sele_load_by_accum 95 0.5
                # run_sele_load_by_accum 98 0.4
                ;;
            motiv2)
                #run_overlap 5
                #run_samekv 10
                #run_overlap 25 facebook/opt-6.7b
                run_samekv 25 facebook/opt-30b
                run_samekv 25 facebook/opt-30b
                #run_overlap 5 facebook/opt-30b
                run_samekv 5 facebook/opt-30b
                run_samekv 5 facebook/opt-30b
                #run_samekv 5 facebook/opt-30b
                #run_overlap 10 facebook/opt-6.7b
                run_samekv 10 facebook/opt-30b
                run_samekv 10 facebook/opt-30b
                #run_overlap 0
                #run_samekv 0
                #run_overlap 50 facebook/opt-6.7b
                run_samekv 50 facebook/opt-30b
                run_samekv 50 facebook/opt-30b
                #run_overlap 80
                #run_samekv 80
                ;;
            sele_load_by_percent)
                # run_sele_load_by_percent 50 0.5
                # run_sele_load_by_percent 25 0.3
                # run_sele_load_by_percent 10 0.17
                # run_sele_load_by_percent 5 0.1
                # run_sele_load_by_percent 1 0.0
                run_sele_load_by_percent 10 0.17
                run_sele_load_by_percent 50 0.5
                run_sele_load_by_percent 25 0.3
                run_sele_load_by_percent 5 0.1
                run_sele_load_by_percent 10 0.17
                run_sele_load_by_percent 50 0.5
                run_sele_load_by_percent 25 0.3
                run_sele_load_by_percent 5 0.1
                ;;
            generate_mapping_list)
                run_generate_mapping_list 57 10
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