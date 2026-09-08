model="opt-6.7b"
sele_percent="25"
gpu_size="10240"
cpu_size="32768"
sim_threds="0.3"
dataset="./fewshots_datasets/input/input_expand/rte-expand.jsonl"
padding_mul="10"
model_path="facebook/${model}"
chunk_sizes=(16 32 64 128 256)
for chunk_size in ${chunk_sizes[@]};do
    echo python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf  --input-path "$dataset" --model-type opt --logits --gen-len 1 --full-only-key-load --sele-percent "$sele_percent" --padding-mul "$padding_mul" --suffix-comp --cache-type LFU --disk-type KV_Division --gpu-size "$gpu_size" --cpu-size "$cpu_size" --chunk-size "$chunk_size"
    time python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf  --input-path "$dataset" --model-type opt --logits --gen-len 1 --full-only-key-load --sele-percent "$sele_percent" --padding-mul "$padding_mul" --suffix-comp --cache-type LFU --disk-type KV_Division --gpu-size "$gpu_size" --cpu-size "$cpu_size" --chunk-size "$chunk_size"
    echo python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf  --input-path "$dataset" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --padding-mul "$padding_mul" --suffix-comp --cache-type CKLFU --fill-keys-zero --gpu-size "$gpu_size" --cpu-size "$cpu_size" --reorder --chunk-size "$chunk_size"
    time python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf  --input-path "$dataset" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --padding-mul "$padding_mul" --suffix-comp --cache-type CKLFU --fill-keys-zero --gpu-size "$gpu_size" --cpu-size "$cpu_size" --reorder --chunk-size "$chunk_size"
done
mkdir logs/sense_sim_chunk_size
mv logs/*.log logs/sense_sim_chunk_size
cd scripts
python sense_chunk_size_log_analys.py
cd ..