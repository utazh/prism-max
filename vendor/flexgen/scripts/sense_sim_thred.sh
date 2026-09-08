model="opt-6.7b"
sele_percent="10"
gpu_size="10240"
cpu_size="32768"
sim_threds=(1	0.554944152828354	0.30796301275838	0.170902273217667	0.094841217227218	0.0526315789473684	0.029207586990966	0.0162085796188621	0.00899485648514035	0.00499164301195884	0.00277008310249307	0.0187794400512351	0.0139896638315216	0.0152816372606734	0.0120745123089769)
dataset="./fewshots_datasets/input/input_expand/rte-expand.jsonl"
padding_mul="1"
model_path="facebook/${model}"
for sim_thred in ${sim_threds[@]};do
    echo python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf  --input-path "$dataset" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --padding-mul "$padding_mul" --suffix-comp --cache-type CKLFU --fill-keys-zero --gpu-size "$gpu_size" --cpu-size "$cpu_size" --reorder
    time python flex_opt.py --gpu-batch-size 1 --overlap false --model "$model_path" --prefix-aware-inf  --input-path "$dataset" --model-type opt --logits --gen-len 1 --sele-load --sele-percent "$sele_percent" --sele-load-by-percent --sim-thred "$sim_thred" --padding-mul "$padding_mul" --suffix-comp --cache-type CKLFU --fill-keys-zero --gpu-size "$gpu_size" --cpu-size "$cpu_size" --reorder
done
mkdir logs/sense_sim_thred
mv logs/*.log logs/sense_sim_thred
cd scripts
python sense_sim_thred_log_analys.py
cd ..