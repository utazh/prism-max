import json
from transformers import AutoTokenizer
from opt_config import get_opt_config
import random
from itertools import accumulate
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
import os

def cal_meet_ntask_percentage(capacity, accu_kvsize):
    ret = []
    meet_ntasks = 0
    for i, value in enumerate(accu_kvsize):
        if value < capacity:
            meet_ntasks += 1
        # ret.append(round(meet_ntasks / (i+1), 2)*100) # 缓存的任务比例
        ret.append(100 - round(meet_ntasks / (i+1), 2)*100) # 需要重计算的任务比例
    return ret

def plot_curves(datalst, labellst, savepath):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    # 设置y轴范围
    plt.ylim(0, 100)

    # 绘制三条曲线
    styles = ['b-D', 'g-s', 'r-o']
    for data, style, label in zip(datalst, styles, labellst):
        plt.plot(range(len(data)), data, style, linewidth=2, markevery=10, markersize=8, label=label)

    # 设置x轴和y轴的标签
    plt.xlabel('Number of tasks')
    plt.ylabel('Recompute percentage (%)')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.savefig(savepath)

def read_jsonl_2wikimqa(filepath):
    retdata = []
    with open(filepath, 'r') as f:
        for line in f:
            retdata.append(json.loads(line))
    print(f'-> total samples = {len(retdata)}')
    print(f'-> keys in sample[0] = {retdata[0].keys()}')
    # for k in retdata[0].keys():
    #     print(f'keys={k}, values={retdata[0][k]}')
    
    # 相同 context 的多个 query 视为一个 task
    tasks = {}
    prefix_dct = {}
    for dct in retdata:
        prefix_hash = hash(dct['context'])
        prefix_dct[prefix_hash] = dct['context']
        
        q = dct['input']
        try:
            a = dct['answers']
        except:
            continue
        if prefix_hash not in tasks:
            tasks[prefix_hash] = [(q, a)]
        else:
            tasks[prefix_hash].append((q,a))
    
    # 统计相同 context 下不同的问题数量，打印具体问题和对应的答案
    # for prefix_hash, qalst in tasks.items():
    #     print(f'-> {len(qalst)} qa pairs:')
    #     for q,a in qalst:
    #         print(f'{q} {a}')
    
    # 统计 tasks 的数量、每个 tasks 的 seqlen、total_kvsize
    # 随机打乱任务，统计累计所需的 kv size
    accu_kvsize = []
    
    # 以 opt-30b 对应的 tokenizer 为例
    print(f'-> total number of tasks: {len(tasks)}')
    config = get_opt_config('facebook/opt-30b')
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left", use_fast=True)
    for prefix_hash, qalst in tasks.items():
        prefix = prefix_dct[prefix_hash]
        input_ids_lst = tokenizer(prefix, padding="max_length",
                          max_length=256, add_special_tokens=False).input_ids
        seqlen, kvsize = len(input_ids_lst), round(config.cache_bytes(1, len(input_ids_lst))/(1<<30))
        print(f'-> seqlen={seqlen}, kvsize={kvsize} GB')
        accu_kvsize.append(kvsize)
    # 打乱 accu_kvsize，并进行向前累加
    random.shuffle(accu_kvsize)
    accu_kvsize = list(accumulate(accu_kvsize))
    print(f'accumulated_kv_requirements: {accu_kvsize}')
    
    capacity_lst = [('GPU',25), ('GPU+CPU',25+128), ('GPU+CPU+SSD',25+128+1024)]
    pltdatalst = []
    pltlabellst = []
    dirname, basename = os.path.split(os.path.abspath(filepath))
    base, ext = os.path.splitext(basename)
    savepath = os.path.join(dirname, f'{base}_recomp_percent.png')
    for label, cap in capacity_lst:
        recomp_ratio_lst = cal_meet_ntask_percentage(cap, accu_kvsize)
        print(f'when capacity is {cap}, recompute_ratio lst={recomp_ratio_lst}')
        pltdatalst.append(recomp_ratio_lst)
        pltlabellst.append(label)
    plot_curves(pltdatalst, pltlabellst, savepath)
    
    return retdata

if __name__ == "__main__":
    read_jsonl_2wikimqa('../data/2wikimqa.jsonl')