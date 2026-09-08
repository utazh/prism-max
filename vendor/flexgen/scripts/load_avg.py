import re
from collections import defaultdict

# 初始化字典来存储每个层的 True 和 False 数量
layer_stats = defaultdict(lambda: {'True': 0, 'False': 0})

# 正则表达式，用于提取层号和是否选中
layer_pattern = re.compile(r"target accum_percent for current attn layer\[(\d+)]")
selection_pattern = re.compile(r"Whether selected\? (True|False)")

# 打开 .log 文件
with open('/home/zrd/impllm/h2o_flexgen/flexgen/logs_sim/fo-30b-gbs1-piqa-padding_mul1-gen1-percent-100-0-100-0-100-0--load_by_percentTrue_[25]sim_thred0.3-cache_typeLRU-disk_typeChunk-prefix_dumpFalse-prefix_aware_infTruegpu-cacheoverlap_ratio.log', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
    # 按每三行进行处理
    for i in range(38, len(lines), 3):
        # 获取当前三行中的第二行（包含层号）
        layer_match = layer_pattern.search(lines[i + 1])
        # 获取当前三行中的第三行（包含 True/False）
        selection_match = selection_pattern.search(lines[i + 2])
        
        if layer_match and selection_match:
            layer_num = int(layer_match.group(1))  # 提取层号
            selected = selection_match.group(1)   # 提取 True/False
            # 统计每个层的 True 和 False
            layer_stats[layer_num][selected] += 1

# 打印统计结果
for layer, counts in layer_stats.items():
    print(f"Layer {layer}: True = {counts['True']}, False = {counts['False']}")
    ture_count = counts['True']
    false_count = counts['False']
    avg_load = (ture_count*0.25+false_count*1)/(ture_count+false_count)
    print(f'avg_laod={avg_load*100}')