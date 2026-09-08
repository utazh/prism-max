import re
from collections import defaultdict
import os, csv
import argparse
# 初始化字典来存储每个层的 True 和 False 数量

# 正则表达式，用于提取层号和是否选中
layer_pattern = re.compile(r"target accum_percent for current attn layer\[(\d+)]")
selection_pattern = re.compile(r"Whether selected\? (True|False)")

# 打开 .log 文件
def extract_load_ratio(filename, args):
    layer_stats = defaultdict(lambda: {'True': 0, 'False': 0})
    parts = filename.split('sim_thred')
    if len(parts) > 1:
        extracted_number = parts[1].split('-', 1)[0]
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        filtered_lines = []
        for line in lines:
            if "used" not in line and "after" not in line:
                filtered_lines.append(line)
        lines = filtered_lines
        
        # 按每三行进行处理
        for i in range(34, len(lines), 3):
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
    #if args.
    load = 0
    for layer, counts in layer_stats.items():
        if extracted_number == "0.3":
            with open('./scripts/figure19.csv', 'a') as f:
                f.write(f"Layer {layer}: True = {counts['True']}, False = {counts['False']}\n")
        ture_count = counts['True']
        false_count = counts['False']
        avg_load = (ture_count*0.25+false_count*1)/(ture_count+false_count)
        load += avg_load
        if extracted_number == "0.3":
            with open('./scripts/figure19.csv', 'a') as f:
                f.write(f'avg_laod={avg_load*100}\n')
    if args.figure11:
        with open('./scripts/figure11.csv', 'a') as f:
            f.write(f'sim_thred{extracted_number}, load_ratio:{(load/48)*100}%\n')
        
def add_parser_arguments(parser):
    parser.add_argument("--figure11", action="store_true")
parser = argparse.ArgumentParser()
add_parser_arguments(parser)
args = parser.parse_args()

folder_path = "./logs_sim"
file_names = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_names.append(os.path.join(root, file))
for file_name in file_names:
    print(file_name)
    if file_name.endswith('.log'):
        extract_load_ratio(file_name, args)

rows = []
try:
    with open('./scripts/figure11.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
    # 按照字典序对rows进行排序，排序依据是每行数据组成的元组（将列表转换为元组进行比较）
    sorted_rows = sorted(rows, key=tuple)
    with open('./scripts/figure11.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for sorted_row in sorted_rows:
            writer.writerow(sorted_row)
except FileNotFoundError:
    print(f"文件 {'./scripts/figure11.csv'} 不存在")
except Exception as e:
    print(f"处理CSV文件时出现错误: {e}")