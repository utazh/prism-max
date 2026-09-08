import os
import re
import argparse
import csv

def add_parser_arguments(parser):
    parser.add_argument("--figure22", action="store_true")
    parser.add_argument("--figure15", action="store_true")
    parser.add_argument("--figure11", action="store_true")
    parser.add_argument("--folder-path", type=str, default="./fewshots_datasets/output")
parser = argparse.ArgumentParser()
add_parser_arguments(parser)
args = parser.parse_args()
folder_path = args.folder_path
# 用于存储分组后的文件，键为结尾字符串，值为对应结尾的文件列表
grouped_files = {
    "ours.log": [],
    "as.log": [],
    "recomp.log": [],
}

def extract_acc_info(grouped_files, args):
    """
    从已分组的.log文件中提取包含'acc'那一行的信息
    """
    dict = {'1':2, '0.55':1.8, '0.3':1.6, '0.17':1.4, '0.09':1.2, '0.05':1, '0.03':0.8, '0.016':0.6, '0.009':0.4, '0.0045':0.2, '0.0027':0}
    for ending, file_list in grouped_files.items():
        print(f"处理以 {ending} 结尾的文件组:")
        for file_path in file_list:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    pattern = r'"acc": [\d.]+'
                    match = re.search(pattern, content)
                    if match:
                        log = f"在文件 {os.path.basename(file_path)} 中提取到: {match.group(0)}"
                        if args.figure11:
                            with open ('./scripts/figure11.csv', 'a') as f:
                                if 'piqa-opt-30b' in log:
                                    parts = file_path.split('sim_thred')
                                    if len(parts) > 1:
                                        extracted_number = parts[1].split('-', 1)[0]
                                    f.write(f'sim_thred{extracted_number},{match.group(0)},'+'\n')
                        # print(f"在文件 {os.path.basename(file_path)} 中提取到: {match.group(0)}")
                        if args.figure22:
                            with open ('./scripts/figure22.csv', 'a') as f:
                                if 'rte-opt-6.7b' in log:
                                    parts = file_path.split('sim_thred')
                                    if len(parts) > 1:
                                        extracted_number = parts[1].split('-', 1)[0]
                                    f.write(f'sim_thred{extracted_number},alpha{dict[extracted_number]}{match.group(0)}\n')
                        if args.figure15:
                            if True:
                                if 'ours' in os.path.basename(file_path):
                                    with open ('./scripts/figure15.csv', 'a') as f:
                                        pattern = r"^((?:[^-]*-){2}[^-]*)-.*percent_(\d+)"
                                        match1 = re.search(pattern, os.path.basename(file_path))
                                        if match1:
                                            dataset_model = match1.group(1)  # 第一个匹配组：第三个'-'之前的子串
                                            percent_value = match1.group(2)  # 第二个匹配组：percent的值
                                            f.write(f'ours_{dataset_model}_{percent_value}_{match.group(0)}\n')
                                if 'as' in os.path.basename(file_path):
                                    with open ('./scripts/figure15.csv', 'a') as f:
                                        pattern = r"^((?:[^-]*-){2}[^-]*)-.*percent_(\d+)"
                                        match2 = re.search(pattern, os.path.basename(file_path))
                                        if match2:
                                            dataset_model = match2.group(1)  # 第一个匹配组：第三个'-'之前的子串
                                            percent_value = match2.group(2)  # 第二个匹配组：percent的值
                                            f.write(f'as+h20+lru_{dataset_model}_{percent_value}_{match.group(0)}\n')
                                if 'recomp' in os.path.basename(file_path):
                                    print('recomp')
                                    with open ('./scripts/figure15.csv', 'a') as f:
                                        pattern = r"^((?:[^-]*-){2}[^-]*)-"
                                        match3 = re.search(pattern, os.path.basename(file_path))
                                        if match3:
                                            dataset_model = match3.group(1)
                                            f.write(f'recomp_{dataset_model}_{match.group(0)}\n')
            except FileNotFoundError:
                print(f"文件 {file_path} 不存在，跳过该文件")
            except Exception as e:
                print(f"读取文件 {file_path} 时出现错误: {e}")
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith("ours.log"):
            grouped_files["ours.log"].append(file_path)
        elif file.endswith("as.log"):
            grouped_files["as.log"].append(file_path)
        elif file.endswith("recomp.log"):
            grouped_files["recomp.log"].append(file_path)

# 打印分组后的结果，你可以根据实际需求进一步处理这些分组数据
for ending, file_list in grouped_files.items():
    print(f"以 {ending} 结尾的文件列表:")
    for file in file_list:
        print(file)
    print("-" * 20)
extract_acc_info(grouped_files, args)
rows = []
if args.figure15:
    try:
        with open('./scripts/figure15.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)
        # 按照字典序对rows进行排序，排序依据是每行数据组成的元组（将列表转换为元组进行比较）
        sorted_rows = sorted(rows, key=tuple)
        with open('./scripts/figure15.csv', 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            for sorted_row in sorted_rows:
                writer.writerow(sorted_row)
    except FileNotFoundError:
        print(f"文件 {'./scripts/figure15.csv'} 不存在")
    except Exception as e:
        print(f"处理CSV文件时出现错误: {e}")
