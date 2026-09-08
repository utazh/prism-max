import json, os, random
input_paths = [
    # '../flexgen/fewshots_datasets/input/openbookqa-expand.jsonl',
    # '../flexgen/fewshots_datasets/input/piqa-expand.jsonl',
    # '../flexgen/fewshots_datasets/input/winogrande-expand.jsonl',
    '../flexgen/fewshots_datasets/input/copa-expand.jsonl',
]

# 初始化计数器
line_count = 0
for input_file in input_paths:
    prefix_dict = {}
    requests= []
# 打开输入文件并读取
    with open(input_file, 'r', encoding='utf-8') as infile:
        # 打开输出文件写入
        outfile_dir = os.path.dirname(input_file)
        outfile_base = os.path.basename(input_file).split('-')[0]+'-sample.jsonl'
        output_file = os.path.join(outfile_dir, outfile_base)
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # 逐行读取输入文件
            for id, line in enumerate(infile, start=1):
                # 将前1000行写入输出文件
                if line.strip() != '':    
                    query = json.loads(line)
                    requests.append(query)
                    prompt = query['prompt']
                    suffix_len = len(query['suffix'])
                    prefix = prompt[:(len(prompt) - suffix_len)]
                    if prefix not in prefix_dict:
                        prefix_dict[prefix] = []
                        prefix_dict[prefix].append(id)
                    else:
                        prefix_dict[prefix].append(id)
            print(f'shared_prefix_count:{len(prefix_dict)}')
            for prefix, index in prefix_dict.items():
                print(f'prefix:{prefix[-5:]}, index_len:{len(index)}')
                if 'openbook' in input_file:
                    num_to_select = len(index) // 20
                elif 'piqa' in input_file:
                    num_to_select = len(index) // 35
                elif 'copa' in input_file:
                    num_to_select = len(index) // 2
                else:
                    num_to_select = len(index) // 25
                selected_indices = random.sample(index, num_to_select)
                print(f'prefix:{prefix[-5:]}, index_len:{len(selected_indices)}')
                for idx in selected_indices:
                    json.dump(requests[idx-1], outfile)
                    outfile.write('\n')
