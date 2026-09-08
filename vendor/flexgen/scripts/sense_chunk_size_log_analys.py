import os
from collections import defaultdict

def extract_log_files(file_paths):
    # 有一个list变量，其中有很多字符串表示文件路径或者文件夹路径，抽取出其中所有以'.log'为结尾的文件路径，放到列表并返回
    log_files = []
    for path in file_paths:
        if os.path.isfile(path) and path.endswith('.log'):
            log_files.append(path)
        elif os.path.isdir(path):
            sub_files = [os.path.join(path, f) for f in os.listdir(path)]
            log_files.extend(extract_log_files(sub_files))
    return log_files


if __name__ == '__main__':
    # number = re.findall("[\d,.]+",str)
    datas = defaultdict(lambda:defaultdict(lambda:{}))
    # 把 analys_list 中的文件夹中的.log日志文件抽取出来并排序，方便观看
    # analys_list = extract_log_files(analys_list)
    analys_list = extract_log_files(['../logs/sense_chunk_size'])
    analys_list = sorted(analys_list, key=lambda x: os.path.basename(x))
    for file_path in analys_list:
        data = defaultdict(lambda:{})
        file_name = os.path.basename(file_path)
        chunk_size = int(file_name.split('cksize')[1].split('gpu')[0])
        with open(file_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'total time:' in line:
                    tol_time = float(line.split('total time: ')[1].split()[0])
                if 'input_path = ' in line:
                    data_set = line.split('input_path = ')[1].split(',')[0].split('/')[-1]
                if "OptConfig(name='" in line:
                    model_name = line.split("OptConfig(name='")[1].split("'")[0]
                if 'cache_type:' in line:
                    cache_type = line.split('cache_type:')[1].split(',')[0]
            if cache_type == 'LFU':
                method = 'H2O+LFU'
            elif cache_type == 'CKLFU':
                method = 'ours'
            datas[model_name+'+'+data_set][chunk_size][method]=tol_time

    # for group,data in datas.items():
    #     data.sort(key=lambda row:row['chunk_size'],reverse=True)

    with open('./sense_chunk_size.csv', 'w') as f:
        for group,data in datas.items():
            f.write(f'{group}, , , \n')
            f.write(f'chunk size , H2O+LFU , ours , speed up\n')
            for chunk_size,row in sorted(data.items()):
                if not isinstance(row,dict):
                    continue
                f.write(f"{chunk_size} , {row['H2O+LFU']} , {row['ours']} , {row['H2O+LFU']/row['ours']}\n")
            f.write(' , , , \n')



            
                
                