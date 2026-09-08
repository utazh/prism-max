import os
from collections import defaultdict

method_order={'recompute':0,
              'Attention Store':1,
              'H2O+LRU':2,
              'H2O+LFU':3,
              'ours':4,
              'ours+tech1':5,
              'ours+tech1+2':6,
              }

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
    datas = defaultdict(lambda:[_ for _ in range(7)])
    # 把 analys_list 中的文件夹中的.log日志文件抽取出来并排序，方便观看
    # analys_list = extract_log_files(analys_list)
    analys_list = extract_log_files(['../logs'])
    analys_list = sorted(analys_list, key=lambda x: os.path.basename(x))
    for file_path in analys_list:
        data = dict()
        file_name = os.path.basename(file_path)
        sele_load = True
        with open(file_path,'r') as f:
            lines = f.readlines()
            cache_type = None
            if 'roFalse' in file_name:
                ro = False
            if 'roTrue' in file_name:
                ro = True
            if 'sele_percent[100]' in file_name:
                sele_load = False
            for line in lines:
                if 'total time:' in line:
                    tol_time = float(line.split('total time: ')[1].split()[0])
                if 'input_path = ' in line:
                    data_set = line.split('input_path = ')[1].split(',')[0].split('/')[-1]
                if 'cache_type:' in line:
                    cache_type = line.split('cache_type:')[1].split(',')[0]
                if "OptConfig(name='" in line:
                    model_name = line.split("OptConfig(name='")[1].split("'")[0]
                if 'Save' in line:
                    continue
                if 'disk_type:' in line:
                    disk_type = line.split('disk_type:')[1].split()[-1]
            if cache_type is None:
                method = 'recompute'
            elif cache_type == 'LRU':
                if sele_load == False:
                    method = 'Attention Store'
                else:
                    if disk_type == 'KV_Division':
                        method = 'H2O+LRU'
                    elif disk_type == 'Chunk':
                        if ro == True:
                            method = 'ours+tech1+2'
                        else:
                            method = 'ours+tech1'
            elif cache_type == 'LFU':
                method = 'H2O+LFU'
            elif cache_type == 'CKLFU':
                method = 'ours'
            # data['model'] = model_name
            # data['dataset'] = data_set
            data['method'] = method
            data['time (s)'] = tol_time
            datas[model_name+'+'+data_set][method_order[method]]=data
    with open('./performance.csv', 'w') as f:
        for group,data in datas.items():
            f.write(f'{group}, \n')
            for row in data:
                if not isinstance(row,dict):
                    continue
                f.write(f"{row['method']},{row['time (s)']}\n")
            f.write(' , \n')



            
                
                