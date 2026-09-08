import torch
import numpy as np
import os
import time
from functools import partial


'''
文件介绍：
使用前在./cache目录下创建hdd文件夹和软连接向/nvme0n1下目录的ssd文件夹
入口处修改shape(KV cache对应的大小)和chunk_size
save函数把kv存储成对应的三种格式(token,chunk,prefix)
三个主要函数load_token load_chunk load_prefix加载对应的文件,拷贝至cpu tensor,传输到gpu,并返回时间(load time, pcie time)
benchmark函数:先warm执行函数10次(不计时),再测量100次时间取均值
torch的拷贝可以直接跨设备(一个tensor在cpu上,另一个在gpu上)进行,因此不需要.to(device)来进行二次拷贝
torch.split返回的是原tensor的view,因此直接向这个返回的tensor内填充就可以写入原本的gpu tensor中
存在的问题：
1.长度较短下ssd和hdd的读取时间差不多,且波动较大,经常出现hdd比ssd快的情况
2.chunk_size=64下chunk读取比token更慢,且token读取速度与prefix接近
3.chunk_size由16下降至8时chunk的读取时间明显变短
4.chunk粒度下pcie时间跟prefix粒度过于接近,有时出现比prefix快的情况
'''

def uncache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

def load_token(path,k_v_num_tokens,tmp_k):
    # 先使用 sync 命令确保所有的缓冲区数据都写入磁盘
    os.system('sync')
    # 清除缓存
    os.system("sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'")
    cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
    cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
    gpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
    gpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
    a = torch.split(cpu_k_tensor,1,dim=0)
    b = torch.split(cpu_v_tensor,1,dim=0)
    torch.cuda.synchronize()
    st = time.monotonic()
    for i in range(k_v_num_tokens):
        uncache(f'./cache/{path}/token_{i}.npy')
        mmap_kv_tensor = np.load(f'./cache/{path}/token_{i}.npy')
        k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
        a[i][0,:] = k[:]
        b[i][0,:] = v[:]
    torch.cuda.synchronize()
    dt  = time.monotonic() - st

    c = torch.split(gpu_k_tensor,1,dim=0)
    d = torch.split(gpu_v_tensor,1,dim=0)
    torch.cuda.synchronize()
    st = time.monotonic()
    for i in range(len(a)):
        c[i][:] = a[i][:].to('cuda:0')
        d[i][:] = b[i][:].to('cuda:0')
    torch.cuda.synchronize()
    dt_pcie  = time.monotonic() - st

    return dt,dt_pcie

def load_chunk(path,chunk_size,tmp_k):
    # 先使用 sync 命令确保所有的缓冲区数据都写入磁盘
    os.system('sync')
    # 清除缓存
    os.system("sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'")
    cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
    cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
    gpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
    gpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
    a = torch.split(cpu_k_tensor,chunk_size,dim=0)
    b = torch.split(cpu_v_tensor,chunk_size,dim=0)
    chunk_num = len(a)
    torch.cuda.synchronize()
    st = time.monotonic()
    for i in range(chunk_num):
        uncache(f'./cache/{path}/chunk_{i}.npy')
        mmap_kv_tensor = np.load(f'./cache/{path}/chunk_{i}.npy')
        k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
        a[i][:] = k[:]
        b[i][:] = v[:]
    torch.cuda.synchronize()
    dt  = time.monotonic() - st


    c = torch.split(gpu_k_tensor,chunk_size,dim=0)
    d = torch.split(gpu_v_tensor,chunk_size,dim=0)
    torch.cuda.synchronize()
    st = time.monotonic()
    for i in range(len(a)):
        c[i][:] = a[i][:].to('cuda:0')
        d[i][:] = b[i][:].to('cuda:0')
    torch.cuda.synchronize()
    dt_pcie  = time.monotonic() - st
    return dt,dt_pcie

def load_prefix(path,tmp_k):
    # 先使用 sync 命令确保所有的缓冲区数据都写入磁盘
    os.system('sync')
    # 清除缓存
    os.system("sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'")
    cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
    cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
    gpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
    gpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
    torch.cuda.synchronize()
    st = time.monotonic()
    uncache(f'./cache/{path}/prefix.npy')
    mmap_kv_tensor = np.load(f'./cache/{path}/prefix.npy')
    k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
    cpu_k_tensor[:] = k[:]
    cpu_v_tensor[:] = v[:]
    torch.cuda.synchronize()
    dt = time.monotonic() - st

    torch.cuda.synchronize()
    st = time.monotonic()
    gpu_k_tensor[:] = k[:].to('cuda:0')
    gpu_v_tensor[:] = v[:].to('cuda:0')
    torch.cuda.synchronize()
    dt_pcie  = time.monotonic() - st
    return dt,dt_pcie

def save(k,v,chunk_size):
    tmp_k,tmp_v = k,v
    a,b = k,v
    prefix = torch.stack([a,b]).numpy()
    a,b = torch.split(tmp_k,chunk_size,dim=0),torch.split(tmp_v,chunk_size,dim=0)
    chunk_num = len(a)
    np.save('./cache/ssd/prefix.npy',prefix)
    np.save('./cache/hdd/prefix.npy',prefix)
    for i in range(k_v_num_tokens):
        prefix = torch.stack([tmp_k[i],tmp_v[i]]).numpy()
        np.save(f'./cache/ssd/token_{i}.npy',prefix)
        np.save(f'./cache/hdd/token_{i}.npy',prefix)
    for i in range(chunk_num):
        prefix = torch.stack([a[i],b[i]]).numpy()
        np.save(f'./cache/ssd/chunk_{i}.npy',prefix)
        np.save(f'./cache/hdd/chunk_{i}.npy',prefix)


def func_test():
    # return load_token('hdd',k_v_num_tokens=k_v_num_tokens,tmp_k=tmp_k)
    # return load_chunk('hdd',chunk_size=chunk_size,tmp_k=tmp_k)
    # return load_prefix('hdd',tmp_k=tmp_k)
    # return load_token('ssd',k_v_num_tokens=k_v_num_tokens,tmp_k=tmp_k)
    # return load_chunk('ssd',chunk_size=chunk_size,tmp_k=tmp_k)
    # return load_prefix('ssd',tmp_k=tmp_k)
    pass

def benchmark(func):
    for i in range(10):
        func()

    costs = []
    pcie_costs = []

    for i in range(5):
        dts = 0
        dts_pcie = 0
        for i in range(20):
            dt,dt_pcie = func()
            dts += dt
            dts_pcie += dt_pcie
        costs.append((dts) / 20)
        pcie_costs.append((dts_pcie) / 20)
    print(np.mean(costs),np.mean(pcie_costs))

# mmap_kv_tensor = np.load('./cache/kvs/tmp.npy', mmap_mode='r')
# tmp_k, tmp_v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
if __name__ == '__main__':
    shape = [233,32,128]
    chunk_size = 16

    tmp_k = torch.rand(size=shape,dtype=torch.float16,device='cpu')
    tmp_v = torch.rand(size=shape,dtype=torch.float16,device='cpu')

    k_v_num_tokens = shape[0]

    save(tmp_k,tmp_v,chunk_size=chunk_size)

    # benchmark()
    benchmark(partial(load_token,'hdd',k_v_num_tokens=k_v_num_tokens,tmp_k=tmp_k))
    benchmark(partial(load_chunk,'hdd',chunk_size=chunk_size,tmp_k=tmp_k))
    benchmark(partial(load_prefix,'hdd',tmp_k=tmp_k))
    benchmark(partial(load_token,'ssd',k_v_num_tokens=k_v_num_tokens,tmp_k=tmp_k))
    benchmark(partial(load_chunk,'ssd',chunk_size=chunk_size,tmp_k=tmp_k))
    benchmark(partial(load_prefix,'ssd',tmp_k=tmp_k))


# print(load_token('hdd',k_v_num_tokens=k_v_num_tokens,tmp_k=tmp_k))
# print(load_chunk('hdd',chunk_size=chunk_size,tmp_k=tmp_k))
# print(load_prefix('hdd',tmp_k=tmp_k))
# print(load_token('ssd',k_v_num_tokens=k_v_num_tokens,tmp_k=tmp_k))
# print(load_chunk('ssd',chunk_size=chunk_size,tmp_k=tmp_k))
# print(load_prefix('ssd',tmp_k=tmp_k))

# prefix = torch.stack([tmp_k,tmp_v]).numpy()
# np.save('./cache/kvs/tmp.npy',prefix)





# token read
# cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# a = torch.split(cpu_k_tensor,1,dim=0)
# b = torch.split(cpu_v_tensor,1,dim=0)
# torch.cuda.synchronize()
# st = time.monotonic()
# # for i in range(k_v_num_tokens):
# #     uncache(f'./cache/hdd/token_{i}.npy')
# #     mmap_kv_tensor = np.load(f'./cache/hdd/token_{i}.npy', mmap_mode='r')
# #     k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
# #     a[i][0,:] = k[:]
# #     b[i][0,:] = v[:]
# # torch.cuda.synchronize()
# # dt_hdd_token  = time.monotonic() - st



# cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# a = torch.split(cpu_k_tensor,1,dim=0)
# b = torch.split(cpu_v_tensor,1,dim=0)

# torch.cuda.synchronize()
# st = time.monotonic()
# for i in range(k_v_num_tokens):
#     uncache(f'./cache/ssd/token_{i}.npy')
#     mmap_kv_tensor = np.load(f'./cache/ssd/token_{i}.npy', mmap_mode='r')
#     k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
#     a[i][0,:] = k[:]
#     b[i][0,:] = v[:]
# torch.cuda.synchronize()
# dt_ssd_token  = time.monotonic() - st


# torch.cuda.synchronize()
# st = time.monotonic()
# for i in range(len(a)):
#     gpu_k_tensor[i,:] = a[i].to('cuda:0')[:]
#     gpu_v_tensor[i,:] = b[i].to('cuda:0')[:]
# torch.cuda.synchronize()
# dt_pcie_token  = time.monotonic() - st



# gpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
# gpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cuda:0')
# # chunk read

# cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# a = torch.split(cpu_k_tensor,chunk_size,dim=0)
# b = torch.split(cpu_v_tensor,chunk_size,dim=0)
# torch.cuda.synchronize()
# st = time.monotonic()
# for i in range(chunk_num):
#     uncache(f'./cache/hdd/chunk_{i}.npy')
#     mmap_kv_tensor = np.load(f'./cache/hdd/chunk_{i}.npy', mmap_mode='r')
#     k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
#     a[i][:] = k[:]
#     b[i][:] = v[:]
# torch.cuda.synchronize()
# dt_hdd_chunk  = time.monotonic() - st


# cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# a = torch.split(cpu_k_tensor,chunk_size,dim=0)
# b = torch.split(cpu_v_tensor,chunk_size,dim=0)
# torch.cuda.synchronize()
# st = time.monotonic()
# for i in range(chunk_num):
#     uncache(f'./cache/ssd/chunk_{i}.npy')
#     mmap_kv_tensor = np.load(f'./cache/ssd/chunk_{i}.npy', mmap_mode='r')
#     k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
#     a[i][:] = k[:]
#     b[i][:] = v[:]
# torch.cuda.synchronize()
# dt_ssd_chunk  = time.monotonic() - st


# torch.cuda.synchronize()
# st = time.monotonic()
# for i in range(len(a)):
#     if i < chunk_num - 1:
#         gpu_k_tensor[i*chunk_size:(i+1)*chunk_size,:] = a[i].to('cuda:0')[:]
#         gpu_v_tensor[i*chunk_size:(i+1)*chunk_size,:] = b[i].to('cuda:0')[:]
#     else:
#         gpu_k_tensor[i*chunk_size:,:] = a[i].to('cuda:0')[:]
#         gpu_v_tensor[i*chunk_size:,:] = b[i].to('cuda:0')[:]
# torch.cuda.synchronize()
# dt_pcie_chunk  = time.monotonic() - st



# torch.cuda.synchronize()
# st = time.monotonic()
# uncache('./cache/hdd/prefix.npy')
# mmap_kv_tensor = np.load('./cache/hdd/prefix.npy', mmap_mode='r')
# k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
# cpu_k_tensor[:] = k[:]
# cpu_v_tensor[:] = v[:]
# torch.cuda.synchronize()
# dt_hdd_prefix  = time.monotonic() - st

# cpu_k_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# cpu_v_tensor = torch.empty(size=tmp_k.shape,dtype=tmp_k.dtype,device='cpu')
# torch.cuda.synchronize()
# st = time.monotonic()
# uncache('./cache/ssd/prefix.npy')
# mmap_kv_tensor = np.load('./cache/ssd/prefix.npy', mmap_mode='r')
# k, v = torch.from_numpy(mmap_kv_tensor[0, :]), torch.from_numpy(mmap_kv_tensor[1, :])
# cpu_k_tensor[:] = k[:]
# cpu_v_tensor[:] = v[:]
# torch.cuda.synchronize()
# dt_ssd_prefix  = time.monotonic() - st

# torch.cuda.synchronize()
# st = time.monotonic()
# gpu_k_tensor[:] = k.to('cuda:0')[:]
# gpu_v_tensor[:] = v.to('cuda:0')[:]
# torch.cuda.synchronize()
# dt_pcie_prefix  = time.monotonic() - st



# with open('./prefix_time.txt','a+') as f:
#     f.write(f'dt_ssd_prefix:{dt_ssd_prefix},dt_hdd_prefix:{dt_hdd_prefix},dt_pcie_prefix:{dt_pcie_prefix}\ndt_ssd_chunk:{dt_ssd_chunk},dt_hdd_chunk:{dt_hdd_chunk},dt_pcie_chunk:{dt_pcie_chunk}\ndt_ssd_token:{dt_ssd_token},dt_hdd_token:{dt_hdd_token},dt_pcie_token:{dt_pcie_token}\n\n')