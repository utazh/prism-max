"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os, sys, shutil
import re
import pickle
import time
from typing import Union, List, Optional
import psutil

import json, tqdm
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from compression import CompressionConfig
from opt_config import OptConfig, get_opt_config, download_opt_weights
from llama_config import LlamaConfig, get_llama_config, download_llama_weights
from pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, LlamaTorchDevice ,general_copy, fix_recursive_import,
    cache_replace, acc_replace)
from timer import timers
from utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, print_cpu_mem_usage,
    write_benchmark_log, read_benchmark_log)

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import math
from my_pcache_async import Pcache

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

prefix_table = {}
max_prefix_id = 0
pcache = None
path = None
layer_num = None
model_name = None
dt_prefix = 0
dt_all = 0  
dt_load = 0
dt_sele = 0
dt_load_head = 0
dt_suffix = 0
dt_key = 0
dt_value = 0
dt_tol = 0
reorder = None
all_ttft = []
orcale_ttft = []
orcale_load_time = 0
dt_async_prefetch_wait = 0
dt_async_prefetch_submit = 0
async_prefetch_jobs = 0
async_prefetch_tokens = 0
async_prefetch_hit_tokens = 0
async_prefetch_miss_tokens = 0

PREFETCH_RATIO = None
CURRENT_SAMPLE_PREFETCH_RATIO = None

PROFILE_KV_IO_LAYER = os.environ.get("PROFILE_KV_IO_LAYER")
PROFILE_KV_IO_LAYER = int(PROFILE_KV_IO_LAYER) if PROFILE_KV_IO_LAYER else None

def tensor_nbytes(tensor):
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement()

def maybe_profile_kv_io(tag, layer_id, seconds, *tensors):
    if PROFILE_KV_IO_LAYER is None or layer_id != PROFILE_KV_IO_LAYER:
        return
    bytes_moved = sum(tensor_nbytes(tensor) for tensor in tensors)
    gb = bytes_moved / 1e9
    bandwidth = gb / seconds if seconds > 0 else 0
    print(
        "[kv_io_profile] "
        f"tag={tag},"
        f"layer={layer_id},"
        f"bytes={bytes_moved},"
        f"gb={gb:.6f},"
        f"time_s={seconds:.8f},"
        f"bandwidth_gbps={bandwidth:.6f}"
    )

def uncache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

def insert_pcache(pcache:Pcache,hash_name):
    global path,layer_num,model_name,max_prefix_id,reorder
    keys = []
    values = []
    if layer_num is None:
        i = 0
        file = f'{model_name}_attnid{i}_{hash_name}.npy'
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path):
            print(f'ERROR: 找不到 Prefix 的 KV文件 {file_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
            sys.exit(-1)
        while os.path.exists(file_path):
            i += 1
            mmap_kv_tensor = np.load(file_path, mmap_mode='r')
            prefix_k, prefix_v = torch.from_numpy(mmap_kv_tensor[0,: , :, :]), torch.from_numpy(mmap_kv_tensor[1,:, :, :])
            keys.append(prefix_k)
            values.append(prefix_v)
            file = f'{model_name}_attnid{i}_{hash_name}.npy'
            file_path = os.path.join(path, file)
        layer_num = i
    else:
        for i in range(layer_num):
            file = f'{model_name}_attnid{i}_{hash_name}.npy'
            file_path = os.path.join(path, file)
            if not os.path.exists(file_path):
                print(f'ERROR: 找不到 Prefix 的 KV文件 {file_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                sys.exit(-1)
            mmap_kv_tensor = np.load(file_path, mmap_mode='r')
            prefix_k, prefix_v = torch.from_numpy(mmap_kv_tensor[0,: , :, :]), torch.from_numpy(mmap_kv_tensor[1,:, :, :])
            keys.append(prefix_k)
            values.append(prefix_v)
    key = torch.stack(keys)
    value = torch.stack(values)
    if args.reorder:
        pcache.insert(prefix_id=max_prefix_id,key=key,value=value,reorder=reorder[hash_name])
    else:
        pcache.insert(prefix_id=max_prefix_id,key=key,value=value)
    prefix_table[hash_name] = max_prefix_id
    max_prefix_id += 1
    print(f'id:{max_prefix_id}')


def init_pcache(pcache:Pcache,folder_path,name):
    # folder_path = './cache/prefix/'
    # name = 'facebook_opt-6.7b'
    # print(f"folder_path = {folder_path}\n")
    files = os.listdir(folder_path)
    i = 0
    layer = 0
    name_list = []
    hash_names = set()
    global prefix_table,path,layer_num,model_name,max_prefix_id,reorder
    path = folder_path
    model_name = name 
    for file in files:
        _,file_type = os.path.splitext(file)
        # print(f'file_type = {file_type}\n')
        if file_type == '.pt':
            reorder=torch.load(os.path.join(path,file))
        elif file_type == '.npy':
            hash_name = file.split('_')[-1].split('.')[0]
            if hash_name not in hash_names:
                hash_names.add(hash_name)
                name_list.append(hash_name)
                i = i + 1
            layer_id = int(file.split('attnid')[-1].split('_')[0])
            layer = max(layer,layer_id)
    layer += 1
    layer_num = layer

    for prefix_id,hash_name in enumerate(name_list):
        keys = []
        values = []
        for i in range(layer):
            file = f'{name}_attnid{i}_{hash_name}.npy'
            file_path = os.path.join(folder_path, file)
            mmap_kv_tensor = np.load(file_path, mmap_mode='r')
            prefix_k, prefix_v = torch.from_numpy(mmap_kv_tensor[0,: , :, :]), torch.from_numpy(mmap_kv_tensor[1,:, :, :])
            keys.append(prefix_k)
            values.append(prefix_v)
        key = torch.stack(keys)
        # print(key.shape)
        value = torch.stack(values)
        prefix_table[hash_name] = prefix_id
        max_prefix_id += 1
        if args.reorder:
            # print(f"reorder = {reorder}\n")
            # print(f"hash name = {hash_name}\n")
            pcache.insert(prefix_id=prefix_id,key=key,value=value,reorder=reorder[hash_name])
        else:
            pcache.insert(prefix_id=prefix_id,key=key,value=value)

def init_prefetch_ratio(folder_path):
    RATIO_COLLECTION = []
    current_sample_ratio = None
    prefetch_ratio_file = os.path.join(folder_path, 'prefetch_ratios.txt')
    with open(prefetch_ratio_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('# Sample'):
                rematch = re.search(r'# Sample (\d+)', line)
                sample_num = int(rematch.group(1))
                if current_sample_ratio is not None:
                    RATIO_COLLECTION.append(current_sample_ratio)
                current_sample_ratio = {
                    'sample': sample_num,
                    'layers': {}
                }
            
            elif line.startswith('Layer'):
                rematch = re.search(r'Layer (\d+): (\d+)', line)
                layer_num = int(rematch.group(1))
                prefetch_ratio = int(rematch.group(2))
                current_sample_ratio['layers'][layer_num] = prefetch_ratio
        
        if current_sample_ratio is not None:
            RATIO_COLLECTION.append(current_sample_ratio)
    return RATIO_COLLECTION

def cal_recall_rate(sele_tokenids, prefix_len):
    # 对 sele_tokenids 每个头计算召回率，对象是对每个头进行随机取的结果
    # print(sele_tokenids)
    sum = 0
    for head in range(sele_tokenids.shape[0]):
        sele_ratio = 25 # 可调节
        sele_ids = sele_tokenids[head].cpu().numpy().tolist()
        # print(f'sele_ids_len = {len(sele_ids)}\n')
        # print(f'sele_ids = {sele_ids}\n')
        tokenids = list(range(prefix_len))
        # print(f"tokenids = {tokenids}\n")
        np.random.shuffle(tokenids)
        random_sele_ids = tokenids[:math.ceil(sele_ratio / 100 * prefix_len)]
        # print(f'random_sele_ids_len = {len(random_sele_ids)}\n')
        # print(f'random_sele_ids = {random_sele_ids}\n')
        hit = 0
        for tokenid in random_sele_ids:
            if tokenid in sele_ids:
                hit += 1
        recall_rate = hit / len(sele_ids) if len(sele_ids) > 0 else 0
        sum += recall_rate

    # with open('./time_logs/recall_rate.log', 'a') as f:
    #     f.write(f'Recall rate: {sum/sele_tokenids.shape[0]}\n')

# @dataclasses.dataclass(frozen=True)
@dataclasses.dataclass
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    # heavy hitter pruning
    hh_ratio: float = 1
    hh_all: bool = False
    hh_long_seq: bool = False
    
    #
    ret_topk_indices: bool = False
    prefix_dump: bool = False
    prefix_aware_inf: bool = False
    sele_load: bool = False # load selected kv
    full_load: bool = False # load full kv
    sele_inf: bool = False # inference with selective kv

    suffix_len: int = 100
    sele_percent: tuple = (100)

    logits: bool = False
    
    accum_percent: tuple = (50)
    sele_head: tuple = (0, 1, 2)
    sim_thred: float = 0.5
    sele_load_by_percent: bool = False
    full_load_sele_inf_by_accum: bool = False
    full_only_key_load: bool = False
    generate_mapping_list: bool = False

    prefetch: bool = False
    suffix_comp: bool = False

    prefetch_ratio: int = 10  # Percentage of tokens to prefetch

    no_prefetch: bool = False # 默认开启预取

    solid_ratio_prefetch: bool = False # 默认非固定比例预取

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy, check_hidden = False):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None
        self.check_hidden = check_hidden
    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate, hh_long_seq=self.policy.hh_long_seq)
        # if self.check_hidden or 1:
            # print('embed+pos:',h.data)
        hidden.val = h


class   OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        
        self.task = None
        self.logits = policy.logits

        self.logits_val = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, oplm = None):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        # print(h.data[0, 0, :5])
        if self.logits:
            h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
                self.task.do_sample, self.task.temperature, record_logits = self.logits, oplm = self)
            oplm.logits_val = self.logits_val
        # print(h.data)
        else:
            h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
                self.task.do_sample, self.task.temperature)
        hidden.val = h

    def prefetch(self):
        pass


class SelfAttention:
    def __init__(self, config, env, policy, layer_id, check_hidden=False, parent_model = None, mlp = None):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        
        # 访问到模型本身，这样可以使用模型创建的cuda流，避免多次创建流
        self.parent_model = parent_model
        self.mlp = mlp
        self.already_assigned_hidden_val = False

        # 层间预取缓存传递，上一层会把自己预取的kv放到这里
        # self.prefetched_cache = None
        # self.prefetched_tokenids = None

        self.task = None
        self.check_hidden = check_hidden
        # 收集关于 hh 的信息
        self.topk_indices_lst = []
        self.prefix_k_v = []
        self.prefix_hash = None
        self.prefetched_head = None
        # 收集prefix中重要的tokenid
        self.imp_dec_mapping_per_layer = []

    # 新增方法，管理当前层预取缓存
    def set_prefetched_cache(self, tokenids, k_tensor, v_tensor):
        self.prefetched_cache = (k_tensor, v_tensor)
        self.prefetched_tokenids = tokenids

    def get_prefetched_cache(self):
        cache = self.prefetched_cache
        tokenids = self.prefetched_tokenids
        self.prefetched_cache = None
        self.prefetched_tokenids = None
        return cache, tokenids

    def set_task(self, task):
        self.task = task
        self.hh_k = int(task.prompt_len * self.policy.hh_ratio)

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy, self.hh_k, self.policy.hh_all)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home, acc = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        pos = min(self.hh_k * 2 - 1, self.task.prompt_len) + 1
        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            if self.policy.hh_all:
                indices = (slice(0, pos),
                           slice(0, k_home.shape[1]))
            else:
                indices = (slice(0, pos + i),
                           slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                    acc.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf, acc_buf = dst.next_attention_compute_workspace()
            if self.policy.hh_all:
                indices = (slice(0, pos),
                           slice(0, k_home.shape[1]))
            else:
                indices = (slice(0, pos + i),
                           slice(0, k_home.shape[1]))

            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                general_copy(acc_buf, indices, acc, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False), (acc_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, pos + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home, acc = cache_home.val
        k_new, v_new, acc_new, kick_ind = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))

        else:  # decoding
            if self.policy.hh_all:
                oldest = ((i - 1) % (self.hh_k - 1)) - (self.hh_k - 1)
                cache_replace(k_home, kick_ind, k_new, self.hh_k, oldest)
                cache_replace(v_home, kick_ind, v_new, self.hh_k, oldest)
                acc_replace(acc, kick_ind, acc_new, self.hh_k, oldest)
                return

            if self.hh_k is None:
                pos = self.task.prompt_len + i
            else:
                pos = min(self.hh_k * 2 - 1, self.task.prompt_len) + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)
        if self.policy.hh_all:
            general_copy(acc, indices, acc_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype
    
    def prefetch(self):
        global pcache
        prefix_hash = str(hash(tuple(self.task.inputs[0][:-self.suffix_len])))
        if prefix_hash not in prefix_table:
            insert_pcache(pcache=pcache,hash_name=prefix_hash)
        self.prefix_k,self.prefix_v = pcache.get(prefix_id=prefix_table[prefix_hash],layer=self.layer_id)

    # 结合了mlp的forward
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, read_buf2 = None):

        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        output_file = get_timelog_filename(args)

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            # print(f'mask = {mask.shape} {mask.data}') # shape (b, s), True if not padding
            # print(f'input: {self.task.inputs[0]}')
            global pcache,dt_prefix,dt_all,dt_load,dt_sele,dt_load_head,dt_suffix,dt_key,dt_value, orcale_load_time, orcale_ttft
            global dt_async_prefetch_submit, async_prefetch_jobs, async_prefetch_hit_tokens, async_prefetch_miss_tokens
            if self.policy.prefix_aware_inf:
                prefix_hash = str(hash(tuple(self.task.inputs[0][:-self.policy.suffix_len].tolist())))
                datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
                model_name = os.path.basename(args.model)
                if args.padding_mul == 1:
                    prefix_kv_path = os.path.join(f'./cache/prefix/{model_name}/{datasetname}', args.model.replace('/', '_') + f'_attnid{self.layer_id}_{prefix_hash}.npy')
                    if self.layer_id < self.config.num_hidden_layers - 1:
                        prefetch_prefix_kv_path = os.path.join(f'./cache/prefix/{model_name}/{datasetname}', args.model.replace('/', '_') + f'_attnid{self.layer_id+1}_{prefix_hash}.npy')
                else:
                    prefix_kv_path = os.path.join(f'./cache/prefix/{model_name}/{datasetname}_{args.padding_mul}', args.model.replace('/', '_') + f'_attnid{self.layer_id}_{prefix_hash}.npy')
                    if self.layer_id < self.config.num_hidden_layers - 1:
                        prefetch_prefix_kv_path = os.path.join(f'./cache/prefix/{model_name}/{datasetname}_{args.padding_mul}', args.model.replace('/', '_') + f'_attnid{self.layer_id+1}_{prefix_hash}.npy')
                    # print(prefix_kv_path, os.path.exists(prefix_kv_path))
                # 完全加载prefix kv (不使用选择性加载 或 在选择性加载的模式下，layer_id >= 10)
                if self.policy.full_load:
                    # 如果 prefix的kv 已经有缓存过，直接加载
                    # if os.path.exists(prefix_kv_path):
                    if True:
                        # print(f'-> Found prefix KV path: {prefix_kv_path}')
                        # k_v_num_tokens = len(self.task.inputs[0][:-self.policy.suffix_len])

                        torch.cuda.synchronize()
                        st = time.monotonic()
                        if not args.no_cache:
                            if prefix_hash not in prefix_table:
                                insert_pcache(pcache=pcache,hash_name=prefix_hash)
                            prefix_k,prefix_v = pcache.get(prefix_id=prefix_table[prefix_hash],layer=self.layer_id)
                        else:
                            mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                            prefix_k, prefix_v = torch.from_numpy(mmap_kv_tensor[0, :, :, :]).cuda(), torch.from_numpy(mmap_kv_tensor[1, :, :, :]).cuda()
                        # print(f'shape of prefix_k = {prefix_k.shape}, shape of prefix_v = {prefix_v.shape}') # (prefix_s, num_head, head_dim)
                        torch.cuda.synchronize()
                        per_layer_dt_load_head = time.monotonic() - st
                        maybe_profile_kv_io("as_full_kv_load", self.layer_id, per_layer_dt_load_head, prefix_k, prefix_v)
                        dt_load += per_layer_dt_load_head

                        if self.policy.sele_inf:
                            # 1008666
                            # 每层选择指定百分比的 kv 进行推理 (加载的时候依然全加载，加载后计算 attention 后推理时选择固定百分比)
                            # 重计算完成输入对应的q，重计算 suffix-k/v；选择部分 prefix-k/v，拼接后得到部分的 k/v
                            # idx = math.ceil((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                            if self.policy.generate_mapping_list:
                                # 按照 full_load_sele_inf 来，区别在于每个请求返回prefix中重要的 tokenid 后记录下来
                                idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                                cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                                h, new_k_cache, new_v_cache, acc, reverse_indices = self.compute.mha_with_sele_percent_prefixkv(h, mask, w_q, b_q,
                                        w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len, cur_sele_percent, generate_mapping_list = self.policy.generate_mapping_list)
                                self.imp_dec_mapping_per_layer = reverse_indices.cpu().tolist()
                                # print("mapping list:", self.imp_dec_mapping_per_layer)
                            else:    # full_load_sele_inf no use
                                if not self.policy.full_load_sele_inf_by_accum:
                                    # 每层选择指定百分比的 kv 进行推理 (加载的时候依然全加载，加载后计算 attention 后推理时选择固定百分比)
                                    # 重计算完成输入对应的q，重计算 suffix-k/v；选择部分 prefix-k/v，拼接后得到部分的 k/v
                                    
                                    idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                                    cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                                    # print(f'cur_sele_percent = {cur_sele_percent}')

                                    if self.policy.suffix_comp:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache,acc = self.compute.mha_with_prefixkv_suffix(h, mask, w_q, b_q,
                                                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                        torch.cuda.synchronize()
                                        dt_suffix  += time.monotonic() - st
                                    else:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache, acc = self.compute.mha_with_sele_percent_prefixkv(h, mask, w_q, b_q,
                                                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len, cur_sele_percent)
                                        torch.cuda.synchronize()
                                        dt_all  = time.monotonic() - st

                                    
                                else: # full_load_sele_inf_by_accum
                                    # 每层选择变长数量的 kv 进行推理直到累积的atten达到指定百分比 (加载的时候依然全加载，加载后计算 attention 后推理时选择变长数量)
                                    # 重计算完成输入对应的q，重计算 suffix-k/v；选择部分 prefix-k/v，拼接后得到部分的 k/v
                                    idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.accum_percent))
                                    cur_accum_percent = self.policy.accum_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.accum_percent[idx]
                                    # print(f'[full_load_sele_inf] target accum_percent for current attn layer[{self.layer_id}] = {cur_accum_percent}%')

                                    if self.policy.suffix_comp:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache,acc = self.compute.mha_with_prefixkv_suffix(h, mask, w_q, b_q,
                                                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                        torch.cuda.synchronize()
                                        dt_suffix  += time.monotonic() - st
                                    else:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache, acc = self.compute.mha_with_var_percent_prefixkv(h, mask, w_q, b_q,
                                                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len, cur_accum_percent)
                                        torch.cuda.synchronize()
                                        dt_all  = time.monotonic() - st

                        else: # full_load_full_inf
                            # 重计算完成输入对应的q，重计算 suffix-k/v；拼接后得到全部的 k/v ==> 与不使用 prefixkv 推理结果相同
                            # h, new_k_cache, new_v_cache,acc = self.compute.mha_with_prefixkv_suffix(h, mask, w_q, b_q,
                            #         w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                            #         self.policy.compress_cache, self.policy.comp_cache_config,
                            #         self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                            if self.policy.suffix_comp:
                                torch.cuda.synchronize()
                                st = time.monotonic()
                                h, new_k_cache, new_v_cache,acc = self.compute.mha_with_prefixkv_suffix(h, mask, w_q, b_q,
                                        w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                torch.cuda.synchronize()
                                
                            else: # 不用
                                torch.cuda.synchronize()
                                st = time.monotonic()
                                h, new_k_cache, new_v_cache, acc = self.compute.mha_with_prefixkv(h, mask, w_q, b_q,
                                        w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                torch.cuda.synchronize()
                                dt_all  = time.monotonic() - st
                            
                            # 把后面的MLP层放到这里做
                            hidden.val = h
                            self.already_assigned_hidden_val = True
                            self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)
                            torch.cuda.synchronize()
                            per_layer_computation = time.monotonic() - st
                            dt_suffix  += per_layer_computation
                            all_ttft.append(per_layer_dt_load_head + per_layer_computation)
                            # with open(output_file, 'a') as f:
                            #     f.write(f'Layer {self.layer_id} load head time = {per_layer_dt_load_head:.8f}s\n')
                            #     f.write(f'Layer {self.layer_id} sele token time = {0:.8f}s\n')
                            #     f.write(f'Layer {self.layer_id} load time = {0:.8f}s\n')
                            #     f.write(f'Layer {self.layer_id} prefetch time = {0:.8f}s\n')
                            #     f.write(f'Layer {self.layer_id} compute time = {per_layer_compute_time:.8f}s\n')
                            #     f.write(f'Layer {self.layer_id} parallel time = {per_layer_compute_time:.8f}s\n')
                            #     f.write(f'===========================================================================\n')
                
                elif self.policy.full_only_key_load: # h2o
                    # full_load_key sele_load_value
                    # 把 key 完全加载，然后找出重要的 v，加载部分的v 填充后进行推理
                    if args.no_cache:
                        if not os.path.exists(prefix_kv_path):
                            print(f'ERROR: 找不到 Prefix 的 KV文件 {prefix_kv_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                            sys.exit(-1)
                    
                    torch.cuda.synchronize()
                    st_key = time.monotonic()
                    if not args.no_cache:
                        if prefix_hash not in prefix_table:
                            insert_pcache(pcache=pcache,hash_name=prefix_hash)
                        complete_prefix_k = pcache.get_key(prefix_id=prefix_table[prefix_hash],pos_id=None, layer=self.layer_id) # ［seq，head num，head dim]
                    else:
                        mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                        complete_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, :, :]).cuda()
                    torch.cuda.synchronize()
                    # dt_key += time.monotonic() - st_key
                    per_layer_dt_key = time.monotonic() - st_key
                    maybe_profile_kv_io("h2o_full_key_load", self.layer_id, per_layer_dt_key, complete_prefix_k)
                    dt_key += per_layer_dt_key
                    # print(f'shape of only prefix_k = {complete_prefix_k.shape}') # (prefix_s, num_head, head_dim)
                    
                    torch.cuda.synchronize()
                    st_sele = time.monotonic()
                    idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                    cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                    # print('cur_sele_percent', cur_sele_percent)
                    # shape: (sele_num, ) in cuda
                    # sele_tokenids, del_tokenids = self.compute.sele_tokenid_with_all_keys(h, mask, w_q, b_q,
                    #             w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                    #             complete_prefix_k, cur_sele_percent)
                    sele_tokenids = self.compute.sele_tokenid_with_all_keys(h, mask, w_q, b_q,
                                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                complete_prefix_k, cur_sele_percent)
                    

                    # print("sele_tokenids = {sele_tokenids}")
                    # 计算召回率
                    # cal_recall_rate(sele_tokenids, complete_prefix_k.shape[0])

                    torch.cuda.synchronize()
                    # dt_sele += time.monotonic() - st_sele
                    per_layer_dt_sele_head = time.monotonic() - st_sele
                    dt_sele += per_layer_dt_sele_head                 


                    # 根据 sele_tokenids 加载 部分 value
                    torch.cuda.synchronize()
                    st_value = time.monotonic()
                    if not args.no_cache:
                        remain_prefix_v = pcache.get_value(prefix_id=prefix_table[prefix_hash],pos_id=sele_tokenids[0],layer=self.layer_id)
                    else:
                        remain_prefix_v = torch.from_numpy(mmap_kv_tensor[1, sele_tokenids, :, :]).cuda()
                    torch.cuda.synchronize()
                    # dt_value += time.monotonic() - st_value
                    per_layer_dt_load_value = time.monotonic() - st_value
                    maybe_profile_kv_io("h2o_selected_value_load", self.layer_id, per_layer_dt_load_value, remain_prefix_v)
                    dt_value += per_layer_dt_load_value
                    
                    # 把 remain_prefix_v 和 complete_prefix_k 一起进行推理
                    k_v_num_tokens = len(self.task.inputs[0][:-self.policy.suffix_len])
                    filled_prefix_v = torch.zeros(size=(k_v_num_tokens, self.config.n_head, self.config.hidden_size // self.config.n_head), dtype=remain_prefix_v.dtype, device="cuda:0")
                    filled_prefix_v[sele_tokenids] = remain_prefix_v
                    
                    
                    if self.policy.suffix_comp:
                        torch.cuda.synchronize()
                        st = time.monotonic()
                        h, new_k_cache, new_v_cache, acc = self.compute.mha_with_prefixkv_suffix(h, mask, w_q, b_q,
                                    w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                    self.policy.compress_cache, self.policy.comp_cache_config,
                                    self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, complete_prefix_k, filled_prefix_v, self.policy.suffix_len)
                        torch.cuda.synchronize()
                        
                        # per_layer_dt_compute_time = time.monotonic() - st
                    else:
                        torch.cuda.synchronize()
                        st = time.monotonic()
                        h, new_k_cache, new_v_cache, acc = self.compute.mha_with_filled_selected_prefixkv(h, mask, w_q, b_q,
                                    w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                    self.policy.compress_cache, self.policy.comp_cache_config,
                                    self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, complete_prefix_k, filled_prefix_v, self.policy.suffix_len, del_tokenids = del_tokenids)
                        torch.cuda.synchronize()
                        dt_all += time.monotonic() - st
                        # per_layer_dt_compute_time = time.monotonic() - st

                    hidden.val = h
                    self.already_assigned_hidden_val = True
                    self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)
                    torch.cuda.synchronize()
                    per_layer_computation = time.monotonic() - st
                    dt_suffix  += per_layer_computation
                    all_ttft.append(per_layer_dt_key + per_layer_dt_sele_head + per_layer_dt_load_value + per_layer_computation)

                    
                else:
                    # the paper's method
                    # sele_load prefix kv
                    if args.no_cache:
                        if not os.path.exists(prefix_kv_path):
                            print(f'ERROR: 找不到 Prefix 的 KV文件 {prefix_kv_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                            sys.exit(-1)
                    if True:
                        # 先加载 sele_head 中的全部的 key 
                        # -> atten_weights 
                        # -> 计算重要的 token id with accum_percent (或者指定每层选择的比例)
                        # -> 计算相似度 sim, 与 sim_thred 比较
                        # get_head_start_event = torch.cuda.Event()
                        per_layer_prefetch_wait = 0.0
                        if self.layer_id > 0 and not self.policy.no_prefetch:
                            per_layer_prefetch_wait = self.parent_model.resolve_pending_prefetch(self.layer_id)
                        torch.cuda.synchronize()
                        st = time.monotonic()
                        k_v_num_tokens = len(self.task.inputs[0][:-self.policy.suffix_len])
                        if not args.no_cache:
                            if self.prefetched_head is None:
                                if prefix_hash not in prefix_table:
                                    insert_pcache(pcache=pcache,hash_name=prefix_hash)
                                fullhead_prefix_k = pcache.get_head(prefix_id=prefix_table[prefix_hash],layer=self.layer_id)
                            else:
                                fullhead_prefix_k = self.prefetched_head
                                self.prefetched_head = None
                            # print(fullhead_prefix_k.shape,prefix_table[prefix_hash],prefix_hash)
                        else:
                            mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                            # fullhead_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, self.policy.sele_head, :])
                            # print(f'fullhead_prefix_k.shape1 = {fullhead_prefix_k.shape}')
                            fullhead_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, self.policy.sele_head, :]).permute(1, 0, 2).cuda()
                            # print(f'fullhead_prefix_k.shape = {fullhead_prefix_k.shape}')
                        torch.cuda.synchronize()
                        per_layer_dt_load_head = time.monotonic() - st
                        maybe_profile_kv_io("probe_head_key_load", self.layer_id, per_layer_dt_load_head, fullhead_prefix_k)
                        dt_load_head += per_layer_dt_load_head
                        # dt_load_head += time.monotonic() - st
                        # print(f'load head time for layer {self.layer_id} = {per_layer_dt_load_head:.4f}s')

                        # mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                        # fullhead_prefix_k = torch.from_numpy(mmap_kv_tensor[:k_v_num_tokens, self.policy.sele_head, :]).cuda()
                        # print(f'fullhead_prefix_k.shape = {fullhead_prefix_k.shape}')
                        
                        st = time.monotonic()

                        # 每层选累积百分比的 token
                        if not self.policy.sele_load_by_percent:
                            idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.accum_percent))
                            cur_accum_percent = self.policy.accum_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.accum_percent[idx]
                            # print(f'target accum_percent for current attn layer[{self.layer_id}] = {cur_accum_percent}%')
                        
                            # shape: (sele_num, ) in cuda
                            sele_tokenids = self.compute.sele_tokenid(h, mask, w_q, b_q,
                                        w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                        self.policy.sele_head, fullhead_prefix_k, cur_accum_percent, self.policy.sim_thred)
                        
                        # 每层选指定百分比的 token
                        else:
                            idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                            cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                            # print(f'target accum_percent for current attn layer[{self.layer_id}] = {cur_sele_percent}%')
                            # print('cur_sele_percent', cur_sele_percent)
                            # shape: (sele_num, ) in cuda
                            sele_tokenids, del_tokenids = self.compute.sele_tokenid(h, mask, w_q, b_q,
                                        w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                        self.policy.sele_head, fullhead_prefix_k, -1, self.policy.sim_thred, cur_sele_percent)

                            # with open('./token_logs/important_token_30b_25_wiki.log', 'a') as f:
                            #     f.write(f'layer id {self.layer_id} sele_tokenids: {sele_tokenids.cpu().numpy()}\n')
                            # 这里是本层需要预取的 tokenids ，而self.parent_model.last_layer_prefetched_tokenids 是上一层预取的 tokenids
                            # self.parent_model.last_layer_prefetched_tokenids 在本层最后被更新为 prefetch_tokenids

                            # 预取逻辑 prefetch logic
                            if not self.policy.no_prefetch and self.layer_id < self.config.num_hidden_layers - 1:
                                if self.policy.solid_ratio_prefetch:
                                    # 固定比例预取
                                    prefetch_token_num = int(len(sele_tokenids) * self.policy.prefetch_ratio / cur_sele_percent)
                                    prefetch_tokenids = sele_tokenids[:prefetch_token_num]
                                    
                                else:
                                    # 使用策略预取，将全部 token_id 塞入 prefetch，由策略选择
                                    prefetch_tokenids = sele_tokenids.clone()

 
                        torch.cuda.synchronize()
                        per_layer_dt_sele_head = time.monotonic() - st
                        dt_sele += per_layer_dt_sele_head
                        
                        # dt_sele += time.monotonic() - st
                        # print(f'sele token time for layer {self.layer_id} = {per_layer_dt_sele_head:.4f}s')
                        # st = time.monotonic()
                        # 预取逻辑
                        # 这里已经选出了要拿上来的sele_tokenids，现在把已经在上一层预取阶段拿上来的tokenids去掉
                        if (not self.layer_id == 0 and not self.policy.no_prefetch and
                                self.parent_model.last_layer_prefetched_tokenids is not None):
                            # 第0层没有之前的预取结果，只有后面层需要做
                            # 将选取出来的 sele_tokenids 和 self.parent_model.last_layer_prefetched_tokenids 做差集，找到还没有拿到的 tokenids
                            prefetched_tokenids_for_compare = self.parent_model.last_layer_prefetched_tokenids.to(sele_tokenids.device)
                            prefetch_mask = ~torch.isin(sele_tokenids, prefetched_tokenids_for_compare)
                            missing_tokenids = sele_tokenids[prefetch_mask]
                            async_prefetch_hit_tokens += int(sele_tokenids.numel() - missing_tokenids.numel())
                            async_prefetch_miss_tokens += int(missing_tokenids.numel())
                            # print(f"Layer {self.layer_id}: 总需要{len(sele_tokenids)}个token,"
                            #       f"可复用预取{len(sele_tokenids)-len(missing_tokenids)}个，"
                            #       f"需额外加载{len(missing_tokenids)}个")
                            # print(f"missing_tokenids: {missing_tokenids}")
                        else:
                            missing_tokenids = sele_tokenids

                        
                        st = time.monotonic()
                        # 原来的 load 逻辑
                        if not args.no_cache:
                            # 这里本来的 pos_id 是赋值为 sele_tokenids，现在改为 missing_tokenids，去掉预取部分
                            remain_prefix_k, remain_prefix_v = pcache.get(prefix_id=prefix_table[prefix_hash],pos_id=missing_tokenids,layer=self.layer_id)
                        else:
                            remain_prefix_k, remain_prefix_v = torch.from_numpy(mmap_kv_tensor[0, sele_tokenids, :, :]).cuda(), torch.from_numpy(mmap_kv_tensor[1, sele_tokenids, :, :]).cuda()
                        torch.cuda.synchronize()
                        per_layer_dt_load = time.monotonic() - st
                        maybe_profile_kv_io("selected_kv_sync_load", self.layer_id, per_layer_dt_load, remain_prefix_k, remain_prefix_v)
                        dt_load += per_layer_dt_load
                        # dt_load += time.monotonic() - st

                        
                        c_stream = self.parent_model.c_stream
                        p_stream = self.parent_model.p_stream

                        if args.fill_keys_zero:
                            filled_prefix_k, filled_prefix_v = torch.zeros(size=(k_v_num_tokens, self.config.n_head, self.config.hidden_size // self.config.n_head), dtype=remain_prefix_k.dtype, device="cuda:0"), torch.zeros(size=(k_v_num_tokens, self.config.n_head, self.config.hidden_size // self.config.n_head), dtype=remain_prefix_v.dtype, device="cuda:0")
                        else:
                            filled_prefix_v = torch.zeros(size=(k_v_num_tokens, self.config.n_head, self.config.hidden_size // self.config.n_head), dtype=remain_prefix_v.dtype, device="cuda:0")
                            # 计算三维上的平均值，得到一个 shape 为 (prefixlen, 128) 的 tensor
                            prefix_len, sele_nheads, head_dim = fullhead_prefix_k.shape
                            # 计算第三维度上的平均值
                            mean_values = fullhead_prefix_k.mean(dim=1, keepdim=True)
                            # 将平均值扩展
                            mean_values_expanded = mean_values.expand(prefix_len, self.config.n_head-sele_nheads, head_dim)
                            # 拼接原始张量和扩展后的平均值张量，形成 (prefix_len, self.config.n_head, head_dim)
                            filled_prefix_k = torch.cat((fullhead_prefix_k, mean_values_expanded), dim=1)
                        # filled_prefix_k[sele_tokenids] = remain_prefix_k
                        # filled_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, :, :]).cuda()
                        # 预取逻辑
                        # 两步走，先填 上一层预取的部分
                        if (not self.layer_id == 0 and not self.policy.no_prefetch and
                                self.parent_model.last_layer_prefetched_tokenids is not None):
                            filled_prefix_k[self.parent_model.last_layer_prefetched_tokenids] = self.parent_model.last_layer_prefetched_prefix_k
                            filled_prefix_v[self.parent_model.last_layer_prefetched_tokenids] = self.parent_model.last_layer_prefetched_prefix_v
                        # 再填 剩余的部分
                        filled_prefix_k[missing_tokenids] = remain_prefix_k
                        #filled_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, :, :]).cuda()
                        filled_prefix_v[missing_tokenids] = remain_prefix_v
                        
                        c_stream_start_event = torch.cuda.Event(enable_timing=True)
                        c_stream_end_event = torch.cuda.Event(enable_timing=True)
                        p_stream_start_event = torch.cuda.Event(enable_timing=True)
                        p_stream_end_event = torch.cuda.Event(enable_timing=True)

                        torch.cuda.synchronize()
                        st = time.monotonic()

                            # print(f'filled_prefix_k.shape={filled_prefix_k.shape}, filled_prefix_v.shape={filled_prefix_v.shape}') # shape: (s, n_head, head_dim)
                        with torch.cuda.stream(c_stream):
                            # 去掉comp之前的 synchronize，让 prefetch 和 comp 同时进行
                            if self.policy.suffix_comp:
                                # torch.cuda.synchronize()
                                # c_stream_st = time.monotonic()
                                c_stream_start_event.record()
                                h, new_k_cache, new_v_cache, acc = self.compute.mha_with_prefixkv_suffix(h, mask, w_q, b_q,
                                        w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, filled_prefix_k, filled_prefix_v, self.policy.suffix_len)
                                # torch.cuda.synchronize()
                                # per_layer_dt_suffix = time.monotonic() - st
                                # dt_suffix  += per_layer_dt_suffix
                                # dt_suffix  += time.monotonic() - st
                                # print(f'suffix comp time for layer {self.layer_id} = {per_layer_dt_suffix:.4f}s')
                            else:
                                # torch.cuda.synchronize()
                                # st = time.monotonic()
                                h, new_k_cache, new_v_cache, acc = self.compute.mha_with_filled_selected_prefixkv(h, mask, w_q, b_q,
                                            w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                            self.policy.compress_cache, self.policy.comp_cache_config,
                                            self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, filled_prefix_k, filled_prefix_v, self.policy.suffix_len, del_tokenids = del_tokenids)
                                # torch.cuda.synchronize()
                                
                            
                            # 把后面的MLP层放到这里做
                            hidden.val = h
                            self.already_assigned_hidden_val = True
                            self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)
                            #self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)
                            # c_stream.synchronize()
                            c_stream_end_event.record()
                            # compute_time = time.monotonic() - c_stream_st
                        
                        # 先提交计算任务，再把下一层 KV 预取提交到后台线程。
                        if (not self.policy.no_prefetch and
                                self.layer_id < self.config.num_hidden_layers - 1 and
                                len(prefetch_tokenids) > 0 and not args.no_cache):
                            submit_st = time.monotonic()
                            self.parent_model.pending_prefetch_handle = pcache.prefetch_async(
                                prefix_id=prefix_table[prefix_hash],
                                pos_id=prefetch_tokenids,
                                layer=self.layer_id + 1,
                                k_v_num_tokens=k_v_num_tokens,
                                time_budget=self.parent_model.prefetch_time_budget,
                            )
                            dt_async_prefetch_submit += time.monotonic() - submit_st
                            async_prefetch_jobs += 1
                            print(f"submitted async prefetch for layer {self.layer_id + 1}")
                        else:
                            self.parent_model.pending_prefetch_handle = None
                                # if hasattr(self.parent_model, 'layers') and self.layer_id + 1 < len(self.parent_model.layers):
                                #     next_layer = self.parent_model.layers[self.layer_id + 1]
                                #     if hasattr(next_layer, 'attention'):
                                #         next_layer.attention.set_prefetched_cache(prefetch_tokenids.to("cuda:0").clone(), 
                                #                                                   prefetch_k.to("cuda:0"), 
                                #                                                   prefetch_v.to("cuda:0"))
                                #         print(f"Layer {self.layer_id}: 为下一层预取了{len(prefetch_tokenids)}个token")

                        c_stream.synchronize()
                        ed = time.monotonic()
                        # print(f"self.parent_model.last_layer_prefetched_prefix_k.shape = {self.parent_model.last_layer_prefetched_prefix_k.shape}")
                        # print(f"self.parent_model.last_layer_prefetched_prefix_v.shape = {self.parent_model.last_layer_prefetched_prefix_v.shape}")
                        parallel_time = ed - st
                        if not self.policy.no_prefetch:
                            prefetch_time = 0.0
                        compute_time = c_stream_start_event.elapsed_time(c_stream_end_event) / 1000
                        dt_suffix += parallel_time
                        
                        all_ttft.append(per_layer_prefetch_wait + per_layer_dt_load_head + per_layer_dt_sele_head + per_layer_dt_load + parallel_time)
                        if self.policy.no_prefetch:
                            orcale_ttft.append(per_layer_dt_sele_head + ((per_layer_dt_load + per_layer_dt_load_head - parallel_time) if parallel_time < (per_layer_dt_load + per_layer_dt_load_head) else 0))
                            orcale_load_time += (per_layer_dt_load + per_layer_dt_load_head - parallel_time) if parallel_time < (per_layer_dt_load + per_layer_dt_load_head) else 0
                        
                        # 预取逻辑
                        if self.layer_id < self.config.num_hidden_layers - 1 and not self.policy.no_prefetch:
                            self.parent_model.prefetch_time_budget = compute_time + per_layer_dt_sele_head

                        
                        

                    else:
                        print(f'ERROR: 找不到 Prefix 的 KV文件 {prefix_kv_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                        sys.exit(-1)
                        
                    
            else:
                if self.policy.ret_topk_indices:
                    # topk_indices 表示一个层中，每个head中最重要的k个token的id shape:(b*n_head, k)
                    h, new_k_cache, new_v_cache, acc, topk_indices = self.compute.mha(h, mask, w_q, b_q,
                                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                self.policy.compress_cache, self.policy.comp_cache_config,
                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices)
                    self.topk_indices_lst = topk_indices.cpu().tolist()
                else:

                    
                    # torch.cuda.synchronize()
                    # st = time.monotonic()
                    # h2, new_k_cache, new_v_cache = self.compute.mha_prefixkv(h, mask, w_q, b_q,
                    #         w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                    #         self.policy.compress_cache, self.policy.comp_cache_config,
                    #         self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, self.policy.suffix_len)
                    # torch.cuda.synchronize()
                    # dt_prefix += time.monotonic() - st

                    print('recompute without prefix k,v for full prefix')
                    torch.cuda.synchronize()
                    st = time.monotonic()
                    
                    h, new_k_cache, new_v_cache, acc = self.compute.mha(h, mask, w_q, b_q,
                                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                                self.policy.compress_cache, self.policy.comp_cache_config,
                                self.hh_k, self.policy.hh_all)
                    
                    if args.sep_layer == False:
                        hidden.val = h
                        self.already_assigned_hidden_val = True
                        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)
                    
                    torch.cuda.synchronize()
                    dt_per_layer = time.monotonic() - st
                    dt_all += dt_per_layer

            
            if self.policy.prefix_dump:
                # 记录下 k,v 方便存入磁盘
                # 假设后 100 个为 query
                self.prefix_k_v = torch.stack((new_k_cache.data[:-self.policy.suffix_len, :, :].cpu(), new_v_cache.data[:-self.policy.suffix_len, :, :].cpu()), dim=0)
                # print(f'move prefix KV into CPU memory with KV shape = {self.prefix_k_v.shape}')
                self.prefix_hash = str(hash(tuple(self.task.inputs[0][:-self.policy.suffix_len].tolist())))
            cache_write_buf.store((new_k_cache, new_v_cache, acc, None))
        else:  # decoding
            # print("hidden", h.shape)
            # print(h.data)
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]), (acc, _) = cache_read_buf.pop()
            # if self.layer_id == 10:
            #     print(k_cache.shape)
            #     cnt = min(self.hh_k * 2, self.task.prompt_len)
            #     print(v_cache.data[:cnt + i][-5:, 0, :2])
            if self.policy.hh_all is not None:
                cnt = min(self.hh_k * 2 - 1, self.task.prompt_len + i)
                mask = mask.device.slice_attention_mask(mask, cnt + 1)
            h, new_k_cache, new_v_cache, acc, kick_ind = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, acc, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config,
                self.hh_k, self.policy.hh_all)
            # if self.layer_id == 10:
            #     print(h.data)
            cache_write_buf.store((new_k_cache, new_v_cache, acc, kick_ind))

        if not self.already_assigned_hidden_val:
            hidden.val = h

    def prefetch(self):
        global pcache,prefix_table
        prefix_hash = str(hash(tuple(self.task.inputs[0][:-self.policy.suffix_len].tolist())))
        if prefix_hash not in prefix_table:
            insert_pcache(pcache=pcache,hash_name=prefix_hash)
        self.prefetched_head = pcache.get_head(prefix_id=prefix_table[prefix_hash],layer=self.layer_id)

    def print_prefetch_tokenids_device(self, sele_tokenids, output_file, prefix_hash):
        # 输出本层重要的token在下一层位于的device位置
        if self.layer_id < self.config.num_hidden_layers - 1:
            next_layer = pcache.cache[prefix_table[prefix_hash]].layers[self.layer_id + 1]
            tokenids = sele_tokenids.cpu().numpy().tolist()
            # print(f"Layer {self.layer_id} 选中的tokenid在下一层的device位置:")
            gpu_count = 0
            cpu_count = 0
            disk_count = 0
            # print(f"device_map = {next_layer.device_map}")
            if args.disk_type == 'KV_Division':
                k_gpu_count = 0
                v_gpu_count = 0
                k_cpu_count = 0
                v_cpu_count = 0
                k_disk_count = 0
                v_disk_count = 0
                for tid in tokenids:
                    k_device = next_layer.device_map[tid // 64]
                    v_device = next_layer.device_map[tid // 64 + next_layer.chunk_num]
                    if k_device == "cuda:0":
                        k_gpu_count += 1
                    elif k_device == "cpu":
                        k_cpu_count += 1
                    elif k_device == "disk":
                        k_disk_count += 1
                    if v_device == "cuda:0":
                        v_gpu_count += 1
                    elif v_device == "cpu":
                        v_cpu_count += 1
                    elif v_device == "disk":
                        v_disk_count += 1
                    # with open(output_file, 'a') as f:
                    #     f.write(f"tid = {tid}, k_device = {k_device}, v_device = {v_device}\n")
                with open(output_file, 'a') as f:
                    if len(tokenids) == 0:
                        f.write(f"Layer {self.layer_id} 选中的tokenid在下一层的device位置: 本层未选择任何token\n")
                        return
                    else:
                        f.write(f"Layer {self.layer_id} 选中的tokenid在下一层的device位置:\n")
                        f.write(f"GPU K: {k_gpu_count}, CPU K: {k_cpu_count}, Disk K: {k_disk_count}\n")
                        f.write(f"GPU K ratio: {k_gpu_count / len(tokenids):.2%}, CPU K ratio: {k_cpu_count / len(tokenids):.2%}, Disk K ratio: {k_disk_count / len(tokenids):.2%}\n")
                        f.write(f"GPU V: {v_gpu_count}, CPU V: {v_cpu_count}, Disk V: {v_disk_count}\n")
                        f.write(f"GPU V ratio: {v_gpu_count / len(tokenids):.2%}, CPU V ratio: {v_cpu_count / len(tokenids):.2%}, Disk V ratio: {v_disk_count / len(tokenids):.2%}\n")

            else:
                gpu_count = 0
                cpu_count = 0
                disk_count = 0
                for tid in tokenids:
                    device = next_layer.device_map[tid // 64]
                    if device == "cuda:0":
                        gpu_count += 1
                    elif device == "cpu":
                        cpu_count += 1
                    elif device == "disk":
                        disk_count += 1
                with open(output_file, 'a') as f:
                    if len(tokenids) == 0:
                        f.write(f"Layer {self.layer_id} 选中的tokenid在下一层的device位置: 本层未选择任何token\n")
                        return
                    else:
                        f.write(f"Layer {self.layer_id} 选中的tokenid在下一层的device位置:\n")
                        f.write(f"GPU: {gpu_count}, CPU: {cpu_count}, Disk: {disk_count}\n")
                        f.write(f"GPU ratio: {gpu_count / len(tokenids):.2%}, CPU ratio: {cpu_count / len(tokenids):.2%}, Disk ratio: {disk_count / len(tokenids):.2%}\n")
                # print(f"GPU: {gpu_count}, CPU: {cpu_count}, Disk: {disk_count}")
                # print(f"GPU ratio: {gpu_count / len(tokenids):.2%}, CPU ratio: {cpu_count / len(tokenids):.2%}, Disk ratio: {disk_count / len(tokenids):.2%}")

class MLP:
    def __init__(self, config, env, policy, layer_id, parent_model = None):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        # we calculate weight at GPU
        self.compute = self.env.gpu
        # if compress weight, we load weight at compressed device, if not, we load weight at GPU
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        # 添加parent model以使用流
        self.parent_model = parent_model
        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k, use_stream = None):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if use_stream is not None:
            with torch.cuda.stream(use_stream):
                h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        else:
            h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        
        hidden.val = h
    
    def prefetch(self):
        pass


class TransformerLayer:
    def __init__(self, config, env, policy, i, parent_model=None):
        # self.attention = SelfAttention(config, env, policy, i, parent_model = parent_model)
        self.mlp = MLP(config, env, policy, i, parent_model= parent_model)
        self.attention = SelfAttention(config, env, policy, i, parent_model = parent_model, mlp = self.mlp)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k, read_buf2 = read_buf2)

        # if self.parent_model is not None:
        #     c_stream = self.parent_model.c_stream
        #     self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k, use_stream=c_stream)
        #     # 在MLP完成后再同步c_stream
        #     c_stream.synchronize()
        # else:
        #     self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)
        # self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)

    def prefetch(self):
        self.attention.prefetch()


class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        self.logits_val = None

        # 预取相关变量
        self.last_layer_prefetched = False  # 上一层有没有预取
        self.last_layer_prefetched_tokenids = None
        self.last_layer_prefetched_prefix_k = None
        self.last_layer_prefetched_prefix_v = None
        self.pending_prefetch_handle = None
        self.prefetch_time_budget = 0.0

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i, parent_model = self))
                layers.append(MLP(self.config, self.env, self.policy, i, parent_model= self))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i, parent_model= self))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        print(f'model.layers={self.layers}')
        self.num_layers = len(layers)

        # we can only place all weights at one place        
        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()
        self.prefetch_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.c_stream = torch.cuda.Stream()
        self.p_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def resolve_pending_prefetch(self, layer_id=None):
        global dt_async_prefetch_wait, async_prefetch_tokens

        if self.pending_prefetch_handle is None:
            return 0.0

        wait_st = time.monotonic()
        prefix_k, prefix_v, tokenids = self.pending_prefetch_handle.result()
        wait_time = time.monotonic() - wait_st
        if layer_id is not None:
            maybe_profile_kv_io("prefetch_wait_resolve", layer_id, wait_time, prefix_k, prefix_v)
        dt_async_prefetch_wait += wait_time
        if tokenids is not None:
            tokenids = tokenids.to("cuda:0")

        self.last_layer_prefetched_prefix_k = prefix_k
        self.last_layer_prefetched_prefix_v = prefix_v
        self.last_layer_prefetched_tokenids = tokenids
        if tokenids is not None:
            async_prefetch_tokens += int(tokenids.numel())
        self.pending_prefetch_handle = None
        return wait_time

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)
        self.hh_k = int(task.prompt_len * self.policy.hh_ratio)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        if j == self.num_layers - 1:
            self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                self.weight_read_buf[j], self.attention_mask[k],
                self.cache_write_buf[j][k], i, k, oplm=self)
        else:
            self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
                self.weight_read_buf[j], self.attention_mask[k],
                self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        # if in the decoding phase, we only need to update the mask for the kth gpu batch
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            # cnt = min(self.hh_k * 2, self.task.prompt_len)
            # if i == 1:
            #     mask.val = mask.val.device.slice_attention_mask(mask.val, cnt)
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            #if self.policy.hh_all:
            #    mask.val = mask.val.device.slice_attention_mask(mask.val, cnt + 1)
            return
        
        # Prefill phase, initialize the attention mask for the kth gpu batch
        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        # the shape of the value in the attention mask is (gpu_batch_size, prompt_len)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        # fill the attention mask with True for the input ids, False for padding
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def prefetch(self,i,j,k):
        if j == self.num_layers:
            return
        with torch.cuda.stream(self.prefetch_stream):
            self.layers[j].prefetch()


    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        print(f"execute_gen_len={self.execute_gen_len}, num_layers={self.num_layers}, num_gpu_batches={self.num_gpu_batches}")
        print(f"inputs.shape={np.array(inputs).shape}")

        # Output token ids
        # initialize the output_id, fill the front with input ids and pad the rest
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                # cache_home is used for storing the kv cache
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy, self.hh_k)
        # print_cpu_mem_usage("after init cache___")

        # Generate
        if debug_mode is None:
            if not overlap: # 当所有数据都在显存中时，是否 overlap 没有区别
                # No overlap, easy to understand, suitable for debugging
                if self.policy.generate_mapping_list:
                    mapping_2dtensor = self.generation_loop_normal()
                else:
                    self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()
        
        if self.policy.generate_mapping_list:
            return mapping_2dtensor, self.output_ids
        else:
            return self.output_ids

    def generation_loop_normal(self):
        try:
            print(f"execute_gen_len={self.execute_gen_len}")
            for i in range(self.execute_gen_len): # 生成第 i 个token
                print(f"== generate token {i} ==")
                timers("generate").start()
                for k in range(self.num_gpu_batches):
                    self.update_attention_mask(i, k)
                for j in range(self.num_layers): # 前传第 j 层
                    # print("step", i, "layer", j)
                    for k in range(self.num_gpu_batches):
                        self.load_weight(i, j, k, overlap=False)

                    for k in range(self.num_gpu_batches): # 计算第 k 个 batch 的前传
                        self.load_cache(i, j, k, overlap=False)
                        self.load_hidden(i, j, k)
                        # print(f'self.compute_layer({i}, {j}, {k})')
                        # if we assigned prefetch, load & compute
                        if self.policy.prefetch:
                            self.prefetch(i,j+1,k)
                            with torch.cuda.stream(self.compute_stream):
                                self.compute_layer(i, j, k)
                            torch.cuda.synchronize()
                        else:
                            self.compute_layer(i, j, k)
                        # hidden is the intermediate computation result for the i-th token
                        self.store_hidden(i, j, k)
                        self.store_cache(i, j, k, overlap=False)
                timers("generate").stop()
                
                if i==0 and self.policy.ret_topk_indices:
                    print('-> Prefill for all layers done. Start analysing similarity...')
                    # 此时所有层的 prefill 都结束
                    # 根据每个 attention 层的 self.topk_indices_lst 列表，
                    # 分析同层不同通道间的相似性高，还是同通道的不同层的相似性高
                    # 如果两个向量中的元素仅顺序不同应该判断为相同的向量，因此选择集合相似性的 Jaccard 系数比较合理
                    all_topk_indices_group_by_head = [None] * self.config.n_head
                    all_topk_indices_group_by_layer = []
                    for lid, layer in enumerate(self.layers):
                        if hasattr(layer, 'topk_indices_lst'):
                            print(f'layer[{lid}].topk_indices_lst[:10] = {layer.topk_indices_lst[:10]}')
                            all_topk_indices_group_by_layer.append(layer.topk_indices_lst)
                            for idx, head in enumerate(layer.topk_indices_lst):
                                if all_topk_indices_group_by_head[idx] is None:
                                    all_topk_indices_group_by_head[idx] = [head]
                                else:
                                    all_topk_indices_group_by_head[idx].append(head)
                    print('='*20, 'Visualization', '='*20)
                    print(len(all_topk_indices_group_by_head[-1]))
                    print(len(all_topk_indices_group_by_layer[-1]))
                    
                    all_sim_mtrx_gby_head = [] # 用于可视化
                    all_sim_mtrx_gby_layer = [] # 用于可视化
                    avg_sim_gby_head = []
                    avg_sim_gby_layer = []
                    for _i in range(len(all_topk_indices_group_by_head)):
                        sim_mtrx = self.compute_pair_similarity(all_topk_indices_group_by_head[_i])
                        # print(f'pair_similarity for head {_i}: {sim_mtrx}')
                        all_sim_mtrx_gby_head.append(sim_mtrx)
                        avg_sim_gby_head.append(sim_mtrx.mean())
                    self.visualize(all_sim_mtrx_gby_head, title='Head')
                    print(f'avg_sim_gby_head = {avg_sim_gby_head}')
                    
                    for _i in range(len(all_topk_indices_group_by_layer)):
                        sim_mtrx = self.compute_pair_similarity(all_topk_indices_group_by_layer[_i])
                        # print(f'pair_similarity for layer {_i}: {sim_mtrx}')
                        all_sim_mtrx_gby_layer.append(sim_mtrx)
                        avg_sim_gby_layer.append(sim_mtrx.mean())
                    self.visualize(all_sim_mtrx_gby_layer, title='Layer')
                    print(f'avg_sim_gby_layer = {avg_sim_gby_layer}')
                
                if i==0 and self.policy.generate_mapping_list:
                    # 收集每层的每个 imp_dec_mapping_per_layer 记录到当前prefixhash对应的字典中
                    imp_dec_mapping_all_layers = []
                    for lid, layer in enumerate(self.layers):
                        if hasattr(layer, 'topk_indices_lst'):
                            # print(f'layer[{lid}].imp_dec_mapping_per_layer[:10] = {layer.imp_dec_mapping_per_layer[:10]}')
                            imp_dec_mapping_all_layers.append(layer.imp_dec_mapping_per_layer)
                            # print("imp_dec_mapping_all_layers", imp_dec_mapping_all_layers.size())
                    print("imp_dec_mapping_all_layers", len(imp_dec_mapping_all_layers))
                    return torch.tensor(imp_dec_mapping_all_layers)
                
                # prefix dump
                if i==0 and self.policy.prefix_dump:
                    for layer in self.layers:
                        dump_layer = layer
                        if not hasattr(dump_layer, 'prefix_k_v') and hasattr(layer, 'attention'):
                            dump_layer = layer.attention
                        if hasattr(dump_layer, 'prefix_k_v'):
                            datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
                            model_name = os.path.basename(args.model)
                            if args.padding_mul == 1:
                                targetdir = f'./cache/prefix/{model_name}/{datasetname}'
                            else:
                                targetdir = f'./cache/prefix/{model_name}/{datasetname}_{args.padding_mul}'
                            prefix_hash = dump_layer.prefix_hash
                            # self.prefix_k_v shape: (prefix_len*2, nhead, dim)
                            save_kv_tensor_name = os.path.join(targetdir, args.model.replace('/', '_') + f'_attnid{dump_layer.layer_id}_{prefix_hash}.npy')
                            # print(save_kv_tensor_name, layer.prefix_k_v.shape, layer.prefix_k_v.is_cuda) # e.g., facebook_opt-6.7b_0 torch.Size([452, 32, 128]) False
                            if not os.path.exists(targetdir):
                                os.makedirs(targetdir)
                            np.save(save_kv_tensor_name, dump_layer.prefix_k_v.numpy())
                            print(f'Save {save_kv_tensor_name} done.')
                    # sys.exit()

        except Exception as e:
            print(f"Error during generation: {e}")
            raise
    
    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 10
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        lowest_avail_mem = float('inf')
        cpu_avail_mem = []
        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
                if i > 0 or j:
                    avail = psutil.virtual_memory().available / GB
                    lowest_avail_mem = min(lowest_avail_mem, avail)
                    cpu_avail_mem.append((i, j, avail))

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # print("(token, layer, available mem)")
        # for mem_info in cpu_avail_mem:
        #     print(f"({mem_info[0]}, {mem_info[1]}, {mem_info[2]:.2f})"),
        print(f"lowest available cpu mem: {lowest_avail_mem:.2f}")

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[execute_num_batches // 2:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        lowest_avail_mem = float('inf')
        cpu_avail_mem = []
        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):

                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
                    if i > 0 or (j + k == 0):
                        avail = psutil.virtual_memory().available / GB
                        lowest_avail_mem = min(lowest_avail_mem, avail)
                        cpu_avail_mem.append((i, j, k, avail))

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # print("(token, layer, batch, available mem)")
        # for mem_info in cpu_avail_mem:
        #     print(f"({mem_info[0]}, {mem_info[1]}, {mem_info[2]}, {mem_info[3]:.2f})"),
        print(f"lowest available cpu mem: {lowest_avail_mem:.2f}")

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()

    def compute_pair_similarity(self, nested_lst):
        """
        compute similarity for each pair of sublist in nested_lst and save the fig

        Args:
            nested_lst ([[], [], ...]): nested topk_indices lst
        """
        # 计算相似性矩阵
        n = len(nested_lst)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = self.jaccard_similarity(nested_lst[i], nested_lst[j])
        return similarity_matrix
        # print(similarity_matrix)
        
    def jaccard_similarity(self, lst1, lst2):
        # 将向量转换为集合
        set1 = set(lst1)
        set2 = set(lst2)

        # 计算杰卡德相似系数
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_score = intersection / union
        return jaccard_score

    def visualize(self, similarity_matrices, title=""):
        num_groups = len(similarity_matrices)
        # 创建子图
        ncols = 4
        nrows = (num_groups + ncols - 1) // ncols  # 计算行数，确保所有子图都显示
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows*5))

        # 绘制每个组的热力图
        axes = axes.flatten()
        for i in range(num_groups):
            heatmap = sns.heatmap(similarity_matrices[i], annot=False, cmap='coolwarm', cbar=False, ax=axes[i], vmin=0.8, xticklabels=5, yticklabels=5)
            axes[i].set_title(f'{title} {i}')
            xy_label = 'Layer' if title == 'Head' else 'Head'
            axes[i].set_xlabel(f'{xy_label} Index', fontsize=22)
            axes[i].set_ylabel(f'{xy_label} Index', fontsize=22)
            axes[i].tick_params(axis='x', labelsize=18)  # 增大x轴刻度字体大小
            axes[i].tick_params(axis='y', labelsize=18)  # 增大y轴刻度字体大小
            
            # 调整 color bar 的标签
            cbar = heatmap.figure.colorbar(heatmap.collections[0], ax=axes[i])
            cbar.set_ticks(np.arange(0.8, 1.01, 0.1))  # 设置间隔为0.1
            cbar.set_ticklabels([f'{x:.2f}' for x in np.arange(0.8, 1.01, 0.1)])  # 保留1位小数
            cbar.ax.tick_params(labelsize=18)  # 增大color bar的字体大小
            
            # 将 x 轴标签移动到顶部
            axes[i].xaxis.set_label_position('top')
            axes[i].xaxis.tick_top()  # 将 x 轴刻度移动到顶部
            # axes[i].set_xlabel('Vector Index', labelpad=10)  # 可以调整标签和边距

        # 处理多余的子图
        for j in range(num_groups, len(axes)):
            fig.delaxes(axes[j])
        
        # 显示图像
        plt.tight_layout()
        model_size = args.model.split('-')[-1]
        plt.savefig(f'Group_by_{title.lower()}_hhratio{args.hh_ratio}_{model_size}.png')
        print(f'-> save to Group_by_{title.lower()}_hhratio{args.hh_ratio}_{model_size}.png')

class LlamaInputEmbed(InputEmbed):
    def __init__(self, config, env, policy):
        super().__init__(config, env, policy)

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst),))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 3
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), = weight_read_buf.pop()
        else:
            (w_token, _), = weight_read_buf.val
        h = self.compute.llama_input_embed(h, mask,
            w_token, self.config.pad_token_id, donate)
        hidden.val = h


class LlamaOutputEmbed(OutputEmbed):
    def __init__(self, config, env, policy):
        super().__init__(config, env, policy)

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "norm.weight"),
            # w_token
            ((v, h), dtype, path + "lm_head.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), w_token.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k,oplm = None):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        if self.logits:
            h = self.compute.llama_output_embed(h, w_ln, w_token, self.config.rms_norm_eps, donate,
                self.task.do_sample, self.task.temperature,record_logits = self.logits, oplm = self)
            oplm.logits_val = self.logits_val
        else:
            h = self.compute.llama_output_embed(h, w_ln, w_token, self.config.rms_norm_eps, donate,
                self.task.do_sample, self.task.temperature)
        hidden.val = h


class LlamaSelfAttention(SelfAttention):
    def __init__(self, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)

    def init_weight(self, weight_home, path):
        h, n_head, n_kv_head, dtype = (self.config.input_dim, self.config.n_head, self.config.num_key_value_heads, self.config.dtype)
        head_dim = h // n_head
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "input_layernorm.weight"),
            # w_q
            ((h, n_head*head_dim), dtype, path + "self_attn.q_proj.weight"),
            # w_k
            ((n_kv_head*head_dim, h), dtype, path + "self_attn.k_proj.weight"),
            # w_v
            ((n_kv_head*head_dim, h), dtype, path + "self_attn.v_proj.weight"),
            # w_re
            ((head_dim//2,), dtype, path + "self_attn.rotary_emb.inv_freq"),
            # w_o
            ((n_head*head_dim, h), dtype, path + "self_attn.o_proj.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_q, w_k, w_v, w_re, w_o = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2),
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1),
                w_v.smart_copy(dst1),
                w_re.smart_copy(dst1),
                w_o.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head
        n_kv_head = self.config.num_key_value_heads

        donate = [False] * 10
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[2]), (w_q, donate[3]), (w_k, donate[4]), (w_v, donate[5]),
             (w_re, donate[6]), (w_o, donate[7])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (w_q, _), (w_k, _), (w_v, _),
             (w_re, _), (w_o, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            position_ids = torch.cumsum(mask.data, dim=1).int() * mask.data - 1
            # print(f'mask = {mask.shape} {mask.data}') # shape (b, s), True if not padding
            # print(f'input: {self.task.inputs[0]}')
            global pcache,dt_prefix,dt_all,dt_load,dt_sele,dt_load_head,dt_suffix,dt_key,dt_value, orcale_ttft, orcale_load_time
            global dt_async_prefetch_submit, async_prefetch_jobs, async_prefetch_hit_tokens, async_prefetch_miss_tokens
            if self.policy.prefix_aware_inf:
                prefix_hash = str(hash(tuple(self.task.inputs[0][:-self.policy.suffix_len].tolist())))
                datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
                model_name = os.path.basename(args.model)
                if args.padding_mul == 1:
                    prefix_kv_path = os.path.join(f'./cache/prefix/{model_name}/{datasetname}', args.model.replace('/', '_') + f'_attnid{self.layer_id}_{prefix_hash}.npy')
                else:
                    prefix_kv_path = os.path.join(f'./cache/prefix/{model_name}/{datasetname}_{args.padding_mul}', args.model.replace('/', '_') + f'_attnid{self.layer_id}_{prefix_hash}.npy')
                    # print(prefix_kv_path, os.path.exists(prefix_kv_path))
                # 完全加载prefix kv (不使用选择性加载 或 在选择性加载的模式下，layer_id >= 10)
                if self.policy.full_load:
                    # 如果 prefix的kv 已经有缓存过，直接加载
                    # if os.path.exists(prefix_kv_path):
                    if True:
                        # print(f'-> Found prefix KV path: {prefix_kv_path}')
                        # k_v_num_tokens = len(self.task.inputs[0][:-self.policy.suffix_len])

                        torch.cuda.synchronize()
                        st = time.monotonic()
                        if not args.no_cache:
                            if prefix_hash not in prefix_table:
                                insert_pcache(pcache=pcache,hash_name=prefix_hash)
                            prefix_k,prefix_v = pcache.get(prefix_id=prefix_table[prefix_hash],layer=self.layer_id)
                        else:
                            mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                            prefix_k, prefix_v = torch.from_numpy(mmap_kv_tensor[0, :, :, :]).cuda(), torch.from_numpy(mmap_kv_tensor[1, :, :, :]).cuda()
                        # print(f'shape of prefix_k = {prefix_k.shape}, shape of prefix_v = {prefix_v.shape}') # (prefix_s, num_head, head_dim)
                        torch.cuda.synchronize()
                        dt_load += time.monotonic() - st

                        if self.policy.sele_inf:
                            # 每层选择指定百分比的 kv 进行推理 (加载的时候依然全加载，加载后计算 attention 后推理时选择固定百分比)
                            # 重计算完成输入对应的q，重计算 suffix-k/v；选择部分 prefix-k/v，拼接后得到部分的 k/v
                            # idx = math.ceil((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                            if self.policy.generate_mapping_list:
                                # 按照 full_load_sele_inf 来，区别在于每个请求返回prefix中重要的 tokenid 后记录下来
                                idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                                cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                                h, new_k_cache, new_v_cache, acc, reverse_indices = self.compute.llama_mha_with_sele_percent_prefixkv(h, position_ids, mask, w_ln,
                                        w_q, w_k, w_v, w_re, w_o, n_head,n_kv_head, donate, self.config.rms_norm_eps,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len, cur_sele_percent, generate_mapping_list = self.policy.generate_mapping_list)
                                self.imp_dec_mapping_per_layer = reverse_indices.cpu().tolist()
                            else:    
                                if not self.policy.full_load_sele_inf_by_accum:
                                    # 每层选择指定百分比的 kv 进行推理 (加载的时候依然全加载，加载后计算 attention 后推理时选择固定百分比)
                                    # 重计算完成输入对应的q，重计算 suffix-k/v；选择部分 prefix-k/v，拼接后得到部分的 k/v
                                    
                                    idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                                    cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                                    # print(f'cur_sele_percent = {cur_sele_percent}')

                                    if self.policy.suffix_comp:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache,acc = self.compute.llama_mha_with_prefixkv_suffix(h, position_ids, mask, w_ln,
                                                w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                        torch.cuda.synchronize()
                                        dt_suffix  += time.monotonic() - st
                                    else:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache, acc = self.compute.llama_mha_with_sele_percent_prefixkv(h, position_ids, mask, w_ln,
                                                w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len, cur_sele_percent)
                                        torch.cuda.synchronize()
                                        dt_all  = time.monotonic() - st
                                else:
                                    # 每层选择变长数量的 kv 进行推理直到累积的atten达到指定百分比 (加载的时候依然全加载，加载后计算 attention 后推理时选择变长数量)
                                    # 重计算完成输入对应的q，重计算 suffix-k/v；选择部分 prefix-k/v，拼接后得到部分的 k/v
                                    idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.accum_percent))
                                    cur_accum_percent = self.policy.accum_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.accum_percent[idx]
                                    # print(f'[full_load_sele_inf] target accum_percent for current attn layer[{self.layer_id}] = {cur_accum_percent}%')

                                    if self.policy.suffix_comp:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache,acc = self.compute.llama_mha_with_prefixkv_suffix(h, position_ids, mask, w_ln,
                                                w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                        torch.cuda.synchronize()
                                        dt_suffix  += time.monotonic() - st
                                    else:
                                        torch.cuda.synchronize()
                                        st = time.monotonic()
                                        h, new_k_cache, new_v_cache, acc = self.compute.llama_mha_with_var_percent_prefixkv(h, position_ids, mask, w_ln,
                                                w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                                self.policy.compress_cache, self.policy.comp_cache_config,
                                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len, cur_accum_percent)
                                        torch.cuda.synchronize()
                                        dt_all  = time.monotonic() - st

                        else:
                            # 重计算完成输入对应的q，重计算 suffix-k/v；拼接后得到全部的 k/v ==> 与不使用 prefixkv 推理结果相同
                            # h, new_k_cache, new_v_cache,acc = self.compute.mha_with_prefixkv_suffix(h, mask, w_q, b_q,
                            #         w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                            #         self.policy.compress_cache, self.policy.comp_cache_config,
                            #         self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                            if self.policy.suffix_comp:
                                torch.cuda.synchronize()
                                st = time.monotonic()
                                h, new_k_cache, new_v_cache,acc = self.compute.llama_mha_with_prefixkv_suffix(h, position_ids, mask, w_ln,
                                        w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                torch.cuda.synchronize()
                                dt_suffix  += time.monotonic() - st
                            else:
                                torch.cuda.synchronize()
                                st = time.monotonic()
                                h, new_k_cache, new_v_cache, acc = self.compute.llama_mha_with_prefixkv(h, position_ids, mask, w_ln,
                                        w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, prefix_k, prefix_v, self.policy.suffix_len)
                                torch.cuda.synchronize()
                                dt_all  = time.monotonic() - st
                    else:
                        print(f'ERROR: 找不到 Prefix 的 KV文件 {prefix_kv_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                        sys.exit(-1)
                
                elif self.policy.full_only_key_load:
                    # full_load_key sele_load_value
                    # 把 key 完全加载，然后找出重要的 v，加载部分的v 填充后进行推理
                    if args.no_cache:
                        if not os.path.exists(prefix_kv_path):
                            print(f'ERROR: 找不到 Prefix 的 KV文件 {prefix_kv_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                            sys.exit(-1)
                    
                    torch.cuda.synchronize()
                    st_key = time.monotonic()
                    if not args.no_cache:
                        if prefix_hash not in prefix_table:
                            insert_pcache(pcache=pcache,hash_name=prefix_hash)
                        complete_prefix_k = pcache.get_key(prefix_id=prefix_table[prefix_hash],pos_id=None, layer=self.layer_id) # ［seq，head num，head dim]
                    else:
                        mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                        complete_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, :, :]).cuda()
                    torch.cuda.synchronize()
                    dt_key += time.monotonic() - st_key
                    # print(f'shape of only prefix_k = {complete_prefix_k.shape}') # (prefix_s, num_head, head_dim)
                    
                    torch.cuda.synchronize()
                    st_sele = time.monotonic()
                    idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                    cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                    # print('cur_sele_percent', cur_sele_percent)
                    # shape: (sele_num, ) in cuda
                    # sele_tokenids, del_tokenids = self.compute.sele_tokenid_with_all_keys(h, mask, w_q, b_q,
                    #             w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                    #             complete_prefix_k, cur_sele_percent)
                    sele_tokenids = self.compute.llama_sele_tokenid_with_all_keys(h, position_ids, mask, w_ln,
                                w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                complete_prefix_k, cur_sele_percent)
                    torch.cuda.synchronize()
                    dt_sele += time.monotonic() - st_sele                  


                    # 根据 sele_tokenids 加载 部分 value
                    torch.cuda.synchronize()
                    st_value = time.monotonic()
                    if not args.no_cache:
                        remain_prefix_v = pcache.get_value(prefix_id=prefix_table[prefix_hash],pos_id=sele_tokenids[0],layer=self.layer_id)
                    else:
                        remain_prefix_v = torch.from_numpy(mmap_kv_tensor[1, sele_tokenids, :, :]).cuda()
                    torch.cuda.synchronize()
                    dt_value += time.monotonic() - st_value
                    
                    # 把 remain_prefix_v 和 complete_prefix_k 一起进行推理
                    k_v_num_tokens = len(self.task.inputs[0][:-self.policy.suffix_len])
                    filled_prefix_v = torch.zeros(size=(k_v_num_tokens, self.config.num_key_value_heads, remain_prefix_v.shape[-1]), dtype=remain_prefix_v.dtype, device="cuda:0")
                    filled_prefix_v[sele_tokenids] = remain_prefix_v
                    
                    
                    if self.policy.suffix_comp:
                        torch.cuda.synchronize()
                        st = time.monotonic()
                        h, new_k_cache, new_v_cache, acc = self.compute.llama_mha_with_prefixkv_suffix(h, position_ids, mask, w_ln,
                                    w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                    self.policy.compress_cache, self.policy.comp_cache_config,
                                    self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, complete_prefix_k, filled_prefix_v, self.policy.suffix_len)
                        torch.cuda.synchronize()
                        dt_suffix  += time.monotonic() - st
                    else:
                        torch.cuda.synchronize()
                        st = time.monotonic()
                        h, new_k_cache, new_v_cache, acc = self.compute.llama_mha_with_filled_selected_prefixkv(h, position_ids, mask, w_ln,
                                    w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                    self.policy.compress_cache, self.policy.comp_cache_config,
                                    self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, complete_prefix_k, filled_prefix_v, self.policy.suffix_len, del_tokenids = del_tokenids)
                        torch.cuda.synchronize()
                        dt_all+= time.monotonic() - st
                else:
                    # sele_load prefix kv
                    if args.no_cache:
                        if not os.path.exists(prefix_kv_path):
                            print(f'ERROR: 找不到 Prefix 的 KV文件 {prefix_kv_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                            sys.exit(-1)
                    if True:
                        # 先加载 sele_head 中的全部的 key 
                        # -> atten_weights 
                        # -> 计算重要的 token id with accum_percent (或者指定每层选择的比例)
                        # -> 计算相似度 sim, 与 sim_thred 比较
                        torch.cuda.synchronize()
                        st = time.monotonic()
                        k_v_num_tokens = len(self.task.inputs[0][:-self.policy.suffix_len])
                        if not args.no_cache:
                            if self.prefetched_head is None:
                                if prefix_hash not in prefix_table:
                                    insert_pcache(pcache=pcache,hash_name=prefix_hash)
                                fullhead_prefix_k = pcache.get_head(prefix_id=prefix_table[prefix_hash],layer=self.layer_id)
                            else:
                                fullhead_prefix_k = self.prefetched_head
                                self.prefetched_head = None
                            # print(fullhead_prefix_k.shape,prefix_table[prefix_hash],prefix_hash)
                        else:
                            mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                            # fullhead_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, self.policy.sele_head, :])
                            # print(f'fullhead_prefix_k.shape1 = {fullhead_prefix_k.shape}')
                            fullhead_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, self.policy.sele_head, :]).permute(1, 0, 2).cuda()
                            print(f'fullhead_prefix_k.shape = {fullhead_prefix_k.shape}')
                        torch.cuda.synchronize()
                        dt_load_head += time.monotonic() - st

                        # mmap_kv_tensor = np.load(prefix_kv_path, mmap_mode='r')
                        # fullhead_prefix_k = torch.from_numpy(mmap_kv_tensor[:k_v_num_tokens, self.policy.sele_head, :]).cuda()
                        # print(f'fullhead_prefix_k.shape = {fullhead_prefix_k.shape}')
                        
                        # 每层选累积百分比的 token
                        if not self.policy.sele_load_by_percent:
                            idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.accum_percent))
                            cur_accum_percent = self.policy.accum_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.accum_percent[idx]
                            # print(f'target accum_percent for current attn layer[{self.layer_id}] = {cur_accum_percent}%')
                        
                            # shape: (sele_num, ) in cuda
                            sele_tokenids = self.compute.llama_sele_tokenid(h, position_ids, mask, w_ln,
                                        w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                        self.policy.sele_head, fullhead_prefix_k, cur_accum_percent, self.policy.sim_thred)
                        
                        # 每层选指定百分比的 token
                        else:
                            idx = int((self.layer_id+1) / self.config.num_hidden_layers * len(self.policy.sele_percent))
                            cur_sele_percent = self.policy.sele_percent[idx-1] if idx == len(self.policy.sele_percent) else self.policy.sele_percent[idx]
                            # print('cur_sele_percent', cur_sele_percent)
                            # shape: (sele_num, ) in cuda
                            sele_tokenids, del_tokenids = self.compute.llama_sele_tokenid(h, position_ids, mask, w_ln,
                                        w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                        self.policy.sele_head, fullhead_prefix_k, -1, self.policy.sim_thred, cur_sele_percent)
                            #print('sele_token',sele_tokenids, sele_tokenids.shape)
                            #print('del_token', del_tokenids, del_tokenids.shape)
                        torch.cuda.synchronize()
                        dt_sele += time.monotonic() - st
                            
                        # torch.cuda.synchronize()
                        # st = time.monotonic()
                        # h2, new_k_cache, new_v_cache = self.compute.mha_prefixkv(h, mask, w_q, b_q,
                        #         w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                        #         self.policy.compress_cache, self.policy.comp_cache_config,
                        #         self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, self.policy.suffix_len)
                        # torch.cuda.synchronize()
                        # dt_prefix += time.monotonic() - st
                        # with open(f'./ids_{cur_sele_percent}_{datasetname}.txt','a+') as f:
                        #     f.write(f'prefix_id:{prefix_table[prefix_hash]},prompt_len:{k_v_num_tokens},sele_token:{sele_tokenids}\n')

                        # -> 根据 sele_tokenids 加载剩余 head 中的 kv
                        # -> 和之前的 key 拼接后一起进行推理
                        # sele_tokenids_k, sele_tokenids_v = sele_tokenids, sele_tokenids + k_v_num_tokens
                        # all_heads = torch.arange(self.config.n_head)
                        # remain_heads = all_heads[~torch.isin(all_heads, torch.tensor(self.policy.sele_head))]
                        torch.cuda.synchronize()
                        st = time.monotonic()
                        if not args.no_cache:
                            remain_prefix_k, remain_prefix_v = pcache.get(prefix_id=prefix_table[prefix_hash],pos_id=sele_tokenids,layer=self.layer_id)
                        else:
                            remain_prefix_k, remain_prefix_v = torch.from_numpy(mmap_kv_tensor[0, sele_tokenids, :, :]).cuda(), torch.from_numpy(mmap_kv_tensor[1, sele_tokenids, :, :]).cuda()
                        torch.cuda.synchronize()
                        dt_load += time.monotonic() - st
                            
                        
                        # print(f'remain_prefix_k.shape={remain_prefix_k.shape}, remain_prefix_v.shape={remain_prefix_v.shape}') # shape: (sele_token_num, n_head, head_dim)
                        # fill with zeros for inference
                        if args.fill_keys_zero:
                            filled_prefix_k, filled_prefix_v = torch.zeros(size=(k_v_num_tokens, self.config.num_key_value_heads, remain_prefix_k.shape[-1]), dtype=remain_prefix_k.dtype, device="cuda:0"), torch.zeros(size=(k_v_num_tokens, self.config.num_key_value_heads, remain_prefix_v.shape[-1]), dtype=remain_prefix_v.dtype, device="cuda:0")
                        else:
                            filled_prefix_v = torch.zeros(size=(k_v_num_tokens, self.config.num_key_value_heads, remain_prefix_v.shape[-1]), dtype=remain_prefix_v.dtype, device="cuda:0")
                            # 计算三维上的平均值，得到一个 shape 为 (prefixlen, 128) 的 tensor
                            prefix_len, sele_nheads, head_dim = fullhead_prefix_k.shape
                            # 计算第三维度上的平均值
                            mean_values = fullhead_prefix_k.mean(dim=1, keepdim=True)
                            # 将平均值扩展
                            mean_values_expanded = mean_values.expand(prefix_len, self.config.n_head-sele_nheads, head_dim)
                            # 拼接原始张量和扩展后的平均值张量，形成 (prefix_len, self.config.n_head, head_dim)
                            filled_prefix_k = torch.cat((fullhead_prefix_k, mean_values_expanded), dim=1)
                        
                        filled_prefix_k[sele_tokenids] = remain_prefix_k
                        # filled_prefix_k = torch.from_numpy(mmap_kv_tensor[0, :, :, :]).cuda()
                        filled_prefix_v[sele_tokenids] = remain_prefix_v
                        
                        # print(f'filled_prefix_k.shape={filled_prefix_k.shape}, filled_prefix_v.shape={filled_prefix_v.shape}') # shape: (s, n_head, head_dim)
                        
                        if self.policy.suffix_comp:
                            torch.cuda.synchronize()
                            st = time.monotonic()
                            h, new_k_cache, new_v_cache, acc = self.compute.llama_mha_with_prefixkv_suffix(h, position_ids, mask, w_ln,
                                    w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                    self.policy.compress_cache, self.policy.comp_cache_config,
                                    self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, filled_prefix_k, filled_prefix_v, self.policy.suffix_len)
                            torch.cuda.synchronize()
                            dt_suffix  += time.monotonic() - st
                        else:
                            torch.cuda.synchronize()
                            st = time.monotonic()
                            h, new_k_cache, new_v_cache, acc = self.compute.llama_mha_with_filled_selected_prefixkv(h, position_ids, mask, w_ln,
                                        w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                        self.policy.compress_cache, self.policy.comp_cache_config,
                                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices, filled_prefix_k, filled_prefix_v, self.policy.suffix_len, del_tokenids = del_tokenids)
                            torch.cuda.synchronize()
                            dt_all+= time.monotonic() - st
                        
                        
                    else:
                        print(f'ERROR: 找不到 Prefix 的 KV文件 {prefix_kv_path}；请先生成 Prefix 并存入磁盘，命令见 README-wj.md')
                        sys.exit(-1)
                        
                    
            else:
                if self.policy.ret_topk_indices:
                    # topk_indices 表示一个层中，每个head中最重要的k个token的id shape:(b*n_head, k)
                    h, new_k_cache, new_v_cache, acc, topk_indices = self.compute.llama_mha(h, position_ids, mask, w_ln,
                                w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                                self.policy.compress_cache, self.policy.comp_cache_config,
                                self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices)
                    self.topk_indices_lst = topk_indices.cpu().tolist()
                else: # recompute

                    # opt
                    # torch.cuda.synchronize()
                    # st = time.monotonic()
                    # h, new_k_cache, new_v_cache, acc = self.compute.mha(h, mask, w_q, b_q,
                    #             w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                    #             self.policy.compress_cache, self.policy.comp_cache_config,
                    #             self.hh_k, self.policy.hh_all)
                    # torch.cuda.synchronize()
                    # dt_prefix += time.monotonic() - st

                    # llama
                    torch.cuda.synchronize()
                    st = time.monotonic()
                    
                    h, new_k_cache, new_v_cache, acc = self.compute.llama_mha(h, position_ids, mask, w_ln,
                        w_q, w_k, w_v, w_re, w_o, n_head, n_kv_head, donate, self.config.rms_norm_eps,
                        self.policy.compress_cache, self.policy.comp_cache_config,
                        self.hh_k, self.policy.hh_all, self.policy.ret_topk_indices)
                    
                    torch.cuda.synchronize()
                    dt_all  += time.monotonic() - st
                    # with open('./times.txt','a+') as f:
                    #     f.write(f'comp without prefix:{dt_comp_no_prefix}\n')
            
            if self.policy.prefix_dump:
                # 记录下 k,v 方便存入磁盘
                # 假设后 100 个为 query
                self.prefix_k_v = torch.stack((new_k_cache.data[:-self.policy.suffix_len, :, :].cpu(), new_v_cache.data[:-self.policy.suffix_len, :, :].cpu()), dim=0)
                # print(f'move prefix KV into CPU memory with KV shape = {self.prefix_k_v.shape}')
                self.prefix_hash = str(hash(tuple(self.task.inputs[0][:-self.policy.suffix_len].tolist())))
            cache_write_buf.store((new_k_cache, new_v_cache, acc, None))
            
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[8]), (v_cache, donate[9]) = cache_read_buf.pop()
            position_ids = torch.cumsum(mask.data, dim=1).int() * mask.data + 1
            position_ids = position_ids[:, -h.shape[1]].unsqueeze(1)
            # 需不需要
            # if self.policy.hh_all is not None:
                # cnt = min(self.hh_k * 2 - 1, self.task.prompt_len + i)
                # mask = mask.device.slice_attention_mask(mask, cnt + 1)
            h, new_k_cache, new_v_cache = self.compute.llama_mha_gen(h, position_ids, mask, w_ln,
                w_q, w_k, w_v, w_re, w_o, self.config.rms_norm_eps, n_head, n_kv_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class LlamaMLP(MLP):
    def __init__(self, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)

    def init_weight(self, weight_home, path):
        h, intermediate, dtype = (self.config.input_dim, self.config.intermediate_size, self.config.dtype)
        path = os.path.join(os.path.join(path, f"layers.{self.layer_id}."))
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "post_attention_layernorm.weight"),
            # w_g
            ((intermediate, h), dtype, path + "mlp.gate_proj.weight"),
            # w_u
            ((intermediate, h), dtype, path + "mlp.up_proj.weight"),
            # w_d
            ((h, intermediate), dtype, path + "mlp.down_proj.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_g, w_u, w_d = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_ln.smart_copy(dst2),
                w_g.smart_copy(dst1),
                w_u.smart_copy(dst1),
                w_d.smart_copy(dst1)))

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_ln, donate[1]), (w_g, donate[2]), (w_u, donate[3]),
             (w_d, donate[4])) = weight_read_buf.pop()
        else:
            ((w_ln, _), (w_g, _), (w_u, _), (w_d, _)) = weight_read_buf.val

        h = self.compute.llama_mlp(h, w_ln, w_g, w_u, w_d, self.config.rms_norm_eps, donate)
        hidden.val = h


class LlamaTransformerLayer(TransformerLayer):
    def __init__(self, config, env, policy, i):
        self.attention = LlamaSelfAttention(config, env, policy, i)
        self.mlp = LlamaMLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute


class LlamaLM(OptLM):
    def __init__(self,
                 config: Union[str, LlamaConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        if isinstance(config, str):
            config = get_llama_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(LlamaInputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(LlamaSelfAttention(self.config, self.env, self.policy, i))
                layers.append(LlamaMLP(self.config, self.env, self.policy, i))
            else:
                layers.append(LlamaTransformerLayer(self.config, self.env, self.policy, i))
        layers.append(LlamaOutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_llama_weights(self.config.name, self.path, self.config.hf_token)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

def get_timelog_filename(args):
    model_size = args.model.split('-')[-1]
    if 'Llama' in args.model:
        model_size = args.model.split('-')[-2]

    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = './time_logs/'
    if args.input_path:
        datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
        filename += f"{model_size}-" \
                f"{datasetname}-padding_mul{args.padding_mul}-suffix_mul{args.suffix_mul}-" \
                f"no_prefetch{args.no_prefetch}-full_load{args.full_load}-sele_load{args.sele_load}-" \
                f"cache_type{args.cache_type}-reorder{args.reorder}"

    return filename

def get_filename(args):
    model_size = args.model.split('-')[-1]
    if 'Llama' in args.model:
        model_size = args.model.split('-')[-2]

    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = './logs/'
    if args.input_path:
        datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
        filename += f"fo-{model_size}-" \
                f"{datasetname}-" \
                f"full_load{args.full_load}-sele_load{args.sele_load}-" \
                f"sele_load_by_p{args.sele_load_by_percent}-sele_percent{args.sele_percent}" \
                f"sim_thred{args.sim_thred}-cache_type{args.cache_type}-disk_type{args.disk_type}-" \
                f"prefix_aware_inf{args.prefix_aware_inf}-" \
                f"ro{args.reorder}-cksize{args.chunk_size}-no_prefetch{args.no_prefetch}-recompute{args.recompute}"
    else:
        filename += f"fo-{model_size}-" \
                f"hhr{args.hh_ratio}-full_load{args.full_load}-sele_load{args.sele_load}-" \
                f"sele_load_by_p{args.sele_load_by_percent}-sele_percent{args.sele_percent}" \
                f"sim_thred{args.sim_thred}-cache_type{args.cache_type}-disk_type{args.disk_type}-" \
                f"prefix_dump{args.prefix_dump}-prefix_aware_inf{args.prefix_aware_inf}-" \
                f"ro{args.reorder}-cksize{args.chunk_size}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_inputs(prompt_len, num_prompts, tokenizer):
    data = []
    # 读取了整个数据集，并且把contex和input提取并拼接起来。
    with open('../../datasets/sys_prompt.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            try:
                context = json.loads(line).get('context','')
                question = json.loads(line).get('input', '')
                data.append([context + question])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue   
    # batchsize = 4 
    inputs = data[5:9]
    # print(inputs)
    inputs_real = []
    for item in inputs:
        input_ids = tokenizer(item, padding="max_length",
                          max_length=prompt_len, add_special_tokens=False).input_ids
        inputs_real.append(input_ids[0])
    # print(input_ids)
    return tuple(inputs_real)

def get_test_inputs(prompt_len, num_prompts, tokenizer):
    # prompts = ["Paris is the capital city of"]
    #prompts = ["Artificial Intelligence (AI) refers to the development of computer systems or machines that can perform tasks that typically require human intelligence. These tasks include problem-solving, learning, understanding natural language, recognizing patterns, perception, and decision-making. AI systems are designed to process vast amounts of data and draw conclusions or make decisions based on that data. There are two main categories of AI: Narrow AI and General AI. Narrow AI, also known as Weak AI, is designed for "]
    
    prompts = ["Question: To start a hole of golf,\nAnswer: drive your ball onto the field from the tee.\n\nQuestion: How do you remove the motherboard from a computer case.\nAnswer: After removing every other component (or you can leave the CPU, CPU fan and RAM), unscrew the screw holding the motherboard on the standoffs, and then unscrew the standoffs from the case.\n\nQuestion: To remove paper glued to wood to serve as a template for drill holes.\nAnswer: Use a heat gun to remove the paper.\n\nQuestion: how to bake salmon\nAnswer: Preheat the oven to 450 degrees F.    Season salmon with salt and pepper. Place salmon, skin side down, on a non-stick baking sheet or in a non-stick pan with an oven-proof handle. Bake until salmon is cooked through, about 12 to 15 minutes. Serve with the Toasted Almond Parsley Salad and squash, if desired.\n\nQuestion: To avoid spreading germs while storing meat in the fridge,\nAnswer: only put the meat on the bottom shelf.\n\nQuestion: Remove seeds from  strawberries\nAnswer: Blend the strawberries, pour the mixture through a fine-mesh strainer with a bowl underneath to catch the pulps and strain out the seeds"]
    # prompts = ["Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.Which NFL team represented the AFC at Super Bowl 50?"]
    #prompts = ["Lennon was murdered by Mark David Chapman outside the Dakota on Dec. 8, 1980.\nQuestion: Mark David Chapman killed Lennon. True or False?\nAnswer: True\n\nSaudi Arabia's production mix will shift to a higher proportion of lighter crudes. Unlike last year's Qatif and Abu Safah developments, nearly all of the proposed projects produce Arab Light or lighter crudes.\nQuestion: Saudi Arabia produces more oil than any other country. True or False?\nAnswer: False\n\nGold mining operations in California and Nevada use cyanide to extract the precious metal.\nQuestion: Cyanide is used in gold mining. True or False?\nAnswer: True\n\nIndia, an enchanting country situated in the southern central peninsula of the Asian continent, covers over 3.28 million square kilometers.\nQuestion: India is on the Asian continent. True or False?\nAnswer: True\n\nWASHINGTON --  A newly declassified narrative of the Bush administration's advice to the CIA on harsh interrogations shows that the small group of Justice Department lawyers who wrote memos authorizing controversial interrogation techniques were operating not on their own but with direction from top administration officials, including then-Vice President Dick Cheney and national security adviser Condoleezza Rice. At the same time, the narrative suggests that then-Defense Secretary Donald H. Rumsfeld and then-Secretary of State Colin Powell were largely left out of the decision-making process.\nQuestion: Dick Cheney was the Vice President of Bush. True or False?\nAnswer: True\n\nIn the history of art, prehistoric art is all art produced in preliterate cultures (prehistory), beginning somewhere in very late geological history.\nQuestion: Prehistoric art discovered in South Africa. True or False?\nAnswer: "]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len,
                            add_special_tokens=False, return_tensors='pt').input_ids
    # input_ids = tokenizer(prompts, add_special_tokens=False).input_ids
    # print(len(prompts))
    return (input_ids[0],) * num_prompts


def run_flexgen(args, logtrace_filename):
    logits_bool = False   #计算返回logits
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    elif 'llama' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left",padding="max_length")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = 2
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", use_fast=True)
        # tokenizer.save_pretrained(f'cache/huggingface/hub/{args.model}') # 避免 token 重复加载
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    # warmup_inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    # inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    # inputs = get_inputs(prompt_len, num_prompts, tokenizer)
    # prompt_len = len(inputs[0])
    # print(f'prompt_len={prompt_len}, inputs={inputs}')

    if 'opt' in args.model:
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(args.offload_dir)
        env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    elif 'llama' in args.model:
        gpu = LlamaTorchDevice("cuda:0")
        cpu = LlamaTorchDevice("cpu")
        disk = TorchDisk(args.offload_dir)
        env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    if args.logits:
        logits_bool = True

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False),
                    hh_ratio=args.hh_ratio,
                    hh_all=args.hh_all,
                    hh_long_seq=args.hh_long_seq,
                    ret_topk_indices=args.ret_topk_indices,
                    prefix_dump=args.prefix_dump,
                    prefix_aware_inf=args.prefix_aware_inf,
                    sele_load=args.sele_load,
                    suffix_len=args.suffix_len,
                    logits=logits_bool,
                    sele_percent=tuple(args.sele_percent),
                    full_load=args.full_load,
                    sele_inf=args.sele_inf,
                    accum_percent=tuple(args.accum_percent),
                    sele_head=tuple(args.sele_head),
                    sim_thred=args.sim_thred,
                    sele_load_by_percent=args.sele_load_by_percent,
                    full_load_sele_inf_by_accum=args.full_load_sele_inf_by_accum,
                    prefetch=args.prefetch,
                    full_only_key_load=args.full_only_key_load,
                    suffix_comp=args.suffix_comp,
                    generate_mapping_list=args.generate_mapping_list,
                    # 预取相关
                    prefetch_ratio = args.prefetch_ratio,
                    no_prefetch = args.no_prefetch,
                    solid_ratio_prefetch = args.solid_ratio_prefetch,
                    )
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    print("init weight...")
    if 'opt' in args.model:
        opt_config = get_opt_config(args.model)
        cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
        hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
        print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
            f"cache size: {cache_size/GB:.3f} GB, "
            f"hidden size (prefill): {hidden_size} ")
        model = OptLM(opt_config, env, args.path, policy)
    elif 'llama' in args.model:
        llama_config = get_llama_config(args.model, pad_token_id=tokenizer.eos_token_id)
        cache_size = llama_config.cache_bytes(num_prompts, prompt_len + gen_len)
        hidden_size = llama_config.hidden_bytes(num_prompts, prompt_len + gen_len)
        model = LlamaLM(llama_config, env, args.path, policy)
        print(f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
            f"cache size: {cache_size/GB:.3f} GB, "
            f"hidden size (prefill): {hidden_size/GB:.3f} GB")
    print(f'model.config = {model.config}')
    print_cpu_mem_usage("after init weight")
    datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
    model_name = os.path.basename(args.model)
    if args.input_path and not args.no_cache and not args.prefix_dump:
        st = time.time()
        global pcache
        print(f'gpu_size:{args.gpu_size},cpu_size:{args.cpu_size},chunk_size:{args.chunk_size}')
        print(f'cache_type:{args.cache_type},disk_type:{args.disk_type}')
        pcache = Pcache(gpu_size=args.gpu_size,cpu_size=args.cpu_size,cache_type=args.cache_type,disk_type=args.disk_type,cpu_gather=True,chunk_size=args.chunk_size)
        datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
        model_name = os.path.basename(args.model)
        if args.padding_mul == 1:
            targetdir = f'./cache/prefix/{model_name}/{datasetname}'
        else:
            targetdir = f'./cache/prefix/{model_name}/{datasetname}_{args.padding_mul}'
        init_pcache(pcache=pcache, folder_path=targetdir, name=args.model.replace('/', '_'))
        print(f'-> Init Pcache done with {time.time() - st} seconds.')
        
        # 加载预取比例，用于 debug
        # if not args.no_prefetch and not args.solid_ratio_prefetch:
        #     global PREFETCH_RATIO
        #     PREFETCH_RATIO = init_prefetch_ratio(folder_path=targetdir)
        #     print(f'-> Init PREFETCH_RATIO done, PREFETCH_RATIO length={len(PREFETCH_RATIO)}')
    
    try:
        # print("warmup - generate")
        # output_ids = model.generate(
        #     warmup_inputs, max_new_tokens=4, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        if logits_bool and not args.perplexity:
            results = []
            input_path = args.input_path
            dirname, filename_ext = os.path.dirname(input_path), os.path.basename(input_path)
            outdirname = os.path.join(os.path.dirname(dirname), 'output')
            filename, ext = os.path.splitext(filename_ext)
            modelname = args.model.split('/')[-1]
            output_path = os.path.join(outdirname, filename+f'-{modelname}-full{ext}')
            print(f'input_path = {input_path}, output_path = {output_path}')
            
            if policy.generate_mapping_list:
                mapping_dct = {}
            
            requests = []
            with open(input_path, 'r') as f:
                for line in f:
                    if line.strip() != '':
                        requests.append(json.loads(line))
            np.random.seed(2024)
            # np.random.seed(2025)
            # np.random.seed(2026)
            np.random.shuffle(requests)
            requests=requests[:]
            request_index = 0
            for request in requests:
                # print(request)
                if 'suffix' not in request:
                    print(f'ERROR: Please preprocessing dataset using add_suffix_item.py (see in README-wj.md)')
                    sys.exit()
                result = {}

                # 取出对应样本预取比例
                global CURRENT_SAMPLE_PREFETCH_RATIO
                if PREFETCH_RATIO is not None:
                    CURRENT_SAMPLE_PREFETCH_RATIO = PREFETCH_RATIO[request_index]
                request_index += 1
                # print(f"CURRENT_SAMPLE_PREFETCH_RATIO = {CURRENT_SAMPLE_PREFETCH_RATIO}")
                # ss
                # request = requests[1]
                prompt = request['prompt']

                # prompt = ['Question: When boiling butter, when it\'s ready, you can\nAnswer: Pour it into a jar\n\nQuestion: To permanently attach metal legs to a chair, you can\nAnswer: Weld the metal together to get it to stay firmly in place\n\nQuestion: how do you indent something?\nAnswer: leave a space before starting the writing\n\nQuestion: how do you shake something?\nAnswer: move it up and down and side to side quickly.\n\nQuestion: Clean tires\nAnswer: Pour water, scrape off caked on dirt. Use a steel wool to clean out crevices and narrow spaces.\n\nQuestion: how do you taste something?\nAnswer: place it in your mouth to taste.\n\nQuestion: To create a makeshift ice pack,\nAnswer: take a sponge and soak it in water. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.\n\nQuestion: What should I use as a stain on a wooden bowl I\'ve just made.\nAnswer: You should coat the wooden bowl with a butcher block oil & finish per manufacturer directions.\n\nQuestion: How to boil eggs.\nAnswer: Place your eggs in a pot and cover with cold water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.\n\nQuestion: Remove seeds from  strawberries\nAnswer: Blend the strawberries, pour the mixture through a fine-mesh strainer with a bowl underneath to catch the pulps and strain out the seeds']
                
                input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
                # print(input_ids.shape)
                prompt_len_real = input_ids.shape[1]
                # # 动态修改 suffix 的长度
                suffix_ids = tokenizer(request['suffix'], add_special_tokens=False, return_tensors='pt').input_ids
                suffix_len_real = suffix_ids.shape[-1]
                # 增大 suffix 长度
                # suffix_mul = max(1, (prompt_len_real - suffix_len_real) * args.padding_mul // suffix_len_real)
                suffix_mul = args.suffix_mul
                _max_len = (prompt_len_real - suffix_len_real) * args.padding_mul + suffix_len_real * suffix_mul
                input_ids = tokenizer(prompt,padding="max_length",
                          max_length=(prompt_len_real - suffix_len_real) * args.padding_mul + suffix_len_real * suffix_mul, add_special_tokens=False, return_tensors='pt').input_ids
                # 这里是将填充字符填充到开头
                # suffix_ids = tokenizer(request['suffix'],padding="max_length",
                        #   max_length=67, add_special_tokens=False, return_tensors='pt').input_ids

                # policy.suffix_len = suffix_ids.shape[-1]
                policy.suffix_len = suffix_len_real * suffix_mul
                # print(f"input_ids[0:(prompt_len_real - suffix_len_real) * (args.padding_mul - 1)] = {input_ids[:,0:(prompt_len_real - suffix_len_real) * (args.padding_mul - 1)]}")
                # print(f"input_ids[-suffix_len_real:] = {input_ids[:,-suffix_len_real:]}")
                
                # input_ids = torch.cat([
                #     input_ids[:, 0:prompt_len_real-policy.suffix_len],           # 前缀部分
                #     input_ids[:, prompt_len_real:],                              # 后缀之后的部分  
                #     input_ids[:, prompt_len_real-policy.suffix_len:prompt_len_real]  # 后缀部分
                # ], dim=1)
                # print(f"prefix = {input_ids[:, (prompt_len_real - suffix_len_real) * (args.padding_mul - 1) + suffix_len_real * (suffix_mul - 1):-suffix_len_real]}")
                # print(f"suffix = {input_ids[:,-suffix_len_real:]}")
                input_ids = torch.cat([
                    input_ids[:,(prompt_len_real - suffix_len_real) * (args.padding_mul - 1) + suffix_len_real * (suffix_mul - 1):-suffix_len_real], # 真实的 prefix 部分
                    input_ids[:,0:(prompt_len_real - suffix_len_real) * (args.padding_mul - 1)], # prefix 填充部分
                    # input_ids[prompt_len_real-suffix_len_real:prompt_len_real]]) # suffix部分
                    input_ids[:,-suffix_len_real:], # 真实 suffix 部分
                    input_ids[:,(prompt_len_real - suffix_len_real) * (args.padding_mul - 1):(prompt_len_real - suffix_len_real) * (args.padding_mul - 1) + suffix_len_real * (suffix_mul - 1)]
                    ], dim = 1) # suffix 填充部分
                # print(f"input2 = {input_ids}")
                # print(f'padding input shape:{input_ids.shape}')

                # print(f'policy.suffix_len = {policy.suffix_len}, prefix_len = {input_ids.shape[1] - policy.suffix_len}')
                with open(get_timelog_filename(args), 'a') as f:
                    f.write(f'suffix_len = {policy.suffix_len}, prefix_len = {input_ids.shape[1] - policy.suffix_len}\n')
                
                request.pop('suffix')
                result['request'] = request

                if policy.generate_mapping_list:
                    mapping_2dtensor, logits = model.generate(
                        input_ids, max_new_tokens=args.gen_len,
                        debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
                else:
                    logits = model.generate(
                        input_ids, max_new_tokens=args.gen_len,
                        debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
                # with open("./time_logs/copa_spec_7_request.txt", 'a') as f:
                #     f.write(f"request_time = {request_time:.8f}s\n")

                logits = model.logits_val.log_softmax(dim=-1)
                # print(logits,logits.shape)
                values, indices = logits.squeeze(0).topk(dim=-1, k=1)
                tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
                gold_indices = input_ids[:, 1:] # skip first
                gold_indices = gold_indices.to('cuda:0')
                logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().tolist()
                top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
                # top_logprobs_cleaned = []
                # for item in top_logprobs[1:]:  # 忽略第一个 None
                #     cleaned_item = {tokenizer.convert_tokens_to_string([token]): prob for token, prob in item.items()}
                #     top_logprobs_cleaned.append(cleaned_item)
                # top_logprobs_cleaned = []
                # for item in top_logprobs[1:]:  # 忽略第一个 None
                #     cleaned_item = {tokenizer.convert_tokens_to_string([token]): prob for token, prob in item.items()}
                #     top_logprobs_cleaned.append(cleaned_item)
                result['result'] = {
                    "choices": [
                        {
                            "text": prompt, 
                            "logprobs": {
                                "tokens": tokens, 
                                "token_logprobs": logprobs, 
                                "top_logprobs": top_logprobs, 
                                "text_offset": []
                            }, 
                            "finish_reason": "length"
                        }
                    ], 
                    "request_time": {
                        "batch_time": 0, 
                        "batch_size": 1}
                }
                # print(result)
                # print(top_logprobs_cleaned)
                # sys.exit()
                results.append(result)
                
                if policy.generate_mapping_list:
                    prefix_hash = str(hash(tuple(input_ids[0][:-policy.suffix_len].tolist())))
                    mapping_dct[prefix_hash] = mapping_2dtensor
                    # print(f'request prefix hash: {prefix_hash}, mapping_2dtensor.shape: {mapping_2dtensor.shape}')
            
            # 所有请求都完成了 generate_mapping_list，结果保存在字典 mapping_dct 中，key是 prefix_hash，value 是一个 2d tensor，行是层数，列是token数
            # 保存为 modelname-datasetname-mul.pt
            if policy.generate_mapping_list:
                # dirname, filename_ext = os.path.dirname(input_path), os.path.basename(input_path)
                # outdirname = os.path.join(os.path.dirname(dirname), 'output') # fewshots_datasets/output
                
                # datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
                # model_name = args.model.split('/')[-1]
                # mapping_save_fname = f"{model_name}-{datasetname}_{args.padding_mul}.pt"
                # mapping_save_path = os.path.join(outdirname, mapping_save_fname)
                
                datasetname, _ = os.path.splitext(os.path.basename(args.input_path))
                model_name = os.path.basename(args.model)
                if args.padding_mul == 1:
                    targetdir = f'./cache/prefix/{model_name}/{datasetname}'
                else:
                    targetdir = f'./cache/prefix/{model_name}/{datasetname}_{args.padding_mul}'
                mapping_save_fname = f"{model_name}-{datasetname}_{args.padding_mul}.pt"
                mapping_save_path = os.path.join(targetdir, mapping_save_fname)

                torch.save(mapping_dct, mapping_save_path)
                print(f'-> mapping dict saved into {mapping_save_path}')
                sys.exit(0)
            
                
            # 把 logs 下面的文件拷贝到 output_file 中存留一份，用于后续数据分析
            output_base_ext = os.path.basename(output_path)
            output_base, _ = os.path.splitext(output_base_ext)
            backup_base_ext = get_backup_filepath(output_base, args)
            backup_filepath = os.path.join(outdirname, backup_base_ext)
            if args.input_path and args.logits:
                shutil.copy(logtrace_filename, backup_filepath)
            
            # 推理结果写入 output_path
            # with open(output_path, 'w') as f:
            #     for result in results:
            #         f.write(json.dumps(result) + '\n')
            # sys.exit()
            # sys.exit(0)
        elif logits_bool and args.perplexity:
            ppl=[]
            print('evaluate perplexity')
            input_path = args.input_path
            requests = []
            with open(input_path, 'r') as f:
                for line in f:
                    if line.strip() != '':
                        requests.append(json.loads(line))
            for i,request in enumerate(requests):
                if 'answers' not in request:
                    print(f'ERROR: This task is not for evaluating perplexity')
                    sys.exit()
                # ss
                request = requests[103]
                prompt = request['prompt']
                suffix_ids = tokenizer(request['suffix'], add_special_tokens=False, return_tensors='pt', #padding="max_length",
                          #max_length=16384,
                          ).input_ids

                input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
                policy.suffix_len = suffix_ids.shape[-1]
                labels = input_ids
                print(f'policy.suffix_len = {policy.suffix_len}, prefix_len = {input_ids.shape[1] - policy.suffix_len}')
                # 2048为模型最大输入长度
                if labels.shape[1]>=2048:
                    continue
                #    input_ids = input_ids[:,:2048]
                print(labels.shape)
                print(i)
                # # 动态修改 suffix 的长度
                
        
                
                # request.pop('suffix')
                
                logits = model.generate(
                    input_ids, max_new_tokens=args.gen_len,
                    debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
                out_logits = model.logits_val
                # sys.exit()
                # print(out_logits, out_logits.shape)
                # print(labels, labels.shape)
                labels = labels.to(out_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                from torch.nn import CrossEntropyLoss
                loss_fct = CrossEntropyLoss()
                # print('vo_size', shift_logits.size(-1))
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # print(loss, loss.shape)
                perplexity = torch.exp(loss)
                print(perplexity)
                ppl.append(perplexity)
            print('ppl_per_request',ppl)
            # sys.exit()  
            sys.exit()
        elif args.rouge: # sys_prompt
            results = []
            input_path = args.input_path
            dirname, filename_ext = os.path.dirname(input_path), os.path.basename(input_path)
            outdirname = os.path.join(os.path.dirname(dirname), 'output')
            filename, ext = os.path.splitext(filename_ext)
            modelname = args.model.split('/')[-1]
            output_path = os.path.join(outdirname, filename+f'-{modelname}-generate.pt')
            print(f'input_path = {input_path}, output_path = {output_path}')
            
            requests = []
            with open(input_path, 'r') as f:
                for line in f:
                    if line.strip() != '':
                        requests.append(json.loads(line))
            for request in requests:
                # request = requests[0]            
                prompt = request['prompt']
                if 'suffix' not in request:
                    print(f'ERROR: Please preprocessing dataset using add_suffix_item.py (see in README-wj.md)')
                    sys.exit()
                
                # prompt = ['Question: When boiling butter, when it\'s ready, you can\nAnswer: Pour it into a jar\n\nQuestion: To permanently attach metal legs to a chair, you can\nAnswer: Weld the metal together to get it to stay firmly in place\n\nQuestion: how do you indent something?\nAnswer: leave a space before starting the writing\n\nQuestion: how do you shake something?\nAnswer: move it up and down and side to side quickly.\n\nQuestion: Clean tires\nAnswer: Pour water, scrape off caked on dirt. Use a steel wool to clean out crevices and narrow spaces.\n\nQuestion: how do you taste something?\nAnswer: place it in your mouth to taste.\n\nQuestion: To create a makeshift ice pack,\nAnswer: take a sponge and soak it in water. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.\n\nQuestion: What should I use as a stain on a wooden bowl I\'ve just made.\nAnswer: You should coat the wooden bowl with a butcher block oil & finish per manufacturer directions.\n\nQuestion: How to boil eggs.\nAnswer: Place your eggs in a pot and cover with cold water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.\n\nQuestion: Remove seeds from  strawberries\nAnswer: Blend the strawberries, pour the mixture through a fine-mesh strainer with a bowl underneath to catch the pulps and strain out the seeds']
                
                input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
                print(input_ids.shape)
                # # 动态修改 suffix 的长度
                suffix_ids = tokenizer(request['suffix'], add_special_tokens=False, return_tensors='pt').input_ids
                policy.suffix_len = suffix_ids.shape[-1]
                print(f'policy.suffix_len = {policy.suffix_len}, prefix_len = {input_ids.shape[1] - policy.suffix_len}')
                output_ids = model.generate(
                    input_ids, max_new_tokens=args.gen_len,
                    debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                prompt_len = len(prompt)
                answer = outputs[0][prompt_len:]
                # 写进结果文件夹里
                results.append(answer)
                print(type(answer))
                #sys.exit()
            
            torch.save(results, output_path)
            sys.exit()
        else: # squadS
            results = []
            input_path = args.input_path
            dirname, filename_ext = os.path.dirname(input_path), os.path.basename(input_path)
            outdirname = os.path.join(os.path.dirname(dirname), 'output')
            filename, ext = os.path.splitext(filename_ext)
            modelname = args.model.split('/')[-1]
            output_path = os.path.join(outdirname, filename+f'-{modelname}-full{ext}')
            print(f'input_path = {input_path}, output_path = {output_path}')
            
            requests = []
            f1 = 0
            f1 = 0
            with open(input_path, 'r') as f:
                for line in f:
                    if line.strip() != '':
                        requests.append(json.loads(line))
            requests = requests[:1000]
            for request in requests:
                # request = requests[0]            
                # request = requests[0]            
                prompt = request['prompt']
                if 'suffix' not in request:
                    print(f'ERROR: Please preprocessing dataset using add_suffix_item.py (see in README-wj.md)')
                    sys.exit()
                result = {}
                
                # prompt = ['Question: When boiling butter, when it\'s ready, you can\nAnswer: Pour it into a jar\n\nQuestion: To permanently attach metal legs to a chair, you can\nAnswer: Weld the metal together to get it to stay firmly in place\n\nQuestion: how do you indent something?\nAnswer: leave a space before starting the writing\n\nQuestion: how do you shake something?\nAnswer: move it up and down and side to side quickly.\n\nQuestion: Clean tires\nAnswer: Pour water, scrape off caked on dirt. Use a steel wool to clean out crevices and narrow spaces.\n\nQuestion: how do you taste something?\nAnswer: place it in your mouth to taste.\n\nQuestion: To create a makeshift ice pack,\nAnswer: take a sponge and soak it in water. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.\n\nQuestion: What should I use as a stain on a wooden bowl I\'ve just made.\nAnswer: You should coat the wooden bowl with a butcher block oil & finish per manufacturer directions.\n\nQuestion: How to boil eggs.\nAnswer: Place your eggs in a pot and cover with cold water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.\n\nQuestion: Remove seeds from  strawberries\nAnswer: Blend the strawberries, pour the mixture through a fine-mesh strainer with a bowl underneath to catch the pulps and strain out the seeds']
                
                input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
                print(input_ids.shape)
                # # 动态修改 suffix 的长度
                suffix_ids = tokenizer(request['suffix'], add_special_tokens=False, return_tensors='pt').input_ids
                policy.suffix_len = suffix_ids.shape[-1]
                print(f'policy.suffix_len = {policy.suffix_len}, prefix_len = {input_ids.shape[1] - policy.suffix_len}')
                output_ids = model.generate(
                    input_ids, max_new_tokens=args.gen_len,
                    debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                prompt_len = len(prompt)
                answer = outputs[0][prompt_len:]
                gold_answers = request['answers']['text']
                if len(outputs)!=1:
                    print('output error')
                    sys.exit()
                print(request['suffix'])
                print(request['suffix'])
                print(answer)
                print(gold_answers)
                f1_score = max(compute_f1(answer, gold_answer) for gold_answer in gold_answers)
                print('f1:', f1_score)
                f1 += f1_score
            f1 = f1/1000*100 # 需要修改
            print('f1 score', f1)
            # 只计算数据集的前1000个，因为数据集太长。
            sele_id_path = os.path.dirname(__file__) + f'/logs_zrd/{datasetname}-6.7b/{datasetname}_{args.sele_percent}_sele_id.pt'
            if '30b' in args.model:
                sele_id_path = os.path.dirname(__file__) + f'/logs_zrd/{datasetname}-30b/{datasetname}_{args.sele_percent}_sele_id.pt'
            if args.motiv2 and not args.samekv:
                model.save_sele_id_dict(sele_id_path)
            output_base_ext = os.path.basename(output_path)
            output_base, _ = os.path.splitext(output_base_ext)
            backup_base_ext = get_backup_filepath(output_base, args)
            backup_filepath = os.path.join(outdirname, backup_base_ext)
            if args.input_path and args.logits:
                shutil.copy(logtrace_filename, backup_filepath)
            sys.exit()  
            f1 += f1_score
            f1 = f1/1000*100 # 需要修改
            print('f1 score', f1)
            # 只计算数据集的前1000个，因为数据集太长。
            sele_id_path = os.path.dirname(__file__) + f'/logs_zrd/{datasetname}-6.7b/{datasetname}_{args.sele_percent}_sele_id.pt'
            if '30b' in args.model:
                sele_id_path = os.path.dirname(__file__) + f'/logs_zrd/{datasetname}-30b/{datasetname}_{args.sele_percent}_sele_id.pt'
            if args.motiv2 and not args.samekv:
                model.save_sele_id_dict(sele_id_path)
            output_base_ext = os.path.basename(output_path)
            output_base, _ = os.path.splitext(output_base_ext)
            backup_base_ext = get_backup_filepath(output_base, args)
            backup_filepath = os.path.join(outdirname, backup_base_ext)
            if args.input_path and args.logits:
                shutil.copy(logtrace_filename, backup_filepath)
            sys.exit()  
            # output_ids = model.generate(
            #     inputs, max_new_tokens=args.gen_len,
            #     debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
            
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    global dt_prefix,dt_all,dt_load,dt_sele,dt_load_head,dt_suffix,dt_tol,dt_key,dt_value,all_ttft, orcale_load_time, orcale_ttft
    global dt_async_prefetch_wait, dt_async_prefetch_submit, async_prefetch_jobs, async_prefetch_tokens
    global async_prefetch_hit_tokens, async_prefetch_miss_tokens
    print(f'-> total time: {dt_tol} seconds.')
    print(f'suffix time:{dt_suffix},all compute:{dt_all},load:{dt_load},sele:{dt_sele},load_head:{dt_load_head}')
    print(f'get key:{dt_key},get value:{dt_value}')
    print(
        f'async prefetch jobs:{async_prefetch_jobs},'
        f'submit_time:{dt_async_prefetch_submit},'
        f'wait_time:{dt_async_prefetch_wait},'
        f'prefetched_tokens:{async_prefetch_tokens},'
        f'hit_tokens:{async_prefetch_hit_tokens},'
        f'miss_tokens:{async_prefetch_miss_tokens}'
    )
    print(f'orcale load time:{orcale_load_time}')
    print(f'orcale_time:', sum(orcale_ttft))
    if pcache is not None:
        pcache.get_hit_rate()
        pcache.get_time()
    if len(all_ttft)>0:
        print(f'ttft_sum:{sum(all_ttft)}')
        p99 = np.percentile(all_ttft,99)
        print(f'p99:{p99}')
    if len(orcale_ttft)>0:
        p99 = np.percentile(orcale_ttft,99)
        print(f'orcale p99:{p99}')
    # print(f'chunk_hit:(prefix_id,layer_id,chunk_id)->(chunk hit num,token hit num)')
    # pcache.get_chunk_hit()
    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path and "output_ids" in locals():
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]: # 打印第一个和最后一个的请求和生成结果
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)
    
    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)

    print('run_flexgen finished.')


def compute_f1(prediction, reference):
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    # 计算交集词的数量
    common_tokens = set(pred_tokens) & set(ref_tokens)
    num_common = len(common_tokens)

    # 计算 Precision 和 Recall
    if len(pred_tokens) > 0:
        precision = num_common / len(pred_tokens)
    else:
        precision = 0.0

    if len(ref_tokens) > 0:
        recall = num_common / len(ref_tokens)
    else:
        recall = 0.0

    # 计算 F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return f1


def get_backup_filepath(outputbase, args):
    # outputbase += '-eval-result'
    if args.full_load:
        outputbase += '-full_load'
        if args.sele_inf:
            if args.full_load_sele_inf_by_accum:
                accum_percents = '-'.join([str(v) for v in args.accum_percent])
                outputbase += '-sele_inf-by_accum'
                outputbase += f'_{accum_percents}-prekv-trace'
            else:
                sele_percents = '-'.join([str(v) for v in args.sele_percent])
                outputbase += f'-sele_inf-{sele_percents}-prekv-trace'
        else:
            outputbase += '-prekv-trace'
    
    if args.sele_load:
        outputbase += '-sele_load'
        if not args.sele_load_by_percent:
            accum_percents = '-'.join([str(v) for v in args.accum_percent])
            outputbase += '-by_accum'
            outputbase += f'_{accum_percents}_{args.sim_thred}-prekv-trace'
        else:
            sele_percents = '-'.join([str(v) for v in args.sele_percent])
            outputbase += '-by_percent'
            outputbase += f'_{sele_percents}_{args.sim_thred}-prekv-trace'
    
    outputbase += '.log'
    
    return outputbase

def add_parser_arguments(parser):
    # 加参数用的
    parser.add_argument("--model", type=str, default="facebook/opt-2.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)  # 一个 batch 有几个 query
    parser.add_argument("--num-gpu-batches", type=int, default=1) # 加载一个权重后算几个 batch
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0], # 默认权重、KV、激活都放 gpu 中
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")

    # heavy hitter pruning
    parser.add_argument("--hh-ratio", type=float, default=1,
                        help="ratio of the prompt seq length")
    parser.add_argument("--hh-all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--hh-long-seq", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?', const=True, default=True)
    
    # 
    parser.add_argument("--no-redir", action="store_true", help="no redirect output of terminal into log file")
    parser.add_argument("--ret-topk-indices", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--prefix-dump", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--prefix-aware-inf", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--sele-load", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--full-only-key-load", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--sele-inf", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--suffix-len", type=int, default=100)
    parser.add_argument("--sele-percent", nargs="+", type=int, default=[100], help="selected kv ration. If multiple values then it will be distributed to each layer evenly.")
    parser.add_argument("--accum-percent", nargs="+", type=int, default=[50], help="selected kv with accumulative attention target. If multiple values then it will be distributed to each layer evenly.")
    parser.add_argument("--sele-head", nargs="+", type=int, default=[0, 1, 2], help="the IDs of heads for full load")
    parser.add_argument("--sim-thred", type=float, default=0.5, help="the similarity threashold to determin whether all kvs in heads will be fetched for prefix kv")
    parser.add_argument("--sele-load-by-percent", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--full-load", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--full-load-sele-inf-by-accum", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--no-cache", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--fill-keys-zero", type=str2bool, nargs='?', const=True, default=False, help="fill remaining keys with zeros when sele_load")
    parser.add_argument("--generate-mapping-list", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--perplexity", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--rouge", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='opt')
    
    parser.add_argument("--prefetch", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--suffix-comp", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--gpu-size",type=int, default=3072, help="GPU mem used for pcache(MB)")
    parser.add_argument("--cpu-size",type=int, default=6144, help="CPU mem used for pcache(MB)")
    parser.add_argument("--chunk-size",type=int, default=64, help="token num of each chunk")
    parser.add_argument("--cache-type", type=str, default="LRU")
    parser.add_argument("--disk-type", type=str, default="Chunk")
    parser.add_argument("--padding-mul", type=int, default=1)
    parser.add_argument("--suffix-mul", type=int, default=1)
    parser.add_argument("--recompute", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--reorder", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--no-prefetch", type=str2bool, nargs='?', const=True, default=False) # 默认开启预取，加了这个参数退回到 IMPRESS
    parser.add_argument("--prefetch-ratio", type=int, default=0)
    parser.add_argument("--solid-ratio-prefetch", type=str2bool, nargs='?', const=True, default=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    # print(f'args: {args}')
    # 保存原始的 sys.stdout 和 sys.stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # 把输出到终端的内容重定向到 logfile
    if args.log_file == "auto":
        filename = get_filename(args) + 'overlap_ratio' + ".log"
    else:
        filename = args.log_file
    # print(f'-> log file: {filename}')
    if args.no_redir:
        print(f'-> log time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        run_flexgen(args, filename)
    else:
        with open(filename, 'w') as out_f:
            # 将 sys.stdout 和 sys.stderr 重定向到文件
            sys.stdout = out_f
            sys.stderr = out_f
            
            assert len(args.percent) == 6
            assert int(args.prompt_len * args.hh_ratio) > 0, "Please increase the ratio to keep at least one token"
            try:
                print(f'-> log time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                run_flexgen(args, filename)
            finally:
                # 恢复原始的 sys.stdout 和 sys.stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
