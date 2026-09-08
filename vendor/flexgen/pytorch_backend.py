"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os, sys
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

# import cupy as cp
# from cupyx import jit
import torch
import torch.nn.functional as F
import numpy as np

import torch
import math

from utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    # from flexgen import compression
    import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            # print(f"Loading from np file {filename} to {self.device.device_type}")
            self.load_from_np(np.load(filename))
            # print(f"filename = {filename}")

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops
        
        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}
        self.layer = 1
        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self
        
    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
            func = torch.zeros
        else:
            func = torch.empty
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = func(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy, hh_k=None):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            if policy.hh_all:
                max_seq_len = hh_k * 2
            else:
                max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                acc = self.allocate(shape[:-1], np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache, acc))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def slice_attention_mask(self, mask, cnt):
        return TorchTensor.create_from_torch(mask.data[:, -cnt:], self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate, hh_long_seq=False):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        if hh_long_seq:
            return TorchTensor.create_from_torch(token_embed, self)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        # print("input_embed", token_embed.shape, token_embed)
        # print("pos_embed", pos_embed.shape, pos_embed)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature, record_logits = False, oplm = None):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        # print('logits',logits, logits.shape)
        logits_softmax = logits.log_softmax(dim=-1)
        # print('logits_softmax', logits_softmax)
        # sys.exit()
        last_token_logits = logits[:,-1,:]
        if record_logits:
            if do_sample and not temperature < 1e-5:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                ids = torch.multinomial(probs, num_samples=1)
            else:
                ids = last_token_logits.argmax(dim=1, keepdim=True)
            oplm.logits_val = logits
            # 记录logits
            return TorchTensor.create_from_torch(ids, self)

        else:
            if do_sample and not temperature < 1e-5:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                ids = torch.multinomial(probs, num_samples=1)
            else:
                ids = last_token_logits.argmax(dim=1, keepdim=True)
            return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy, hh_k=None, hh_all=False):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        if not policy.prefix_aware_inf:
            if hh_all:
                shape = (hh_k * 2, gpu_batch_size * num_head, hidden_size // num_head)
            elif hh_k * 2 < prompt_len:
                shape = (hh_k * 2 + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
            else:
                shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        else:
            shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        acc = self.allocate(shape[:-1], np.float16, pin_memory=pin_memory)
        return k_cache, v_cache, acc

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)
        # print(f'k[0, 0, :, :]={k[0, 0, :, :]}')
        # print(f'k_part[0,0,:,:]={F.linear(hidden[:, :10, :], w_k.data, bias=b_k.data).view(b, 10, n_head, head_dim)[0,0,:,:]}')
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        #print(f'attn_weights.shape={attn_weights.shape}')
        
        # # 判断其中一个 head 是否为对称矩阵 —— 本身就不应该是对称的
        # # 遍历每个 s * s 的矩阵
        # for i in range(attn_weights.size(0)):
        #     matrix = attn_weights[i] # s * s
        #     is_symmetric = torch.equal(matrix, matrix.t())
        #     print(is_symmetric)
        
        # # 计算每个 head 的 attention 中有几个非零值
        # for i in range(attn_weights.size(0)):
        #     matrix = attn_weights[i] # s * s
        #     print(torch.count_nonzero(matrix, dim=1))
        
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data, bias=b_out.data)
        
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        # select the heavy hitters and recent tokens
        if hh_k is not None:
            if not ret_topk_indices:
                k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
                # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
                # print(f'k_ori={k}')
                # print(f'v_ori={v}')
                # print(f'acc_ori={acc}')
            else:
                k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
                # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)
            
        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v, acc, topk_indices
        return TorchTensor.create_from_torch(value, self), k, v, acc

    def mha_with_prefixkv_suffix(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            prefix_k=None, prefix_v=None, suffix_len=100):
        """Multi-head attention only with suffix (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        # b -> batch size, s -> sequence length, h -> hidden size
        head_dim = h // n_head
        scaling = head_dim ** -0.5 # 这就是根号d，只不过在分母上，一会直接变成乘法
        # inputs.data 是输入张量，形状一般为 (b, s, h)
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        # print(suffix_len,b,s,h,attention_mask.shape)
        # print(f'q.shape = {q.shape}')
        # 修改 hidden
        hidden = hidden[:, -suffix_len:, :]
        # attention_mask = attention_mask[:, -suffix_len:]
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, suffix_len, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)
        v = v.view(b, suffix_len, n_head, head_dim)
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, suffix_len, head_dim)
        # q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, suffix_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, suffix_len)
        # shape: (b * n_head, suffix_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, suffix_len, head_dim)
        
        # 拼接 prefix kv
        k = torch.cat((prefix_k.permute(1, 2, 0), k), dim=2)
        v = torch.cat((prefix_v.permute(1, 0, 2), v), dim=1)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        causal_mask = causal_mask[:,:,-suffix_len:,:]
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, suffix_len, s)
        # attn_weights = attn_weights.view(b, n_head, s, s)
        # print(mask.shape,attn_weights.shape)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, suffix_len, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # print(f'attn_weights.shape={attn_weights.shape}')
        
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, suffix_len, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, suffix_len, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data[:, -suffix_len:, :])

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        # select the heavy hitters and recent tokens
        # if hh_k is not None:
        #     if not ret_topk_indices:
        #         k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
        #         # print(f'k={k}')
        #         # print(f'v={v}')
        #         # print(f'acc={acc}')
        #     else:
        #         k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        acc = torch.empty((s,n_head),dtype=hidden.dtype,device='cuda:0')
        k = torch.empty((s,n_head,head_dim),dtype=hidden.dtype,device='cuda:0')
        v = torch.empty((s,n_head,head_dim),dtype=hidden.dtype,device='cuda:0')
        value = torch.empty((1,s,h),dtype=hidden.dtype,device='cuda:0')
        acc = TorchTensor.create_from_torch(acc, self)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            # acc = TorchTensor.create_from_torch(acc, self)
        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v,acc
        return TorchTensor.create_from_torch(value, self), k, v,acc
    
    def mha_prefixkv(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            suffix_len=100):
        """Multi-head attention only with suffix (prefill phase)."""
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        s = s - suffix_len
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        # 减去均值，除以标准差，然后应用权重缩放和bias进行偏移调整。这俩参数是训练得到的
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # 只计算suffix部分的qkv，利用prefix部分的缓存
        hidden = hidden[:, 0:-suffix_len, :]

        # shape: (b, s, h)
        # 这里是指矩阵的线性变换，X * W转置 + bias
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling # -》
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        # pytorch 的 view 方法，直接改变张量的形状，而不改变数据
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)
        # print(f'k[0, 0, :, :]={k[0, 0, :, :]}')
        # print(f'k_part[0,0,:,:]={F.linear(hidden[:, :10, :], w_k.data, bias=b_k.data).view(b, 10, n_head, head_dim)[0,0,:,:]}')
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data[:,0:s].view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        #print(f'attn_weights.shape={attn_weights.shape}')
        
        # # 判断其中一个 head 是否为对称矩阵 —— 本身就不应该是对称的
        # # 遍历每个 s * s 的矩阵
        # for i in range(attn_weights.size(0)):
        #     matrix = attn_weights[i] # s * s
        #     is_symmetric = torch.equal(matrix, matrix.t())
        #     print(is_symmetric)
        
        # # 计算每个 head 的 attention 中有几个非零值
        # for i in range(attn_weights.size(0)):
        #     matrix = attn_weights[i] # s * s
        #     print(torch.count_nonzero(matrix, dim=1))
        
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data, bias=b_out.data)
        
        value.add_(inputs.data[:, 0:-suffix_len, :])

        # if donate[0]: inputs.delete()
        # if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        # select the heavy hitters and recent tokens
        # if hh_k is not None:
        #     if not ret_topk_indices:
        #         k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
        #         # print(f'k_ori={k}')
        #         # print(f'v_ori={v}')
        #         # print(f'acc_ori={acc}')
        #     else:
        #         k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            # acc = TorchTensor.create_from_torch(acc, self)
            
        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v
        return TorchTensor.create_from_torch(value, self), k, v
    
    def mha_with_prefixkv(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            prefix_k=None, prefix_v=None, suffix_len=100):
        """Multi-head attention only with suffix (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        # print(f'q.shape = {q.shape}')
        # 修改 hidden
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)
        v = v.view(b, suffix_len, n_head, head_dim)
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, suffix_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, suffix_len)
        # shape: (b * n_head, suffix_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, suffix_len, head_dim)
        
        # 拼接 prefix kv
        k = torch.cat((prefix_k.permute(1, 2, 0), k), dim=2)
        v = torch.cat((prefix_v.permute(1, 0, 2), v), dim=1)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        # print(f'attn_weights.shape={attn_weights.shape}')
        
        self.mask_with_zero_cols(attn_weights, 1-100*0.01, suffix_len)

        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        aggr_attn = torch.sum(attn_weights, 1)
        aggr_attn = aggr_attn.transpose(0, 1)
        acc = aggr_attn
        
        # # select the heavy hitters and recent tokens
        # if hh_k is not None:
        #     if not ret_topk_indices:
        #         k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
        #         # print(f'k={k}')
        #         # print(f'v={v}')
        #         # print(f'acc={acc}')
        #     else:
        #         k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)
            
        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v, acc, topk_indices
        return TorchTensor.create_from_torch(value, self), k, v, acc

    def mask_with_zero(self, attn_3d, del_percent):
        """ deprecated. """
        bnhead = attn_3d.shape[0]
        for i in range(bnhead):
            attn = attn_3d[i]
            # 确保是上三角矩阵
            assert attn.shape[0] == attn.shape[1], "矩阵必须是方阵"
            s = attn.shape[0]
            
            # 对每一行进行处理
            for i in range(s):
                row = attn[i, :]
                non_zero_elements = row[row > 0]
                if len(non_zero_elements) > 0:
                    # 将非零元素转换为float32以便计算quantile
                    non_zero_elements_float32 = non_zero_elements.to(torch.float32)
                    # 计算 del_percent 位置的值
                    threshold = torch.quantile(non_zero_elements_float32, del_percent)
                    # 将小于等于阈值的元素设为0
                    row[row < threshold] = 0
        
        return attn_3d

    def mask_with_zero_cols_motiv2(self, attn_weights, del_percent, suffix_len, ):
        # suffix 全保留，prefix 中保留 1-del_percent 比例
        aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        prefix_len = aggr_attn.shape[1]-suffix_len
        del_num = int(prefix_len * del_percent)
        sele_num = prefix_len - del_num
        _, del_indcies = torch.topk(aggr_attn[:, :prefix_len], del_num, dim=1, largest=False) # 每个 head 中选择要删除的 token id
        _, sele_indices = torch.topk(aggr_attn[:, :prefix_len], sele_num, dim=1, largest=True) # 每个 head 中选择要保留的 token id
        #print('del_indices:', del_indcies, del_indcies.shape)
        # print('sele_indices:', sele_indices, sele_indices.shape)
        # 将张量移到 CPU 并转为 NumPy
        # np_array = sele_indices.cpu().numpy()
        # np.set_printoptions(threshold=np.inf, linewidth=10000)
        # print('sele_incices:')
        # print(np_array)
        # print('Shape:', np_array.shape)

         # === 新增：处理并保存 token ID 到文件 ===
        # 创建输出目录
        # output_dir = "./token_logs"
        # os.makedirs(output_dir, exist_ok=True)
        # output_file = os.path.join(output_dir, f"full_load_selected_tokens.txt")
        
        # with open(output_file, 'a') as f:
        #     f.write(f"Selected Tokens per Head: {sele_num}\n")
        #     for head_idx, head_tokens in enumerate(sele_indices.cpu().numpy()):
        #         f.write(f"Head {head_idx}: {','.join(map(str, head_tokens))}\n\n")
        
        # print(f"Selected tokens saved to: {output_file}")
        
        all_indices = torch.arange(prefix_len).unsqueeze(0).expand(aggr_attn.shape[0], prefix_len)

        #print('all_indices:', all_indices.shape)
        all_indices = all_indices.to(self.dev)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask = mask.to(self.dev)
        mask.scatter_(1, del_indcies, False)
        keep_indices = all_indices[mask].view(aggr_attn.shape[0], -1)
        #print('keep_incices:', keep_indices, keep_indices.shape)
        if keep_indices.shape[-1]!=sele_num:
            print('ERROR:检查代码，选中id不对')
            sys.exit(-1)
        
        # 把要丢弃的 prefix 对应的 attn_weights 置零
        attn_weights[torch.arange(aggr_attn.shape[0]).unsqueeze(1), :, del_indcies] = 0
        # 设置打印选项
        # torch.set_printoptions(profile="full")
        
        return attn_weights, del_indcies, keep_indices
        
    def mask_with_zero_cols_samekv(self, attn_weights, del_percent, suffix_len, sele_indices):
        # suffix 全保留，prefix 中保留 1-del_percent 比例
        aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        prefix_len = aggr_attn.shape[1]-suffix_len
        del_num = int(prefix_len * del_percent)
        
        all_indices = torch.arange(prefix_len).unsqueeze(0).expand(aggr_attn.shape[0], prefix_len)
        all_indices = all_indices.to(self.dev)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask = mask.to(self.dev)
        mask.scatter_(1, sele_indices, False)
        del_indcies = all_indices[mask].view(aggr_attn.shape[0], -1)
        # 把要丢弃的 prefix 对应的 attn_weights 置零
        attn_weights[torch.arange(aggr_attn.shape[0]).unsqueeze(1), :, del_indcies] = 0
        if del_indcies.shape[-1]!=del_num:
            print('error:same kv sele_id 形状错误')
            sys.exit(-1)
        # 设置打印选项
        # torch.set_printoptions(profile="full")
        self.layer += 1
        
        return attn_weights, del_indcies

    def mask_with_zero_cols(self, attn_weights, del_percent, suffix_len):
        # suffix 全保留，prefix 中保留 1-del_percent 比例
        aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        prefix_len = aggr_attn.shape[1]-suffix_len
        del_num = int(prefix_len * del_percent)
        _, del_indcies = torch.topk(aggr_attn[:, :prefix_len], del_num, dim=1, largest=False) # 每个 head 中选择要删除的 token id
        # 把要丢弃的 prefix 对应的 attn_weights 置零
        attn_weights[torch.arange(aggr_attn.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        # print('delnum', del_num, del_indcies.shape)
        # 设置打印选项
        # torch.set_printoptions(profile="full")
        return attn_weights, del_indcies

        # # 让 prefix + suffix 中 suffix 保留的 token 数和 prefix 保留的 token 数一样
        # aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        # input_len = aggr_attn.shape[1]
        # recent_len = int((1-del_percent) * input_len)
        # del_num = input_len - 2*recent_len
        # _, del_indcies = torch.topk(aggr_attn[:, :input_len-recent_len], del_num, dim=1, largest=False) # 每个 head 中选择要删除的 token id
        # # reserve_indices = torch.arange(input_len-recent_len, input_len, device='cuda:0').view(1, -1).expand(aggr_attn.shape[0], -1)
        # # del_indcies = torch.cat((del_indcies, reserve_indices), dim=1)
        # print(f'del_indcies.shape = {del_indcies.shape}')
        # attn_weights[torch.arange(aggr_attn.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        # return attn_weights, del_indcies
    
    def mask_with_zero_cols_by_accum(self, attn_weights, del_accum_percent, suffix_len):
        # suffix 全保留，prefix 中保留 1-del_percent 比例
        aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        prefix_len = aggr_attn.shape[1]-suffix_len
        aggr_attn = aggr_attn[:, :prefix_len]
        
        row_sums = torch.sum(aggr_attn, 1) # shape: (b*nhead, 1)
        target_thred = row_sums * del_accum_percent# shape: (b*nhead, 1)
        # 升序排列
        sorted_values, sorted_indices = torch.topk(aggr_attn, prefix_len, dim=1, largest=False)
        # 按行累计求和
        cum_sums = sorted_values.cumsum(dim=1) # shape: (b*nhead, 1)
        # 逐行找出每个head中 累计到 target_attention_threshold 需要的 token 数
        mask = cum_sums >= target_thred.unsqueeze(1) # shape: (b*nhead, s)
        # 计算平均每个 head 达到 target_attention_threshold 所需的 token 数量
        avg_del_num = int((~mask).sum(dim=1).float().mean().item())
        # 根据 sorted_indices 找出每个 head 的最重要的 token indices
        del_indices = sorted_indices[:, :avg_del_num] # shape: (b*nhead, avg_del_num)
        sele_indices = sorted_indices[:, avg_del_num:]
        
        # print(f'sele {prefix_len - avg_del_num} / {prefix_len} = {(1-avg_del_num/prefix_len)*100}% of all tokens')
        
        # 把要丢弃的 prefix 对应的 attn_weights 置零
        attn_weights[torch.arange(aggr_attn.shape[0]).unsqueeze(1), :, del_indices] = 0
        
        return attn_weights, del_indices, sele_indices
    
    def mha_with_sele_percent_prefixkv(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            prefix_k=None, prefix_v=None, suffix_len=100, cur_sele_percent=100, motiv2 = False, samekv=False, sele_indices = None, generate_mapping_list = False):
        # print(f'generate_mapping_list = {generate_mapping_list}')
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        # print(f'q.shape = {q.shape}')
        # 修改 hidden
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)
        v = v.view(b, suffix_len, n_head, head_dim)
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, suffix_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, suffix_len)
        # shape: (b * n_head, suffix_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, suffix_len, head_dim)
        
        # 拼接 prefix kv
        k = torch.cat((prefix_k.permute(1, 2, 0), k), dim=2)
        v = torch.cat((prefix_v.permute(1, 0, 2), v), dim=1)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)
        
        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)

        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # inplace softmax
        attn_weights -= torch.max(attn_weights,dim=2)[0].view(attn_weights.shape[0],attn_weights.shape[1],1)
        attn_weights.exp_()
        sums = torch.sum(attn_weights,dim=2).view(attn_weights.shape[0],attn_weights.shape[1],1)
        attn_weights /= sums
        
        # # 把每行 attn_weights 中最小的 (1-cur_sele_percent) 的值置为 0
        # attn_weights = self.mask_with_zero(attn_weights, 1-cur_sele_percent*0.01)
        
        # 把每列的 attn_weights 累加，找出和最小的 token，把对应列的 attn 值变为0
        if motiv2 and not samekv:
            attn_weights, del_indcies, keep_indices = self.mask_with_zero_cols_motiv2(attn_weights, 1-cur_sele_percent*0.01, suffix_len, )
        elif motiv2 and samekv:
            attn_weights, del_indcies = self.mask_with_zero_cols_samekv(attn_weights, 1-cur_sele_percent*0.01, suffix_len, sele_indices)
        
        else:
            if generate_mapping_list:
                aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
                summed_aggr_attn = torch.sum(aggr_attn, dim=0) # shape: (s, )
                prefix_len = aggr_attn.shape[1]-suffix_len
                _, sorted_indices = torch.topk(summed_aggr_attn[:prefix_len], prefix_len)
                
                # 建立反向索引
                reverse_indices = torch.full_like(sorted_indices, -1)
                reverse_indices[sorted_indices] = torch.arange(len(sorted_indices), device="cuda:0")
                
            attn_weights, del_indcies = self.mask_with_zero_cols(attn_weights, 1-cur_sele_percent*0.01, suffix_len)
            
        '''# 根据 del_indcies 将 k / v 置零，然后重新执行 atten 计算
        k[torch.arange(b*n_head).unsqueeze(1), :, del_indcies] = 0
        v[torch.arange(b*n_head).unsqueeze(1), del_indcies, :] = 0
        attn_weights = torch.bmm(q, k)
        
        attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)'''

        # 根据 del_indcies 将 k / v 置零，然后重新执行 atten 计算
        # k[torch.arange(b*n_head).unsqueeze(1), :, del_indcies] = 0
        v[torch.arange(b*n_head).unsqueeze(1), del_indcies, :] = 0
        attn_weights = torch.bmm(q, k)
        # attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)


        # tmp_attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = -1e4
        # attn_weights = F.softmax(tmp_attn_weights, dim=2, dtype=torch.float32).to(torch.float16)

        
        # shape: (b, n_head, s, head_dim)   
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        aggr_attn = torch.sum(attn_weights, 1)
        aggr_attn = aggr_attn.transpose(0, 1)
        acc = aggr_attn[sele_indices]
        
        # # select the heavy hitters and recent tokens
        # if hh_k is not None:
        #     if not ret_topk_indices:
        #         k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
        #         # print(f'k={k}')
        #         # print(f'v={v}')
        #         # print(f'acc={acc}')
        #     else:
        #         k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)
            
        # print(f'mha shape:acc:{aggr_attn.shape},k:{k.shape},v:{v.shape},value:{value.shape}')
        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v, acc, topk_indices
        elif generate_mapping_list:
            return TorchTensor.create_from_torch(value, self), k, v, acc, reverse_indices
        
        # if motiv2 and not samekv:
            # return TorchTensor.create_from_torch(value, self), k, v, acc, keep_indices
        return TorchTensor.create_from_torch(value, self), k, v, acc

    def mha_with_var_percent_prefixkv(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            prefix_k=None, prefix_v=None, suffix_len=100, cur_accum_percent=100):
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        # print(f'q.shape = {q.shape}')
        # 修改 hidden
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)
        v = v.view(b, suffix_len, n_head, head_dim)
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, suffix_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, suffix_len)
        # shape: (b * n_head, suffix_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, suffix_len, head_dim)
        
        # 拼接 prefix kv
        k = torch.cat((prefix_k.permute(1, 2, 0), k), dim=2)
        v = torch.cat((prefix_v.permute(1, 0, 2), v), dim=1)
        
        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        
        # # 把每行 attn_weights 中最小的 (1-cur_sele_percent) 的值置为 0
        # attn_weights = self.mask_with_zero(attn_weights, 1-cur_sele_percent*0.01)
        
        # 把每列的 attn_weights 累加，找出和最小的 token，把对应列的 attn 值变为0
        attn_weights, del_indcies, sele_indices = self.mask_with_zero_cols_by_accum(attn_weights, 1-cur_accum_percent*0.01, suffix_len)
        
        # 根据 del_indcies 将 k / v 置零，然后重新执行 atten 计算
        k[torch.arange(b*n_head).unsqueeze(1), :, del_indcies] = 0
        v[torch.arange(b*n_head).unsqueeze(1), del_indcies, :] = 0
        attn_weights = torch.bmm(q, k)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        # select the heavy hitters and recent tokens
        # if hh_k is not None:
        #     if not ret_topk_indices:
        #         k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
        #         # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
        #         # print(f'k={k}')
        #         # print(f'v={v}')
        #         # print(f'acc={acc}')
        #     else:
                # k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
                # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        aggr_attn = torch.sum(attn_weights, 1)
        aggr_attn = aggr_attn.transpose(0, 1)
        acc = aggr_attn[sele_indices]
        
        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)
            
        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v, acc, topk_indices
        return TorchTensor.create_from_torch(value, self), k, v, acc

    def mha_with_filled_selected_prefixkv(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            filled_prefix_k=None, filled_prefix_v=None, suffix_len=100, select_kv_tokenid=None, del_tokenids = None):
        """Multi-head attention only with suffix (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        
        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        # 只计算 suffix 的 kv
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)
        v = v.view(b, suffix_len, n_head, head_dim)
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, suffix_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, suffix_len)
        # shape: (b * n_head, suffix_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, suffix_len, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        
        # 拼接 prefix kv
        k = torch.cat((filled_prefix_k.permute(1, 2, 0), k), dim=2)
        v = torch.cat((filled_prefix_v.permute(1, 0, 2), v), dim=1)
                
        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)
        # print(f'attn_weights.shape={attn_weights.shape}')
        
        # 为了测精度时与其他的比较对象对齐
        attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_tokenids] = torch.finfo(attn_weights.dtype).min

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)
        
        # # 再把 kv 中不需要的剔除，从而后续不会被 store_cache 存入
        # # print(inputs.shape[1]-suffix_len, inputs.shape[1])
        # select_kv_tokenid = np.concatenate((select_kv_tokenid, np.arange(inputs.shape[1]-suffix_len, inputs.shape[1])))
        # k = k[select_kv_tokenid, :, :]
        # v = v[select_kv_tokenid, :, :]

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        aggr_attn = torch.sum(attn_weights, 1)
        aggr_attn = aggr_attn.transpose(0, 1)
        acc = aggr_attn[select_kv_tokenid]
        
        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)
        
        # print(f'mha shape:acc:{aggr_attn.shape},k:{k.shape},v:{v.shape},value:{value.shape}')
        return TorchTensor.create_from_torch(value, self), k, v, acc

    def sele_tokenid(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, sele_head, fullhead_prefix_k, accum_percent, sim_thred, cur_sele_percent=-1,
            ):
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        
        # print(fullhead_prefix_k.shape)

        prefix_len = fullhead_prefix_k.shape[0]

        b, s, h = inputs.shape
        suffix_len = s - prefix_len
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        # v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)[:, :, sele_head, :]
        k = k.view(b, suffix_len, n_head, head_dim)[:, :, sele_head, :]
        # v = v.view(b, s, n_head, head_dim)
        # print(f'k[0, 0, :, :]={k[0, 0, :, :]}')
        # print(f'k_part[0,0,:,:]={F.linear(hidden[:, :10, :], w_k.data, bias=b_k.data).view(b, 10, n_head, head_dim)[0,0,:,:]}')
        
        q = q.permute(0, 2, 1, 3).reshape(b * len(sele_head), s, head_dim)
        k = k.permute(0, 2, 3, 1).reshape(b * len(sele_head), head_dim, suffix_len)
        # v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        
        # 拼接 prefix kv
        k = torch.cat((fullhead_prefix_k.permute(1, 2, 0), k), dim=2)
        # print(f'k.shape={k.shape}, q.shape={q.shape}')
        
        attn_weights = torch.bmm(q, k)
        

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, len(sele_head), s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * len(sele_head), s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # print(f'attn_weights.shape={attn_weights.shape}')
        
        sele_tokenid, del_tokenid = self._sele_important_tokens(attn_weights, prefix_len, accum_percent, sim_thred, cur_sele_percent)
        # sele_tokenid, del_tokenid = self._sele_important_tokens(attn_weights, prefix_len, accum_percent, sim_thred, cur_sele_percent)
        # print(f'sele_tokenid={sele_tokenid}')
        # print(f'prefetch_sele_percent={prefetch_sele_percent}')
        # print(sele_tokenid.shape, del_tokenid.shape)
        return sele_tokenid, del_tokenid
        # 测试加入 prefetch 之后 sele 计算量变化
        # return sele_tokenid, del_tokenid
    
    def _sele_important_tokens(self, attn_weights, prefix_len, accum_percent, sim_thred, cur_sele_percent):
        aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        # 只有 Prefix 需要进行选择
        aggr_attn = aggr_attn[:, :prefix_len]
        bnhead, prefix_len = aggr_attn.shape
        
        if cur_sele_percent >= 0:
            sele_len = math.ceil(prefix_len * cur_sele_percent / 100)
            
            _, sele_indices = torch.topk(aggr_attn, sele_len, dim=1, largest=True) # shape: (b*nhead, sele_len)
            # print(f'sele_indices={sele_indices}')
            avg_sele_num = sele_len
            
        else:
            # 按行求和，计算出 target_attention_threshold
            row_sums = torch.sum(aggr_attn, 1) # shape: (b*nhead, 1)
            target_thred = row_sums * accum_percent / 100 # shape: (b*nhead, 1)
            # 按行降序排列
            sorted_values, sorted_indices = torch.topk(aggr_attn, prefix_len, dim=1) # shape: (b*nhead, s)
            # 按行累计求和
            cum_sums = sorted_values.cumsum(dim=1) # shape: (b*nhead, 1)
            # 逐行找出每个head中 累计到 target_attention_threshold 需要的 token 数
            mask = cum_sums >= target_thred.unsqueeze(1) # shape: (b*nhead, s)
            # 计算平均每个 head 达到 target_attention_threshold 所需的 token 数量
            avg_sele_num = int((~mask).sum(dim=1).float().mean().item())
            avg_sele_num = max(avg_sele_num, 1)
            # 根据 sorted_indices 找出每个 head 的最重要的 token indices
            sele_indices = sorted_indices[:, :avg_sele_num] # shape: (b*nhead, avg_sele_num)
        
        # 计算不同 head 间选择的 indices 的相似度
        sim = self.cal_jaccard_sim(sele_indices)
        print(f'select {avg_sele_num} / {prefix_len} = {avg_sele_num / prefix_len * 100}% of all tokens, with sim = {sim}. Whether selected? {sim >= sim_thred}')
        
        if sim >= sim_thred:
            unique_indices, counts = torch.unique(sele_indices.reshape(-1), return_counts=True)
            sorted_indices = torch.argsort(counts, descending=True)
            top_sele_num_indices = unique_indices[sorted_indices[:avg_sele_num]]
            # print(f'top_sele_num_indices = {top_sele_num_indices}')
            # print(f'unique_indices={unique_indices}')
            # record the top sele_num_indices for 5%, 10%, 15%, 20% of prefix_len
            # used for searching the relation between select ratio and recall ratio
            # top_sele_num_indices_prefetch = unique_indices[sorted_indices[:math.ceil(prefix_len * prefetch_sele_percent / 100)]]
            # top_sele_num_indices_5percent = unique_indices[sorted_indices[:math.ceil(prefix_len * 5 / 100)]]
            # top_sele_num_indices_10percent = unique_indices[sorted_indices[:math.ceil(prefix_len * 10 / 100)]]
            # top_sele_num_indices_15percent = unique_indices[sorted_indices[:math.ceil(prefix_len * 15 / 100)]]
            # top_sele_num_indices_20percent = unique_indices[sorted_indices[:math.ceil(prefix_len * 20 / 100)]]
            # with open('./token_logs/sele_load_openbookqa_sample_5101520.txt', 'a') as f:
            #     f.write(f'Selected Tokens per Head: {avg_sele_num}\n')
            #     f.write(f'sele_num_indices: {top_sele_num_indices.tolist()}\n')
            #     f.write(f'top_sele_num_indices_5percent: {top_sele_num_indices_5percent.tolist()}\n')
            #     f.write(f'top_sele_num_indices_10percent: {top_sele_num_indices_10percent.tolist()}\n')
            #     f.write(f'top_sele_num_indices_15percent: {top_sele_num_indices_15percent.tolist()}\n')
            #     f.write(f'top_sele_num_indices_20percent: {top_sele_num_indices_20percent.tolist()}\n')
            
            # 通过布尔掩码去掉在 sele_indices 中出现过的数字，返回 del_indices
            top_sele_num_indices = top_sele_num_indices.cpu()
            # print('top_sele_num_indices', top_sele_num_indices, top_sele_num_indices.shape)
            arange_tensor = torch.arange(prefix_len, device='cpu')
            mask = ~torch.isin(arange_tensor, top_sele_num_indices)
            del_indices = arange_tensor[mask]
            # print('del', del_indices, del_indices.shape)
            # print(f'top_sele_num_indices = {top_sele_num_indices}, del_indices = {del_indices}')
            # print('del_indices', del_indices.shape)
            return top_sele_num_indices, del_indices
            # return top_sele_num_indices, del_indices
        else:
            return torch.arange(prefix_len, device='cpu'), torch.arange(0, device='cpu')
    
    def cal_jaccard_sim(self, sele_indices):
        bnhead, sele_token_num = sele_indices.shape
        # 用于存储所有行对之间的Jaccard相似度
        jaccard_similarities = []

        # 计算所有可能的行对 (i, j) 之间的 Jaccard 相似度
        for i in range(bnhead):
            for j in range(i + 1, bnhead):
                set_i = set(sele_indices[i].tolist())
                set_j = set(sele_indices[j].tolist())
                
                # 计算交集和并集
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                
                # 计算Jaccard相似度
                jaccard_similarity = intersection / union if union > 0 else 0
                jaccard_similarities.append(jaccard_similarity)

        # 求Jaccard相似度的均值
        mean_jaccard_similarity = sum(jaccard_similarities) / len(jaccard_similarities)
        return mean_jaccard_similarity
    
    def sele_tokenid_with_all_keys(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, complete_prefix_k, cur_sele_percent=-1):
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        
        prefix_len = complete_prefix_k.shape[0]

        b, s, h = inputs.shape
        suffix_len = s - prefix_len
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        # v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)
        # v = v.view(b, s, n_head, head_dim)
        # print(f'k[0, 0, :, :]={k[0, 0, :, :]}')
        # print(f'k_part[0,0,:,:]={F.linear(hidden[:, :10, :], w_k.data, bias=b_k.data).view(b, 10, n_head, head_dim)[0,0,:,:]}')
        
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, suffix_len)
        # v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        
        # 拼接 prefix kv
        k = torch.cat((complete_prefix_k.permute(1, 2, 0), k), dim=2)
        # print(f'k.shape={k.shape}, q.shape={q.shape}')
        
        attn_weights = torch.bmm(q, k)
        del q,k,hidden

        # shape: (b, 1, s, s)
        # idx = torch.arange(s, device=self.dev)
        # causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        # mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)

        attn_weights = attn_weights.view(b, n_head, s, s)
        # attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # print(f'attn_weights.shape={attn_weights.shape}')

        # inplace softmax
        attn_weights -= torch.max(attn_weights,dim=2)[0].view(attn_weights.shape[0],attn_weights.shape[1],1)
        attn_weights.exp_()
        sums = torch.sum(attn_weights,dim=2).view(attn_weights.shape[0],attn_weights.shape[1],1)
        attn_weights /= sums
        
        # 选择重要的 key
        aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        # 只有 Prefix 需要进行选择
        aggr_attn = aggr_attn[:, :prefix_len]
        bnhead, prefix_len = aggr_attn.shape
        
        if cur_sele_percent >= 0:
            sele_len = math.ceil(prefix_len * cur_sele_percent / 100)
            
            _, sele_indices = torch.topk(aggr_attn, sele_len, dim=1) # shape: (b*nhead, sele_len)
            avg_sele_num = sele_len
            
            # sele_indices = sele_indices.cpu()
            # arange_tensor = torch.arange(prefix_len, device='cuda').expand(bnhead, -1)
            # mask = torch.zeros_like(arange_tensor, dtype=torch.bool)
            # for i in range(arange_tensor.shape[0]):
            #     mask[i] = torch.isin(arange_tensor[i], sele_indices[i])
            # del_indices = arange_tensor.masked_fill(mask, -1)
            # del_indices = del_indices[del_indices != -1].view(bnhead, -1)
            
            # print(f'select {avg_sele_num} / {prefix_len} = {avg_sele_num / prefix_len * 100}% of all prefix tokens, del_indices.shape = {del_indices.shape}')
        else:
            print(f'cur_sele_percent should by >= 0 rather than {cur_sele_percent}')
            sys.exit(-1)
        return sele_indices.cpu()
        # return sele_indices, del_indices
     
    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, acc, donate,
                attn_sparsity, compress_cache, comp_config,
                hh_k=None, hh_all=False):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        # if True:
        #     print("hidden_states", hidden.shape, hidden)
        #     print("query_states", q.shape, q)
        #     k_tmp = k.permute(0, 2, 1, 3).reshape(b * n_head, -1, head_dim)
        #     v_tmp = v.permute(0, 2, 1, 3).reshape(b * n_head, -1, head_dim)
        #     print("key_states", k_tmp.shape, k_tmp)
        #     print("value_states", v_tmp.shape, v_tmp)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                # k_print = k.permute(0, 2, 1)
                # print("selected_key_states", k_print.shape, k_print)
                # print("selected_value_states", v.shape, v)

                if k.is_cuda:
                    value, attn_weights = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value, attn_weights = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                    value = value.cuda().half()
                    attn_weights = attn_weights.cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)

        # print("after bmm (gen)", value.shape, np.array((value[0,0, :100]).tolist()))

        value = F.linear(value, w_out.data, bias=b_out.data)
        # print("after value_proj (gen)", value.shape, np.array((value[0, 0, :100]).tolist()))

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        # get the least heavy hitter (except recent tokens)
        kick_ind = None
        if hh_all:
            # k shape: (b * n_head, head_dim, s)
            # v shape: (b * n_head, s, head_dim)
            # attn_weights shape: (b * n_head, 1, s)
            # print(attn_weights.shape)
            attn_weights = attn_weights.squeeze(1).transpose(0, 1)
            # (s, b * n_head)
            acc.data = acc.data.cuda()
            acc.data[-1] = 0
            acc.data = acc.data + attn_weights
            # print("acc.data", acc.data.shape, acc.data[:, -4])
            kick_ind = self._get_light_hitter(acc.data[:src_s - hh_k, :])
            if not k.is_cuda:
                acc.data = acc.data.float().cpu()
            # kick_ind = self._get_light_hitter(acc.data[:src_s - hh_k + 1, :])

        return TorchTensor.create_from_torch(value, self), k_new, v_new, acc, kick_ind


    def _get_light_hitter(self, acc):
        # return torch.zeros(attn_weights.shape[0])
        if acc.shape[0] > 0:
            kick_ind = acc.argmin(dim=0).squeeze()
            # print(kick_ind)
            # fake_ind = torch.randint(low=0, high=acc.shape[0] - 1, size=kick_ind.shape)
            # return fake_ind
            # kick_ind[:] = 6
            # print("kick_ind", kick_ind.shape, kick_ind)
            return kick_ind
        return None


    def _heavy_hitter_pruning(self, k, v, attn_weights, hh_k, ret_topk_indices=False):
        # k, v: (s, b * n_head, head_dim)
        # attn_weights: (b * n_head, s, s)

        aggr_attn = torch.sum(attn_weights, 1) # (b * n_head, s)
        # topk_indices.shape = (b * n_head, hh_k)
        _, topk_indices = aggr_attn[:, :-hh_k + 1].topk(
            min(hh_k, aggr_attn.shape[1] - hh_k + 1), dim=1)
        # topk_indices, _ = topk_indices.sort()
        #print("topk_indices", topk_indices.shape) # (k, 1)

        # select heavy-hitters
        # (b * n_head, s, head_dim)
        k_t = k.transpose(1, 0)
        v_t = v.transpose(1, 0)
        dim0_indices = torch.arange(k_t.size(0))[:, None]
        dim0_indices = dim0_indices.expand_as(topk_indices) 
        # dim0_indices.shape is same as topk_indices but with all 0 from all b*nhead-1
        
        # (b * n_head, hh_k, head_dim)
        k_hh_t = k_t[dim0_indices, topk_indices]
        v_hh_t = v_t[dim0_indices, topk_indices]
        # (hh_k, b * n_head, head_dim)
        k_hh = k_hh_t.transpose(1, 0)
        v_hh = v_hh_t.transpose(1, 0)
        # (hh_k * 2 -1, b * n_head, head_dim)
        k = torch.cat([k_hh, k[-hh_k + 1:]], dim=0)
        v = torch.cat([v_hh, v[-hh_k + 1:]], dim=0)
        aggr_attn = aggr_attn.transpose(0, 1) # (s, b*n_head)
        dim1_indices = torch.arange(aggr_attn.size(1)).unsqueeze(0) # (1, b*n_head)
        acc_hh = aggr_attn[topk_indices.transpose(0, 1), dim1_indices] # (hh_k, b*n_head)
        acc = torch.cat([acc_hh, aggr_attn[-hh_k + 1:]], dim=0) # (hh_k * 2-1, b * n_head)
        if ret_topk_indices:
            return k, v, acc, topk_indices
        return k, v, acc


    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # print("attn_weights (first bmm)", attn_weights.shape, attn_weights[-1])
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        #attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        # print("attn_weights (before softmax)", attn_weights.shape, attn_weights[-1])
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(k.dtype)
        # print("attn_weights (after softmax)", attn_weights.shape, attn_weights[-4])
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim), attn_weights

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu, _ = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu, _ = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id), daemon=True
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        acc = self.allocate(shape[:-1], np.float16)
        return k_cache, v_cache, acc

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy, hh_k=None, hh_all=False):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        if hh_all:
            shape = (hh_k * 2, gpu_batch_size * num_head, hidden_size // num_head)
        elif hh_k * 2 < prompt_len:
            shape = (hh_k * 2 + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        else:
            shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


# @jit.rawkernel()
# def evict(dst, ind, src, oldest, s1, s2):
#     d1 = jit.blockIdx.x
#     d2 = jit.threadIdx.x
#     # dst[ind[d1], d1, d2] = dst[oldest, d1, d2]
#     di = ind[d1] * s1 * s2 + d1 * s2 + d2
#     old = oldest * s1 * s2 + d1 * s2 + d2
#     dst[di] = dst[old]
#     # dst[oldest, d1, d2] = src[0, d1, d2]
#     si = d1 * s2 + d2
#     dst[old] = src[si]


# least_recent = torch.ones((1, 2048, 128), dtype=torch.float16).cuda()


def cache_replace(dst: TorchTensor, dst_indices, src: torch.Tensor, hh_k, oldest):
    # mask = torch.ones(dst.shape[0], dst.shape[1], dtype=torch.bool).to(dst.data.device)
    # indices = dst_indices.unsqueeze(0).to(dst.data.device)
    # mask.scatter_(0, indices, torch.zeros_like(indices, dtype=torch.bool))
    # mask = mask.transpose(0, 1)
    # dst.data[:-1] = dst.data.transpose(0, 1)[mask].reshape(dst.shape[1], dst.shape[0] - 1, -1).transpose(0, 1)
    # # dst.data[:-1] = dst.data[1:]
    # dst.data[-2] = src.data.squeeze()
    # return

    # dst: (s, b * n_head, head_dim)
    # dst_indices: (h)
    # src: (1, b * n_head, head_dim)
    if abs(oldest) >= min(dst.shape[0], hh_k * 2): return

    oldest = hh_k * 2 - 1 + oldest

    #dst_cupy = cp.asarray(dst.data)
    #ind_cupy = cp.asarray(dst_indices)
    #src_cupy = cp.asarray(src.data)
    #evict((2048,), (128,), (dst_cupy, ind_cupy, src_cupy, oldest, dst.shape[1], dst.shape[2]))
    #dst_dlpack = dst_cupy.toDlpack()
    #dst.data = torch.utils.dlpack.from_dlpack(dst_dlpack)

    # print(dst.shape)
    # print(dst_indices.shape)
    # print(src.shape)
    # exit()

    #for i, idx in enumerate(dst_indices):
    #    dst.data[idx, i] = dst.data[oldest, i]
    #dst.data[oldest] = src.data.squeeze()
    #return

    # least_recent[0, :, :] = dst.data[oldest][:, :]
    least_recent = torch.tensor(dst.data[oldest]).unsqueeze(0).to(dst.data.device)
    indices = dst_indices.view(-1, 1).expand(-1, dst.shape[2]).unsqueeze(0).to(dst.data.device)
    dst.data.scatter_(0, indices, least_recent)
    dst.data[oldest] = src.data.squeeze()


def acc_replace(dst, dst_indices, src, hh_k, oldest):
    # dst.data = src.data
    # mask = torch.ones(dst.shape[0], dst.shape[1], dtype=torch.bool).to(dst.data.device)
    # indices = dst_indices.unsqueeze(0).to(dst.data.device)
    # mask.scatter_(0, indices, torch.zeros_like(indices, dtype=torch.bool))
    # mask = mask.transpose(0, 1)
    # # print(indices)
    # # print(mask[22])
    # # print("acc", mask.shape)
    # dst.data[:-1] = dst.data.transpose(0, 1)[mask].reshape(dst.shape[1], dst.shape[0] - 1).transpose(0, 1)
    # # dst.data[:-1] = src.data[1:].clone()
    # dst.data[-2] = src.data[-1]
    # return
 
    if abs(oldest) >= min(dst.shape[0], hh_k * 2): return
    dst.data = src.data
    oldest = hh_k * 2 - 1 + oldest
    least_recent = torch.tensor(src.data[oldest]).unsqueeze(0).to(dst.data.device)
    indices = dst_indices.unsqueeze(0).to(dst.data.device)
    dst.data.scatter_(0, indices, least_recent)
    dst.data[oldest] = src.data[-1]


def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=False)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()

# llama support
# https://github.com/FMInference/FlexLLMGen/pull/135/commits/8e2cc943eb888b6133e2c0eb1c4a62f79c71f980

def rms_norm(input, weight, eps) -> torch.Tensor:
    input_dtype = input.dtype
    hidden_states = input.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def rotary_embedding(x, inv_freq, seq_len):
    t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq.to(x.device))
    emb = torch.cat((freqs, freqs), dim=-1)
    return (
        emb.cos().to(x.dtype)[:seq_len].to(dtype=x.dtype),
        emb.sin().to(x.dtype)[:seq_len].to(dtype=x.dtype),
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2,position_idsq=None):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if position_idsq is None:
        position_idsq = position_ids
    cosk = cos[:,position_ids[:]].squeeze(0).unsqueeze(unsqueeze_dim)
    sink = sin[:,position_ids[:]].squeeze(0).unsqueeze(unsqueeze_dim)
    cosq = cos[:,position_idsq[:]].squeeze(0).unsqueeze(unsqueeze_dim)
    sinq = sin[:,position_idsq[:]].squeeze(0).unsqueeze(unsqueeze_dim)
    q_embed = (q * cosq) + (rotate_half(q) * sinq)
    k_embed = (k * cosk) + (rotate_half(k) * sink)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class LlamaTorchDevice(TorchDevice):

    def llama_input_embed(self, inputs, attention_mask, w_token, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        return TorchTensor.create_from_torch(token_embed, self)

    def llama_output_embed(self, inputs, w_ln, w_token, eps, donate, do_sample, temperature,record_logits = False, oplm = None):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]
        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        if record_logits:
            oplm.logits_val = logits
        return TorchTensor.create_from_torch(ids, self)

    def llama_mha(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config,hh_k=None, hh_all=None, ret_topk_indices=False):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_head, head_dim)
        v = v.view(b, s, n_kv_head, head_dim)

        kv_seq_len = k.shape[-3]
        cos, sin = rotary_embedding(v, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if hh_k is not None:
            if not ret_topk_indices:
                k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
                # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
                # print(f'k_ori={k}')
                # print(f'v_ori={v}')
                # print(f'acc_ori={acc}')
            else:
                k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
                # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)

        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v, acc, topk_indices
        return TorchTensor.create_from_torch(value, self), k, v, acc
    
    def llama_mha_with_prefixkv_suffix(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            prefix_k=None, prefix_v=None,suffix_len=100):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        # 修改 hidden
        kv_seq_len = hidden.shape[-2]
        hidden = hidden[:, -suffix_len:, :]
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, suffix_len, n_head, head_dim)
        k = k.view(b, suffix_len, n_kv_head, head_dim)
        v = v.view(b, suffix_len, n_kv_head, head_dim)


        cos, sin = rotary_embedding(v, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids[:,-suffix_len:],position_idsq=position_ids[:,-suffix_len:])

        k = torch.cat((prefix_k.unsqueeze(0), k), dim=1)
        v = torch.cat((prefix_v.unsqueeze(0), v), dim=1)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, suffix_len, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        causal_mask = causal_mask[:,:,-suffix_len:,:]
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, suffix_len, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, suffix_len, s)
        attn_weights = F.softmax(attn_weights, dim=2,dtype=torch.float16)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, suffix_len, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, suffix_len, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data[:, -suffix_len:, :])

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        acc = torch.empty((s,n_head),dtype=hidden.dtype,device='cuda:0')
        k = torch.empty((s,n_head,head_dim),dtype=hidden.dtype,device='cuda:0')
        v = torch.empty((s,n_head,head_dim),dtype=hidden.dtype,device='cuda:0')
        value = torch.empty((1,s,h),dtype=hidden.dtype,device='cuda:0')
        acc = TorchTensor.create_from_torch(acc, self)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v,acc
        return TorchTensor.create_from_torch(value, self), k, v,acc
    
    def llama_mha_prefixkv(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config,hh_k=None, hh_all=None, ret_topk_indices=False,suffix_len=100):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        s = s - suffix_len
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
        kv_seq_len = hidden.shape[-2]
        hidden = hidden[:, 0:-suffix_len, :]

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_head, head_dim)
        v = v.view(b, s, n_kv_head, head_dim)

        cos, sin = rotary_embedding(v, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids,position_idsq=position_ids[:])

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2,dtype=torch.float16)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data[:, 0:-suffix_len, :])

        # if donate[0]: inputs.delete()
        # if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v
        return TorchTensor.create_from_torch(value, self), k, v
    
    def llama_mha_with_prefixkv(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config,prefix_k=None,prefix_v=None,suffix_len=100):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        # 修改 hidden
        kv_seq_len = hidden.shape[-2]
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_kv_head, head_dim)
        v = v.view(b, suffix_len, n_kv_head, head_dim)


        cos, sin = rotary_embedding(v, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids[:,-suffix_len:],position_idsq=position_ids[:])

        k = torch.cat((prefix_k.unsqueeze(0), k), dim=1)
        v = torch.cat((prefix_v.unsqueeze(0), v), dim=1)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        self.mask_with_zero_cols(attn_weights, 1-100*0.01, suffix_len)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    def llama_mha_gen(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
                w_re, w_out, eps, n_head, n_kv_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data) * scaling
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_kv_head, head_dim)
        v = v.view(b, tgt_s, n_kv_head, head_dim)

        cos, sin = rotary_embedding(v, w_re.data, seq_len=position_ids.max().item() + 1)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    def llama_mha_with_sele_percent_prefixkv(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            prefix_k=None, prefix_v=None, suffix_len=100, cur_sele_percent=100, motiv2 = False, samekv=False, sele_indices = None, generate_mapping_list = False):
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        # print(f'q.shape = {q.shape}')
        # 修改 hidden
        kv_seq_len = hidden.shape[-2]
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_kv_head, head_dim)
        v = v.view(b, suffix_len, n_kv_head, head_dim)

        cos, sin = rotary_embedding(v, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids[:,-suffix_len:],position_idsq=position_ids[:])

        k = torch.cat((prefix_k.unsqueeze(0), k), dim=1)
        v = torch.cat((prefix_v.unsqueeze(0), v), dim=1)

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)
        
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, suffix_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, -1)
        # shape: (b * n_head, suffix_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, -1, head_dim)
        
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)
        
        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        tmp_attn_weights = attn_weights.clone()
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        
        # # 把每行 attn_weights 中最小的 (1-cur_sele_percent) 的值置为 0
        # attn_weights = self.mask_with_zero(attn_weights, 1-cur_sele_percent*0.01)
        
        # 把每列的 attn_weights 累加，找出和最小的 token，把对应列的 attn 值变为0
        if motiv2 and not samekv:
            attn_weights, del_indcies, keep_indices = self.mask_with_zero_cols_motiv2(attn_weights, 1-cur_sele_percent*0.01, suffix_len, )
        elif motiv2 and samekv:
            attn_weights, del_indcies = self.mask_with_zero_cols_samekv(attn_weights, 1-cur_sele_percent*0.01, suffix_len, sele_indices)
        
        else:
            if generate_mapping_list:
                aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
                summed_aggr_attn = torch.sum(aggr_attn, dim=0) # shape: (s, )
                prefix_len = aggr_attn.shape[1]-suffix_len
                _, sorted_indices = torch.topk(summed_aggr_attn[:prefix_len], prefix_len)
                
                # 建立反向索引
                reverse_indices = torch.full_like(sorted_indices, -1)
                reverse_indices[sorted_indices] = torch.arange(len(sorted_indices), device="cuda:0")
                
            attn_weights, del_indcies = self.mask_with_zero_cols(attn_weights, 1-cur_sele_percent*0.01, suffix_len)
            
        '''# 根据 del_indcies 将 k / v 置零，然后重新执行 atten 计算
        k[torch.arange(b*n_head).unsqueeze(1), :, del_indcies] = 0
        v[torch.arange(b*n_head).unsqueeze(1), del_indcies, :] = 0
        attn_weights = torch.bmm(q, k)
        
        attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)'''

        # 根据 del_indcies 将 k / v 置零，然后重新执行 atten 计算
        # k[torch.arange(b*n_head).unsqueeze(1), :, del_indcies] = 0
        v[torch.arange(b*n_head).unsqueeze(1), del_indcies, :] = 0
        attn_weights = torch.bmm(q, k)
        # attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = torch.finfo(attn_weights.dtype).min
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)


        # tmp_attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_indcies] = -1e4
        # attn_weights = F.softmax(tmp_attn_weights, dim=2, dtype=torch.float32).to(torch.float16)

        
        # shape: (b, n_head, s, head_dim)   
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        aggr_attn = torch.sum(attn_weights, 1)
        aggr_attn = aggr_attn.transpose(0, 1)
        acc = aggr_attn[sele_indices]
        
        # # select the heavy hitters and recent tokens
        if hh_k is not None:
            if not ret_topk_indices:
                k, v, acc = self._heavy_hitter_pruning(k, v, attn_weights, hh_k)
                # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}')
                # print(f'k={k}')
                # print(f'v={v}')
                # print(f'acc={acc}')
            else:
                k, v, acc, topk_indices = self._heavy_hitter_pruning(k, v, attn_weights, hh_k, ret_topk_indices=True)
                # print(f'after applying hh_k: k.shape={k.shape}, v.shape={v.shape}, acc.shape={acc.shape}, topk_indices.shape={topk_indices.shape}')

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)
            
        # print(f'mha shape:acc:{aggr_attn.shape},k:{k.shape},v:{v.shape},value:{value.shape}')
        if ret_topk_indices:
            return TorchTensor.create_from_torch(value, self), k, v, acc, topk_indices
        elif generate_mapping_list:
            return TorchTensor.create_from_torch(value, self), k, v, acc, reverse_indices
        
        if motiv2 and not samekv:
            return TorchTensor.create_from_torch(value, self), k, v, acc, keep_indices
        return TorchTensor.create_from_torch(value, self), k, v, acc
    
    def llama_mha_with_filled_selected_prefixkv(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, compress_cache, comp_config, hh_k=None, hh_all=None, ret_topk_indices=False,
            filled_prefix_k=None, filled_prefix_v=None, suffix_len=100, select_kv_tokenid=None, del_tokenids = None):
        """Multi-head attention only with suffix (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)
        
        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        # 只计算 suffix 的 kv
        kv_seq_len = hidden.shape[-2]
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data)
        v = F.linear(hidden, w_v.data)
        
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)
        v = v.view(b, suffix_len, n_head, head_dim)

        cos, sin = rotary_embedding(v, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids[:,-suffix_len:],position_idsq=position_ids[:])

        k = torch.cat((filled_prefix_k.unsqueeze(0), k), dim=1)
        v = torch.cat((filled_prefix_v.unsqueeze(0), v), dim=1)
        

        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        v = repeat_kv(v, n_kv_groups)
        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, suffix_len)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, suffix_len, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        
                
        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)
        # print(f'attn_weights.shape={attn_weights.shape}')
        
        # 为了测精度时与其他的比较对象对齐
        attn_weights[torch.arange(attn_weights.shape[0]).unsqueeze(1), :, del_tokenids] = torch.finfo(attn_weights.dtype).min

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, torch.finfo(attn_weights.dtype).min)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)

        # print("after bmm (prompt) prompt.shape", value.shape)

        value = F.linear(value, w_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)
        
        # # 再把 kv 中不需要的剔除，从而后续不会被 store_cache 存入
        # # print(inputs.shape[1]-suffix_len, inputs.shape[1])
        # select_kv_tokenid = np.concatenate((select_kv_tokenid, np.arange(inputs.shape[1]-suffix_len, inputs.shape[1])))
        # k = k[select_kv_tokenid, :, :]
        # v = v[select_kv_tokenid, :, :]

        # print(f'before applying hh_k: k.shape={k.shape}, v.shape={v.shape}')
        
        aggr_attn = torch.sum(attn_weights, 1)
        aggr_attn = aggr_attn.transpose(0, 1)
        acc = aggr_attn[select_kv_tokenid]
        
        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
            acc = TorchTensor.create_from_torch(acc, self)
        
        # print(f'mha shape:acc:{aggr_attn.shape},k:{k.shape},v:{v.shape},value:{value.shape}')
        return TorchTensor.create_from_torch(value, self), k, v, acc



    def llama_sele_tokenid_with_all_keys(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, complete_prefix_k, cur_sele_percent=-1):
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)
        
        prefix_len = complete_prefix_k.shape[0]

        b, s, h = inputs.shape
        suffix_len = s - prefix_len
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        kv_seq_len = hidden.shape[-2]
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data)
        # v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)

        cos, sin = rotary_embedding(k, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids[:,-suffix_len:],position_idsq=position_ids[:])

        k = torch.cat((complete_prefix_k.unsqueeze(0), k), dim=1)
        n_kv_groups = n_head // n_kv_head
        k = repeat_kv(k, n_kv_groups)
        # v = v.view(b, s, n_head, head_dim)
        # print(f'k[0, 0, :, :]={k[0, 0, :, :]}')
        # print(f'k_part[0,0,:,:]={F.linear(hidden[:, :10, :], w_k.data, bias=b_k.data).view(b, 10, n_head, head_dim)[0,0,:,:]}')
        
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        
        # 拼接 prefix kv
        # print(f'k.shape={k.shape}, q.shape={q.shape}')
        
        attn_weights = torch.bmm(q, k)
        del q,k,hidden

        # shape: (b, 1, s, s)
        # idx = torch.arange(s, device=self.dev)
        # causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        # mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)

        attn_weights = attn_weights.view(b, n_head, s, s)
        # attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float32).to(torch.float16)
        # attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # print(f'attn_weights.shape={attn_weights.shape}')

        # inplace softmax
        attn_weights -= torch.max(attn_weights,dim=2)[0].view(attn_weights.shape[0],attn_weights.shape[1],1)
        attn_weights.exp_()
        sums = torch.sum(attn_weights,dim=2).view(attn_weights.shape[0],attn_weights.shape[1],1)
        attn_weights /= sums
        
        # 选择重要的 key
        aggr_attn = torch.sum(attn_weights, 1) # shape: (b*nhead, s)
        # 只有 Prefix 需要进行选择
        aggr_attn = aggr_attn[:, :prefix_len]
        bnhead, prefix_len = aggr_attn.shape
        
        if cur_sele_percent >= 0:
            sele_len = math.ceil(prefix_len * cur_sele_percent / 100)
            
            _, sele_indices = torch.topk(aggr_attn, sele_len, dim=1) # shape: (b*nhead, sele_len)
            avg_sele_num = sele_len
            
            # sele_indices = sele_indices.cpu()
            # arange_tensor = torch.arange(prefix_len, device='cuda').expand(bnhead, -1)
            # mask = torch.zeros_like(arange_tensor, dtype=torch.bool)
            # for i in range(arange_tensor.shape[0]):
            #     mask[i] = torch.isin(arange_tensor[i], sele_indices[i])
            # del_indices = arange_tensor.masked_fill(mask, -1)
            # del_indices = del_indices[del_indices != -1].view(bnhead, -1)
            
            # print(f'select {avg_sele_num} / {prefix_len} = {avg_sele_num / prefix_len * 100}% of all prefix tokens, del_indices.shape = {del_indices.shape}')
        else:
            print(f'cur_sele_percent should by >= 0 rather than {cur_sele_percent}')
            sys.exit(-1)
        return sele_indices.cpu()


    def llama_sele_tokenid(self, inputs, position_ids, attention_mask, w_ln, w_q, w_k, w_v,
            w_re, w_out, n_head, n_kv_head, donate, eps, sele_head, fullhead_prefix_k, accum_percent, sim_thred, cur_sele_percent=-1):
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_re = w_re.device.decompress(w_re)
            w_out = w_out.device.decompress(w_out)
        
        # print(fullhead_prefix_k.shape)

        prefix_len = fullhead_prefix_k.shape[0]

        b, s, h = inputs.shape
        suffix_len = s - prefix_len
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        hidden = rms_norm(inputs.data, weight=w_ln.data, eps=eps)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data) * scaling
        kv_seq_len = hidden.shape[-2]
        hidden = hidden[:, -suffix_len:, :]
        k = F.linear(hidden, w_k.data)
        # v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)

        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, suffix_len, n_head, head_dim)

        cos, sin = rotary_embedding(k, w_re.data, seq_len=kv_seq_len)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids[:,-suffix_len:],position_idsq=position_ids[:])

        q = q[:, :, sele_head, :]
        k = k[:, :, sele_head, :]
        k = torch.cat((fullhead_prefix_k.unsqueeze(0), k), dim=1)

        # v = v.view(b, s, n_head, head_dim)
        # print(f'k[0, 0, :, :]={k[0, 0, :, :]}')
        # print(f'k_part[0,0,:,:]={F.linear(hidden[:, :10, :], w_k.data, bias=b_k.data).view(b, 10, n_head, head_dim)[0,0,:,:]}')
        
        q = q.permute(0, 2, 1, 3).reshape(b * len(sele_head), s, head_dim)
        k = k.permute(0, 2, 3, 1).reshape(b * len(sele_head), head_dim, s)
        # v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # print(f'q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}')
        
        # print(f'k.shape={k.shape}, q.shape={q.shape}')
        
        attn_weights = torch.bmm(q, k)
        

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, len(sele_head), s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * len(sele_head), s, s)
        attn_weights = F.softmax(attn_weights, dim=2, dtype=torch.float16)
        # print(f'attn_weights.shape={attn_weights.shape}')
        
        sele_tokenid, del_tokenid = self._sele_important_tokens(attn_weights, prefix_len, accum_percent, sim_thred, cur_sele_percent)
        
        # print(sele_tokenid.shape, del_tokenid.shape)
        return sele_tokenid, del_tokenid
    
        
    def llama_mlp(self, inputs, w_ln, w_g, w_u, w_d, eps, donate):
        # decompress weights
        if w_ln.device.device_type == DeviceType.COMPRESSED:
            w_g = w_g.device.decompress(w_g)
            w_u = w_g.device.decompress(w_u)
            w_d = w_g.device.decompress(w_d)

        out = rms_norm(inputs.data, weight=w_ln.data, eps=eps)
        gate_out = F.linear(out, w_g.data)
        F.silu(gate_out, inplace=True)
        up_out = F.linear(out, w_u.data)
        out = F.linear(gate_out * up_out, w_d.data)
        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)