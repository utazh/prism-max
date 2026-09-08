import os
import torch
import numpy as np
from myheap import Heap
from functools import partial
import time
import threading

from contiguous_fuxian.priority_executor import PriorityExecutor

AVG_CHUNK_TIME = {}
AVG_CHUNK_TIME['cuda:0'] = 0.00006998
AVG_CHUNK_TIME['cpu'] = 0.00008312
AVG_CHUNK_TIME['disk'] = 0.00040818


class AsyncPrefetchHandle:
    def __init__(self, future):
        self.future = future

    def done(self):
        return self.future.done()

    def result(self):
        result = self.future.result()
        if len(result) == 5:
            key, value, tokenids, event, pinned_buffers = result
        else:
            key, value, tokenids, event = result
            pinned_buffers = None
        if event is not None:
            event.synchronize()
        # Keep pinned CPU staging buffers alive until the CUDA copy event is done.
        # Dropping this local reference after synchronize lets PyTorch reuse them.
        pinned_buffers = None
        return key, value, tokenids

def uncache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)
            os.rmdir(file_path)

class DummyLRU():
    def __init__(self) -> None:
        self.prev = self
        self.next = self
    
    def insert(self,item):  #insert to next
        if isinstance(item,TokenCacheGeneral):
            item = item.control
        self.next.prev = item
        item.next = self.next
        item.prev = self
        self.next = item

    def delete(self):
        if self.prev is not None and self.next is not None:
            self.prev.next = self.next
            self.next.prev = self.prev
            self.next = self
            self.prev = self
        return self


class TokenCache():
    # device2id = {'cuda:0':0,'cpu':1,'disk':2}
    # 我的理解是，TokenCache是一个一个token进行存储的，所以，它的寻址策略是以token为单位进行寻址
    def __init__(self,key,value,prefix_cache,layer_id,control,pos_id,device='cuda:0',key_exist=True,value_exist=True):
        self.key = key
        self.value = value
        self.key_exist=key_exist
        self.value_exist=value_exist
        if self.key is None:
            self.key_exist = False
        if self.value is None:
            self.value_exist = False
        self.device = None # 本 token 的 cache 所在设备
        self.prefix_cache = prefix_cache # 对应一个 prefixkvlayer 实例，表示这个token属于哪个prefix的kv缓存层
        # 一个 token cache 需要通过 self.prefix_cache.device_map 来记录它的设备位置
        self.pos_id = pos_id
        # control 就是替换策略
        self.control = control
        self.layer_id = layer_id # 一个 kv cache 对应某一层
        self.size = 0
        if self.key_exist:
            self.size += (key.element_size() * key.nelement()) / 1024 / 1024
        if self.value_exist:
            self.size += (value.element_size() * value.nelement()) / 1024 / 1024

        # 关键字参数写法，也可以写成self.to(device)，这种是位置参数写法
        self.to(device=device)

    def to(self, device):
        if device == self.device:
            return
        if device == 'disk':
            self.to_disk()
            return
        if self.device == 'disk':
            if self.pos_id is None:
                self.key = self.prefix_cache.get_disk_key_head()
            else:
                if self.prefix_cache.disk_map is not None:
                    if self.key_exist:
                        self.key = self.prefix_cache.key_buffer[self.prefix_cache.disk_map[self.pos_id]].to(device)
                    if self.value_exist:
                        self.value = self.prefix_cache.value_buffer[self.prefix_cache.disk_map[self.pos_id]].to(device)
                else:
                    if self.key_exist:
                        self.key,_ = self.prefix_cache.get_disk_token(self.pos_id,device)
                    if self.value_exist:
                        _,self.value = self.prefix_cache.get_disk_token(self.pos_id,device)
        else:
            if self.key_exist:
                self.key = self.key.to(device)
            if self.value_exist:
                self.value = self.value.to(device)
        # 更新设备映射
        self.device = device
        if self.pos_id is not None:
            # 移动的时候需要修改自己的设备位置
            self.prefix_cache.device_map[self.pos_id] = device 

    def to_disk(self):
        # 把自己类的东西清空，改一下device_map
        self.key = None
        self.value = None
        self.device = 'disk'
        if self.pos_id is not None:
            self.prefix_cache.device_map[self.pos_id] = 'disk'

    def get(self,device='cuda:0',head=None):
        key = None
        value = None
        if self.device == 'disk':
            if self.pos_id is None:
                # if no pos_id, it means we are getting the head token
                key,value = self.prefix_cache.get_disk_head()
            else:
                key,value = self.prefix_cache.get_disk_token(self.pos_id)
        else:
            if self.key_exist:
                key = self.key
            if self.value_exist:
                value = self.value
        if head is not None:
            if self.key_exist:
                key = key[head]
            if self.value_exist:
                value = value[head]
        if self.pos_id is not None:
            self.control.update()
        return key.to(device),value.to(device)
    
    def get_key(self,device='cuda:0',head=None):
        if not self.key_exist:
            return None
        if self.device == 'disk':
            if self.pos_id is None:
                key = self.prefix_cache.get_disk_key_head()
            else:
                key = self.prefix_cache.get_disk_key_token(self.pos_id)
        else:
            key = self.key
        if head is not None:
            key = key[head]
        if self.pos_id is not None:
            self.control.update()
        return key.to(device)
    
    def get_value(self,device='cuda:0',head=None):
        if not self.value_exist:
            return None
        if self.device == 'disk':
            if self.pos_id is None:
                return None
            else:
                value = self.prefix_cache.get_disk_value_token(self.pos_id)
        else:
            value = self.value
        if head is not None:
            value = value[head]
        if self.pos_id is not None:
            self.control.update()
        return value.to(device)

        
class TokenCacheControl():
    def __init__(self,token,pcache_control):
        self.token = token
        self.pcache_control = pcache_control

    def insert(self):
        pass

    def update(self,score=1):
        pass

    def delete(self):
        pass

    def to(self,device):
        self.token.to(device=device)

    def to_disk(self):
        self.token.to_disk()


class TokenCacheLRU(TokenCacheControl):
    def __init__(self,token,pcache_control):
        super().__init__(token=token,pcache_control=pcache_control)
        # super().__init__(key=key,value=value,prefix_cache=prefix_cache,pos_id=pos_id,device=device)
        self.prev = self
        self.next = self
        self.score = [0,0]

    def update(self,score=[1,1]):
        self.score[0] += score[0]
        self.score[1] += score[1]
        self.pcache_control.update(self.token)

    def insert(self,item):  #insert to next
        if isinstance(item,TokenCacheGeneral):
            item = item.control
        self.next.prev = item
        item.next = self.next
        item.prev = self
        self.next = item

    def delete(self):
        if self.prev is not None and self.next is not None:
            self.prev.next = self.next
            self.next.prev = self.prev
            self.next = self
            self.prev = self
        return self.token

class TokenCacheScore(TokenCacheControl):
    def __init__(self,token,pcache_control):
        super().__init__(token=token,pcache_control=pcache_control)
        self.heap_pos = 0
        self.score = [0,0]
        self.heap = None

    def __lt__(self, other):
        return self.score < other.score
    def __le__(self, other):
        return self.score <= other.score
    def __eq__(self, other):
        return self.score == other.score
    def __ne__(self, other):
        return self.score != other.score
    def __gt__(self, other):
        return self.score > other.score
    def __le__(self, other):
        return self.score >= other.score

    def insert(self,heap,score=None):
        self.heap = heap
        if score is not None:
            self.score = score
        heap.push(self)

    def update(self,score=[1,1]):
        score[0] = self.score[0] + score[0]
        score[1] = self.score[1] + score[1]
        self.pcache_control.update_score(self.token,score)


    def update_score(self,new_score):
        # score = self.score
        self.score = new_score
        # if self.heap is not None:
        #     self.heap.delete(self)
        # if self.heap:
        #     if new_score > score:
        #         self.heap.siftup(self.heap_pos)
        #     elif new_score < score:
        #         self.heap.siftdown(self.heap_pos)
        # print(1,self.heap_pos,score,self.score)

class TokenCacheDisk(TokenCache):
    def __init__(self,key,value,layer_id,prefix_cache,control,pos_id,device='cuda:0',key_exist=True,value_exist=True):
        super().__init__(key=key,value=value,layer_id=layer_id,prefix_cache=prefix_cache,control=control,pos_id=pos_id,device=device,key_exist=key_exist,value_exist=value_exist)
        filename,_ = os.path.splitext(self.prefix_cache.path)
        self.path = f'{filename}_{self.pos_id}'
        if self.key_exist:
            self.path += '_k'
        if self.value_exist:
            self.path += '_v'
        self.path += '.npy'
        self.file_exist = False

    def to(self, device):
        if device == 'disk':
            self.to_disk()
            return
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.key_exist and self.value_exist:
                self.key = torch.from_numpy(data[0,:]).to(device)
                self.value = torch.from_numpy(data[1,:]).to(device)
            elif self.key_exist:
                self.key = torch.from_numpy(data[:]).to(device)
            elif self.value_exist:
                self.value = torch.from_numpy(data[:]).to(device)     
        else:
            if self.key_exist:
                self.key = self.key.to(device)
            if self.value_exist:
                self.value = self.value.to(device)
        self.device = device
        if self.pos_id is not None:
            self.prefix_cache.device_map[self.pos_id] = device 

    def to_disk(self):
        if self.device == 'disk':
            return 
        self.device = 'disk'
        if self.pos_id is not None:
            self.prefix_cache.device_map[self.pos_id] = 'disk'
        if not self.file_exist:
            if self.key_exist and self.value_exist:
                key = np.array(self.key.to('cpu'))
                value = np.array(self.value.to('cpu'))
                self.shape = [2]
                self.shape.extend(key.shape)
                self.shape = tuple(self.shape)
                # self.shape = (2,key.shape[0],key.shape[1])
                self.dtype = key.dtype
                data = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=self.shape)
                data[0,:] = key
                data[1,:] = value
                data.flush()
                self.file_exist = True
            else:
                kv = None
                if self.key_exist:
                    kv = np.array(self.key.to('cpu'))
                elif self.value_exist:
                    kv = np.array(self.value.to('cpu'))
                
                self.shape = kv.shape
                # self.shape = (2,key.shape[0],key.shape[1])
                self.dtype = kv.dtype
                data = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=self.shape)
                data[:] = kv
                data.flush()
                self.file_exist = True

        self.key = None
        self.value = None

    def get(self,device='cuda:0',head=None):
        key = None
        value = None
        if self.pos_id is not None:
            self.control.update()
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.key_exist and self.value_exist:
                key,value = torch.from_numpy(data[0,:]),torch.from_numpy(data[1,:])
            elif self.key_exist:
                key = torch.from_numpy(data[:])
            elif self.value_exist:
                value = torch.from_numpy(data[:])
        else:
            if self.key_exist:
                key = self.key.to(device)
            if self.value_exist:
                value = self.value.to(device)
        if head is not None:
            if self.key_exist:
                key = key[head]
            if self.value_exist:
                value = value[head]
        return key,value
    
    def get_key(self,device='cuda:0',head=None):
        if not self.key_exist:
            return None
        if self.pos_id is not None:
            self.control.update()
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.value_exist:
                key = torch.from_numpy(data[0,:])
            else:
                key = torch.from_numpy(data[:])
        else:
            key = self.key
        if head is not None:
            key = key[head]
        return key.to(device)
    
    def get_value(self,device='cuda:0',head=None):
        if not self.value_exist:
            return None
        if self.pos_id is not None:
            self.control.update()
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.key_exist:
                value = torch.from_numpy(data[1,:])
            else:
                value = torch.from_numpy(data[:])
        else:
            value = self.value
        if head is not None:
            value = value[head]
        return value.to(device)
    
class ChunkCache(TokenCacheDisk):
    def __init__(self,key,value,layer_id,prefix_cache,control,pos_id,device='cuda:0',chunk_size=64,key_exist=True,value_exist=True):
        # key_exist and value_exist are used to determine whether the key and value tensors are present
        # used for supporting KV division
        self.key_exist=key_exist
        self.value_exist=value_exist
        if key is None:
            self.key_exist = False
        if value is None:
            self.value_exist = False
        if self.key_exist:
            kv_shape = list(key.shape)
            self.torch_dtype = key.dtype
        else:
            kv_shape = list(value.shape)
            self.torch_dtype = value.dtype
        # kv_shape[0] = chunk_size

        kv_shape = tuple(kv_shape)

        self.device = None
        self.prefix_cache = prefix_cache
        self.pos_id = pos_id
        self.control = control
        self.layer_id = layer_id
        self.size = 0
        filename,_ = os.path.splitext(self.prefix_cache.path)
        self.path = f'{filename}_{self.pos_id}'

        if self.key_exist:
            self.key = torch.empty(dtype=self.torch_dtype,size=kv_shape,device=device)
            self.key[:key.shape[0]] = key[:]

            self.path += '_k'
            self.size += (self.key.element_size() * self.key.nelement()) / 1024 / 1024
        if self.value_exist:
            self.value = torch.empty(dtype=self.torch_dtype,size=kv_shape,device=device)
            self.value[:value.shape[0]] = value[:]

            self.path += '_v'
            self.size += (self.value.element_size() * self.value.nelement()) / 1024 / 1024

        self.path += '.npy'

        self.file_exist = False
        self.to(device=device)

        if not self.file_exist:
            if self.key_exist and self.value_exist:
                key = np.array(self.key.to('cpu'))
                value = np.array(self.value.to('cpu'))
                self.shape = [2]
                self.shape.extend(key.shape)
                self.shape = tuple(self.shape)
                # self.shape = (2,key.shape[0],key.shape[1])
                self.dtype = key.dtype
                data = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=self.shape)
                data[0,:] = key
                data[1,:] = value
                data.flush()
                self.file_exist = True
            else:
                if self.key_exist:
                    kv = np.array(self.key.to('cpu'))
                if self.value_exist:
                    kv = np.array(self.value.to('cpu'))
                self.shape = kv.shape
                # self.shape = (2,key.shape[0],key.shape[1])
                self.dtype = kv.dtype
                data = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=self.shape)
                data[:] = kv
                data.flush()
                self.file_exist = True

    def _index_for_source(self, pos_id, source):
        if pos_id is None:
            return None
        if source.device.type == 'cpu':
            return pos_id.to('cpu')
        return pos_id.to(source.device)

    def get(self,pos_id,device='cuda:0',head=None, is_prefetching=False):
        if self.pos_id is not None and is_prefetching == False:
            st = time.monotonic()
            if pos_id is None:
                self.control.update([1,self.shape[1]])
            else:
                self.control.update([1,pos_id.shape[0]])
            Pcache.control_update_time += time.monotonic() - st
        keys = None
        values = None
        if self.device == 'disk':
            # print(f'self.device: {self.device}')
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.key_exist and self.value_exist:
                keys,values = torch.from_numpy(data[0,:]),torch.from_numpy(data[1,:])
            elif self.key_exist:
                keys = torch.from_numpy(data[:])
            elif self.value_exist:
                values = torch.from_numpy(data[:])
        else:
            # print(f'self.device: {self.device}')
            keys,values = self.key, self.value
        if self.key_exist:
            key_pos_id = self._index_for_source(pos_id, keys)
            if head is None:
                if pos_id is None:
                    keys = keys.to(device)
                else:
                    keys = keys[key_pos_id].to(device)
            else:
                if pos_id is None:
                    keys = keys[:,head].to(device)
                else:
                    keys = keys[key_pos_id,head].to(device)
        if self.value_exist:
            value_pos_id = self._index_for_source(pos_id, values)
            if head is None:
                if pos_id is None:
                    values = values.to(device)
                else:
                    values = values[value_pos_id].to(device)
            else:
                if pos_id is None:
                    values = values[:,head].to(device)
                else:
                    values = values[value_pos_id,head].to(device)
        return keys,values
        # return keys.to(device),values.to(device)
        # pos_id = pos_id.to(self.device)
        # return torch.index_select(keys, 0, pos_id.to(self.device)).to(device),torch.index_select(values, 0, pos_id.to(self.device)).to(device)

    def get_key(self,pos_id,device='cuda:0',head=None, is_prefetching=False):
        if not self.key_exist:
            return None
        if self.pos_id is not None and is_prefetching==False:
            st = time.monotonic()
            if pos_id is None:
                self.control.update([1,self.shape[1]])
            else:
                self.control.update([1,pos_id.shape[0]])
            Pcache.control_update_time += time.monotonic() - st
        keys = None
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.value_exist:
                keys = torch.from_numpy(data[0,:])
            else:
                keys = torch.from_numpy(data[:])
            # pin keys
            if not keys.is_pinned():
                keys = keys.pin_memory()
        else:
            keys = self.key
        # 加入 non_blocking=True
        key_pos_id = self._index_for_source(pos_id, keys)
        if head is None:
            if pos_id is None:
                keys = keys.to(device, non_blocking=True)
            else:
                keys = keys[key_pos_id].to(device, non_blocking=True)
        else:
            if pos_id is None:
                keys = keys[:,head].to(device, non_blocking=True)
            else:
                keys = keys[key_pos_id,head].to(device, non_blocking=True)
        return keys

    def asyn_get_key(self,pos_id,device='cuda:0',head=None, output_buffer=None):
        """
        使用 copy_ 方法优化的版本
        
        Args:
            pos_id: 位置索引
            device: 目标设备
            head: 头索引（可选）
            output_buffer: 预分配的输出缓冲区（可选）
        """
        if not self.key_exist:
            return None
            
        if self.pos_id is not None:
            self.control.update([1, pos_id.shape[0]])
        
        # 1. 获取源数据
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.value_exist:
                source_keys = torch.from_numpy(data[0,:])
            else:
                source_keys = torch.from_numpy(data[:])
            
            # Pin memory for faster transfer
            if not source_keys.is_pinned():
                source_keys = source_keys.pin_memory()
        else:
            source_keys = self.key
        
        # 2. 选择需要的数据
        if head is None:
            selected_keys = source_keys[pos_id]
        else:
            selected_keys = source_keys[pos_id, head]
        
        # 3. 使用 copy_ 方法传输数据
        if output_buffer is not None:
            # 使用预分配的缓冲区
            if output_buffer.shape != selected_keys.shape:
                raise ValueError(f"Buffer shape {output_buffer.shape} doesn't match data shape {selected_keys.shape}")
            output_buffer.copy_(selected_keys, non_blocking=True)
            return output_buffer
        else:
            # 分配新张量并拷贝（仍比 to() 更高效）
            target_tensor = torch.empty_like(selected_keys, device=device)
            target_tensor.copy_(selected_keys, non_blocking=True)
            return target_tensor
    
    def get_value(self,pos_id,device='cuda:0',head=None, is_prefetching=False):
        if not self.value_exist:
            return None
        if self.pos_id is not None and is_prefetching==False:
            st = time.monotonic()
            if pos_id is None:
                self.control.update([1,self.shape[1]])
            else:
                self.control.update([1,pos_id.shape[0]])
            Pcache.control_update_time += time.monotonic() - st
        values = None
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.key_exist:
                values = torch.from_numpy(data[1,:])
            else:
                values = torch.from_numpy(data[:])
            # pin values
            if not values.is_pinned():
                values = values.pin_memory()
        else:
            values = self.value
        # 加入 non_blocking=True
        value_pos_id = self._index_for_source(pos_id, values)
        if head is None:
            if pos_id is None:
                values = values.to(device, non_blocking=True)
            else:
                values = values[value_pos_id].to(device, non_blocking=True)
        else:
            if pos_id is None:
                values = values[:,head].to(device, non_blocking=True)
            else:
                values = values[value_pos_id,head].to(device, non_blocking=True)
        return values

    def asyn_get_value(self,pos_id,device='cuda:0',head=None, output_buffer=None):
        """
        使用 copy_ 方法优化的版本
        
        Args:
            pos_id: 位置索引
            device: 目标设备
            head: 头索引（可选）
            output_buffer: 预分配的输出缓冲区（可选）
        """
        if not self.value_exist:
            return None
            
        if self.pos_id is not None:
            self.control.update([1, pos_id.shape[0]])
        
        # 1. 获取源数据
        if self.device == 'disk':
            uncache(self.path)
            data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
            if self.key_exist:
                source_values = torch.from_numpy(data[1,:])
            else:
                source_values = torch.from_numpy(data[:])
            
            # Pin memory for faster transfer
            if not source_values.is_pinned():
                source_values = source_values.pin_memory()
        else:
            source_values = self.value
        
        # 2. 选择需要的数据
        if head is None:
            selected_values = source_values[pos_id]
        else:
            selected_values = source_values[pos_id, head]
        
        # 3. 使用 copy_ 方法传输数据
        if output_buffer is not None:
            # 使用预分配的缓冲区
            if output_buffer.shape != selected_values.shape:
                raise ValueError(f"Buffer shape {output_buffer.shape} doesn't match data shape {selected_values.shape}")
            output_buffer.copy_(selected_values, non_blocking=True)
            return output_buffer
        else:
            # 分配新张量并拷贝（仍比 to() 更高效）
            target_tensor = torch.empty_like(selected_values, device=device)
            target_tensor.copy_(selected_values, non_blocking=True)
            return target_tensor


class TokenCacheGeneral():
    def __init__(self,key,value,pos_id,prefix_cache,layer_id,pcache_control,cache_type='LRU',disk_type='Single',chunk_size=64,key_exist=True,value_exist=True):
        if disk_type == 'Batched':
            # 将所有的 KV cache 存储成一个大文件，无法进行部分加载，因为一次就把一整个文件加载进来了
            self.token = TokenCache(key=key,value=value,layer_id=layer_id,control=self,pos_id=pos_id,prefix_cache=prefix_cache,key_exist=key_exist,value_exist=value_exist)
        elif disk_type == 'Single':
            self.token = TokenCacheDisk(key=key,value=value,layer_id=layer_id,control=self,pos_id=pos_id,prefix_cache=prefix_cache,key_exist=key_exist,value_exist=value_exist)
        elif disk_type == 'Chunk':
            self.token = ChunkCache(key=key,value=value,layer_id=layer_id,control=self,pos_id=pos_id,prefix_cache=prefix_cache,chunk_size=chunk_size,key_exist=key_exist,value_exist=value_exist)
        else:
            assert 'unsupport disk type'
        # control 就是缓存替换策略
        if cache_type == 'LRU':
            self.control = TokenCacheLRU(token=self,pcache_control=pcache_control)
        elif cache_type == 'LFU':
            self.control = TokenCacheScore(token=self,pcache_control=pcache_control)
        elif cache_type == 'CKLFU':
            self.control = TokenCacheScore(token=self,pcache_control=pcache_control)
        else:
            assert 'unsupport cache type'
        self.disk_type = disk_type
        self.cache_type = cache_type
        self.control.pcache_control.insert(self)

    def to(self, device):
        # if self.device != device:
        #     print(f"[TokenCache] Token pos_id={self.pos_id} from {self.device} -> {device}")
        self.token.to(device=device)

    def to_disk(self):
        # if self.device != 'disk':
        #     print(f"[TokenCache] Token pos_id={self.pos_id} from {self.device} -> disk")
        self.token.to_disk()

    def get(self,pos_id=None,device='cuda:0',head=None,is_prefetching=False):
        if self.disk_type == 'Chunk':
            # print(f'self.disk_type: {self.disk_type}')
            return self.token.get(pos_id=pos_id,device=device,head=head,is_prefetching=is_prefetching)
        return self.token.get(device=device,head=head)
    
    def insert(self,item):
        self.control.insert(item=item)

    def delete(self):
        return self.control.delete()
    
    def update(self,score=[1,1]):
        self.control.update(score=score)

    def update_score(self,score):
        self.control.update_score(score)

    def get_key(self,pos_id=None,device='cuda:0',head=None,is_prefetching=False):
        if self.disk_type == 'Chunk':
            return self.token.get_key(pos_id=pos_id,device=device,head=head,is_prefetching=is_prefetching)
        return self.token.get_key(device=device,head=head)

    def asyn_get_key(self,pos_id=None,device='cuda:0',head=None, output_buffer=None):
        """
        使用 copy_ 方法优化的版本
        
        Args:
            pos_id: 位置索引
            device: 目标设备
            head: 头索引（可选）
            output_buffer: 预分配的输出缓冲区（可选）
        """
        if self.disk_type == 'Chunk':
            return self.token.asyn_get_key(pos_id=pos_id,device=device,head=head, output_buffer=output_buffer)
        return self.token.get_key(pos_id=pos_id,device=device,head=head)
    
    def get_value(self,pos_id=None,device='cuda:0',head=None,is_prefetching=False):
        if self.disk_type == 'Chunk':
            return self.token.get_value(pos_id=pos_id,device=device,head=head,is_prefetching=is_prefetching)
        return self.token.get_value(device=device,head=head)
    
    def asyn_get_value(self,pos_id=None,device='cuda:0',head=None, output_buffer=None):
        """
        使用 copy_ 方法优化的版本
        
        Args:
            pos_id: 位置索引
            device: 目标设备
            head: 头索引（可选）
            output_buffer: 预分配的输出缓冲区（可选）
        """
        if self.disk_type == 'Chunk':
            return self.token.asyn_get_value(pos_id=pos_id,device=device,head=head, output_buffer=output_buffer)
        return self.token.get_value(pos_id=pos_id,device=device,head=head)
    
    def get_chunk_hit(self):
        return self.control.score

    @property
    def device(self):
        return self.token.device
    
    @property
    def pcache_control(self):
        return self.control.pcache_control
    
    @property
    def score(self):
        return self.control.score
    
    @property
    def pos_id(self):
        return self.token.pos_id
    
    @property
    def size(self):
        return self.token.size
    
    @property
    def key(self):
        return self.token.key
    
    @property
    def value(self):
        return self.token.value
        

class PrefixKVLayer():
    def __init__(self,prefix_id,key,value,layer_id,pcache_control,cache_type='LRU',disk_type='Single',cpu_gather=False,head_ids=[0,1,2],chunk_size=64,reorder=None):
        # 一个 prefix_id + layer_id 对应一个 PrefixKVLayer 实例，表示第 prefix_id 个 prefix 的第 layer_id 层 kv 缓存
        self.prefix_id = prefix_id
        self.path = os.path.join(PrefixKV.disk_dir,f'{prefix_id}_{layer_id}.npy')
        # self.tokens = np.array([TokenCacheGeneral(key=key[i,3:,:],value=value[i,3:,:],layer_id=layer_id,prefix_cache=self,pos_id=i,pcache_control=pcache_control,cache_type=cache_type,disk_type=disk_type) for i in range(key.shape[0])])
        self.reorder = reorder
        if reorder is not None:
            self.reverse_reorder = torch.full((len(reorder),), -1, device=reorder.device) # 建立 reorder 的反向映射
            self.reverse_reorder[reorder] = torch.arange(len(reorder), device=reorder.device)
        else:
            self.reverse_reorder = None
        key = key[reorder]
        value = value[reorder]
        if disk_type == 'Chunk':
            self.real_len = key.shape[0]
            keys = torch.split(key,chunk_size,dim=0)
            values = torch.split(value,chunk_size,dim=0)
            self.chunk_num = len(keys)
            self.device_map = np.array(['cuda:0' for _ in range(len(keys))]) # gpu:0,cpu:1,disk:2
            self.tokens = np.array([TokenCacheGeneral(key=keys[i],value=values[i],layer_id=layer_id,prefix_cache=self,pos_id=i,pcache_control=pcache_control,cache_type=cache_type,disk_type=disk_type,chunk_size=chunk_size) for i in range(len(keys))])
            self.head_token = TokenCacheGeneral(key=key[:,head_ids,:],value=None,layer_id=layer_id,prefix_cache=self,pos_id=None,pcache_control=pcache_control,cache_type=cache_type,disk_type='Single',value_exist=False)
        elif disk_type == 'KV_Division':
            # so key and value are not placed together, there's actual need to do this!
            self.real_len = key.shape[0]
            keys = torch.split(key,chunk_size,dim=0)
            values = torch.split(value,chunk_size,dim=0)
            self.chunk_num = len(keys)
            self.device_map = np.array(['cuda:0' for _ in range(2*len(keys))]) # gpu:0,cpu:1,disk:2
            self.key_tokens = np.array([TokenCacheGeneral(key=keys[i],value=None,layer_id=layer_id,prefix_cache=self,pos_id=i,pcache_control=pcache_control,cache_type=cache_type,disk_type='Chunk',chunk_size=chunk_size,key_exist=True,value_exist=False) for i in range(len(keys))])
            self.value_tokens = np.array([TokenCacheGeneral(key=None,value=values[i],layer_id=layer_id,prefix_cache=self,pos_id=i+self.chunk_num,pcache_control=pcache_control,cache_type=cache_type,disk_type='Chunk',chunk_size=chunk_size,key_exist=False,value_exist=True) for i in range(len(keys))])
            self.head_token = TokenCacheGeneral(key=key[:,head_ids,:],value=None,layer_id=layer_id,prefix_cache=self,pos_id=None,pcache_control=pcache_control,cache_type=cache_type,disk_type='Single',value_exist=False)
        else: # single
            self.device_map = np.array(['cuda:0' for _ in range(key.shape[0])]) # gpu:0,cpu:1,disk:2
            self.tokens = np.array([TokenCacheGeneral(key=key[i,:,:],value=value[i,:,:],layer_id=layer_id,prefix_cache=self,pos_id=i,pcache_control=pcache_control,cache_type=cache_type,disk_type=disk_type) for i in range(key.shape[0])])
            self.head_token = TokenCacheGeneral(key=key[:,head_ids,:],value=None,layer_id=layer_id,prefix_cache=self,pos_id=None,pcache_control=pcache_control,cache_type=cache_type,disk_type=disk_type,value_exist=False)
        self.cpu_gather = cpu_gather
        self.head_ids=head_ids

        self.layer_id = layer_id
        self.disk_type = disk_type # 与其叫 disk_type，不如叫在disk上的存储形式
        # torch_dtype 是 torch 的数据类型，和dtype不一样，dtype是numpy的。所以张量操作需要使用torch_dtype
        self.torch_dtype = key.dtype
        self.disk_map = None
        # 一次性获取多个token的kv的时候，作为临时的GPU张量缓冲区，用于收集所有的kv，同意返回
        # 我们做预取的时候也需要使用这样的缓冲区
        self.key_buffer = None
        self.value_buffer = None
        self.chunk_size = chunk_size

        self.gpu_hit = 0
        self.cpu_hit = 0
        self.disk_hit = 0
        self.disk_chunk_hit = 0

        self.dt_gpu = 0
        self.dt_cpu = 0
        self.dt_disk = 0
        # kv = torch.cat([key.unsqueeze(0),value.unsqueeze(0)],dim=0)

        key = np.array(key.cpu())
        value = np.array(value.cpu())
        self.shape = [2]
        self.shape.extend(key.shape)
        self.shape = tuple(self.shape)
        # self.shape = (2,key.shape[0],key.shape[1],key.shape[2])
        self.dtype = key.dtype
        if disk_type == 'Batched':
            data = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=self.shape)
            data[0,:] = key
            data[1,:] = value
            data.flush()

    # def print_token_device_ratio(self, sele_tokenids):
    #     # sele_tokenids: 1D tensor，表示重要token的索引
    #     gpu_cnt = 0
    #     cpu_cnt = 0
    #     disk_cnt = 0
    #     for tid in sele_tokenids:
    #         device = self.device_map[tid // self.chunk_size]
    #         if device == 'cuda:0':
    #             gpu_cnt += 1
    #         elif device == 'cpu':
    #             cpu_cnt += 1
    #         elif device == 'disk':
    #             disk_cnt += 1
    #     total = len(sele_tokenids)
    #     with open("./time_logs/sele_load_rte_time.txt", "a") as f:
    #         f.write(f"Layer {self.layer_id}: GPU={gpu_cnt/total:.4%}, CPU={cpu_cnt/total:.4%}, Disk={disk_cnt/total:.4%} (total={total})\n")

    def to(self,pos_id,device):
        for pos in pos_id:
            self.device_map[pos] = device
            if device == 'disk':
                self.to_disk(pos_id)
            else:
                self.tokens[pos].to(device)

    def to_disk(self,pos_id):
        key,value = self.tokens[pos_id].get()
        key = np.array(key)
        value = np.array(value)
        # data = np.memmap(self.path, dtype=self.dtype, mode='w+', shape=self.shape)
        # data[0,:] = key
        # data[1,:] = value
        # data.flush()
        self.tokens[pos_id].to_disk()
        self.device_map[pos_id] = 'disk'

    def get(self,pos_id=None,need_key=True,need_value=True,is_prefetching=False):
        # pos_id -> tensor
        if self.disk_type == 'Chunk' or self.disk_type == 'KV_Division':
            # print('chunk')
            return self.get_by_chunk(pos_id=pos_id,need_key=need_key,need_value=need_value,is_prefetching=is_prefetching)
        else:
            return self.get_parallel(pos_id=pos_id,need_key=need_key,need_value=need_value)
        # keys = []
        # values = []
        # if pos_id is None:
        #     for id in range(len(self.tokens)):
        #         key,value = self.get_single(id)
        #         keys.append(key)
        #         values.append(value)
        # else:
        #     for id in pos_id:
        #         key,value = self.get_single(id)
        #         keys.append(key)
        #         values.append(value)
        # key = torch.stack(keys)
        # value = torch.stack(values)

        # key,value = self.get_single(pos_id[0],layer=layer)
        # key = key.unsqueeze(0)
        # value = value.unsqueeze(0)
        # for i in range(1,len(pos_id)):
        #     new_key,new_value = self.get_single(pos_id=pos_id[i],layer=layer)
        #     new_key = new_key.unsqueeze(0)
        #     new_value = new_value.unsqueeze(0)
        #     key = torch.cat([key,new_key],dim=0)
        #     value = torch.cat([value,new_value],dim=0)
        # return key,value

    def prefetch(self,pos_id=None,need_key=True,need_value=True,k_v_num_tokens=None,time_budget=0.0):
        if self.disk_type == 'Chunk' or self.disk_type == 'KV_Division':
            # print('chunk')
            return self.prefetch_by_chunk(pos_id=pos_id,need_key=need_key,need_value=need_value,k_v_num_tokens=k_v_num_tokens,time_budget=time_budget)

    def prefetch_update(self,pos_id=None,chunk_ids=None,need_key=True,need_value=True):
        if self.disk_type == 'Chunk' or self.disk_type == 'KV_Division':
            # print('chunk')
            return self.prefetch_update_by_chunk(pos_id=pos_id,chunk_ids=chunk_ids,need_key=need_key,need_value=need_value)
        
    def prefetch_update_by_chunk(self,pos_id=None,chunk_ids=None,need_key=True,need_value=True):
        if not (need_key or need_value):
            return None 
        if pos_id is None:
            pos_id = torch.arange(self.real_len)
        if len(pos_id.shape) == 1:
            # if pos_id is a 1D tensor, it means we want to get the key and value of multiple tokens
            if chunk_ids is None:
                # 说明我们需要根据 pos_id 来计算 chunk_ids
                pos_id = self.reorder[pos_id]
                pos2chunk = pos_id // self.chunk_size
                chunk_ids = torch.unique(pos2chunk)
            # 如果有 chunk_ids 直接使用 chunk_ids 就行了
            # print(f'len_chunk_ids: {chunk_ids.shape[0]}')
            # print(f'Is key_chunks contiguous? {key_chunks.is_contiguous()}')
            # print(f'Is value_chunks contiguous? {value_chunks.is_contiguous()}')
            # import sys
            # sys.exit(10086)
            # chunk_start_move_event = torch.cuda.Event(enable_timing=True)
            # chunk_end_move_event = torch.cuda.Event(enable_timing=True)

            for chunk_id in chunk_ids:
                # chunk_start_move_event.record()
                if self.disk_type == 'KV_Division':
                    if need_key:
                        chunk = self.key_tokens[chunk_id]
                        # chunk_device = chunk.device
                        chunk.pcache_control.prefetch_update(chunk)
                        # print(f'chunk data shape: {chunk.token.key.shape}')
                        # sys.exit(10086)
                    if need_value:
                        chunk = self.value_tokens[chunk_id]
                        # chunk_device = chunk.device
                        chunk.pcache_control.prefetch_update(chunk)
                        # print(f'chunk data shape: {chunk.token.value.shape}')
                        # sys.exit(10086)
                # chunk_end_move_event.record()

                # torch.cuda.synchronize()
                # chunk_move_time = chunk_start_move_event.elapsed_time(chunk_end_move_event) / 1000

                # with open("./time_logs/chunk_move_time.txt", "a") as f:
                #     if chunk_device == 'cuda:0':
                #         f.write(f"Layer {self.layer_id}: chunk_id {chunk_id} GPU -> GPU, time: {chunk_move_time:.8f} s\n")
                #     elif chunk_device == 'cpu':
                #         f.write(f"Layer {self.layer_id}: chunk_id {chunk_id} CPU -> GPU, time: {chunk_move_time:.8f} s\n")
                #     elif chunk_device == 'disk':
                #         f.write(f"Layer {self.layer_id}: chunk_id {chunk_id} Disk -> GPU, time: {chunk_move_time:.8f} s\n")
            return chunk_ids.shape[0]

    def get_head(self):
        return self.head_token.get_key()
    
    def get_by_chunk(self,pos_id=None,need_key=True,need_value=True,is_prefetching=False):
        # give a pos_id, return the key and value of the tokens in the pos_id
        # these tokens can be distributed in different chunks
        # and these chunks can be in different devices
        # one token has one pos id, and pos_id -> chunk_id * chunk_size + token_id in the chunk
        # device_map[chunk_id] -> device
        # why there 's need_key need_value ? because when we are selecting important tokens, we may only need key or value
        if not (need_key or need_value):
            return None 
        fast_result = self._get_by_chunk_fast_cpu(
            pos_id=pos_id,
            need_key=need_key,
            need_value=need_value,
        )
        if fast_result is not None:
            return fast_result
        if pos_id is None:
            # TODO: need reorder
            # pos_id = torch.arange(self.real_len)
            key_buffer = None
            value_buffer = None
            if need_key:
                key_buffer = torch.empty(dtype=self.torch_dtype,size=self.shape[1:],device='cuda:0')
            if need_value:
                value_buffer = torch.empty(dtype=self.torch_dtype,size=self.shape[1:],device='cuda:0')
            for chunk_id in range(self.chunk_num-1):
                if self.disk_type == 'KV_Division':
                    if need_key:
                        device_key = self.device_map[chunk_id]
                    if need_value:
                        device_value = self.device_map[chunk_id+self.chunk_num]
                else:
                    device = self.device_map[chunk_id]

                # torch.cuda.synchronize()
                # st = time.monotonic()

                if self.disk_type == 'KV_Division':
                    if need_key:
                        chunk = self.key_tokens[chunk_id]
                        key_buffer[(chunk_id)*self.chunk_size:(chunk_id+1)*self.chunk_size] = chunk.get_key()
                    if need_value:
                        chunk = self.value_tokens[chunk_id]
                        value_buffer[(chunk_id)*self.chunk_size:(chunk_id+1)*self.chunk_size] = chunk.get_value()
                else:
                    chunk = self.tokens[chunk_id]
                    if need_key and need_value:
                        key_buffer[(chunk_id)*self.chunk_size:(chunk_id+1)*self.chunk_size],value_buffer[(chunk_id)*self.chunk_size:(chunk_id+1)*self.chunk_size] = chunk.get()
                    elif not need_key:
                        value_buffer[(chunk_id)*self.chunk_size:(chunk_id+1)*self.chunk_size] = chunk.get_value()
                    elif not need_value:
                        key_buffer[(chunk_id)*self.chunk_size:(chunk_id+1)*self.chunk_size] = chunk.get_key()
                # torch.cuda.synchronize()
                # dt = time.monotonic() - st
                if self.disk_type == 'KV_Division':
                    if need_key:
                        if device_key == 'cuda:0':
                            self.gpu_hit += self.chunk_size
                            # self.dt_gpu += dt
                        elif device_key == 'cpu':
                            self.cpu_hit += self.chunk_size
                            # self.dt_cpu += dt
                        elif device_key == 'disk':
                            self.disk_hit += self.chunk_size
                            self.disk_chunk_hit += 1
                            # self.dt_disk += dt
                    if need_value:
                        if device_value == 'cuda:0':
                            self.gpu_hit += self.chunk_size
                            # self.dt_gpu += dt
                        elif device_value == 'cpu':
                            self.cpu_hit += self.chunk_size
                            # self.dt_cpu += dt
                        elif device_value == 'disk':
                            self.disk_hit += self.chunk_size
                            self.disk_chunk_hit += 1
                            # self.dt_disk += dt
                else:
                    if device == 'cuda:0':
                        self.gpu_hit += self.chunk_size
                        # self.dt_gpu += dt
                    elif device == 'cpu':
                        self.cpu_hit += self.chunk_size
                        # self.dt_cpu += dt
                    elif device == 'disk':
                        self.disk_hit += self.chunk_size
                        self.disk_chunk_hit += 1
                        # self.dt_disk += dt
            chunk_id = self.chunk_num - 1
            if self.disk_type == 'KV_Division':
                if need_key:
                    device_key = self.device_map[chunk_id]
                if need_value:
                    device_value = self.device_map[chunk_id+self.chunk_num]
            else:
                device = self.device_map[chunk_id]

            # torch.cuda.synchronize()
            # st = time.monotonic()
            if self.disk_type == 'KV_Division':
                if need_key:
                    chunk = self.key_tokens[chunk_id]
                    key_buffer[chunk_id * self.chunk_size:] = chunk.get_key()
                if need_value:
                    chunk = self.value_tokens[chunk_id]
                    value_buffer[chunk_id * self.chunk_size:] = chunk.get_value()
            else:
                chunk = self.tokens[chunk_id]
                if need_key and need_value:
                    key_buffer[chunk_id * self.chunk_size:],value_buffer[chunk_id * self.chunk_size:] = chunk.get()
                elif not need_key:
                    value_buffer[chunk_id * self.chunk_size:] = chunk.get_value()
                elif not need_value:
                    key_buffer[chunk_id * self.chunk_size:] = chunk.get_key()
            # torch.cuda.synchronize()
            # dt = time.monotonic() - st
            if self.disk_type == 'KV_Division':
                if need_key:
                    if device_key == 'cuda:0':
                        self.gpu_hit += self.shape[1] % self.chunk_size
                        # self.dt_gpu += dt
                    elif device_key == 'cpu':
                        self.cpu_hit += self.shape[1] % self.chunk_size
                        # self.dt_cpu += dt
                    elif device_key == 'disk':
                        self.disk_hit += self.shape[1] % self.chunk_size
                        self.disk_chunk_hit += 1
                        # self.dt_disk += dt
                if need_value:
                    if device_value == 'cuda:0':
                        self.gpu_hit += self.shape[1] % self.chunk_size
                        # self.dt_gpu += dt
                    elif device_value == 'cpu':
                        self.cpu_hit += self.shape[1] % self.chunk_size
                        # self.dt_cpu += dt
                    elif device_value == 'disk':
                        self.disk_hit += self.shape[1] % self.chunk_size
                        self.disk_chunk_hit += 1
                        # self.dt_disk += dt
            else:
                if device == 'cuda:0':
                    self.gpu_hit += self.shape[1] % self.chunk_size
                    # self.dt_gpu += dt
                elif device == 'cpu':
                    self.cpu_hit += self.shape[1] % self.chunk_size
                    # self.dt_cpu += dt
                elif device == 'disk':
                    self.disk_hit += self.shape[1] % self.chunk_size
                    self.disk_chunk_hit += 1
                    # self.dt_disk += dt

            return key_buffer,value_buffer

        if len(pos_id.shape) == 1:
            # if pos_id is a 1D tensor, it means we want to get the key and value of multiple tokens
            shape = list(self.shape[1:])
            shape[0] = pos_id.shape[0]
            shape = tuple(shape)
            key_buffer = None
            value_buffer = None
            # if isinstance(pos_id, torch.Tensor):
            #     pos_id = pos_id.cpu()
            input_device = pos_id.device
            if self.reorder is not None:
                pos_id = self.reorder[pos_id.to(self.reorder.device)].to(input_device)

            if need_key:
                key_buffer = torch.empty(dtype=self.torch_dtype,size=shape,device='cuda:0')
            if need_value:
                value_buffer = torch.empty(dtype=self.torch_dtype,size=shape,device='cuda:0')

            pos2chunk = pos_id // self.chunk_size
            token_ids = pos_id % self.chunk_size
            # print(f'chunksize: {self.chunk_size}\n')
            chunk_ids = torch.unique(pos2chunk)    # get rid of replicas
            # dt = 0
            for chunk_id in chunk_ids:
                mask = pos2chunk == chunk_id 
                # mask is used to select the tokens in the current chunk
                # if not in the current chunk, mask will be false, and this token will not be selected
                token_id = token_ids[mask]
                chunk_index = int(chunk_id.item())
                # print(f"token_id in the chunk = {token_id}")
                if self.disk_type == 'KV_Division':
                    if need_key:
                        device_key = self.device_map[chunk_index]
                    if need_value:
                        device_value = self.device_map[chunk_index+self.chunk_num]
                else:
                    device = self.device_map[chunk_index]

                # torch.cuda.synchronize()
                # st = time.monotonic()

                if self.disk_type == 'KV_Division':
                    if need_key:
                        chunk = self.key_tokens[chunk_index]
                        # torch.cuda.synchronize()
                        # get_key_st = time.monotonic()
                        key_buffer[mask] = chunk.get_key(token_id,is_prefetching=is_prefetching)
                        # torch.cuda.synchronize()
                        # get_key_ed = time.monotonic()
                        # if not is_prefetching :
                        #     with open("prefetch_profile/get_token_profile.txt", 'a') as f:
                        #         f.write(f"get time: {get_key_ed - get_key_st:.8f} s, token_id len: {len(token_id)}, device: {device_key}\n")
                    if need_value:
                        chunk = self.value_tokens[chunk_index]
                        # torch.cuda.synchronize()
                        # get_value_st = time.monotonic()
                        value_buffer[mask] = chunk.get_value(token_id,is_prefetching=is_prefetching)
                        # torch.cuda.synchronize()
                        # get_value_ed = time.monotonic()
                        # if not is_prefetching:
                        #     with open("prefetch_profile/get_token_profile.txt", 'a') as f:
                        #         f.write(f"get time: {get_value_ed - get_value_st:.8f} s, token_id len: {len(token_id)}, device: {device_value}\n")
                else:
                    chunk = self.tokens[chunk_index]
                    if need_key and need_value:
                        key_buffer[mask],value_buffer[mask] = chunk.get(token_id,is_prefetching=is_prefetching)
                    elif not need_key:
                        value_buffer[mask] = chunk.get_value(token_id,is_prefetching=is_prefetching)
                    elif not need_value:
                        key_buffer[mask] = chunk.get_key(token_id,is_prefetching=is_prefetching)
                # torch.cuda.synchronize()
                # dt = time.monotonic() - st
                if self.disk_type == 'KV_Division':
                    if need_key:
                        if device_key == 'cuda:0':
                            self.gpu_hit += token_id.shape[0]
                            # self.dt_gpu += dt
                        elif device_key == 'cpu':
                            self.cpu_hit += token_id.shape[0]
                            # self.dt_cpu += dt
                        elif device_key == 'disk':
                            self.disk_hit += token_id.shape[0]
                            self.disk_chunk_hit += 1
                            # self.dt_disk += dt
                    if need_value:
                        if device_value == 'cuda:0':
                            self.gpu_hit += token_id.shape[0]
                            # self.dt_gpu += dt
                        elif device_value == 'cpu':
                            self.cpu_hit += token_id.shape[0]
                            # self.dt_cpu += dt
                        elif device_value == 'disk':
                            self.disk_hit += token_id.shape[0]
                            self.disk_chunk_hit += 1
                            # self.dt_disk += dt
                else:
                    if device == 'cuda:0':
                        self.gpu_hit += token_id.shape[0]
                        # self.dt_gpu += dt
                    elif device == 'cpu':
                        self.cpu_hit += token_id.shape[0]
                        # self.dt_cpu += dt
                    elif device == 'disk':
                        self.disk_hit += token_id.shape[0]
                        self.disk_chunk_hit += 1
                        # self.dt_disk += dt
        else:
            # if pos_id is a 2D tensor, it means we want to get the key and value of multiple tokens in multiple batches ?
            shape = list(self.shape[1:])
            shape[0] = pos_id.shape[1]
            shape = tuple(shape)
            key_buffer = None
            value_buffer = None
            pos_id = self.reorder[pos_id]
            if need_key:
                key_buffer = torch.empty(dtype=self.torch_dtype,size=shape,device='cuda:0')
            if need_value:
                value_buffer = torch.empty(dtype=self.torch_dtype,size=shape,device='cuda:0')
            for i in range(pos_id.shape[0]):
                pos2chunk = pos_id[i] // self.chunk_size
                token_ids = pos_id[i] % self.chunk_size
                chunk_ids = torch.unique(pos2chunk)
                # dt = 0
                for chunk_id in chunk_ids:
                    mask = pos2chunk == chunk_id
                    token_id = token_ids[mask]
                    if self.disk_type == 'KV_Division':
                        if need_key:
                            device_key = self.device_map[chunk_id]
                        if need_value:
                            device_value = self.device_map[chunk_id+self.chunk_num]
                    else:
                        device = self.device_map[chunk_id]

                    torch.cuda.synchronize()
                    st = time.monotonic()

                    if self.disk_type == 'KV_Division':
                        if need_key:
                            chunk = self.key_tokens[chunk_id]
                            key_buffer[mask,i] = chunk.get_key(token_id,head=i)
                        if need_value:
                            chunk = self.value_tokens[chunk_id]
                            value_buffer[mask,i] = chunk.get_value(token_id,head=i)
                    else:
                        chunk = self.tokens[chunk_id]
                        if need_key and need_value:
                            key_buffer[mask,i],value_buffer[mask,i] = chunk.get(token_id,head=i)
                        elif not need_key:
                            value_buffer[mask,i] = chunk.get_value(token_id,head=i)
                        elif not need_value:
                            key_buffer[mask,i] = chunk.get_key(token_id,head=i)
                    torch.cuda.synchronize()
                    dt = time.monotonic() - st
                    if self.disk_type == 'KV_Division':
                        if need_key:
                            if device_key == 'cuda:0':
                                self.gpu_hit += token_id.shape[0]
                                self.dt_gpu += dt
                            elif device_key == 'cpu':
                                self.cpu_hit += token_id.shape[0]
                                self.dt_cpu += dt
                            elif device_key == 'disk':
                                self.disk_hit += token_id.shape[0]
                                self.disk_chunk_hit += 1
                                self.dt_disk += dt
                        if need_value:
                            if device_value == 'cuda:0':
                                self.gpu_hit += token_id.shape[0]
                                self.dt_gpu += dt
                            elif device_value == 'cpu':
                                self.cpu_hit += token_id.shape[0]
                                self.dt_cpu += dt
                            elif device_value == 'disk':
                                self.disk_hit += token_id.shape[0]
                                self.disk_chunk_hit += 1
                                self.dt_disk += dt
                    else:
                        if device == 'cuda:0':
                            self.gpu_hit += token_id.shape[0]
                            self.dt_gpu += dt
                        elif device == 'cpu':
                            self.cpu_hit += token_id.shape[0]
                            self.dt_cpu += dt
                        elif device == 'disk':
                            self.disk_hit += token_id.shape[0]
                            self.disk_chunk_hit += 1
                            self.dt_disk += dt

        # print(123,dt)

        # torch.cuda.synchronize()
        # st = time.monotonic()

        # # gpu gather
        # gpu_mask = np.where(self.device_map[pos_id]=='cuda:0')[0]
        # gpu_ids = pos_id[gpu_mask]
        # if gpu_ids.shape[0] > 0:
        #     for i,gpu_id in enumerate(gpu_ids):
        #         key_buffer[gpu_mask[i],:],value_buffer[gpu_mask[i],:] = self.tokens[gpu_id].get(0)     
        # torch.cuda.synchronize()
        # print(456,time.monotonic() - st)
        return key_buffer,value_buffer
    
    def _new_cpu_buffer(self, shape):
        # Pinning can fail in CPU-only tests; fall back to normal CPU memory there.
        try:
            return torch.empty(dtype=self.torch_dtype, size=shape, pin_memory=torch.cuda.is_available())
        except RuntimeError:
            return torch.empty(dtype=self.torch_dtype, size=shape)

    def _select_chunk_to_cpu(self, chunk, token_id, want_key):
        token = chunk.token
        if token.device == 'disk':
            uncache(token.path)
            data = np.memmap(token.path, dtype=token.dtype, mode='r', shape=token.shape)
            if token.key_exist and token.value_exist:
                source = torch.from_numpy(data[0 if want_key else 1])
            else:
                source = torch.from_numpy(data[:])
        else:
            source = token.key if want_key else token.value

        index = token_id.to(source.device) if source.device.type != 'cpu' else token_id
        selected = source[index]
        if selected.device.type != 'cpu':
            selected = selected.cpu()
        return selected

    def _chunk_source_to_cpu(self, chunk, want_key):
        token = chunk.token
        if token.device == 'disk':
            uncache(token.path)
            data = np.memmap(token.path, dtype=token.dtype, mode='r', shape=token.shape)
            if token.key_exist and token.value_exist:
                source = torch.from_numpy(data[0 if want_key else 1])
            else:
                source = torch.from_numpy(data[:])
        else:
            source = token.key if want_key else token.value

        if source.device.type != 'cpu':
            source = source.cpu()
        return source

    def _chunk_source_readonly(self, chunk, want_key):
        # Prefetch must not call chunk.get()/control.update(); it only reads data.
        token = chunk.token
        if token.device == 'disk':
            uncache(token.path)
            data = np.memmap(token.path, dtype=token.dtype, mode='r', shape=token.shape)
            if token.key_exist and token.value_exist:
                return torch.from_numpy(data[0 if want_key else 1])
            return torch.from_numpy(data[:])
        return token.key if want_key else token.value

    def _copy_readonly_source_to_gpu(self, target, out_pos, take, source, pinned_buffers):
        out_end = out_pos + take
        part = source[:take]
        if part.device.type == 'cuda':
            target[out_pos:out_end].copy_(part, non_blocking=True)
            return

        cpu_part = self._new_cpu_buffer(part.shape)
        cpu_part.copy_(part)
        target[out_pos:out_end].copy_(cpu_part, non_blocking=True)
        pinned_buffers.append(cpu_part)

    def _prefetch_chunks_direct_pinned(self, chunk_ids, k_v_num_tokens, need_key, need_value):
        counts = []
        total_tokens = 0
        for chunk_id in chunk_ids:
            chunk_index = int(chunk_id.item())
            chunk_start = chunk_index * self.chunk_size
            take = min(self.chunk_size, max(0, int(k_v_num_tokens) - chunk_start))
            counts.append(take)
            total_tokens += take

        shape = list(self.shape[1:])
        shape[0] = total_tokens
        shape = tuple(shape)

        key_buffer = torch.empty(dtype=self.torch_dtype, size=shape, device='cuda:0') if need_key else None
        value_buffer = torch.empty(dtype=self.torch_dtype, size=shape, device='cuda:0') if need_value else None
        tokenids_cpu = torch.empty(total_tokens, dtype=torch.long)
        pinned_buffers = []

        out_pos = 0
        for chunk_id, take in zip(chunk_ids, counts):
            if take <= 0:
                continue
            chunk_index = int(chunk_id.item())
            chunk_start = chunk_index * self.chunk_size
            out_end = out_pos + take
            tokenids_cpu[out_pos:out_end] = torch.arange(
                chunk_start,
                chunk_start + take,
                dtype=torch.long,
            )

            if self.disk_type == 'KV_Division':
                if need_key:
                    source = self._chunk_source_readonly(self.key_tokens[chunk_index], want_key=True)
                    self._copy_readonly_source_to_gpu(key_buffer, out_pos, take, source, pinned_buffers)
                if need_value:
                    source = self._chunk_source_readonly(self.value_tokens[chunk_index], want_key=False)
                    self._copy_readonly_source_to_gpu(value_buffer, out_pos, take, source, pinned_buffers)
            else:
                chunk = self.tokens[chunk_index]
                if need_key:
                    source = self._chunk_source_readonly(chunk, want_key=True)
                    self._copy_readonly_source_to_gpu(key_buffer, out_pos, take, source, pinned_buffers)
                if need_value:
                    source = self._chunk_source_readonly(chunk, want_key=False)
                    self._copy_readonly_source_to_gpu(value_buffer, out_pos, take, source, pinned_buffers)
            out_pos = out_end

        tokenids = tokenids_cpu.to(chunk_ids.device)
        return key_buffer, value_buffer, tokenids, pinned_buffers

    def _prefetch_by_chunk_pinned(self, chunk_ids, prefetch_tokenids, need_key, need_value):
        token_ids = prefetch_tokenids % self.chunk_size
        shape = list(self.shape[1:])
        shape[0] = token_ids.shape[0]
        shape = tuple(shape)

        key_cpu = self._new_cpu_buffer(shape) if need_key else None
        value_cpu = self._new_cpu_buffer(shape) if need_value else None

        prefetch_tokenids_cpu = prefetch_tokenids.to('cpu')
        token_ids_cpu = token_ids.to('cpu')

        for chunk_id in chunk_ids:
            chunk_index = int(chunk_id.item())
            mask_cpu = (prefetch_tokenids_cpu // self.chunk_size == chunk_index)
            token_id = token_ids_cpu[mask_cpu]

            if self.disk_type == 'KV_Division':
                if need_key:
                    chunk = self.key_tokens[chunk_index]
                    key_cpu[mask_cpu] = self._select_chunk_to_cpu(chunk, token_id, want_key=True)
                if need_value:
                    chunk = self.value_tokens[chunk_index]
                    value_cpu[mask_cpu] = self._select_chunk_to_cpu(chunk, token_id, want_key=False)
            else:
                chunk = self.tokens[chunk_index]
                if need_key:
                    key_cpu[mask_cpu] = self._select_chunk_to_cpu(chunk, token_id, want_key=True)
                if need_value:
                    value_cpu[mask_cpu] = self._select_chunk_to_cpu(chunk, token_id, want_key=False)

        key_buffer = None
        value_buffer = None
        if need_key:
            key_buffer = torch.empty(dtype=self.torch_dtype, size=shape, device='cuda:0')
            key_buffer.copy_(key_cpu, non_blocking=True)
        if need_value:
            value_buffer = torch.empty(dtype=self.torch_dtype, size=shape, device='cuda:0')
            value_buffer.copy_(value_cpu, non_blocking=True)

        return key_buffer, value_buffer, (key_cpu, value_cpu)

    def _gpu_cache_disabled(self):
        if self.disk_type != 'KV_Division':
            return False
        if self.chunk_num == 0:
            return False
        control = self.key_tokens[0].pcache_control
        return getattr(control, "gpu_size", 0) <= 0

    def _chunks_are_cpu_resident(self, chunk_ids, need_key, need_value):
        for chunk_id in chunk_ids:
            chunk_index = int(chunk_id.item())
            if need_key and self.device_map[chunk_index] != 'cpu':
                return False
            if need_value and self.device_map[chunk_index + self.chunk_num] != 'cpu':
                return False
        return True

    def _update_fast_path_stats(self, chunk_ids, token_count, need_key, need_value):
        if need_key:
            self.cpu_hit += token_count
        if need_value:
            self.cpu_hit += token_count

    def _gather_cpu_chunks_to_gpu(self, chunk_ids, storage_tokenids, need_key, need_value):
        token_ids = storage_tokenids % self.chunk_size
        shape = list(self.shape[1:])
        shape[0] = storage_tokenids.shape[0]
        shape = tuple(shape)

        key_cpu = self._new_cpu_buffer(shape) if need_key else None
        value_cpu = self._new_cpu_buffer(shape) if need_value else None

        storage_tokenids_cpu = storage_tokenids.to('cpu')
        token_ids_cpu = token_ids.to('cpu')

        for chunk_id in chunk_ids:
            chunk_index = int(chunk_id.item())
            mask_cpu = (storage_tokenids_cpu // self.chunk_size == chunk_index)
            token_id = token_ids_cpu[mask_cpu]
            if need_key:
                chunk = self.key_tokens[chunk_index]
                key_cpu[mask_cpu] = self._select_chunk_to_cpu(chunk, token_id, want_key=True)
            if need_value:
                chunk = self.value_tokens[chunk_index]
                value_cpu[mask_cpu] = self._select_chunk_to_cpu(chunk, token_id, want_key=False)

        key_buffer = None
        value_buffer = None
        if need_key:
            key_buffer = torch.empty(dtype=self.torch_dtype, size=shape, device='cuda:0')
            # Synchronous get() must keep CPU staging buffers alive until copy ends.
            key_buffer.copy_(key_cpu, non_blocking=False)
        if need_value:
            value_buffer = torch.empty(dtype=self.torch_dtype, size=shape, device='cuda:0')
            value_buffer.copy_(value_cpu, non_blocking=False)
        return key_buffer, value_buffer

    def _get_by_chunk_fast_cpu(self, pos_id, need_key, need_value):
        if not self._gpu_cache_disabled():
            return None

        if pos_id is None:
            storage_tokenids = torch.arange(self.real_len, dtype=torch.long)
        elif len(pos_id.shape) == 1:
            if self.reorder is not None:
                storage_tokenids = self.reorder[pos_id.to(self.reorder.device)].to('cpu')
            else:
                storage_tokenids = pos_id.to('cpu')
        else:
            return None

        pos2chunk = storage_tokenids // self.chunk_size
        chunk_ids = torch.unique(pos2chunk)
        if not self._chunks_are_cpu_resident(chunk_ids, need_key, need_value):
            return None

        key_buffer, value_buffer = self._gather_cpu_chunks_to_gpu(
            chunk_ids=chunk_ids,
            storage_tokenids=storage_tokenids,
            need_key=need_key,
            need_value=need_value,
        )
        self._update_fast_path_stats(
            chunk_ids=chunk_ids,
            token_count=int(storage_tokenids.shape[0]),
            need_key=need_key,
            need_value=need_value,
        )
        return key_buffer, value_buffer

    def prefetch_by_chunk(self,pos_id=None,need_key=True,need_value=True,k_v_num_tokens=None,time_budget=0.0):
        if not (need_key or need_value):
            return None 
        if len(pos_id.shape) == 1:
            # print(f"Layer {self.layer_id} pos_id before reorder = {len(pos_id)} tokens")
            input_device = pos_id.device
            if self.reorder is not None:
                pos_id = self.reorder[pos_id.to(self.reorder.device)].to(input_device) # 变成实际的 token id
            # print(f"Layer {self.layer_id} pos_id after reorder = {pos_id.cpu().numpy()}")
            pos2chunk = pos_id // self.chunk_size
            ordered_chunk_ids = []
            seen_chunk_ids = set()
            for raw_chunk_id in pos2chunk.detach().cpu().tolist():
                chunk_id_value = int(raw_chunk_id)
                if chunk_id_value not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id_value)
                    ordered_chunk_ids.append(chunk_id_value)
            chunk_ids = torch.tensor(
                ordered_chunk_ids,
                dtype=pos2chunk.dtype,
                device=pos2chunk.device,
            )
            if os.environ.get("HYPERINFER_PREFETCH_DEBUG") == "1":
                print(f"Layer {self.layer_id} chunk_ids before time budget = {chunk_ids.cpu().numpy()}")

            if time_budget > 0.0:

                all_chunk_ids = chunk_ids
                for chunk_offset, chunk_id in enumerate(chunk_ids):
                    # 仅实现了 kv division 情况下的预取时间估算
                    chunk_index = int(chunk_id.item())
                    est_time = 0.0
                    global AVG_CHUNK_TIME
                    if self.disk_type == 'KV_Division':
                        if need_key:
                            est_time += AVG_CHUNK_TIME[self.device_map[chunk_index]]
                        if need_value:
                            est_time += AVG_CHUNK_TIME[self.device_map[chunk_index + self.chunk_num]]
                    else:
                        est_time += AVG_CHUNK_TIME[self.device_map[chunk_index]]
                    if time_budget - est_time >= 0.0:
                        time_budget -= est_time
                    else:
                        chunk_ids = all_chunk_ids[:chunk_offset]
                        break
            else:
                # 固定比例预取，取一半的 chunk
                chunk_ids = chunk_ids[:chunk_ids.shape[0] // 2]

            if os.environ.get("HYPERINFER_PREFETCH_DEBUG") == "1":
                print(f"Layer {self.layer_id} chunk_ids after time budget = {chunk_ids.cpu().numpy()}")

            key_buffer, value_buffer, prefetch_tokenids, pinned_buffers = self._prefetch_chunks_direct_pinned(
                chunk_ids=chunk_ids,
                k_v_num_tokens=k_v_num_tokens,
                need_key=need_key,
                need_value=need_value,
            )
        if self.reorder is not None:
            tokenids = self.reverse_reorder[prefetch_tokenids.to(self.reverse_reorder.device)].to(input_device)
            return key_buffer,value_buffer, tokenids, pinned_buffers
        else:
            return key_buffer,value_buffer, prefetch_tokenids, pinned_buffers
    
    def get_parallel(self,pos_id,need_key=True,need_value=True):
        if pos_id is None:
            pos_id = np.arange(len(self.tokens))
        pos_id = self.reorder[pos_id]

        gpu_mask = np.where(self.device_map[pos_id]=='cuda:0')[0]
        cpu_mask = np.where(self.device_map[pos_id]=='cpu')[0]
        disk_mask = np.where(self.device_map[pos_id]=='disk')[0]
        self.gpu_hit += gpu_mask.shape[0]
        self.cpu_hit += cpu_mask.shape[0]
        self.disk_hit += disk_mask.shape[0]

        shape = list(self.shape[1:])
        shape[0] = pos_id.shape[0]
        self.key_buffer = None
        self.value_buffer = None
        shape = tuple(shape)
        if need_key:
            self.key_buffer = torch.empty(dtype=self.torch_dtype,size=shape,device='cuda:0')
        if need_value:
            self.value_buffer = torch.empty(dtype=self.torch_dtype,size=shape,device='cuda:0')


        # cpu gather
        torch.cuda.synchronize()
        st = time.monotonic()


        cpu_ids = pos_id[cpu_mask]
        if cpu_ids.shape[0] > 0:
            if self.cpu_gather:
                shape[0] = cpu_mask.shape[0]
                cpu_key = None
                cpu_value = None
                if need_key:
                    cpu_key = torch.empty(dtype=self.torch_dtype,size=shape)
                if need_value:
                    cpu_value = torch.empty(dtype=self.torch_dtype,size=shape)
                if self.disk_type == 'KV_Division':
                    if need_key:
                        cpu_key[i,:] = self.tokens[cpu_id].get(device='cpu')
                    if need_value:
                        cpu_value[i,:] = self.tokens[cpu_id+self.chunk_num].get(device='cpu')
                else:
                    if need_key and need_value:
                        for i,cpu_id in enumerate(cpu_ids):
                            cpu_key[i,:],cpu_value[i,:] = self.tokens[cpu_id].get(device='cpu')
                    elif not need_key:
                        for i,cpu_id in enumerate(cpu_ids):
                            cpu_value[i,:] = self.tokens[cpu_id].get_value(device='cpu')
                    elif not need_value:
                        for i,cpu_id in enumerate(cpu_ids):
                            cpu_key[i,:] = self.tokens[cpu_id].get_key(device='cpu')
                if need_key:
                    cpu_key = cpu_key.to('cuda:0') 
                    self.key_buffer[cpu_mask,:] = cpu_key
                if need_value:
                    cpu_value = cpu_value.to('cuda:0') 
                    self.value_buffer[cpu_mask,:] = cpu_value
            else:
                if need_key and need_value:
                    for i,cpu_id in enumerate(cpu_ids):
                        self.key_buffer[cpu_mask[i],:],self.value_buffer[cpu_mask[i],:] = self.tokens[cpu_id].get(device='cuda:0')
                elif not need_key:
                    for i,cpu_id in enumerate(cpu_ids):
                        self.value_buffer[cpu_mask[i],:] = self.tokens[cpu_id].get_value(device='cuda:0')
                elif not need_value:
                    for i,cpu_id in enumerate(cpu_ids):
                        self.key_buffer[cpu_mask[i],:] = self.tokens[cpu_id].get_key(device='cuda:0')
 
        torch.cuda.synchronize()
        self.dt_cpu  += time.monotonic() - st


        torch.cuda.synchronize()
        st = time.monotonic()

        # gpu gather
        gpu_ids = pos_id[gpu_mask]
        if gpu_ids.shape[0] > 0:
            if need_key and need_value:
                for i,cpu_id in enumerate(gpu_ids):
                    self.key_buffer[gpu_mask[i],:],self.value_buffer[gpu_mask[i],:] = self.tokens[gpu_id].get(device='cuda:0')
            elif not need_key:
                for i,gpu_id in enumerate(gpu_ids):
                    self.value_buffer[gpu_mask[i],:] = self.tokens[gpu_id].get_value(device='cuda:0')
            elif not need_value:
                for i,gpu_id in enumerate(gpu_ids):
                    self.key_buffer[gpu_mask[i],:] = self.tokens[gpu_id].get_key(device='cuda:0')    
        torch.cuda.synchronize()
        self.dt_gpu  += time.monotonic() - st

        torch.cuda.synchronize()
        st = time.monotonic()


        # disk gather
        disk_ids = pos_id[disk_mask]
        self.disk_map = torch.empty(dtype=torch.int,size=[len(self.tokens)])
        if disk_ids.shape[0] > 0:
            if self.disk_type == 'Batched':
                for i,disk_id in enumerate(disk_ids):
                    self.disk_map[disk_id] = disk_mask[i]
                uncache(self.path)
                data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
                if need_key and need_value:
                    self.key_buffer[disk_mask,:],self.value_buffer[disk_mask,:] = torch.from_numpy(data[0,disk_ids,:,:]).to('cuda:0'),torch.from_numpy(data[1,disk_ids,:,:]).to('cuda:0')
                elif not need_key:
                    self.value_buffer[disk_mask,:] = torch.from_numpy(data[1,disk_ids,:,:]).to('cuda:0')
                elif not need_value:
                    self.key_buffer[disk_mask,:] = torch.from_numpy(data[0,disk_ids,:,:]).to('cuda:0')
                for disk_id in disk_ids:
                    self.tokens[disk_id].control.update()
            elif self.disk_type == 'Single':
                for i,disk_id in enumerate(disk_ids):
                    if need_key and need_value:
                        self.key_buffer[disk_mask[i],:],self.value_buffer[disk_mask[i],:] = self.tokens[disk_id].get()
                    elif not need_key:
                        self.value_buffer[disk_mask[i],:] = self.tokens[disk_id].get_value()
                    elif not need_value:
                        self.key_buffer[disk_mask[i],:] = self.tokens[disk_id].get_key()

                     
        torch.cuda.synchronize()
        self.dt_disk  += time.monotonic() - st

        key = self.key_buffer
        value = self.value_buffer
        self.key_buffer = None
        self.value_buffer = None
        self.disk_map = None
        return key,value
        
            
    def get_single(self,pos_id):
        # pos_id -> int
        key,value = self.tokens[pos_id].get()
        return key.to('cuda:0'),value.to('cuda:0')

    def get_disk_token(self,pos_id,device='cuda:0'):
        uncache(self.path)
        data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
        key,value = torch.from_numpy(data[0,pos_id,:,:]).to(device),torch.from_numpy(data[1,pos_id,:,:]).to(device)
        return key,value
    
    def get_disk_head(self,device='cuda:0'):
        uncache(self.path)
        data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
        key,value = torch.from_numpy(data[0][:,self.head_ids]).to(device),torch.from_numpy(data[1][:,self.head_ids]).to(device)
        return key,value
    
    def get_disk_key_token(self,pos_id,device='cuda:0'):
        uncache(self.path)
        data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
        key = torch.from_numpy(data[0,pos_id,:,:]).to(device)
        return key
    
    def get_disk_key_head(self,device='cuda:0'):
        uncache(self.path)
        data = np.memmap(self.path, dtype=self.dtype, mode='r', shape=self.shape)
        # key = torch.from_numpy(data[0,:,0:3,:]).to(device)
        key = torch.from_numpy(data[0][:,self.head_ids]).to(device)
        return key
    
    def get_key(self,pos_id=None):
        return self.get(pos_id=pos_id,need_key=True,need_value=False)

    def get_value(self,pos_id=None):
        return self.get(pos_id=pos_id,need_key=False,need_value=True)
    
    def get_chunk_hit(self):
        chunk_hit = {}
        if self.disk_type == 'KV_Division':
            for token in self.key_tokens:
                chunk_hit[(self.prefix_id,self.layer_id,token.pos_id)] = token.get_chunk_hit()
            for token in self.value_tokens:
                chunk_hit[(self.prefix_id,self.layer_id,token.pos_id)] = token.get_chunk_hit()
        else:
            for token in self.tokens:
                chunk_hit[(self.prefix_id,self.layer_id,token.pos_id)] = token.get_chunk_hit()
        return chunk_hit

    def check_device(self):
        if self.disk_type == 'KV_Division':
            for i in range(self.chunk_num):
                if self.key_tokens[i].key is None:
                    device = 'disk'
                else:
                    device = self.key_tokens[i].key.device
                assert str(device) == str(self.device_map[i])
                if self.value_tokens[i].value is None:
                    device = 'disk'
                else:
                    device = self.value_tokens[i].value.device
                assert str(device) == str(self.device_map[i+self.chunk_num])
        else:
            for i in range(len(self.tokens)):
                token = self.tokens[i]
                if token.key is None and token.value is None:
                    device = 'disk'
                else:
                    device = token.key.device
                assert str(device) == str(self.device_map[i])



class PrefixKV():
    disk_dir = None
    def __init__(self,prefix_id,key,value,pcache_control,cache_type='LRU',disk_type='Single',cpu_gather=False,head_ids=[0,1,2],chunk_size=64,reorder=None):
        self.layers = [PrefixKVLayer(prefix_id=prefix_id,key=key[i,:],value=value[i,:],layer_id=i,pcache_control=pcache_control,cache_type=cache_type,disk_type=disk_type,cpu_gather=cpu_gather,head_ids=head_ids,chunk_size=chunk_size,reorder=reorder[i]) for i in range(key.shape[0])]

    def to(self,pos_id,device,layer=None):
        if layer is None:
            for cache in self.layers:
                cache.to(pos_id=pos_id,device=device)
        else:
            cache = self.layers[layer]

    def to_disk(self,pos_id,layer=None):
        if layer is None:
            for cache in self.layers:
                cache.to_disk(pos_id=pos_id)
        else:
            cache = self.layers[layer]

    def get(self,pos_id=None,layer=None,is_prefetching=False):
        # pos_id -> tensor

        if layer is None:
            keys = []
            values = []
            for cache in self.layers:
                # 没有layer的话，不会出现 prefetch 的情况
                key,value = cache.get(pos_id=pos_id)
                keys.append(key)
                values.append(value)
            key = torch.stack(keys)
            value = torch.stack(values)
        else:
            cache = self.layers[layer]
            key,value = cache.get(pos_id=pos_id,is_prefetching=is_prefetching)
        return key,value
    
    def prefetch(self,pos_id=None,layer=None,k_v_num_tokens=None,time_budget=0.0):
        cache = self.layers[layer]
        return cache.prefetch(pos_id=pos_id,k_v_num_tokens=k_v_num_tokens,time_budget=time_budget)
    
    def prefetch_update(self,pos_id=None,chunk_ids=None,layer=None):
        if layer is None:
            for cache in self.layers:
                cache.prefetch_update(pos_id=pos_id, chunk_ids=chunk_ids)
        else:
            cache = self.layers[layer]
            return cache.prefetch_update(pos_id=pos_id, chunk_ids=chunk_ids)
    
    def get_head(self,layer):
        return self.layers[layer].get_head()
    
    def get_key(self,pos_id,layer):
        if layer is None:
            keys = []
            for cache in self.layers:
                key,_ = cache.get_key(pos_id=pos_id)
                keys.append(key)
            key = torch.stack(keys)
        else:
            cache = self.layers[layer]
            key,_ = cache.get_key(pos_id=pos_id)
        return key

    
    def get_value(self,pos_id,layer):
        if layer is None:
            values = []
            for cache in self.layers:
                _,value = cache.get_value(pos_id=pos_id)
                values.append(value)
            value = torch.stack(values)
        else:
            cache = self.layers[layer]
            _,value = cache.get_value(pos_id=pos_id)
        return value
    
    def get_chunk_hit(self):
        chunk_hit = {}
        for cache in self.layers:
            chunk_hit.update(cache.get_chunk_hit())
        return chunk_hit
    
    def check_device(self):
        for layer in self.layers:
            layer.check_device()

    

class Pcache():
    control_update_time=0
    def __init__(self, cpu_size, gpu_size, kv_dir='./cache/kvs', cache_type='LRU', disk_type='Batched',cpu_gather=False,head_ids=[0,1,2],chunk_size=64):
        '''
        cpu/gpu/disk_size: num tokens * layers 
        return kv shape:[seq,head,head_dim]
        kv_dir: the dir used to store KV cache on disk 
        cache_type:LRU,LFU
        disk_type:Single(one token one file),Batched(one prefix one file)
        cpu_gather: 
            when get() called
            True: cpu KV cache will gather together before move to GPU (fully use PCIe but double copy)
            False: each cpu KV cache will move to GPU directly (PCIe underutilized)
        head_ids:the heads of get_head() returned
        '''
        # self.cpu_size = cpu_size
        # self.gpu_size = gpu_size
        # self.disk_size = disk_size
        self.cache_type = cache_type
        self.disk_type = disk_type
        self.cpu_gather = cpu_gather
        self.head_ids=head_ids
        self.chunk_size=chunk_size
        self._cache_lock = threading.RLock()
        self._prefetch_executor = PriorityExecutor(
            thread_name_prefix="hyperinfer-prefetch"
        )
        self._prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        if cache_type == 'LRU':
            self.control = LRUUnit(cpu_size=cpu_size,gpu_size=gpu_size)
        elif cache_type == 'LFU':
            self.control = ScoreUnit(cpu_size=cpu_size,gpu_size=gpu_size,cache_type=cache_type)
        elif cache_type == 'CKLFU':
            self.control = ScoreUnit(cpu_size=cpu_size,gpu_size=gpu_size,cache_type=cache_type)

        PrefixKV.disk_dir = kv_dir
        if not os.path.exists(PrefixKV.disk_dir):
            os.mkdir(PrefixKV.disk_dir)
        clear_folder(PrefixKV.disk_dir)
    
        self.cache = []

    def insert(self, prefix_id, key, value, reorder=None):
        # insert kv shape:[layer,seq,head,head_dim]
        assert prefix_id == len(self.cache)
        if reorder is None:
            reorder = torch.stack([torch.arange(key.shape[1]) for _ in range(key.shape[0])])
        
        cache = PrefixKV(prefix_id=prefix_id,key=key,value=value,pcache_control=self.control,cache_type=self.cache_type,disk_type=self.disk_type,cpu_gather=self.cpu_gather,head_ids=self.head_ids,chunk_size=self.chunk_size,reorder=reorder)
        self.cache.append(cache)

    def insert_check(self, prefix_id, key, value):
        size = 2*(key.element_size() * key.nelement()) / 1024 / 1024
        full = self.control.gpu_num + self.control.cpu_num + size <= self.control.gpu_size + self.control.cpu_size
        self.insert(prefix_id, key, value)
        return full
        
    
    
    def get(self, prefix_id, pos_id=None,layer=None,is_prefetching=False):
        with self._cache_lock:
            cache = self.cache[prefix_id]
            key,value = cache.get(pos_id,layer,is_prefetching = is_prefetching)
        # if pos_id is None:
        #     for id in range(len(cache.tokens)):
        #         self.control.update(cache.tokens[id])
        # else:
        #     for id in pos_id:
        #         self.control.update(cache.tokens[id])
        return key,value

    def prefetch(self, prefix_id, pos_id=None, layer=None, k_v_num_tokens=None,time_budget=0.0):
        with self._cache_lock:
            cache = self.cache[prefix_id]
            result = cache.prefetch(pos_id, layer, k_v_num_tokens=k_v_num_tokens,time_budget=time_budget)
        return result

    def prefetch_async(
        self,
        prefix_id,
        pos_id=None,
        layer=None,
        k_v_num_tokens=None,
        time_budget=0.0,
        priority=1,
        prefetch_kind="default",
    ):
        def _run_prefetch():
            if self._prefetch_stream is None:
                result = self.prefetch(
                    prefix_id=prefix_id,
                    pos_id=pos_id,
                    layer=layer,
                    k_v_num_tokens=k_v_num_tokens,
                    time_budget=time_budget,
                )
                if len(result) == 4:
                    key, value, tokenids, pinned_buffers = result
                else:
                    key, value, tokenids = result
                    pinned_buffers = None
                return key, value, tokenids, None, pinned_buffers

            with torch.cuda.stream(self._prefetch_stream):
                result = self.prefetch(
                    prefix_id=prefix_id,
                    pos_id=pos_id,
                    layer=layer,
                    k_v_num_tokens=k_v_num_tokens,
                    time_budget=time_budget,
                )
                if len(result) == 4:
                    key, value, tokenids, pinned_buffers = result
                else:
                    key, value, tokenids = result
                    pinned_buffers = None
                event = torch.cuda.Event()
                event.record(self._prefetch_stream)
            return key, value, tokenids, event, pinned_buffers

        return AsyncPrefetchHandle(
            self._prefetch_executor.submit(
                _run_prefetch,
                priority=priority,
                label=prefetch_kind,
            )
        )

    def prefetch_scheduler_metrics(self):
        return self._prefetch_executor.metrics_snapshot()

    def prefetch_update(self, prefix_id, pos_id=None, chunk_ids=None, layer=None):
        # Fast HyperInfer prefetch is read-only: it returns a temporary buffer and
        # updates cache state only when the caller later consumes missing KVs.
        return None
    
    def get_head(self,prefix_id,layer):
        with self._cache_lock:
            return self.cache[prefix_id].get_head(layer)
    
    def get_key(self,prefix_id,pos_id,layer=None):
        with self._cache_lock:
            return self.cache[prefix_id].get_key(layer=layer,pos_id=pos_id)
    
    def get_value(self,prefix_id,pos_id,layer=None):
        with self._cache_lock:
            return self.cache[prefix_id].get_value(layer=layer,pos_id=pos_id)

    def update_score(self, prefix_id, pos_id, layer_id, new_score):
        cache = self.cache[prefix_id]
        item = cache.layers[layer_id].tokens[pos_id]
        self.control.update_score(item=item,new_score=new_score)
        # self.control.check()

    def get_hit_rate(self):
        gpu_hit = 0
        cpu_hit = 0
        disk_hit = 0
        disk_chunk_hit = 0
        for cache in self.cache:
            for layer in cache.layers:
                gpu_hit += layer.gpu_hit
                cpu_hit += layer.cpu_hit
                disk_hit += layer.disk_hit
                disk_chunk_hit += layer.disk_chunk_hit
        total_hit = gpu_hit + cpu_hit + disk_hit
        if total_hit == 0:
            print(f'gpu_hit:{gpu_hit},cpu_hit:{cpu_hit},disk_hit:{disk_hit},gpu_hit_rate:0,cpu_hit_rate:0,disk_hit_rate:0')
            print(f'disk_chunk_hit:{disk_chunk_hit}')
            return
        print(f'gpu_hit:{gpu_hit},cpu_hit:{cpu_hit},disk_hit:{disk_hit},gpu_hit_rate:{gpu_hit/total_hit},cpu_hit_rate:{cpu_hit/total_hit},disk_hit_rate:{disk_hit/total_hit}')
        print(f'disk_chunk_hit:{disk_chunk_hit}')


    def get_time(self):
        dt_gpu = 0
        dt_cpu = 0
        dt_disk = 0
        for cache in self.cache:
            for layer in cache.layers:
                dt_gpu += layer.dt_gpu
                dt_cpu += layer.dt_cpu
                dt_disk += layer.dt_disk
        total_time = dt_gpu + dt_cpu + dt_disk
        if total_time == 0:
            print(f'dt_gpu:{dt_gpu},dt_cpu:{dt_cpu},dt_disk:{dt_disk},control update time:{Pcache.control_update_time},gpu_rate:0,cpu_rate:0,disk_rate:0')
            return
        print(f'dt_gpu:{dt_gpu},dt_cpu:{dt_cpu},dt_disk:{dt_disk},control update time:{Pcache.control_update_time},gpu_rate:{dt_gpu/total_time},cpu_rate:{dt_cpu/total_time},disk_rate:{dt_disk/total_time}')

    # def _get_kv_size(self,key):
    #     return (2*key.element_size() * key.nelement())/1024/1024

    def get_chunk_hit(self):
        chunk_hit = {}
        # for cache in self.cache:
        #     chunk_hit.update(cache.get_chunk_hit())
        # with open(f'./chunk_hit_rte.txt','w+') as f:
        #     f.write(f'{chunk_hit}')
        print(chunk_hit)

    def check_device(self):
        for cache in self.cache:
            cache.check_device()


class ControlUnit():
    def __init__(self,cpu_size,gpu_size):
        self.cpu_size = cpu_size
        self.gpu_size = gpu_size
        self.cpu_num = 0
        self.gpu_num = 0

    def update(self,item):
        pass

    def update_no_move(self,item):
        # update the LRU cache but not move data from CPU to GPU
        pass

    def insert(self):
        pass

    def evict_gpu(self):
        pass

    def evict_cpu(self):
        pass

class LRUUnit(ControlUnit):
    def __init__(self,cpu_size,gpu_size):
        super().__init__(cpu_size=cpu_size,gpu_size=gpu_size)
        self.gpu_lru_cache = DummyLRU()
        self.cpu_lru_cache = DummyLRU()

    def update(self,item):
        item.delete()
        if item.device == 'cuda:0':
            self.gpu_num -= item.size
        elif item.device == 'cpu':
            self.cpu_num -= item.size
        self.insert(item=item)

    def prefetch_update(self,item):
        if item.device == 'cuda:0':
            # self.gpu_num -= item.size
            return
        elif item.device == 'cpu':
            item.delete()
            self.cpu_num -= item.size
        self.insert(item=item)

    def update_no_move(self,item):
        item.delete()
        if item.device == 'cuda:0':
            self.gpu_lru_cache.insert(item=item)
        elif item.device == 'cpu':
            self.cpu_lru_cache.insert(item=item)

    def insert(self,item):
        # print(f"[LRU] Insert pos_id={item.pos_id}, device={item.device}, size={item.size}")
        self.gpu_num += item.size
        self.gpu_lru_cache.insert(item=item)
        item.to('cuda:0')
        while self.gpu_num > self.gpu_size:
            if self.gpu_num == 0:
                break
            self.evict_gpu()
            while self.cpu_num > self.cpu_size:
                if self.cpu_num == 0:
                    break
                self.evict_cpu()

    def evict_gpu(self):
        if self.gpu_num == 0:
            return
        item = self.gpu_lru_cache.prev.delete()
        # print(f"[{self.__class__.__name__}] Evict GPU: pos_id={item.pos_id}, size={item.size}")
        self.gpu_num -= item.size
        item.to('cpu')
        self.cpu_lru_cache.insert(item=item)
        self.cpu_num += item.size

    def evict_cpu(self):
        if self.cpu_num == 0:
            return
        item = self.cpu_lru_cache.prev.delete()
        # print(f"[{self.__class__.__name__}] Evict CPU: pos_id={item.pos_id}, size={item.size}")
        self.cpu_num -= item.size
        item.to_disk()
        

class ScoreUnit(ControlUnit):
    def __init__(self,cpu_size,gpu_size,cache_type):
        super().__init__(cpu_size=cpu_size,gpu_size=gpu_size)
        self.cache_type=cache_type
        if self.cache_type == 'CKLFU':
            # score id:
            # 1: sum of all token hit num
            # 0: chunk hit num
            self.cpu_score_id = 1
            self.gpu_score_id = 1
            self.cpu_heap = Heap(self.cpu_score_id)
            self.gpu_heap = Heap(self.gpu_score_id)
        elif self.cache_type == 'LFU':
            self.cpu_score_id = 0
            self.gpu_score_id = 0
            self.cpu_heap = Heap(self.cpu_score_id)
            self.gpu_heap = Heap(self.gpu_score_id)
    
    def update_score(self,item,new_score):
        # print(f"[LFU] Update score pos_id={item.pos_id}, new_score={new_score}")
        item.update_score(new_score)

        # self.delete(item=item)
        # self.insert(item=item)
        if item.device == 'cuda:0':
            self.gpu_heap.update(item=item)
        elif item.device == 'cpu':
            if item.score > self.gpu_heap.get_min().score:
                self.delete(item=item)
                self.insert(item=item)
            else:
                self.cpu_heap.update(item=item)
        else:
            self.insert(item=item)

        # self.check()

    def update_no_move(self,item):
        pass

    def delete(self,item):
        if item.token.device == 'cuda:0':
            self.gpu_num -= item.size
        if item.token.device == 'cpu':
            self.cpu_num -= item.size
        if item.control.heap is not None:
            item.control.heap.delete(item.control)

    def insert(self,item):
        if self.gpu_num + item.size < self.gpu_size or item.score[self.gpu_score_id] > self.gpu_heap.get_min().score[self.gpu_score_id]:
            self.gpu_num += item.size
            self.gpu_heap.push(item=item)
            item.to('cuda:0')
            while self.gpu_num > self.gpu_size:
                if self.gpu_num == 0:
                    break
                self.evict_gpu()
                while self.cpu_num > self.cpu_size:
                    if self.cpu_num == 0:
                        break
                    self.evict_cpu()
        elif self.cpu_num + item.size < self.cpu_size or item.score[self.cpu_score_id] > self.cpu_heap.get_min().score[self.cpu_score_id]:
            self.cpu_num += item.size
            self.cpu_heap.push(item=item)
            item.to('cpu')
            while self.cpu_num > self.cpu_size:
                if self.cpu_num == 0:
                    break
                self.evict_cpu()
        else:
            item.to('disk')
            item.heap = None
            

    def evict_gpu(self):
        item = self.gpu_heap.pop()
        # print(f"[{self.__class__.__name__}] Evict GPU: pos_id={item.pos_id}, size={item.size}")
        self.gpu_num -= item.size
        if item.score[self.cpu_score_id] > self.cpu_heap.get_min().score[self.cpu_score_id] or self.cpu_num + item.size < self.cpu_size:
            item.to('cpu')
            self.cpu_heap.push(item=item)
            self.cpu_num += item.size
        else:
            item.to('disk')
            item.heap = None

    def evict_cpu(self):
        item = self.cpu_heap.pop()
        # print(f"[{self.__class__.__name__}] Evict CPU: pos_id={item.pos_id}, size={item.size}")
        self.cpu_num -= item.size
        item.to('disk')
        item.heap = None
        

    # def check(self):
    #     while self.gpu_num > self.gpu_size:
    #         self.evict_gpu()
    #         while self.cpu_num > self.cpu_size:
    #             self.evict_cpu()

    #     while self.gpu_heap.get_min().score < self.cpu_heap.get_max().score or self.cpu_heap.get_min().score < self.disk_heap.get_max().score:
    #         if self.gpu_heap.get_min().score < self.cpu_heap.get_max().score:
    #             # print(self.gpu_heap.get_min().score,self.cpu_heap.get_max().score)
    #             if self.gpu_num < self.gpu_size:
    #                 cpu_item = self.cpu_heap.pop_max()
    #                 cpu_item.to('cuda:0')
    #                 self.gpu_heap.push(cpu_item)
    #                 self.gpu_num += 1
    #                 self.cpu_num -= 1
    #             else:
    #                 gpu_item = self.gpu_heap.pop()
    #                 gpu_item.to('cpu')
    #                 self.cpu_heap.push(gpu_item)
    #                 cpu_item = self.cpu_heap.pop_max()
    #                 cpu_item.to('cuda:0')
    #                 self.gpu_heap.push(cpu_item)
    #         elif self.cpu_heap.get_min().score < self.disk_heap.get_max().score:
    #             if self.cpu_num < self.cpu_size:
    #                 disk_item = self.disk_heap.pop_max()
    #                 disk_item.to('cpu')
    #                 self.cpu_heap.push(disk_item)
    #                 self.cpu_num += 1
    #             else:
    #                 cpu_item = self.cpu_heap.pop()
    #                 cpu_item.to('disk')
    #                 self.disk_heap.push(cpu_item)
    #                 disk_item = self.disk_heap.pop_max()
    #                 disk_item.to('cpu')
    #                 self.cpu_heap.push(disk_item)
                

# 输出一个map中每个设备位置的数量
def count_pos(map):
    cpu = 0
    gpu = 0
    disk = 0
    for pos in map:
        if pos == 'cuda:0':
            gpu += 1
        elif pos == 'cpu':
            cpu += 1
        elif pos == 'disk':
            disk += 1
    print(f'cpu:{cpu},gpu:{gpu},disk:{disk}')

# prefix_table = {}
# def init_pcache(pcache:Pcache,folder_path,model_name):
#     folder_path = './cache/prefix/'
#     model_name = 'facebook_opt-6.7b'
#     files = os.listdir(folder_path)
#     i = 0
#     layer = 0
#     name_list = []
#     global prefix_table
#     for file in files:
#         hash_name = file.split('_')[-1].split('.')[0]
#         if hash_name not in prefix_table:
#             prefix_table[hash_name] = i
#             name_list.append(hash_name)
#             i = i + 1
#         layer_id = int(file.split('attnid')[-1].split('_')[0])
#         layer = max(layer,layer_id)
#     layer += 1

#     for prefix_id,hash_name in enumerate(name_list):
#         keys = []
#         values = []
#         for i in range(layer):
#             file = f'{model_name}_attnid{i}_{hash_name}.npy'
#             file_path = os.path.join(folder_path, file)

#             # mmap_kv_tensor = np.memmap(file_path, dtype=np.float16, mode='r', shape=[2,181,32,128])
#             mmap_kv_tensor = np.load(file_path, mmap_mode='r')
#             prefix_k, prefix_v = torch.from_numpy(mmap_kv_tensor[0,: , :, :]).to('cuda:0'), torch.from_numpy(mmap_kv_tensor[1,:, :, :]).to('cuda:0')
#             print(f'dt:{dt},{prefix_k.shape}')

#             keys.append(prefix_k)
#             values.append(prefix_v)
#         key = torch.stack(keys)
#         value = torch.stack(values)
#         pcache.insert(prefix_id=prefix_id,key=key,value=value)

if __name__ == '__main__':
    layer = 2
    a = Pcache(gpu_size=2,cpu_size=4,cache_type='LRU',disk_type='Chunk',cpu_gather=False,chunk_size=64)
    # init_pcache(a)
    # st = time.monotonic()
    # prefix_k,prefix_v = a.get(prefix_id=prefix_table['8584608586784889307'],layer=0)
    # torch.cuda.synchronize()
    # dt  = time.monotonic() - st
    # print(f'dt:{dt},{dt0},{dt1}')
    

    n = 1280
    head = 3
    hidden = 128
    prefix_id = 0

    key = torch.rand([layer,n,head,hidden])
    # value = torch.rand([layer,n,head,hidden])
    value = torch.arange(layer*n*head*hidden,dtype=torch.float32).reshape([layer,n,head,hidden])
    pos_id = indexb = torch.arange(0,n).reshape(n)
    a.insert(prefix_id=prefix_id,key=key,value=value)



    keyb = torch.rand([layer,n,head,hidden])
    valueb = torch.rand([layer,n,head,hidden])

    a.insert(prefix_id=prefix_id+1,key=keyb,value=valueb)

    ret_key,ret_value = a.get(prefix_id,pos_id)
    torch.cuda.synchronize()
    st = time.monotonic()
    ret_key,ret_value = a.get(prefix_id+1)
    print(valueb.shape,ret_value.shape)
    print(123,torch.equal(valueb[:,:,:,:],ret_value.to('cpu')))
    torch.cuda.synchronize()
    dt  = time.monotonic() - st
    print(f'dt:{dt}')

    ret_key = a.get_key(0,torch.tensor([1,3,5,7,10]))
    mask = [1,3,5,7,10]
    keyc = key[:,mask,:]
    print(torch.equal(keyc[:,:,:,:],ret_key.to('cpu')))

    ret_value = a.get_value(1,torch.tensor([2,4,6,10]))
    mask = [2,4,6,10]
    valued = valueb[:,mask,:]
    print(torch.equal(valued[:,:,:,:],ret_value.to('cpu')))

    ret_value = a.get_value(1,torch.tensor([[2,4,6,10],[2,4,6,10],[2,4,6,10]]))
    mask = [2,4,6,10]
    valued = valueb[:,mask,:]
    print(valued.shape,ret_value.shape)
    print(torch.equal(valued[:,:,:,:],ret_value.to('cpu')))
    # a.update_score(0,1,1,1)
    # a.update_score(0,3,1,2)
    # a.update_score(0,5,1,5)
    # a.update_score(1,2,1,4)
    # a.update_score(1,4,1,3)
    # a.update_score(0,11,1,10)
    # a.update_score(0,13,1,9)
    # a.update_score(0,15,1,6)
    # a.update_score(1,12,1,7)
    # a.update_score(1,14,1,8)
    torch.cuda.synchronize()
    st = time.monotonic()
    ret_key,ret_value = a.get(0,layer=1)
    torch.cuda.synchronize()
    dt  = time.monotonic() - st
    print(f'dt:{dt}')

    torch.cuda.synchronize()
    st = time.monotonic()

    ret_key = a.get_head(prefix_id,layer=0)
    torch.cuda.synchronize()
    dt  = time.monotonic() - st
    print(f'dt2:{dt}')
    # print(ret_value,value[:,:,3:,:])
    print(ret_key.shape,value[0,:,0:3,:].shape)

    print(torch.equal(key[0,:,0:3,:],ret_key.to('cpu')))
    a.get_time()
    a.get_hit_rate()
    a.get_chunk_hit()

    a.check_device()

    count_pos(a.cache[0].layers[0].device_map)
    count_pos(a.cache[0].layers[1].device_map)
    count_pos(a.cache[1].layers[0].device_map)
    count_pos(a.cache[1].layers[1].device_map)
    # print(a.cache[0].layers[0].device_map)
    # print(a.cache[0].layers[1].device_map)
    # print(a.cache[1].layers[0].device_map)
    # print(a.cache[1].layers[1].device_map)
    # print(a.cache[0].layers[1].tokens[1].control.score)
    # for i in range (5):
    #     print(a.control.gpu_heap.min_heap[i][0])
