import math
from torch import tensor
# import torch
import random

'''
不同粒度命中率测试
在flex_opt中添加
with open(path,'a+') as f:
    f.write(f'prefix_id:{prefix_table[prefix_hash]},prompt_len:{k_v_num_tokens},sele_token:{sele_tokenids}\n')
保存下对应的token访问序列,再执行此文件
文件开头处的参数rate表示缓存token的比例,cache_type分为token,chunk,prefix三个粒度,输出结果在out_path中
'''


class DummyLRU():
    def __init__(self) -> None:
        self.prev = self
        self.next = self
    
    def insert(self,item):  #insert to next
        self.next.prev = item
        item.next = self.next
        item.prev = self
        self.next = item
        item.device = 'cuda:0'

    def delete(self):
        if self.prev is not None and self.next is not None:
            self.prev.next = self.next
            self.next.prev = self.prev
            self.next = self
            self.prev = self
            self.device = 'disk'
        return self

class Item():
    def __init__(self,size,id,layer) -> None:
        self.id = id
        self.size = size
        self.layer = layer
        self.prev = self
        self.next = self
    
    def insert(self,item):  #insert to next
        self.next.prev = item
        item.next = self.next
        item.prev = self
        self.next = item
        item.device = 'cuda:0'

    def delete(self):
        if self.prev is not None and self.next is not None:
            self.prev.next = self.next
            self.next.prev = self.prev
            self.next = self
            self.prev = self
            self.device = 'disk'
        return self
    
class Cache():
    def __init__(self,size,cache_type = 'token',chunk_size = 64,layer_num=32) -> None:
        self.cache_type = cache_type # token,chunk,prefix
        self.chunk_size = chunk_size
        self.size = size
        self.used_mem = 0
        self.map = {} #(prefix_id,layer) ->list(map(id->tokens))
        self.lru = DummyLRU()
        self.disk_hit = 0
        self.gpu_hit = 0
        self.layer_num = layer_num

    def insert_prefix(self,prefix_id,token_num):
        for layer in range(self.layer_num):
            if prefix_id in self.map:
                return
            if self.cache_type == 'token':
                cache = [Item(size=1,id=i,layer=layer) for i in range(token_num)]
            elif self.cache_type == 'chunk':
                cache = [Item(size=self.chunk_size if i < math.ceil(token_num/self.chunk_size)-1 else token_num-i*self.chunk_size ,id=i,layer=layer) for i in range(math.ceil(token_num/self.chunk_size))]
            elif self.cache_type == 'prefix':
                cache = [Item(size=token_num,id=0,layer=layer)]
            for token in cache:
                self.insert_token(token)
            self.map[(prefix_id,layer)] = cache

    def insert_token(self,token):
        self.lru.insert(token)
        self.used_mem += token.size
        while self.used_mem > self.size:
            item = self.lru.prev.delete()
            self.used_mem -= item.size

    def get(self,prefix_id,token_ids,layer_id):
        disk = 0
        gpu = 0
        prefixKV = self.map[(prefix_id,layer_id)]
        for token_id in token_ids:
            if self.cache_type == 'token':
                token = prefixKV[token_id]
            elif self.cache_type == 'prefix':
                token = prefixKV[0]
            elif self.cache_type == 'chunk':
                token = prefixKV[token_id//self.chunk_size]

            if token.device == 'disk':
                self.disk_hit += 1
                disk += 1
            elif token.device == 'cuda:0':
                self.gpu_hit += 1
                gpu += 1
        
        for token_id in token_ids:
            if self.cache_type == 'token':
                token = prefixKV[token_id]
            elif self.cache_type == 'prefix':
                token = prefixKV[0]
            elif self.cache_type == 'chunk':
                token = prefixKV[token_id//self.chunk_size]

            if token.device == 'cuda:0':
                self.used_mem -= token.size
            token.delete()
            self.insert_token(token)
                


    def get_hit_rate(self):
        print(self.gpu_hit,self.disk_hit)
        return self.gpu_hit/(self.gpu_hit + self.disk_hit)
    
def get_input(path):
    prefixs = []
    layer_num=32
    total_request = []
    requests = []
    with open(path,'r') as f:
        i = 0
        line = f.readline()
        while line:
            if line[-2:] != ')\n':
                line += f.readline()
                continue
            prefix_id = int(line.split('prefix_id:')[1].split(',')[0])
            prompt_len = int(line.split('prompt_len:')[1].split(',')[0])
            request = (prefix_id,eval(line.split('sele_token:')[1]))
            requests.append(request)
            if int(prefix_id) >= len(prefixs):
                prefixs.append(prompt_len)
            i += 1
            if i % layer_num == 0:
                total_request.append(requests)
                requests = []
            line = f.readline()

    random.shuffle(total_request)
    return prefixs,total_request

    
def sim(prefixs,total_request,cache_type='token',rate=0.1,chunk_size=16):
    layer_num = 32
    total_tokens = sum(prefixs) * layer_num
    print(sum(prefixs)/len(prefixs),len(prefixs))

    # init
    a=Cache(size=rate*total_tokens,cache_type=cache_type,chunk_size=chunk_size,layer_num=layer_num)
    for i,token_num in enumerate(prefixs):
        a.insert_prefix(prefix_id=i,token_num=token_num)

    # sim
    for requests in total_request:
        for i,request in enumerate(requests):
            a.get(prefix_id=request[0],token_ids=request[1],layer_id=i)

    with open('./result.txt','a+') as f:
        f.write(f'rate:{rate},type:{cache_type},hit_rate:{a.get_hit_rate()}\n')

if __name__ == '__main__':
    cache_type = 'token'
    rate = 0.1
    chunk_size = 64
    path = './ids.txt'
    out_path = './result.txt'
    prefixs,total_request = get_input(path) 
    for rate in [0.1,0.2,0.3,0.4,0.5]:
        for cache_type in ['token','chunk','prefix']:
            sim(prefixs=prefixs,total_request=total_request,cache_type=cache_type,rate=rate,chunk_size=chunk_size)
        with open(out_path,'a+') as f:
            f.write('\n')

