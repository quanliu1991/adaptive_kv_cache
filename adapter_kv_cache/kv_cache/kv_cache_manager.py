import torch
from pydantic import BaseModel
import logging

GB = 1024 ** 3
MB = 1024 ** 2


class HeadKVCacke(BaseModel):
    seqlen: int = 0
    start_pos: int = 0


class KVCacheManager:
    def __init__(self, cache_k: torch.Tensor, cacke_v: torch.Tensor):
        self.kvcache = None
        self.cache_k = cache_k.transpose(2, 1)
        self.cache_v = cacke_v.transpose(2, 1)
        self.bsz = 1  # todo bsz>1
        self.head_dim = self.cache_v.shape[-1]
        self.dtype = "float16"
        self.num_heads = self.cache_v.shape[1]
        self.head_kv_cache = {i: HeadKVCacke() for i in range(self.cache_k.shape[1])}
        self.cached_head = set()
        self.mem_info = dict.fromkeys(['origin_mem', 'compress_mem', 'save_mem'], 0)

    def get(self):
        k_cache = {}
        v_cache = {}
        for head_id, head_cache in self.head_kv_cache.items():
            k_cache[head_id] = self.cache_k[:self.bsz, [head_id], : head_cache.start_pos]
            v_cache[head_id] = self.cache_v[:self.bsz, [head_id], : head_cache.start_pos]
        return k_cache, v_cache

    def add(self, k: torch.Tensor, v: torch.Tensor, is_recover_head: int=None, token_index:list=None, policy:list=None, layer_id:int=None, original_length:int=None):
        seq_len = len(token_index)

        self.cache_k = self.cache_k.to(k)
        self.cache_v = self.cache_v.to(v)
        bsz, _, num_heads, head_dim_ = k.shape
        start_pos = self.head_kv_cache[is_recover_head].start_pos
        c_k = k[:, token_index, :, :].transpose(2, 1)
        c_v = v[:, token_index, :, :].transpose(2, 1)
        c_k = c_k[:, [is_recover_head], :, :]
        c_v = c_v[:, [is_recover_head], :, :]
        self.cache_k[:bsz, is_recover_head, start_pos: start_pos + seq_len] = c_k
        self.cache_v[:bsz, is_recover_head, start_pos: start_pos + seq_len] = c_v

        self.head_kv_cache[is_recover_head] = HeadKVCacke(seqlen=seq_len, start_pos=start_pos + seq_len)
        self.cached_head.add(is_recover_head)
        self.kv_cache_info(layer_id, is_recover_head, policy, token_index, original_length)

    def update(self, k:torch.Tensor, v:torch.Tensor):
        k = k.transpose(2, 1)
        v = v.transpose(2, 1)
        for head_id, head_cache in self.head_kv_cache.items():
            self.cache_k[:self.bsz, [head_id], head_cache.start_pos: head_cache.start_pos + 1] = k[:, [head_id], :, :]
            self.cache_v[:self.bsz, [head_id], head_cache.start_pos: head_cache.start_pos + 1] = v[:, [head_id], :, :]
            head_cache.start_pos += 1

    def clean(self):
        self.head_kv_cache = {i: HeadKVCacke() for i in range(self.cache_k.shape[-2])}

    def kv_cache_info(self, layer_id:int, is_recover_head:int, policy:int, token_index:list, original_length:int):
        logging.info(
            f"layer_id:{layer_id}(heads:{is_recover_head})->{policy}-> kv cache len:{original_length}->{len(token_index)}")
        original_head_mem = original_length * self.head_dim * self.get_num_bytes(self.dtype) * 2 / MB
        compress_head_mem = len(token_index) * self.head_dim * self.get_num_bytes(self.dtype) * 2 / MB
        self.mem_info["origin_mem"] += original_head_mem
        self.mem_info["compress_mem"] += compress_head_mem
        self.mem_info["save_mem"] += original_head_mem - compress_head_mem

        if len(self.cached_head) == self.num_heads:
            logging.info(
                f"layer_id:{layer_id} cache:\n"
                f"{self.mem_info['origin_mem']:.3f} MB -> {self.mem_info['compress_mem']:.3f}MB\n"
                f"saving:\n"
                f"{self.mem_info['save_mem']:.3f}MB ({self.mem_info['save_mem']/self.mem_info['origin_mem']*100:.3f})%")

    def get_num_bytes(self, dtype):
        if dtype == "float16":
            return 2
        else:
            assert 0, f"{dtype} NotImplemented"
