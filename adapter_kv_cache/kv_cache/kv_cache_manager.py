from pydantic import BaseModel


class HeadKVCacke(BaseModel):
    seqlen: int = 0
    start_pos: int = 0


class KVCacheManager:
    def __init__(self, cache_k, cacke_v):
        self.kvcache = None
        self.cache_k = cache_k.transpose(2, 1)
        self.cache_v = cacke_v.transpose(2, 1)
        self.bsz = 1  # todo bsz>1
        self.num_heads = self.cache_v.shape[1]
        self.head_kv_cache = {i: HeadKVCacke() for i in range(self.cache_k.shape[1])}

    def get(self):
        k_cache = {}
        v_cache = {}
        for head_id, head_cache in self.head_kv_cache.items():
            k_cache[head_id] = self.cache_k[:self.bsz, : head_cache.start_pos]
            v_cache[head_id] = self.cache_v[:self.bsz, : head_cache.start_pos]
        return k_cache, v_cache

    def add(self, k, v, is_recover_head=None, token_index=[]):
        seqlen = len(token_index)

        self.cache_k = self.cache_k.to(k)
        self.cache_v = self.cache_v.to(v)
        bsz, _, num_heads, head_dim_ = k.shape
        start_pos = self.head_kv_cache[is_recover_head].start_pos
        c_k = k[:, token_index, :,:].transpose(2,1)
        c_v = v[:, token_index, :, :].transpose(2,1)
        c_k = c_k[:, [is_recover_head],:, :]
        c_v = c_v[:,  [is_recover_head], :,:]
        self.cache_k[:bsz, is_recover_head, start_pos: start_pos + seqlen] = c_k
        self.cache_v[:bsz, is_recover_head, start_pos: start_pos + seqlen] = c_v

        self.head_kv_cache[is_recover_head] = HeadKVCacke(seqlen=seqlen, start_pos=start_pos + seqlen)

    def update(self, k, v):
        for head_id, head_cache in self.head_kv_cache.items():
            self.cache_k[:self.bsz, [head_id], head_cache.start_pos: head_cache.start_pos + 1] = k
            self.cache_v[:self.bsz, [head_id], head_cache.start_pos: head_cache.start_pos + 1] = v

    def clean(self):
        self.head_kv_cache = {i: HeadKVCacke() for i in range(self.cache_k.shape[-2])}

    def kvcache_info(self):
        pass
