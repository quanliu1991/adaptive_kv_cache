from typing import Optional
import torch
from adapter_kv_cache.kv_cache_adapter import KVCacheAdapter
from llama.model import Attention, apply_rotary_emb


class AdaptiveCacheAttention(Attention):
    def __init__(self, args):
        super().__init__(args)
        self.kv_cache_adapter = KVCacheAdapter(t=0.95)

    def forward(self, x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                tokens: torch.Tensor = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # todo 判断是否需要adaptive_kv_cache
        is_prefill = xq.shape[1] > 1
        if is_prefill:
            output = self.kv_cache_adapter.prefill(xq, xk, xv, tokens)  #
        else:
            output = self.kv_cache_adapter.decode(xq, xk, xv)
        return self.wo(output)
