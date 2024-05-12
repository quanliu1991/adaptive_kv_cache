from adapter_kv_cache.compression_policies import PoliciesManager
from adapter_kv_cache.kv_cache.kv_cache_manager import KVCacheManager

class KVCacheAdapter:
    def __init__(self,t=0.95):
        self.policies_manager=PoliciesManager()
        self.kv_cache_manager = KVCacheManager()
        self.t=t

    def standard_score(self):
        """
        Calculate the uncompressed kv attention score.
        :return:
        """
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        pass

    def policy_score(self):
        """
        Calculate the compressed kv attention score by policy.
        :return:
        """
        pass

    def is_recover_standard_score_with_t(self):
        """
        Check whether policy score recover standard score with threshold T.
        :return:
        """
        pass
        is_recover, output= None,None
        return is_recover ,output


    def prefill(self,q,k,v,tokens):
        self.kv_cache_manager.clean(k, v)
        standard_score= self.standard_score()
        for policy in self.policies_manager.get_policies_list(policies_type="greed"):
            token_index = self.policies_manager.policies.get_adapter_tokens_index(policy, tokens, standard_score)
            # 策略是否ok，如果OK，则返回out， 否则继续下一个策略
            is_recover, output = self.is_recover_standard_score_with_t(q,k,v,token_index)
            if not is_recover:
                continue
            self.kv_cache_manager.add(k,v)
            return output

    def decode(self,q,k,v):
        k_cache,v,cache= self.kv_cache_manager.get(k, v)

        self.kv_cache_manager.append(k,v)
        pass



