import math
import torch
import torch.nn.functional as F
from adapter_kv_cache.compression_policies import PoliciesManager
from adapter_kv_cache.kv_cache.kv_cache_manager import KVCacheManager
from adapter_kv_cache.utils import image_scores


class KVCacheAdapter:
    policies_manager = PoliciesManager()

    def __init__(self, t=0.95,cache_k=None,cacke_v=None):
        self.t = t
        self.kv_cache_manager = KVCacheManager(cache_k,cacke_v)
        self.is_recover_head=set()
    def standard_score(self, xq, keys, mask):
        """
        Calculate the uncompressed kv attention score.
        :return:
        """
        head_dim = xq.shape[2]
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
        mask =None
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # image_scores(scores)
        return scores

    def policy_score(self,xq, keys, mask, token_index):
        """
        Calculate the compressed kv attention score by policy.
        :return:
        """
        head_scores_dict={}
        for head, token_index in token_index.items():
            c_keys = keys[:,list(token_index),:,:]
            head_c_keys = c_keys[:,:,[head],:]

            head_dim = xq.shape[2]
            head_xq = xq[:,:,[head],:]
            head_xq = head_xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            head_c_keys = head_c_keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            # values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            head_scores = torch.matmul(head_xq, head_c_keys.transpose(2, 3)) / math.sqrt(head_dim)
            # todo
            mask=None
            if mask is not None:
                head_scores = head_scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            head_scores = F.softmax(head_scores.float(), dim=-1).type_as(xq)
            head_scores_dict[head]=head_scores
        return head_scores_dict

    def is_recover_standard_score_with_t(self,xq, keys, mask, token_index):
        """
        Check whether policy score recover standard score with threshold T.
        :return:
        """

        head_policy_scores= self.policy_score(xq, keys, mask, token_index)
        is_recover_head=[]
        for h_policy_scores,h_token_index in zip(head_policy_scores.items(),token_index.items()):
            standard_score = self.standard_score[:, :, :, list(h_token_index[1])]
            head_standard_score=standard_score[:, [h_policy_scores[0]], :, :]
            recover_value = torch.max(abs(h_policy_scores[1] - head_standard_score), dim=-2)
            is_recover_mask = recover_value.values< (1-self.t)
            is_recover_head_by_token = torch.where(is_recover_mask == True)[1].tolist()
            if len(is_recover_head_by_token)==len(h_token_index[1]):
                is_recover_head.append(h_policy_scores[0])
        this_policy_head = set(is_recover_head)-self.is_recover_head
        for head in list(this_policy_head):
            self.is_recover_head.add(head)
        return list(this_policy_head)


    def prefill(self, q, k, v, mask, tokens):
        # self.kv_cache_manager.clean(k, v)
        self.standard_score = self.standard_score(q, k, mask)
        for policy in self.policies_manager.get_policies_list(policies_type="greed"):
            token_index = self.policies_manager.policies.get_adapter_tokens_index(policy, tokens, self.standard_score)
            # 策略是否ok，如果OK，则返回out， 否则继续下一个策略

            is_recover_head = self.is_recover_standard_score_with_t(q, k, mask, token_index)
            for head in is_recover_head:
                head_token_index = sorted(list(token_index[head]))
                self.kv_cache_manager.add( k, v, head,head_token_index)
            if len(self.is_recover_head)!=q.shape[-2]:
                continue
        values = v.transpose(1, 2)
        output = torch.matmul(self.standard_score, values)  # (bs, n_local_heads, seqlen, head_dim)
        bsz,seqlen,_,_ = v.shape
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return output

    def decode(self, q, k, v):
        k_cache, v, cache = self.kv_cache_manager.get()

        self.kv_cache_manager.update(k, v)
        pass
