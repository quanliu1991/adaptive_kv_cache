import inspect
import json
import os
import torch


class Policies:
    """ KV CACHE COMPRESSION POLICIES, had five policies now.
    [special_policies, punctuation_policies, locality_policies, frequency_policies, full_policies]"""
    def __init__(self):
        self.policies_map = {method.replace("_policies", ""): eval("self." + method)
                             for method in dir(self) if
                             inspect.ismethod(getattr(self, method)) and
                             method.split("_")[-1] == "policies"}

        self.config_path = os.path.dirname(__file__).replace("adaptive_kv_cache/adapter_kv_cache/kv_cache",
                                                             "adaptive_kv_cache/adapter_kv_cache/configs/config.json")
        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.num_heads = 32  # todo
        self.special_token_table = config.get("special_token_ids")
        self.punctuation_table = config.get("punctuation_token_ids")
        self.locality_length = config.get("locality_length")
        self.top_frequency = config.get("top_p_frequency")

    def special_policies(self, token_ids: torch.Tensor):
        id_indexs:set = set()
        for special_token in self.special_token_table:
            indices = (token_ids == special_token).nonzero().tolist()
            for i in indices:
                id_indexs.add(i[1])
        head_token_indexs = {i: id_indexs for i in range(self.num_heads)}
        return head_token_indexs

    def punctuation_policies(self, token_ids: torch.Tensor):
        id_indexs:set = set()
        for punctuation_token in self.punctuation_table:
            indices = (token_ids == punctuation_token).nonzero().tolist()
            for i in indices:
                id_indexs.add(i[1])
        head_token_indexs = {i: id_indexs for i in range(self.num_heads)}
        return head_token_indexs

    def locality_policies(self, token_ids: torch.Tensor):
        seq_len = token_ids.shape[-1]
        compose_seq_len = int(seq_len * self.locality_length)
        id_indexs = [i for i in range(seq_len - compose_seq_len, seq_len)]
        head_token_indexs = {i: set(id_indexs) for i in range(self.num_heads)}
        return head_token_indexs

    def frequency_policies(self, attention_score: torch.Tensor):
        head_token_indexs = {}
        attention_score_sum = torch.sum(attention_score, dim=-2)
        for head in range(self.num_heads):
            attention_score_sum_head = attention_score_sum[0, head, :]
            top_k = int(self.top_frequency * len(attention_score_sum_head))
            top_values, top_indices = torch.topk(attention_score_sum_head, k=top_k)
            head_token_indexs.update({head: set(top_indices.tolist())})
        return head_token_indexs

    def full_policies(self, token_id: torch.Tensor):
        id_indexs = [i for i in range(token_id.shape[1])]
        head_token_indexs = {i: set(id_indexs) for i in range(self.num_heads)}
        return head_token_indexs

    def get_adapter_tokens_index(self, policies_hybrid: list, token_id: torch.Tensor=None, attention_score: torch.Tensor=None):
        tokens_index = set()
        head_tokens_index_union = {i: tokens_index for i in range(self.num_heads)}
        for policies in policies_hybrid:
            assert policies in list(self.policies_map.keys()), f"{policies} not in policies_map."
            input_ = attention_score if policies == "frequency" else token_id
            head_token_indexs = self.policies_map[policies](input_)
            for head in head_tokens_index_union.keys():
                head_tokens_index_union[head] = head_token_indexs[head].union(head_tokens_index_union[head])
        return head_tokens_index_union


class PoliciesManager:
    """The strategy search method manager, currently supports only greedy search """
    def __init__(self):
        self.policies_priority = ["special", "punctuation", "frequency", "locality", "full"]
        self.policies = Policies()

    def greed(self):
        policies_hybrid = []
        for i in range(len(self.policies_priority)):
            policies_hybrid.append(self.policies_priority[:i + 1])
        return policies_hybrid

    def get_policies_list(self, policies_type: str = "greed"):
        """
        Get different policies types of policy combinations, currently only greed is supported
        :param policies_type:
        :return:
        """
        if policies_type == "greed":
            return self.greed()
        else:
            assert 0, f"{policies_type} not support now."
