import inspect
import json

import torch


class Policies:
    def __init__(self):
        self.policies_map = {method.replace("_policies", ""): eval("self." + method)
                             for method in dir(self) if
                             inspect.ismethod(getattr(self, method)) and
                             method.split("_")[-1] == "policies"}
        with open("config.json","r") as f:
            config = json.load(f)

        self.special_token_table = config.special_token_table
        self.punctuation_table = config.punctuation_table
        self.locality_length = config.locality_length
        self.top_frequency = config.top_frequency
    def special_policies(self, token_id) -> tuple[torch.Tensor, list]:
        """
        :param token_id:
        :return:
        """
        id_index = []
        special_token_id = token_id

        return special_token_id, id_index

    def punctuation_policies(self, token_id) -> tuple[torch.Tensor, list]:
        id_index = []
        punctuation_token_id = token_id
        return punctuation_token_id, id_index

    def locality_policies(self, token_id) -> tuple[torch.Tensor, list]:
        id_index = []
        locality_tokrn_id = token_id[:self.locality_length]
        return locality_tokrn_id, id_index

    def frequency_policies(self, attention_score) -> tuple[torch.Tensor, list]:
        id_index = []
        frequency_tokrn_id = attention_score[:self.top_frequency]
        return frequency_tokrn_id, id_index

    def full_policies(self, token_id) -> tuple[torch.Tensor, list]:
        id_index = [i for i in range(token_id.shape[1])]
        return id_index


    def get_adapter_tokens_index(self, policies_hybrid: list, token_id=None, attention_score=None):
        tokens_index = []
        for policies in policies_hybrid:
            assert policies in list(self.policies_map.keys()), f"{policies} not in policies_map."
            input_ = attention_score if policies == "frequency" else token_id
            tokens_index.append(self.policies_map[policies](input_))
        return tokens_index


class PoliciesManager:
    def __init__(self):
        self.policies_priority=["special", "punctuation","frequency","locality","full"]
        self.policies = Policies()
    def greed(self):
        policies_hybrid = []
        for i in range(len(self.policies_priority)):
            policies_hybrid.append(self.policies_priority[:i+1])
        return policies_hybrid

    def get_policies_list(self, policies_type: str = "greed"):
        """
        Get different policies types of policy combinations, currently only greed is supported
        :param policies_type:
        :return:
        """
        if policies_type == "greed":
            return self.greed()

