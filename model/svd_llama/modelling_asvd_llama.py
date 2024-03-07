from transformers import LlamaForCausalLM
from .configuration_asvd_llama import ASVDLlamaConfig
import torch.nn as nn

class ASVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        return self.ALinear(self.BLinear(input))

class ASVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = ASVDLlamaConfig
    def __init__(self, config:ASVDLlamaConfig):
        super().__init__(config)
        self.truncation_ranks=config.truncation_ranks

        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(raw_linear)


        for name,module in self.named_modules():
            if name in self.truncation_ranks:
                info=linear_info[module]
                new_layer=ASVDLinear(module.in_features,module.out_features,self.truncation_ranks[name],bias=module.bias is not None)
                setattr(info["father"], info["name"], new_layer)
                
        