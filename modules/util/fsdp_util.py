import torch
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from diffusers.models.attention import BasicTransformerBlock

def transformer_auto_wrap_policy(module):
    return isinstance(module, BasicTransformerBlock)
