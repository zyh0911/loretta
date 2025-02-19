import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose
from ..utils import decompose_weight

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

if is_bnb_available():
    print("bitsandbytes 模块可用")
else:
    print("bitsandbytes 模块不可用")

@dataclass
class vcConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft_new.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """
    sigma=1

    k =20

    vector_length = 4

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.VC


class vcModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft_new import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_vc_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        # Check if the model is loaded in 8-bit mode
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        # Initialize flags and arguments
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "merge_weights": self.peft_config.inference_mode and not is_hf_device_map_available,
        }
        # Iterate over all named modules in the model
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            # Check if the module name matches the target modules
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                # # Handle 8-bit quantization
                # if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                #     kwargs.update(
                #         {
                #             "has_fp16_weights": target.state.has_fp16_weights,
                #             "memory_efficient_backward": target.state.memory_efficient_backward,
                #             "threshold": target.state.threshold,
                #             "index": target.index,
                #         }
                #     )
                #     if self.peft_config.enable_lora is None:
                #         new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                #     else:
                #         kwargs.update({"enable_lora": self.peft_config.enable_lora})
                #         new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                # Handle standard Linear modules
                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                # Handle other modules with LoRA enabled
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({"enable_lora": self.peft_config.enable_lora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = Linear(in_features, out_features,  bias=bias, **kwargs)
                # Replace the target module with the new module
                self._replace_module(parent, target_name, new_module, target)
        # Raise an error if no target modules are found
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, VCLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_vc_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, VCLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError

class VCLayer(nn.Module):
    def __init__(
        self, 
        out_features,
        vector_length, 
        k):
        self.out_features = out_features
        self.vector_length = vector_length
        self.k = k
        
    

class Special_linear_Layer(nn.Linear,nn.Module):
    def __init__(self,          
                input_feature,
                out_features, 
                vector_length, 
                k,
                bias=True):
        super(Special_linear_Layer, self).__init__()
        self.input_feature = input_feature
        self.out_features = out_features
        self.vector_length = vector_length
        self.k = k

        weight1,weight2=decompose_weight(nn.Linear.weight,vector_length)
        self.layer1 = nn.Parameter(weight1)
        self.layer2 = nn.Parameter(weight2)

        # self.layer1 = nn.Parameter(torch.Tensor(input_feature, out_features/k))
        # self.layer2 = nn.Parameter(torch.Tensor(out_features/k*input_feature/vector_length, out_features))
        if bias == False:
            self.bias = 0
        else:
            self.bias = nn.Linear.bias

    def get_weight(self):
        return self.layer1, self.layer2
    
    def forward(self, x: torch.Tensor):
        s, m = x.shape  
        m, n =self.layer1.shape  
        x.view(s, m // self.vector_length, self.vector_length).permute(1, 0, 2)
        self.layer1.view(s, m // self.vector_length, self.out_features/self.k).permute(1, 0, 2)  
        self.layer2.view(s, self.input_feature/self.vector_length,self.out_features).permute(1, 0, 2)  
        result=torch.matmul(x, self.layer1)
        result= torch.matmul(result, self.layer2)
        result = result.sum(dim=0) +self.bias
        self.weight =result
        return result
    


class Linear(nn.Linear):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        vector_length: int,
        k: int,
        out_features: int,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
    
        self.slayer = Special_linear_Layer(nn.Linear,out_features, vector_length, k)
        self.sigma = 0.99
        self.weight.requires_grad = False    

    def update_sigma(self, current_step: int, total_steps: int):
        self.sigma = max(0, 0.99 * (1 - current_step / total_steps))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.slayer.train(mode)

    def eval(self):
        self.slayer.eval()

    def forward(self, x: torch.Tensor):
        self.slayer.weight = self.sigma * x + (1 - self.sigma) * self.slayer.weight
        return self.slayer.weight

    
# if is_bnb_available():

#     class Linear8bitLt(bnb.nn.Linear8bitLt):
#         # Lora implemented in a dense layer
#         def __init__(
#             self,
#             in_features,
#             out_features,
#             **kwargs,
#         ):
#             bnb.nn.Linear8bitLt.__init__(
#                 self,
#                 in_features,
#                 out_features,
#                 bias=kwargs.get("bias", True),
#                 has_fp16_weights=kwargs.get("has_fp16_weights", True),
#                 memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
#                 threshold=kwargs.get("threshold", 0.0),
#                 index=kwargs.get("index", None),
#             )
        
#             # Actual trainable parameters
#             if r > 0:
#                 self.lora_A = nn.Linear(in_features, r, bias=False)
#                 self.lora_B = nn.Linear(r, out_features, bias=False)
#                 self.scaling = self.lora_alpha / self.r
#                 # Freezing the pre-trained weight matrix
#                 self.weight.requires_grad = False
#             self.reset_parameters()

#         def reset_parameters(self):
#             if hasattr(self, "lora_A"):
#                 # initialize A the same way as the default for nn.Linear and B to zero
#                 nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.lora_B.weight)

#         def forward(self, x: torch.Tensor):
#             result = super().forward(x)

#             if self.disable_adapters:
#                 return result
#             elif self.r > 0:
#                 if not torch.is_autocast_enabled():
#                     expected_dtype = result.dtype

#                     if x.dtype != torch.float32:
#                         x = x.float()
#                     output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
#                     result += output
#                 else:
#                     output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
#                     result += output
#             return result

#     class MergedLinear8bitLt(bnb.nn.Linear8bitLt, vcLayer):
#         # Lora implemented in a dense layer
#         def __init__(
#             self,
#             in_features: int,
#             out_features: int,
#             r: int = 0,
#             lora_alpha: int = 1,
#             lora_dropout: float = 0.0,
#             enable_lora: List[bool] = [False],
#             **kwargs,
#         ):
#             bnb.nn.Linear8bitLt.__init__(
#                 self,
#                 in_features,
#                 out_features,
#                 bias=kwargs.get("bias", True),
#                 has_fp16_weights=kwargs.get("has_fp16_weights", True),
#                 memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
#                 threshold=kwargs.get("threshold", 0.0),
#                 index=kwargs.get("index", None),
#             )
#             vcLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
#             if out_features % len(enable_lora) != 0:
#                 raise ValueError("The length of enable_lora must divide out_features")
#             self.enable_lora = enable_lora
#             # Actual trainable parameters
#             if r > 0 and any(enable_lora):
#                 self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
#                 self.lora_B = nn.Conv1d(
#                     r * sum(enable_lora),
#                     out_features // len(enable_lora) * sum(enable_lora),
#                     kernel_size=1,
#                     groups=2,
#                     bias=False,
#                 )
#                 self.scaling = self.lora_alpha / self.r
#                 # Freezing the pre-trained weight matrix
#                 self.weight.requires_grad = False
#                 # Compute the indices
#                 self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
#                 self.lora_ind[enable_lora, :] = True
#                 self.lora_ind = self.lora_ind.view(-1)
#             self.reset_parameters()

#         def reset_parameters(self):
#             if hasattr(self, "lora_A"):
#                 # initialize A the same way as the default for nn.Linear and B to zero
#                 nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.lora_B.weight)

#         def zero_pad(self, x):
#             result = x.new_zeros((*x.shape[:-1], self.out_features))
#             result = result.view(-1, self.out_features)
#             result[:, self.lora_ind] = x.reshape(
#                 -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
#             )
#             return result.view((*x.shape[:-1], self.out_features))

#         def forward(self, x: torch.Tensor):
#             result = super().forward(x)
#             if self.disable_adapters:
#                 return result
#             elif self.r > 0:
#                 if not torch.is_autocast_enabled():
#                     expected_dtype = result.dtype
#                     if x.dtype != torch.float32:
#                         x = x.float()
#                     after_A = self.lora_A(self.lora_dropout(x))
#                     after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
#                     output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
#                     result += output
#                 else:
#                     after_A = self.lora_A(self.lora_dropout(x))
#                     after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
#                     output = self.zero_pad(after_B) * self.scaling
#                     result += output
#             return result
