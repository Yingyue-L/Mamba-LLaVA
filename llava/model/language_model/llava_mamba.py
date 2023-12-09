#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, PreTrainedModel

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.utils.generation import GenerationMixin

from collections import namedtuple
from functools import partial

from torch.nn import CrossEntropyLoss

class MambaConfig(LlamaConfig):
    model_type = "mamba"
    d_model: int = 512
    n_layer: int = 12
    pad_vocab_size_multiple: int = 1
    ssm_cfg: Optional[dict] = None
    norm_epsilon: float = 1e-5
    rms_norm: bool = False
    initializer_cfg: Optional[dict] = None
    residual_in_fp32: bool = False
    fused_add_norm: bool = False
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class LlavaConfig(MambaConfig):
    model_type = "llava"


class MambaModel(PreTrainedModel):
    def __init__(
        self, config: MambaConfig,
    ) -> None:
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        super().__init__(config)
        self.residual_in_fp32 = config.residual_in_fp32

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = config.fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / config.RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(config.initializer_cfg if config.initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inputs_embeds=None, inference_params=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            inputs_embeds, residual = layer(
                inputs_embeds, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (inputs_embeds + residual) if residual is not None else inputs_embeds
            inputs_embeds = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            inputs_embeds = fused_add_norm_fn(
                inputs_embeds,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return inputs_embeds


class LlavaMambaModel(LlavaMetaModel, MambaModel):
    config_class = LlavaConfig

    def __init__(self, config):
         super().__init__(config)


class LlavaMambaForCausalLM(PreTrainedModel, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    def __init__(
        self,
        config: LlavaConfig,
    ) -> None:
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        config.hidden_size = config.d_model
        PreTrainedModel.__init__(self, config)
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
        
        self.model = LlavaMambaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(config.initializer_cfg if config.initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.model.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config, device=device, dtype=dtype, **kwargs)
        model = cls(config)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith("backbone."):
                if key.startswith("backbone.embedding."):
                    state_dict[key.replace("backbone.embedding", "model.embed_tokens")] = state_dict.pop(key)
                else:
                    state_dict[key.replace("backbone", "model")] = state_dict.pop(key)
                
        model.load_state_dict(state_dict)
        return model

    def get_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None, inference_params=None, num_last_tokens=0
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.model(input_ids, inputs_embeds, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        lm_logits = lm_logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaMambaForCausalLM)
