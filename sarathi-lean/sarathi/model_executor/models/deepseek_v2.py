from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_world_size,
)


@dataclass(frozen=True)
class DeepseekV2MLADims:
    hidden_size: int
    tensor_parallel_world_size: int
    total_num_heads: int
    num_heads: int
    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_head_dim: int
    q_proj_output_dim_local: int
    kv_up_proj_output_dim_local: int
    o_proj_input_dim_local: int
    resident_cache_dim: int

    @classmethod
    def from_config(
        cls,
        config,
        tensor_parallel_world_size: Optional[int] = None,
    ) -> "DeepseekV2MLADims":
        tp_world_size = (
            get_tensor_model_parallel_world_size()
            if tensor_parallel_world_size is None
            else tensor_parallel_world_size
        )
        total_num_heads = config.num_attention_heads
        if total_num_heads % tp_world_size != 0:
            raise ValueError(
                "DeepSeek-V2 attention heads must divide evenly across tensor parallel ranks"
            )

        num_heads = total_num_heads // tp_world_size
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        return cls(
            hidden_size=config.hidden_size,
            tensor_parallel_world_size=tp_world_size,
            total_num_heads=total_num_heads,
            num_heads=num_heads,
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_head_dim=q_head_dim,
            q_proj_output_dim_local=num_heads * q_head_dim,
            kv_up_proj_output_dim_local=num_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            o_proj_input_dim_local=num_heads * config.v_head_dim,
            resident_cache_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        )


class DeepseekV2MLAAttention(nn.Module):

    def __init__(
        self,
        config,
        tensor_parallel_world_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.mla_dims = DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 MLA attention execution is not implemented yet. "
            "This scaffold only defines tensor-parallel MLA dimensions."
        )


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        tensor_parallel_world_size: Optional[int] = None,
    ):
        super().__init__()
        self.self_attn = DeepseekV2MLAAttention(
            config,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
        # MoE and RMSNorm execution are still pending.
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 decoder execution is not implemented yet. "
            "The contiguous MLA attention path and MoE blocks are still pending."
        )


class DeepseekV2Model(nn.Module):

    def __init__(
        self,
        config,
        *,
        tensor_parallel_world_size: Optional[int] = None,
        pipeline_parallel_world_size: Optional[int] = None,
        pipeline_parallel_rank: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.tensor_parallel_world_size = (
            get_tensor_model_parallel_world_size()
            if tensor_parallel_world_size is None
            else tensor_parallel_world_size
        )
        self.pipeline_parallel_world_size = (
            get_pipeline_model_parallel_world_size()
            if pipeline_parallel_world_size is None
            else pipeline_parallel_world_size
        )
        self.pipeline_parallel_rank = (
            get_pipeline_model_parallel_rank()
            if pipeline_parallel_rank is None
            else pipeline_parallel_rank
        )
        if config.num_hidden_layers % self.pipeline_parallel_world_size != 0:
            raise ValueError(
                "DeepSeek-V2 hidden layers must divide evenly across pipeline stages"
            )
        self.num_layers = config.num_hidden_layers // self.pipeline_parallel_world_size
        self.layer_offset = self.pipeline_parallel_rank * self.num_layers
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    tensor_parallel_world_size=self.tensor_parallel_world_size,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 model execution is not implemented yet. "
            "The contiguous MLA model path is still pending."
        )


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)
        self.mla_dims = DeepseekV2MLADims.from_config(config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 causal LM execution is not implemented yet. "
            "The MLA attention/model runtime path still needs to be added."
        )

    def load_weights(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 weight loading is not implemented yet. "
            "The MLA attention/model path still needs to be added."
        )
