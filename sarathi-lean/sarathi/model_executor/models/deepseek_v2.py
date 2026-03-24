import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sarathi.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
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


@dataclass(frozen=True)
class DeepseekV2MLAResidentCache:
    kv_latent: torch.Tensor
    k_rope: torch.Tensor

    def __post_init__(self):
        if self.kv_latent.ndim != 2:
            raise ValueError("kv_latent must have shape [tokens, kv_lora_rank]")
        if self.k_rope.ndim != 3:
            raise ValueError("k_rope must have shape [tokens, num_heads, qk_rope_head_dim]")
        if self.kv_latent.shape[0] != self.k_rope.shape[0]:
            raise ValueError("kv_latent and k_rope must agree on token count")

    @property
    def num_tokens(self) -> int:
        return self.kv_latent.shape[0]


@dataclass(frozen=True)
class DeepseekV2MLAProjectionWeights:
    q_proj: Optional[torch.Tensor]
    q_a_proj: Optional[torch.Tensor]
    q_a_layernorm_weight: Optional[torch.Tensor]
    q_b_proj: Optional[torch.Tensor]
    kv_latent_proj: torch.Tensor
    kv_a_layernorm_weight: Optional[torch.Tensor]
    k_rope_proj: torch.Tensor
    kv_up_proj: torch.Tensor
    o_proj: torch.Tensor


@dataclass(frozen=True)
class DeepseekV2MLPWeights:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


@dataclass(frozen=True)
class DeepseekV2MoEWeights:
    gate: torch.Tensor
    experts: Tuple[DeepseekV2MLPWeights, ...]
    shared_experts: Optional[DeepseekV2MLPWeights] = None
    top_k: int = 1
    norm_topk_prob: bool = True


@dataclass(frozen=True)
class DeepseekV2LayerCache:
    kv_cache: object
    resident_cache: Optional[DeepseekV2MLAResidentCache] = None


@dataclass(frozen=True)
class DeepseekV2ComponentMLAKVCache:
    kv_latent: torch.Tensor
    k_rope: torch.Tensor


@dataclass(frozen=True)
class DeepseekV2MLAWrapperInputs:
    query: torch.Tensor
    kv_cache: object
    kv_up_proj_weight: torch.Tensor
    past_resident_cache: Optional[DeepseekV2MLAResidentCache]
    new_resident_cache: DeepseekV2MLAResidentCache
    softmax_scale: float
    layer_id: Optional[int]
    mla_dims: DeepseekV2MLADims


DeepseekV2AttentionBackend = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Optional[DeepseekV2MLAResidentCache], float],
    torch.Tensor,
]


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


def split_query_projection(
    query_states: torch.Tensor,
    mla_dims: DeepseekV2MLADims,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if query_states.ndim != 2:
        raise ValueError("query_states must have shape [tokens, num_heads * q_head_dim]")
    expected_width = mla_dims.q_proj_output_dim_local
    if query_states.shape[1] != expected_width:
        raise ValueError("query_states width does not match local MLA query projection size")

    query_states = query_states.view(-1, mla_dims.num_heads, mla_dims.q_head_dim)
    q_nope, q_rope = torch.split(
        query_states,
        [mla_dims.qk_nope_head_dim, mla_dims.qk_rope_head_dim],
        dim=-1,
    )
    return q_nope, q_rope


def make_resident_cache(
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    mla_dims: DeepseekV2MLADims,
) -> DeepseekV2MLAResidentCache:
    if kv_latent.ndim != 2 or kv_latent.shape[1] != mla_dims.kv_lora_rank:
        raise ValueError("kv_latent must have shape [tokens, kv_lora_rank]")
    if k_rope.ndim != 3:
        raise ValueError("k_rope must have shape [tokens, num_heads, qk_rope_head_dim]")
    expected_k_rope_shape = (kv_latent.shape[0], mla_dims.num_heads, mla_dims.qk_rope_head_dim)
    if tuple(k_rope.shape) != expected_k_rope_shape:
        raise ValueError("k_rope shape does not match local MLA rope dimensions")
    return DeepseekV2MLAResidentCache(kv_latent=kv_latent, k_rope=k_rope)


def append_resident_cache(
    cache: Optional[DeepseekV2MLAResidentCache],
    new_cache: DeepseekV2MLAResidentCache,
) -> DeepseekV2MLAResidentCache:
    if cache is None:
        return new_cache
    return DeepseekV2MLAResidentCache(
        kv_latent=torch.cat([cache.kv_latent, new_cache.kv_latent], dim=0),
        k_rope=torch.cat([cache.k_rope, new_cache.k_rope], dim=0),
    )


def reconstruct_dense_kv(
    cache: DeepseekV2MLAResidentCache,
    kv_up_proj_weight: torch.Tensor,
    mla_dims: DeepseekV2MLADims,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expected_weight_shape = (
        mla_dims.kv_lora_rank,
        mla_dims.kv_up_proj_output_dim_local,
    )
    if tuple(kv_up_proj_weight.shape) != expected_weight_shape:
        raise ValueError("kv_up_proj_weight shape does not match local MLA up-projection size")

    kv_dense = cache.kv_latent @ kv_up_proj_weight
    kv_dense = kv_dense.view(
        cache.num_tokens,
        mla_dims.num_heads,
        mla_dims.qk_nope_head_dim + mla_dims.v_head_dim,
    )
    k_nope, value = torch.split(
        kv_dense,
        [mla_dims.qk_nope_head_dim, mla_dims.v_head_dim],
        dim=-1,
    )
    key = torch.cat([k_nope, cache.k_rope], dim=-1)
    return key, value


def make_projection_weights(
    *,
    q_proj: Optional[torch.Tensor] = None,
    q_a_proj: Optional[torch.Tensor] = None,
    q_a_layernorm_weight: Optional[torch.Tensor] = None,
    q_b_proj: Optional[torch.Tensor] = None,
    kv_latent_proj: torch.Tensor,
    kv_a_layernorm_weight: Optional[torch.Tensor] = None,
    k_rope_proj: torch.Tensor,
    kv_up_proj: torch.Tensor,
    o_proj: torch.Tensor,
    mla_dims: DeepseekV2MLADims,
) -> DeepseekV2MLAProjectionWeights:
    expected_q_proj = (mla_dims.hidden_size, mla_dims.q_proj_output_dim_local)
    if mla_dims.q_lora_rank is not None:
        expected_q_a_proj = (mla_dims.hidden_size, mla_dims.q_lora_rank)
        expected_q_a_layernorm_weight = (mla_dims.q_lora_rank,)
        expected_q_b_proj = (mla_dims.q_lora_rank, mla_dims.q_proj_output_dim_local)
    else:
        expected_q_a_proj = None
        expected_q_a_layernorm_weight = None
        expected_q_b_proj = None
    expected_kv_latent_proj = (mla_dims.hidden_size, mla_dims.kv_lora_rank)
    expected_kv_a_layernorm_weight = (mla_dims.kv_lora_rank,)
    expected_k_rope_proj = (
        mla_dims.hidden_size,
        mla_dims.num_heads * mla_dims.qk_rope_head_dim,
    )
    expected_kv_up_proj = (
        mla_dims.kv_lora_rank,
        mla_dims.kv_up_proj_output_dim_local,
    )
    expected_o_proj = (mla_dims.o_proj_input_dim_local, mla_dims.hidden_size)
    if q_proj is not None:
        if tuple(q_proj.shape) != expected_q_proj:
            raise ValueError("q_proj shape does not match local MLA query projection size")
    else:
        if mla_dims.q_lora_rank is None:
            raise ValueError("q_proj is required when q_lora_rank is not configured")
        if q_a_proj is None or q_a_layernorm_weight is None or q_b_proj is None:
            raise ValueError(
                "q_a_proj, q_a_layernorm_weight, and q_b_proj are required when q_proj is absent"
            )
    if q_a_proj is not None and tuple(q_a_proj.shape) != expected_q_a_proj:
        raise ValueError("q_a_proj shape does not match local MLA query latent size")
    if (
        q_a_layernorm_weight is not None
        and tuple(q_a_layernorm_weight.shape) != expected_q_a_layernorm_weight
    ):
        raise ValueError("q_a_layernorm_weight shape does not match q_lora_rank")
    if q_b_proj is not None and tuple(q_b_proj.shape) != expected_q_b_proj:
        raise ValueError("q_b_proj shape does not match local MLA query projection size")
    if tuple(kv_latent_proj.shape) != expected_kv_latent_proj:
        raise ValueError("kv_latent_proj shape does not match local MLA latent projection size")
    if (
        kv_a_layernorm_weight is not None
        and tuple(kv_a_layernorm_weight.shape) != expected_kv_a_layernorm_weight
    ):
        raise ValueError("kv_a_layernorm_weight shape does not match kv_lora_rank")
    if tuple(k_rope_proj.shape) != expected_k_rope_proj:
        raise ValueError("k_rope_proj shape does not match local MLA rope projection size")
    if tuple(kv_up_proj.shape) != expected_kv_up_proj:
        raise ValueError("kv_up_proj shape does not match local MLA up-projection size")
    if tuple(o_proj.shape) != expected_o_proj:
        raise ValueError("o_proj shape does not match local MLA output projection size")
    return DeepseekV2MLAProjectionWeights(
        q_proj=q_proj,
        q_a_proj=q_a_proj,
        q_a_layernorm_weight=q_a_layernorm_weight,
        q_b_proj=q_b_proj,
        kv_latent_proj=kv_latent_proj,
        kv_a_layernorm_weight=kv_a_layernorm_weight,
        k_rope_proj=k_rope_proj,
        kv_up_proj=kv_up_proj,
        o_proj=o_proj,
    )


def make_mlp_weights(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    hidden_size: int,
) -> DeepseekV2MLPWeights:
    intermediate_size = gate_proj.shape[1]
    expected_gate_proj = (hidden_size, intermediate_size)
    expected_up_proj = (hidden_size, intermediate_size)
    expected_down_proj = (intermediate_size, hidden_size)
    if tuple(gate_proj.shape) != expected_gate_proj:
        raise ValueError("gate_proj shape does not match MLP hidden/intermediate dimensions")
    if tuple(up_proj.shape) != expected_up_proj:
        raise ValueError("up_proj shape does not match MLP hidden/intermediate dimensions")
    if tuple(down_proj.shape) != expected_down_proj:
        raise ValueError("down_proj shape does not match MLP intermediate/hidden dimensions")
    return DeepseekV2MLPWeights(
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
    )


def apply_mlp(
    hidden_states: torch.Tensor,
    mlp_weights: DeepseekV2MLPWeights,
) -> torch.Tensor:
    gate = torch.nn.functional.silu(hidden_states @ mlp_weights.gate_proj)
    up = hidden_states @ mlp_weights.up_proj
    return (gate * up) @ mlp_weights.down_proj


def make_moe_weights(
    gate: torch.Tensor,
    experts: Tuple[DeepseekV2MLPWeights, ...],
    *,
    shared_experts: Optional[DeepseekV2MLPWeights] = None,
    top_k: int = 1,
    norm_topk_prob: bool = True,
    hidden_size: int,
) -> DeepseekV2MoEWeights:
    if gate.ndim != 2 or gate.shape[1] != hidden_size:
        raise ValueError("gate must have shape [num_experts, hidden_size]")
    if len(experts) == 0:
        raise ValueError("experts must provide at least one routed expert")
    if gate.shape[0] != len(experts):
        raise ValueError("gate rows must match the number of routed experts")
    if top_k <= 0 or top_k > len(experts):
        raise ValueError("top_k must be between 1 and the number of routed experts")
    return DeepseekV2MoEWeights(
        gate=gate,
        experts=experts,
        shared_experts=shared_experts,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
    )


def apply_moe(
    hidden_states: torch.Tensor,
    moe_weights: DeepseekV2MoEWeights,
) -> torch.Tensor:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != moe_weights.gate.shape[1]:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")

    gate_scores = hidden_states @ moe_weights.gate.t()
    if gate_scores.ndim != 2:
        raise ValueError("gate scores must have shape [tokens, num_experts]")
    routing_probs = torch.softmax(gate_scores, dim=-1)
    topk_probs, topk_indices = torch.topk(
        routing_probs,
        k=moe_weights.top_k,
        dim=-1,
    )
    if moe_weights.norm_topk_prob:
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    output = hidden_states.new_zeros(hidden_states.shape)
    for expert_idx, expert_weights in enumerate(moe_weights.experts):
        expert_output = apply_mlp(hidden_states, expert_weights)
        expert_mask = topk_indices == expert_idx
        if not expert_mask.any():
            continue
        expert_prob = (topk_probs * expert_mask.to(dtype=topk_probs.dtype)).sum(
            dim=-1,
            keepdim=True,
        )
        output = output + expert_output * expert_prob

    if moe_weights.shared_experts is not None:
        output = output + apply_mlp(hidden_states, moe_weights.shared_experts)
    return output


def make_layer_cache(
    kv_cache: object,
    resident_cache: Optional[DeepseekV2MLAResidentCache] = None,
) -> DeepseekV2LayerCache:
    return DeepseekV2LayerCache(
        kv_cache=kv_cache,
        resident_cache=resident_cache,
    )


def make_component_mla_kv_cache(
    batch_size: int,
    max_seq_len: int,
    mla_dims: DeepseekV2MLADims,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> DeepseekV2ComponentMLAKVCache:
    return DeepseekV2ComponentMLAKVCache(
        kv_latent=torch.zeros(
            batch_size,
            max_seq_len,
            mla_dims.kv_lora_rank,
            device=device,
            dtype=dtype,
        ),
        k_rope=torch.zeros(
            batch_size,
            max_seq_len,
            mla_dims.num_heads,
            mla_dims.qk_rope_head_dim,
            device=device,
            dtype=dtype,
        ),
    )


def make_runtime_mla_kv_caches(
    num_layers: int,
    batch_size: int,
    max_seq_len: int,
    mla_dims: DeepseekV2MLADims,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[DeepseekV2ComponentMLAKVCache, ...]:
    return tuple(
        make_component_mla_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            mla_dims=mla_dims,
            device=device,
            dtype=dtype,
        )
        for _ in range(num_layers)
    )


def is_component_mla_kv_cache(kv_cache: object) -> bool:
    return isinstance(kv_cache, DeepseekV2ComponentMLAKVCache)


def read_component_mla_kv_cache(
    kv_cache: DeepseekV2ComponentMLAKVCache,
    batch_idx: int,
    seq_len: int,
) -> Optional[DeepseekV2MLAResidentCache]:
    if seq_len == 0:
        return None
    return DeepseekV2MLAResidentCache(
        kv_latent=kv_cache.kv_latent[batch_idx, :seq_len].clone(),
        k_rope=kv_cache.k_rope[batch_idx, :seq_len].clone(),
    )


def write_component_mla_kv_cache(
    kv_cache: DeepseekV2ComponentMLAKVCache,
    batch_idx: int,
    token_offset: int,
    resident_cache: DeepseekV2MLAResidentCache,
) -> None:
    next_offset = token_offset + resident_cache.num_tokens
    kv_cache.kv_latent[batch_idx, token_offset:next_offset].copy_(resident_cache.kv_latent)
    kv_cache.k_rope[batch_idx, token_offset:next_offset].copy_(resident_cache.k_rope)


def resolve_layer_cache(
    layer_cache_or_kv_cache,
    resident_cache: Optional[DeepseekV2MLAResidentCache] = None,
) -> Tuple[object, Optional[DeepseekV2MLAResidentCache]]:
    if isinstance(layer_cache_or_kv_cache, DeepseekV2LayerCache):
        if resident_cache is not None and layer_cache_or_kv_cache.resident_cache is not None:
            raise ValueError(
                "resident_cache must not be provided separately when layer_cache already carries one"
            )
        return (
            layer_cache_or_kv_cache.kv_cache,
            layer_cache_or_kv_cache.resident_cache
            if resident_cache is None
            else resident_cache,
        )
    return layer_cache_or_kv_cache, resident_cache


def get_layer_cache_kv_handle(layer_cache_or_kv_cache) -> object:
    if isinstance(layer_cache_or_kv_cache, DeepseekV2LayerCache):
        return layer_cache_or_kv_cache.kv_cache
    return layer_cache_or_kv_cache


def project_mla_from_hidden_states(
    hidden_states: torch.Tensor,
    projection_weights: DeepseekV2MLAProjectionWeights,
    mla_dims: DeepseekV2MLADims,
) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != mla_dims.hidden_size:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")

    if projection_weights.q_proj is not None:
        query_states = hidden_states @ projection_weights.q_proj
    else:
        q_latent = hidden_states @ projection_weights.q_a_proj
        variance = q_latent.pow(2).mean(dim=-1, keepdim=True)
        q_latent = q_latent * torch.rsqrt(variance + 1e-6)
        q_latent = q_latent * projection_weights.q_a_layernorm_weight
        query_states = q_latent @ projection_weights.q_b_proj
    kv_latent = hidden_states @ projection_weights.kv_latent_proj
    if projection_weights.kv_a_layernorm_weight is not None:
        variance = kv_latent.pow(2).mean(dim=-1, keepdim=True)
        kv_latent = kv_latent * torch.rsqrt(variance + 1e-6)
        kv_latent = kv_latent * projection_weights.kv_a_layernorm_weight
    k_rope = hidden_states @ projection_weights.k_rope_proj
    k_rope = k_rope.view(-1, mla_dims.num_heads, mla_dims.qk_rope_head_dim)
    return query_states, make_resident_cache(kv_latent, k_rope, mla_dims)


def contiguous_mla_attention_forward(
    query_states: torch.Tensor,
    new_kv_latent: torch.Tensor,
    new_k_rope: torch.Tensor,
    kv_up_proj_weight: torch.Tensor,
    mla_dims: DeepseekV2MLADims,
    cache: Optional[DeepseekV2MLAResidentCache] = None,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
    q_nope, q_rope = split_query_projection(query_states, mla_dims)
    new_cache = make_resident_cache(new_kv_latent, new_k_rope, mla_dims)
    full_cache = append_resident_cache(cache, new_cache)
    key, value = reconstruct_dense_kv(full_cache, kv_up_proj_weight, mla_dims)
    query = torch.cat([q_nope, q_rope], dim=-1)

    if softmax_scale is None:
        softmax_scale = mla_dims.q_head_dim ** -0.5

    past_len = 0 if cache is None else cache.num_tokens
    scores = torch.einsum("thd,shd->hts", query, key) * softmax_scale

    source_positions = torch.arange(key.shape[0], device=query.device)
    target_positions = past_len + torch.arange(query.shape[0], device=query.device)
    causal_mask = source_positions.unsqueeze(0) <= target_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("hts,shv->thv", attn_weights, value)
    return output.reshape(query.shape[0], -1), full_cache


def contiguous_mla_attention_from_hidden_states(
    hidden_states: torch.Tensor,
    projection_weights: DeepseekV2MLAProjectionWeights,
    mla_dims: DeepseekV2MLADims,
    cache: Optional[DeepseekV2MLAResidentCache] = None,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
    query_states, new_cache = project_mla_from_hidden_states(
        hidden_states,
        projection_weights,
        mla_dims,
    )
    return contiguous_mla_attention_forward(
        query_states=query_states,
        new_kv_latent=new_cache.kv_latent,
        new_k_rope=new_cache.k_rope,
        kv_up_proj_weight=projection_weights.kv_up_proj,
        mla_dims=mla_dims,
        cache=cache,
        softmax_scale=softmax_scale,
    )


def mla_attention_with_backend(
    hidden_states: torch.Tensor,
    projection_weights: DeepseekV2MLAProjectionWeights,
    mla_dims: DeepseekV2MLADims,
    backend: DeepseekV2AttentionBackend,
    cache: Optional[DeepseekV2MLAResidentCache] = None,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
    query_states, new_cache = project_mla_from_hidden_states(
        hidden_states,
        projection_weights,
        mla_dims,
    )
    q_nope, q_rope = split_query_projection(query_states, mla_dims)
    full_cache = append_resident_cache(cache, new_cache)
    key, value = reconstruct_dense_kv(full_cache, projection_weights.kv_up_proj, mla_dims)
    query = torch.cat([q_nope, q_rope], dim=-1)

    if softmax_scale is None:
        softmax_scale = mla_dims.q_head_dim ** -0.5

    output = backend(query, key, value, cache, softmax_scale)
    if (
        output.ndim != 2
        or output.shape[0] != hidden_states.shape[0]
        or output.shape[1] != mla_dims.o_proj_input_dim_local
    ):
        raise ValueError("attention backend must return [tokens, o_proj_input_dim_local]")
    return output @ projection_weights.o_proj, full_cache


def _prepare_mla_attention_tensors(
    hidden_states: torch.Tensor,
    projection_weights: DeepseekV2MLAProjectionWeights,
    mla_dims: DeepseekV2MLADims,
    cache: Optional[DeepseekV2MLAResidentCache] = None,
    softmax_scale: Optional[float] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    DeepseekV2MLAResidentCache,
    Optional[DeepseekV2MLAResidentCache],
    float,
]:
    query_states, new_cache = project_mla_from_hidden_states(
        hidden_states,
        projection_weights,
        mla_dims,
    )
    q_nope, q_rope = split_query_projection(query_states, mla_dims)
    full_cache = append_resident_cache(cache, new_cache)
    key, value = reconstruct_dense_kv(full_cache, projection_weights.kv_up_proj, mla_dims)
    query = torch.cat([q_nope, q_rope], dim=-1)

    if softmax_scale is None:
        softmax_scale = mla_dims.q_head_dim ** -0.5

    return query, key, value, full_cache, cache, softmax_scale


def prepare_mla_wrapper_inputs(
    hidden_states: torch.Tensor,
    projection_weights: DeepseekV2MLAProjectionWeights,
    mla_dims: DeepseekV2MLADims,
    kv_cache: object,
    layer_id: Optional[int] = None,
    cache: Optional[DeepseekV2MLAResidentCache] = None,
    softmax_scale: Optional[float] = None,
) -> Tuple[DeepseekV2MLAWrapperInputs, DeepseekV2MLAResidentCache]:
    kv_cache_carries_resident_state = (
        isinstance(kv_cache, DeepseekV2LayerCache) and cache is None
    )
    _, resolved_cache = resolve_layer_cache(kv_cache, cache)
    query_states, new_cache = project_mla_from_hidden_states(
        hidden_states,
        projection_weights,
        mla_dims,
    )
    q_nope, q_rope = split_query_projection(query_states, mla_dims)
    query = torch.cat([q_nope, q_rope], dim=-1)

    if softmax_scale is None:
        softmax_scale = mla_dims.q_head_dim ** -0.5

    return (
        DeepseekV2MLAWrapperInputs(
            query=query,
            kv_cache=kv_cache,
            kv_up_proj_weight=projection_weights.kv_up_proj,
            past_resident_cache=(
                None if kv_cache_carries_resident_state else resolved_cache
            ),
            new_resident_cache=new_cache,
            softmax_scale=softmax_scale,
            layer_id=layer_id,
            mla_dims=mla_dims,
        ),
        append_resident_cache(resolved_cache, new_cache),
    )


def mla_attention_with_wrapper(
    hidden_states: torch.Tensor,
    projection_weights: DeepseekV2MLAProjectionWeights,
    mla_dims: DeepseekV2MLADims,
    kv_cache,
    layer_id: Optional[int] = None,
    attention_wrapper=None,
    cache: Optional[DeepseekV2MLAResidentCache] = None,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
    if attention_wrapper is None:
        from sarathi.model_executor.attention import get_attention_wrapper

        attention_wrapper = get_attention_wrapper()

    wrapper_inputs, full_cache = prepare_mla_wrapper_inputs(
        hidden_states=hidden_states,
        projection_weights=projection_weights,
        mla_dims=mla_dims,
        kv_cache=kv_cache,
        layer_id=layer_id,
        cache=cache,
        softmax_scale=softmax_scale,
    )

    if hasattr(attention_wrapper, "forward_mla"):
        output = attention_wrapper.forward_mla(wrapper_inputs)
    else:
        runtime_kv_cache = get_layer_cache_kv_handle(wrapper_inputs.kv_cache)
        key, value = reconstruct_dense_kv(
            full_cache,
            projection_weights.kv_up_proj,
            mla_dims,
        )
        output = attention_wrapper.forward(
            wrapper_inputs.query.reshape(wrapper_inputs.query.shape[0], -1),
            key.reshape(key.shape[0], -1),
            value.reshape(value.shape[0], -1),
            runtime_kv_cache,
            wrapper_inputs.softmax_scale,
            wrapper_inputs.layer_id,
        )
    if (
        output.ndim != 2
        or output.shape[0] != hidden_states.shape[0]
        or output.shape[1] != mla_dims.o_proj_input_dim_local
    ):
        raise ValueError(
            "attention wrapper must return [tokens, o_proj_input_dim_local]"
        )
    return output @ projection_weights.o_proj, full_cache


def batched_contiguous_mla_attention_from_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    projection_weights: DeepseekV2MLAProjectionWeights,
    mla_dims: DeepseekV2MLADims,
    caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
    softmax_scale: Optional[float] = None,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[DeepseekV2MLAResidentCache, ...]]:
    if caches is None:
        caches = tuple(None for _ in hidden_states)
    if len(hidden_states) != len(caches):
        raise ValueError("hidden_states and caches must have the same batch length")

    outputs = []
    next_caches = []
    for seq_hidden_states, seq_cache in zip(hidden_states, caches):
        seq_output, seq_next_cache = contiguous_mla_attention_from_hidden_states(
            hidden_states=seq_hidden_states,
            projection_weights=projection_weights,
            mla_dims=mla_dims,
            cache=seq_cache,
            softmax_scale=softmax_scale,
        )
        seq_output = seq_output @ projection_weights.o_proj
        outputs.append(seq_output)
        next_caches.append(seq_next_cache)
    return tuple(outputs), tuple(next_caches)


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

    def split_query_projection(
        self,
        query_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return split_query_projection(query_states, self.mla_dims)

    def make_resident_cache(
        self,
        kv_latent: torch.Tensor,
        k_rope: torch.Tensor,
    ) -> DeepseekV2MLAResidentCache:
        return make_resident_cache(kv_latent, k_rope, self.mla_dims)

    def reconstruct_dense_kv(
        self,
        cache: DeepseekV2MLAResidentCache,
        kv_up_proj_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return reconstruct_dense_kv(cache, kv_up_proj_weight, self.mla_dims)

    def make_projection_weights(
        self,
        *,
        q_proj: Optional[torch.Tensor] = None,
        q_a_proj: Optional[torch.Tensor] = None,
        q_a_layernorm_weight: Optional[torch.Tensor] = None,
        q_b_proj: Optional[torch.Tensor] = None,
        kv_latent_proj: torch.Tensor,
        kv_a_layernorm_weight: Optional[torch.Tensor] = None,
        k_rope_proj: torch.Tensor,
        kv_up_proj: torch.Tensor,
        o_proj: torch.Tensor,
    ) -> DeepseekV2MLAProjectionWeights:
        return make_projection_weights(
            q_proj=q_proj,
            q_a_proj=q_a_proj,
            q_a_layernorm_weight=q_a_layernorm_weight,
            q_b_proj=q_b_proj,
            kv_latent_proj=kv_latent_proj,
            kv_a_layernorm_weight=kv_a_layernorm_weight,
            k_rope_proj=k_rope_proj,
            kv_up_proj=kv_up_proj,
            o_proj=o_proj,
            mla_dims=self.mla_dims,
        )

    def project_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
    ) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
        return project_mla_from_hidden_states(
            hidden_states,
            projection_weights,
            self.mla_dims,
        )

    def forward_contiguous(
        self,
        query_states: torch.Tensor,
        new_kv_latent: torch.Tensor,
        new_k_rope: torch.Tensor,
        kv_up_proj_weight: torch.Tensor,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
        return contiguous_mla_attention_forward(
            query_states=query_states,
            new_kv_latent=new_kv_latent,
            new_k_rope=new_k_rope,
            kv_up_proj_weight=kv_up_proj_weight,
            mla_dims=self.mla_dims,
            cache=cache,
            softmax_scale=softmax_scale,
        )

    def forward_hidden_states_contiguous(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
        output, cache = contiguous_mla_attention_from_hidden_states(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mla_dims=self.mla_dims,
            cache=cache,
            softmax_scale=softmax_scale,
        )
        return output @ projection_weights.o_proj, cache

    def forward_hidden_states_with_backend(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
        backend: DeepseekV2AttentionBackend,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
        return mla_attention_with_backend(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mla_dims=self.mla_dims,
            backend=backend,
            cache=cache,
            softmax_scale=softmax_scale,
        )

    def forward_hidden_states_with_attention_wrapper(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
        kv_cache,
        layer_id: Optional[int] = None,
        attention_wrapper=None,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2LayerCache]:
        output, next_cache = mla_attention_with_wrapper(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mla_dims=self.mla_dims,
            kv_cache=kv_cache,
            layer_id=layer_id,
            attention_wrapper=attention_wrapper,
            cache=cache,
            softmax_scale=softmax_scale,
        )
        return output, make_layer_cache(get_layer_cache_kv_handle(kv_cache), next_cache)

    def forward_hidden_states_contiguous_batched(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        projection_weights: DeepseekV2MLAProjectionWeights,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[DeepseekV2MLAResidentCache, ...]]:
        return batched_contiguous_mla_attention_from_hidden_states(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mla_dims=self.mla_dims,
            caches=caches,
            softmax_scale=softmax_scale,
        )


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_id: Optional[int] = None,
        tensor_parallel_world_size: Optional[int] = None,
    ):
        super().__init__()
        self.self_attn = DeepseekV2MLAAttention(
            config,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
        self.layer_id = layer_id
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size,
            eps=rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
        mlp_weights: Optional[DeepseekV2MLPWeights] = None,
        moe_weights: Optional[DeepseekV2MoEWeights] = None,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
        if mlp_weights is not None and moe_weights is not None:
            raise ValueError("mlp_weights and moe_weights are mutually exclusive")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, cache = self.self_attn.forward_hidden_states_contiguous(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            cache=cache,
            softmax_scale=softmax_scale,
        )
        hidden_states = residual + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)
        if mlp_weights is not None:
            hidden_states = hidden_states + apply_mlp(hidden_states, mlp_weights)
        elif moe_weights is not None:
            hidden_states = hidden_states + apply_moe(hidden_states, moe_weights)
        return hidden_states, cache

    def forward_with_attention_wrapper(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
        kv_cache,
        mlp_weights: Optional[DeepseekV2MLPWeights] = None,
        moe_weights: Optional[DeepseekV2MoEWeights] = None,
        attention_wrapper=None,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2LayerCache]:
        if mlp_weights is not None and moe_weights is not None:
            raise ValueError("mlp_weights and moe_weights are mutually exclusive")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, layer_cache = self.self_attn.forward_hidden_states_with_attention_wrapper(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            kv_cache=kv_cache,
            layer_id=self.layer_id,
            attention_wrapper=attention_wrapper,
            cache=cache,
            softmax_scale=softmax_scale,
        )
        hidden_states = residual + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)
        if mlp_weights is not None:
            hidden_states = hidden_states + apply_mlp(hidden_states, mlp_weights)
        elif moe_weights is not None:
            hidden_states = hidden_states + apply_moe(hidden_states, moe_weights)
        return hidden_states, layer_cache


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
        self.is_pipeline_first_stage = self.pipeline_parallel_rank == 0
        self.is_pipeline_last_stage = (
            self.pipeline_parallel_rank == self.pipeline_parallel_world_size - 1
        )
        self.vocab_size = getattr(config, "vocab_size", None)
        self.embed_tokens = None
        if self.is_pipeline_first_stage and self.vocab_size is not None:
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.num_layers = config.num_hidden_layers // self.pipeline_parallel_world_size
        self.layer_offset = self.pipeline_parallel_rank * self.num_layers
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    layer_id=self.layer_offset + layer_index,
                    tensor_parallel_world_size=self.tensor_parallel_world_size,
                )
                for layer_index in range(self.num_layers)
            ]
        )
        self.norm = (
            DeepseekV2RMSNorm(
                config.hidden_size,
                eps=getattr(config, "rms_norm_eps", 1e-6),
            )
            if self.is_pipeline_last_stage
            else None
        )
        self.layer_projection_weights: Optional[
            Tuple[DeepseekV2MLAProjectionWeights, ...]
        ] = None
        self.layer_mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None
        self.layer_moe_weights: Optional[Tuple[Optional[DeepseekV2MoEWeights], ...]] = None

    def _prepare_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dtype in (torch.int32, torch.int64, torch.long):
            if hidden_states.ndim != 1:
                raise ValueError("token input must have shape [tokens]")
            if self.embed_tokens is None:
                raise ValueError(
                    "token input is only supported on the first pipeline stage with embeddings"
                )
            return self.embed_tokens(hidden_states)
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.config.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        return hidden_states

    def set_scaffold_weights(
        self,
        projection_weights: Tuple[DeepseekV2MLAProjectionWeights, ...],
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        moe_weights: Optional[Tuple[Optional[DeepseekV2MoEWeights], ...]] = None,
    ) -> None:
        if len(projection_weights) != self.num_layers:
            raise ValueError("projection_weights must provide one entry per local layer")
        if mlp_weights is None:
            mlp_weights = tuple(None for _ in range(self.num_layers))
        if len(mlp_weights) != self.num_layers:
            raise ValueError("mlp_weights must provide one entry per local layer")
        if moe_weights is None:
            moe_weights = tuple(None for _ in range(self.num_layers))
        if len(moe_weights) != self.num_layers:
            raise ValueError("moe_weights must provide one entry per local layer")
        for layer_idx, (layer_mlp_weights, layer_moe_weights) in enumerate(
            zip(mlp_weights, moe_weights)
        ):
            if layer_mlp_weights is not None and layer_moe_weights is not None:
                raise ValueError(
                    f"layer {layer_idx} cannot install both dense MLP and MoE weights"
                )
        self.layer_projection_weights = tuple(projection_weights)
        self.layer_mlp_weights = tuple(mlp_weights)
        self.layer_moe_weights = tuple(moe_weights)

    def _resolve_scaffold_weights(
        self,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]],
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]],
        moe_weights: Optional[Tuple[Optional[DeepseekV2MoEWeights], ...]],
    ) -> Tuple[
        Tuple[DeepseekV2MLAProjectionWeights, ...],
        Tuple[Optional[DeepseekV2MLPWeights], ...],
        Tuple[Optional[DeepseekV2MoEWeights], ...],
    ]:
        if projection_weights is None:
            projection_weights = self.layer_projection_weights
        if projection_weights is None:
            raise ValueError(
                "projection_weights must be provided unless scaffold weights are installed"
            )
        if len(projection_weights) != self.num_layers:
            raise ValueError("projection_weights must provide one entry per local layer")

        if mlp_weights is None:
            mlp_weights = self.layer_mlp_weights
        if mlp_weights is None:
            mlp_weights = tuple(None for _ in range(self.num_layers))
        if len(mlp_weights) != self.num_layers:
            raise ValueError("mlp_weights must provide one entry per local layer")
        if moe_weights is None:
            moe_weights = self.layer_moe_weights
        if moe_weights is None:
            moe_weights = tuple(None for _ in range(self.num_layers))
        if len(moe_weights) != self.num_layers:
            raise ValueError("moe_weights must provide one entry per local layer")
        for layer_idx, (layer_mlp_weights, layer_moe_weights) in enumerate(
            zip(mlp_weights, moe_weights)
        ):
            if layer_mlp_weights is not None and layer_moe_weights is not None:
                raise ValueError(
                    f"layer {layer_idx} cannot use both dense MLP and MoE weights"
                )
        return tuple(projection_weights), tuple(mlp_weights), tuple(moe_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        moe_weights: Optional[Tuple[Optional[DeepseekV2MoEWeights], ...]] = None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2MLAResidentCache, ...]]:
        projection_weights, mlp_weights, moe_weights = self._resolve_scaffold_weights(
            projection_weights,
            mlp_weights,
            moe_weights,
        )
        if caches is None:
            caches = tuple(None for _ in range(self.num_layers))
        if len(caches) != self.num_layers:
            raise ValueError("caches must provide one entry per local layer")

        hidden_states = self._prepare_hidden_states(hidden_states)
        next_caches = []
        for (
            layer,
            layer_projection_weights,
            layer_mlp_weights,
            layer_moe_weights,
            layer_cache,
        ) in zip(
            self.layers,
            projection_weights,
            mlp_weights,
            moe_weights,
            caches,
        ):
            hidden_states, next_cache = layer(
                hidden_states=hidden_states,
                projection_weights=layer_projection_weights,
                mlp_weights=layer_mlp_weights,
                moe_weights=layer_moe_weights,
                cache=layer_cache,
                softmax_scale=softmax_scale,
            )
            next_caches.append(next_cache)
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(next_caches)

    def forward_with_attention_wrapper(
        self,
        hidden_states: torch.Tensor,
        kv_caches: Tuple[object, ...],
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        moe_weights: Optional[Tuple[Optional[DeepseekV2MoEWeights], ...]] = None,
        attention_wrapper=None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2LayerCache, ...]]:
        projection_weights, mlp_weights, moe_weights = self._resolve_scaffold_weights(
            projection_weights,
            mlp_weights,
            moe_weights,
        )
        if len(kv_caches) != self.num_layers:
            raise ValueError("kv_caches must provide one entry per local layer")
        if caches is None:
            caches = tuple(None for _ in range(self.num_layers))
        if len(caches) != self.num_layers:
            raise ValueError("caches must provide one entry per local layer")

        hidden_states = self._prepare_hidden_states(hidden_states)
        next_caches = []
        for (
            layer,
            layer_projection_weights,
            layer_mlp_weights,
            layer_moe_weights,
            layer_kv_cache,
            layer_cache,
        ) in zip(
            self.layers,
            projection_weights,
            mlp_weights,
            moe_weights,
            kv_caches,
            caches,
        ):
            hidden_states, next_cache = layer.forward_with_attention_wrapper(
                hidden_states=hidden_states,
                projection_weights=layer_projection_weights,
                mlp_weights=layer_mlp_weights,
                moe_weights=layer_moe_weights,
                kv_cache=layer_kv_cache,
                attention_wrapper=attention_wrapper,
                cache=layer_cache,
                softmax_scale=softmax_scale,
            )
            next_caches.append(next_cache)
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(next_caches)

    def make_runtime_mla_kv_caches(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[DeepseekV2ComponentMLAKVCache, ...]:
        mla_dims = self.layers[0].self_attn.mla_dims
        return make_runtime_mla_kv_caches(
            num_layers=self.num_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            mla_dims=mla_dims,
            device=device,
            dtype=dtype,
        )


class DeepseekV2ForCausalLM(nn.Module):

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
        self.model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=tensor_parallel_world_size,
            pipeline_parallel_world_size=pipeline_parallel_world_size,
            pipeline_parallel_rank=pipeline_parallel_rank,
        )
        self.mla_dims = DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=self.model.tensor_parallel_world_size,
        )
        self.lm_head = None
        if self.model.is_pipeline_last_stage and getattr(config, "vocab_size", None) is not None:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def set_scaffold_weights(
        self,
        projection_weights: Tuple[DeepseekV2MLAProjectionWeights, ...],
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        moe_weights: Optional[Tuple[Optional[DeepseekV2MoEWeights], ...]] = None,
    ) -> None:
        self.model.set_scaffold_weights(
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            moe_weights=moe_weights,
        )

    @staticmethod
    def _get_scaffold_tensor(
        state_dict: Mapping[str, torch.Tensor],
        *keys: str,
    ) -> Optional[torch.Tensor]:
        for key in keys:
            tensor = state_dict.get(key)
            if tensor is not None:
                return tensor
        return None

    @staticmethod
    def _candidate_layer_prefixes(
        layer_idx: int,
        layer_offset: int,
        suffix: str,
    ) -> Tuple[str, ...]:
        local_idx = layer_idx
        global_idx = layer_offset + layer_idx
        return (
            f"model.layers.{local_idx}.{suffix}",
            f"model.layers.{global_idx}.{suffix}",
            f"layers.{local_idx}.{suffix}",
            f"layers.{global_idx}.{suffix}",
        )

    @staticmethod
    def _make_optional_mlp_weights(
        tensors: Mapping[str, Optional[torch.Tensor]],
        *,
        hidden_size: int,
    ) -> Optional[DeepseekV2MLPWeights]:
        present_keys = [name for name, tensor in tensors.items() if tensor is not None]
        if not present_keys:
            return None
        if len(present_keys) != len(tensors):
            missing = sorted(set(tensors) - set(present_keys))
            raise KeyError("Missing scaffold MLP weights: " + ", ".join(missing))
        return make_mlp_weights(
            gate_proj=tensors["gate_proj"],
            up_proj=tensors["up_proj"],
            down_proj=tensors["down_proj"],
            hidden_size=hidden_size,
        )

    @staticmethod
    def _has_moe_layer_weights(
        state_dict: Mapping[str, torch.Tensor],
        layer_prefixes: Tuple[str, ...],
    ) -> bool:
        moe_prefixes = []
        for prefix in layer_prefixes:
            moe_prefixes.extend(
                (
                    f"{prefix}.gate.weight",
                    f"{prefix}.shared_experts.gate_proj.weight",
                    f"{prefix}.shared_experts.up_proj.weight",
                    f"{prefix}.shared_experts.down_proj.weight",
                    f"{prefix}.experts.",
                )
            )
        return any(
            any(name == candidate or name.startswith(candidate) for candidate in moe_prefixes)
            for name in state_dict
        )

    @staticmethod
    def _coerce_scaffold_tensor(
        tensor: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.to(device=reference.device, dtype=reference.dtype)

    def load_scaffold_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        local_projection_weights = []
        local_mlp_weights = []
        local_moe_weights = []
        hidden_size = self.config.hidden_size

        if self.model.embed_tokens is not None:
            embed_key = "model.embed_tokens.weight"
            embed_weight = self._get_scaffold_tensor(
                state_dict,
                embed_key,
                "embed_tokens.weight",
            )
            if embed_weight is None:
                if strict:
                    raise KeyError(f"Missing scaffold weight: {embed_key}")
            else:
                expected_shape = (
                    self.config.vocab_size,
                    hidden_size,
                )
                if tuple(embed_weight.shape) != expected_shape:
                    raise ValueError(
                        "model.embed_tokens.weight shape does not match scaffold embedding size"
                )
                self.model.embed_tokens.weight.data.copy_(embed_weight)

        if self.model.norm is not None:
            norm_weight = self._get_scaffold_tensor(
                state_dict,
                "model.norm.weight",
                "norm.weight",
            )
            if norm_weight is not None:
                expected_shape = (hidden_size,)
                if tuple(norm_weight.shape) != expected_shape:
                    raise ValueError(
                        "model.norm.weight shape does not match scaffold final norm size"
                    )
                self.model.norm.weight.data.copy_(norm_weight)

        if self.lm_head is not None:
            lm_head_key = "lm_head.weight"
            lm_head_weight = self._get_scaffold_tensor(
                state_dict,
                lm_head_key,
                "model.lm_head.weight",
            )
            if lm_head_weight is None:
                if strict:
                    raise KeyError(f"Missing scaffold weight: {lm_head_key}")
            else:
                expected_shape = (
                    self.config.vocab_size,
                    hidden_size,
                )
                if tuple(lm_head_weight.shape) != expected_shape:
                    raise ValueError(
                        "lm_head.weight shape does not match scaffold vocabulary projection size"
                    )
                self.lm_head.weight.data.copy_(lm_head_weight)

        for layer_idx, layer in enumerate(self.model.layers):
            input_norm_weight = self._get_scaffold_tensor(
                state_dict,
                *(
                    f"{prefix}.weight"
                    for prefix in self._candidate_layer_prefixes(
                        layer_idx,
                        self.model.layer_offset,
                        "input_layernorm",
                    )
                ),
            )
            if input_norm_weight is not None:
                if tuple(input_norm_weight.shape) != (hidden_size,):
                    raise ValueError(
                        "input_layernorm.weight shape does not match scaffold hidden size"
                    )
                input_norm_weight = self._coerce_scaffold_tensor(
                    input_norm_weight,
                    reference=layer.input_layernorm.weight,
                )
                layer.input_layernorm.weight.data.copy_(input_norm_weight)

            post_attn_norm_weight = self._get_scaffold_tensor(
                state_dict,
                *(
                    f"{prefix}.weight"
                    for prefix in self._candidate_layer_prefixes(
                        layer_idx,
                        self.model.layer_offset,
                        "post_attention_layernorm",
                    )
                ),
            )
            if post_attn_norm_weight is not None:
                if tuple(post_attn_norm_weight.shape) != (hidden_size,):
                    raise ValueError(
                        "post_attention_layernorm.weight shape does not match scaffold hidden size"
                    )
                post_attn_norm_weight = self._coerce_scaffold_tensor(
                    post_attn_norm_weight,
                    reference=layer.post_attention_layernorm.weight,
                )
                layer.post_attention_layernorm.weight.data.copy_(post_attn_norm_weight)

            projection_prefixes = self._candidate_layer_prefixes(
                layer_idx,
                self.model.layer_offset,
                "self_attn",
            )
            projection_tensors = {
                "q_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.q_proj.weight" for prefix in projection_prefixes),
                ),
                "q_a_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.q_a_proj.weight" for prefix in projection_prefixes),
                ),
                "q_a_layernorm_weight": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.q_a_layernorm.weight" for prefix in projection_prefixes),
                ),
                "q_b_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.q_b_proj.weight" for prefix in projection_prefixes),
                ),
                "kv_latent_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.kv_latent_proj.weight" for prefix in projection_prefixes),
                ),
                "kv_a_layernorm_weight": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.kv_a_layernorm.weight" for prefix in projection_prefixes),
                ),
                "k_rope_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.k_rope_proj.weight" for prefix in projection_prefixes),
                ),
                "kv_up_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.kv_up_proj.weight" for prefix in projection_prefixes),
                ),
                "o_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.o_proj.weight" for prefix in projection_prefixes),
                ),
            }
            kv_a_proj_with_mqa = self._get_scaffold_tensor(
                state_dict,
                *(f"{prefix}.kv_a_proj_with_mqa.weight" for prefix in projection_prefixes),
            )
            kv_a_proj_with_mqa_kv_latent_proj, kv_a_proj_with_mqa_k_rope_proj = (
                self._split_combined_kv_a_proj_with_mqa_weight(
                    kv_a_proj_with_mqa,
                    hidden_size=hidden_size,
                    mla_dims=layer.self_attn.mla_dims,
                )
            )
            if (
                projection_tensors["kv_latent_proj"] is None
                and projection_tensors["k_rope_proj"] is None
                and kv_a_proj_with_mqa_kv_latent_proj is not None
                and kv_a_proj_with_mqa_k_rope_proj is not None
            ):
                projection_tensors["kv_latent_proj"] = kv_a_proj_with_mqa_kv_latent_proj
                projection_tensors["k_rope_proj"] = kv_a_proj_with_mqa_k_rope_proj
            kv_b_proj = self._get_scaffold_tensor(
                state_dict,
                *(f"{prefix}.kv_b_proj.weight" for prefix in projection_prefixes),
            )
            if projection_tensors["kv_up_proj"] is None and kv_b_proj is not None:
                projection_tensors["kv_up_proj"] = kv_b_proj
            projection_tensors["q_proj"] = self._coerce_and_slice_tensor_parallel_linear_weight(
                projection_tensors["q_proj"],
                expected_local_shape=(
                    hidden_size,
                    layer.self_attn.mla_dims.q_proj_output_dim_local,
                ),
                shard_dim=1,
            )
            if layer.self_attn.mla_dims.q_lora_rank is not None:
                projection_tensors["q_a_proj"] = self._coerce_linear_weight_layout(
                    projection_tensors["q_a_proj"],
                    expected_shape=(hidden_size, layer.self_attn.mla_dims.q_lora_rank),
                )
                projection_tensors["q_b_proj"] = self._coerce_and_slice_tensor_parallel_linear_weight(
                    projection_tensors["q_b_proj"],
                    expected_local_shape=(
                        layer.self_attn.mla_dims.q_lora_rank,
                        layer.self_attn.mla_dims.q_proj_output_dim_local,
                    ),
                    shard_dim=1,
                )
            projection_tensors["kv_latent_proj"] = self._coerce_linear_weight_layout(
                projection_tensors["kv_latent_proj"],
                expected_shape=(hidden_size, layer.self_attn.mla_dims.kv_lora_rank),
            )
            projection_tensors["k_rope_proj"] = self._coerce_and_slice_tensor_parallel_linear_weight(
                projection_tensors["k_rope_proj"],
                expected_local_shape=(
                    hidden_size,
                    layer.self_attn.mla_dims.num_heads
                    * layer.self_attn.mla_dims.qk_rope_head_dim,
                ),
                shard_dim=1,
            )
            projection_tensors["kv_up_proj"] = self._coerce_and_slice_tensor_parallel_linear_weight(
                projection_tensors["kv_up_proj"],
                expected_local_shape=(
                    layer.self_attn.mla_dims.kv_lora_rank,
                    layer.self_attn.mla_dims.kv_up_proj_output_dim_local,
                ),
                shard_dim=1,
            )
            projection_tensors["o_proj"] = self._coerce_and_slice_tensor_parallel_linear_weight(
                projection_tensors["o_proj"],
                expected_local_shape=(
                    layer.self_attn.mla_dims.o_proj_input_dim_local,
                    hidden_size,
                ),
                shard_dim=0,
            )
            query_uses_q_lora = projection_tensors["q_proj"] is None and any(
                projection_tensors[name] is not None
                for name in ("q_a_proj", "q_a_layernorm_weight", "q_b_proj")
            )
            required_query_keys = (
                ("q_a_proj", "q_a_layernorm_weight", "q_b_proj")
                if query_uses_q_lora
                else ("q_proj",)
            )
            required_projection_keys = required_query_keys + (
                "kv_latent_proj",
                "k_rope_proj",
                "kv_up_proj",
                "o_proj",
            )
            missing_projection_keys = [
                name for name in required_projection_keys if projection_tensors[name] is None
            ]
            if missing_projection_keys:
                raise KeyError(
                    "Missing scaffold weights for layer "
                    f"{layer_idx}: {', '.join(missing_projection_keys)}"
                )
            projection_reference = layer.input_layernorm.weight
            projection_tensors = {
                name: self._coerce_scaffold_tensor(
                    tensor,
                    reference=projection_reference,
                )
                for name, tensor in projection_tensors.items()
            }
            local_projection_weights.append(
                make_projection_weights(
                    q_proj=projection_tensors["q_proj"],
                    q_a_proj=projection_tensors["q_a_proj"],
                    q_a_layernorm_weight=projection_tensors["q_a_layernorm_weight"],
                    q_b_proj=projection_tensors["q_b_proj"],
                    kv_latent_proj=projection_tensors["kv_latent_proj"],
                    kv_a_layernorm_weight=projection_tensors["kv_a_layernorm_weight"],
                    k_rope_proj=projection_tensors["k_rope_proj"],
                    kv_up_proj=projection_tensors["kv_up_proj"],
                    o_proj=projection_tensors["o_proj"],
                    mla_dims=layer.self_attn.mla_dims,
                )
            )

            mlp_prefixes = self._candidate_layer_prefixes(
                layer_idx,
                self.model.layer_offset,
                "mlp",
            )
            global_layer_idx = self.model.layer_offset + layer_idx
            first_k_dense_replace = getattr(self.config, "first_k_dense_replace", None)
            n_routed_experts = getattr(self.config, "n_routed_experts", 0)
            mlp_tensors = {
                "gate_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.gate_proj.weight" for prefix in mlp_prefixes),
                ),
                "up_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.up_proj.weight" for prefix in mlp_prefixes),
                ),
                "down_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.down_proj.weight" for prefix in mlp_prefixes),
                ),
            }
            mlp_tensors = self._normalize_mlp_tensor_layouts(
                mlp_tensors,
                hidden_size=hidden_size,
                intermediate_size=getattr(self.config, "intermediate_size", None),
            )
            present_mlp_keys = [
                name for name, tensor in mlp_tensors.items() if tensor is not None
            ]
            mlp_reference = layer.post_attention_layernorm.weight
            mlp_tensors = {
                name: self._coerce_scaffold_tensor(
                    tensor,
                    reference=mlp_reference,
                )
                for name, tensor in mlp_tensors.items()
            }
            routed_gate = self._get_scaffold_tensor(
                state_dict,
                *(f"{prefix}.gate.weight" for prefix in mlp_prefixes),
            )
            routed_gate = self._coerce_scaffold_tensor(
                routed_gate,
                reference=mlp_reference,
            )
            shared_expert_tensors = {
                "gate_proj": self._coerce_scaffold_tensor(
                    self._get_scaffold_tensor(
                        state_dict,
                        *(f"{prefix}.shared_experts.gate_proj.weight" for prefix in mlp_prefixes),
                    ),
                    reference=mlp_reference,
                ),
                "up_proj": self._coerce_scaffold_tensor(
                    self._get_scaffold_tensor(
                        state_dict,
                        *(f"{prefix}.shared_experts.up_proj.weight" for prefix in mlp_prefixes),
                    ),
                    reference=mlp_reference,
                ),
                "down_proj": self._coerce_scaffold_tensor(
                    self._get_scaffold_tensor(
                        state_dict,
                        *(f"{prefix}.shared_experts.down_proj.weight" for prefix in mlp_prefixes),
                    ),
                    reference=mlp_reference,
                ),
            }
            shared_expert_tensors = self._normalize_mlp_tensor_layouts(
                shared_expert_tensors,
                hidden_size=hidden_size,
                intermediate_size=getattr(self.config, "moe_intermediate_size", None),
            )
            routed_experts = []
            for expert_idx in range(n_routed_experts):
                expert_tensors = {
                    "gate_proj": self._coerce_scaffold_tensor(
                        self._get_scaffold_tensor(
                            state_dict,
                            *(
                                f"{prefix}.experts.{expert_idx}.gate_proj.weight"
                                for prefix in mlp_prefixes
                            ),
                        ),
                        reference=mlp_reference,
                    ),
                    "up_proj": self._coerce_scaffold_tensor(
                        self._get_scaffold_tensor(
                            state_dict,
                            *(
                                f"{prefix}.experts.{expert_idx}.up_proj.weight"
                                for prefix in mlp_prefixes
                            ),
                        ),
                        reference=mlp_reference,
                    ),
                    "down_proj": self._coerce_scaffold_tensor(
                        self._get_scaffold_tensor(
                            state_dict,
                            *(
                                f"{prefix}.experts.{expert_idx}.down_proj.weight"
                                for prefix in mlp_prefixes
                            ),
                        ),
                        reference=mlp_reference,
                    ),
                }
                expert_tensors = self._normalize_mlp_tensor_layouts(
                    expert_tensors,
                    hidden_size=hidden_size,
                    intermediate_size=getattr(self.config, "moe_intermediate_size", None),
                )
                try:
                    expert_weights = self._make_optional_mlp_weights(
                        expert_tensors,
                        hidden_size=hidden_size,
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Incomplete routed expert weights for layer {global_layer_idx} "
                        f"expert {expert_idx}: {exc}"
                    ) from exc
                if expert_weights is not None:
                    routed_experts.append(expert_weights)
            has_moe_weights = (
                routed_gate is not None
                or self._has_moe_layer_weights(state_dict, mlp_prefixes)
                or any(tensor is not None for tensor in shared_expert_tensors.values())
                or bool(routed_experts)
            )
            if present_mlp_keys:
                if has_moe_weights:
                    raise ValueError(
                        f"Layer {global_layer_idx} provides both dense MLP and MoE weights"
                    )
                if len(present_mlp_keys) != len(mlp_tensors):
                    if strict:
                        missing = sorted(set(mlp_tensors) - set(present_mlp_keys))
                        raise KeyError(
                            "Missing scaffold MLP weights for layer "
                            f"{layer_idx}: {', '.join(missing)}"
                        )
                    local_mlp_weights.append(None)
                    local_moe_weights.append(None)
                else:
                    local_mlp_weights.append(
                        make_mlp_weights(
                            gate_proj=mlp_tensors["gate_proj"],
                            up_proj=mlp_tensors["up_proj"],
                            down_proj=mlp_tensors["down_proj"],
                            hidden_size=hidden_size,
                        )
                    )
                    local_moe_weights.append(None)
            elif has_moe_weights:
                if first_k_dense_replace is None or not n_routed_experts:
                    raise NotImplementedError(
                        "DeepSeek-V2 MoE weights require MoE-aware config fields"
                    )
                if global_layer_idx < first_k_dense_replace:
                    raise ValueError(
                        f"Layer {global_layer_idx} carries MoE weights before first_k_dense_replace"
                    )
                if routed_gate is None:
                    raise KeyError(f"Missing scaffold MoE gate weights for layer {global_layer_idx}")
                if len(routed_experts) != n_routed_experts:
                    raise KeyError(
                        f"Missing routed expert weights for layer {global_layer_idx}: "
                        f"expected {n_routed_experts}, found {len(routed_experts)}"
                    )
                try:
                    shared_experts = self._make_optional_mlp_weights(
                        shared_expert_tensors,
                        hidden_size=hidden_size,
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Incomplete shared expert weights for layer {global_layer_idx}: {exc}"
                    ) from exc
                local_mlp_weights.append(None)
                local_moe_weights.append(
                    make_moe_weights(
                        gate=routed_gate,
                        experts=tuple(routed_experts),
                        shared_experts=shared_experts,
                        top_k=getattr(self.config, "num_experts_per_tok", 1),
                        norm_topk_prob=getattr(self.config, "norm_topk_prob", True),
                        hidden_size=hidden_size,
                    )
                )
            else:
                local_mlp_weights.append(None)
                local_moe_weights.append(None)

        self.set_scaffold_weights(
            projection_weights=tuple(local_projection_weights),
            mlp_weights=tuple(local_mlp_weights),
            moe_weights=tuple(local_moe_weights),
        )

    @staticmethod
    def _load_state_dict_file(path: str) -> Mapping[str, torch.Tensor]:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path)
        loaded = torch.load(path, map_location="cpu")
        if "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping):
            raise ValueError("checkpoint file must contain a state-dict mapping")
        return loaded

    @classmethod
    def _load_scaffold_state_dict_from_path(
        cls,
        model_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ) -> Mapping[str, torch.Tensor]:
        if os.path.isdir(model_path):
            state_dict = {}
            for name, tensor in hf_model_weights_iterator(
                model_path,
                cache_dir,
                load_format,
                revision,
            ):
                state_dict[name] = convert_pyslice_to_tensor(tensor)
            return state_dict
        if os.path.isfile(model_path):
            return cls._load_state_dict_file(model_path)
        state_dict = {}
        for name, tensor in hf_model_weights_iterator(
            model_path,
            cache_dir,
            load_format,
            revision,
        ):
            state_dict[name] = convert_pyslice_to_tensor(tensor)
        return state_dict

    @staticmethod
    def _coerce_linear_weight_layout(
        tensor: Optional[torch.Tensor],
        *,
        expected_shape: Tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        if tensor is None or tensor.ndim != 2:
            return tensor
        if tuple(tensor.shape) == expected_shape:
            return tensor
        if tuple(tensor.t().shape) == expected_shape:
            return tensor.t().contiguous()
        return tensor

    def _coerce_and_slice_tensor_parallel_linear_weight(
        self,
        tensor: Optional[torch.Tensor],
        *,
        expected_local_shape: Tuple[int, int],
        shard_dim: int,
    ) -> Optional[torch.Tensor]:
        if tensor is None or tensor.ndim != 2:
            return tensor
        tensor = self._coerce_linear_weight_layout(
            tensor,
            expected_shape=expected_local_shape,
        )
        if tuple(tensor.shape) == expected_local_shape:
            return tensor

        expected_global_shape = list(expected_local_shape)
        expected_global_shape[shard_dim] *= self.model.tensor_parallel_world_size
        expected_global_shape = tuple(expected_global_shape)
        tensor = self._coerce_linear_weight_layout(
            tensor,
            expected_shape=expected_global_shape,
        )
        if tuple(tensor.shape) != expected_global_shape:
            return tensor

        shard_size = expected_local_shape[shard_dim]
        tp_rank = self._get_tensor_model_parallel_rank()
        shard_start = tp_rank * shard_size
        shard_end = shard_start + shard_size
        if shard_dim == 0:
            return tensor[shard_start:shard_end, :].contiguous()
        return tensor[:, shard_start:shard_end].contiguous()

    def _split_combined_kv_a_proj_with_mqa_weight(
        self,
        tensor: Optional[torch.Tensor],
        *,
        hidden_size: int,
        mla_dims: DeepseekV2MLADims,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tensor is None:
            return None, None

        local_shape = (
            hidden_size,
            mla_dims.kv_lora_rank + mla_dims.num_heads * mla_dims.qk_rope_head_dim,
        )
        tensor = self._coerce_linear_weight_layout(
            tensor,
            expected_shape=local_shape,
        )
        if tuple(tensor.shape) == local_shape:
            kv_latent_width = mla_dims.kv_lora_rank
            return (
                tensor[:, :kv_latent_width].contiguous(),
                tensor[:, kv_latent_width:].contiguous(),
            )

        global_shape = (
            hidden_size,
            mla_dims.kv_lora_rank
            + mla_dims.total_num_heads * mla_dims.qk_rope_head_dim,
        )
        tensor = self._coerce_linear_weight_layout(
            tensor,
            expected_shape=global_shape,
        )
        if tuple(tensor.shape) != global_shape:
            return tensor, tensor

        kv_latent_proj = tensor[:, : mla_dims.kv_lora_rank].contiguous()
        rope_width_local = mla_dims.num_heads * mla_dims.qk_rope_head_dim
        rope_start = mla_dims.kv_lora_rank + self._get_tensor_model_parallel_rank() * rope_width_local
        rope_end = rope_start + rope_width_local
        k_rope_proj = tensor[:, rope_start:rope_end].contiguous()
        return kv_latent_proj, k_rope_proj

    @staticmethod
    def _get_tensor_model_parallel_rank() -> int:
        try:
            return get_tensor_model_parallel_rank()
        except (AssertionError, RuntimeError):
            return 0

    def _normalize_mlp_tensor_layouts(
        self,
        tensors: Mapping[str, Optional[torch.Tensor]],
        *,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
    ) -> Mapping[str, Optional[torch.Tensor]]:
        if intermediate_size is not None:
            if intermediate_size % self.model.tensor_parallel_world_size != 0:
                raise ValueError(
                    "DeepSeek-V2 intermediate size must divide evenly across tensor parallel ranks"
                )
            local_intermediate_size = (
                intermediate_size // self.model.tensor_parallel_world_size
            )
            return {
                "gate_proj": self._coerce_and_slice_tensor_parallel_linear_weight(
                    tensors.get("gate_proj"),
                    expected_local_shape=(hidden_size, local_intermediate_size),
                    shard_dim=1,
                ),
                "up_proj": self._coerce_and_slice_tensor_parallel_linear_weight(
                    tensors.get("up_proj"),
                    expected_local_shape=(hidden_size, local_intermediate_size),
                    shard_dim=1,
                ),
                "down_proj": self._coerce_and_slice_tensor_parallel_linear_weight(
                    tensors.get("down_proj"),
                    expected_local_shape=(local_intermediate_size, hidden_size),
                    shard_dim=0,
                ),
            }

        normalized = dict(tensors)
        for name in ("gate_proj", "up_proj"):
            tensor = normalized.get(name)
            if tensor is None or tensor.ndim != 2:
                continue
            intermediate_size = (
                tensor.shape[1] if tensor.shape[0] == hidden_size else tensor.shape[0]
            )
            normalized[name] = self._coerce_linear_weight_layout(
                tensor,
                expected_shape=(hidden_size, intermediate_size),
            )
        down_proj = normalized.get("down_proj")
        if down_proj is not None and down_proj.ndim == 2:
            intermediate_size = (
                down_proj.shape[0]
                if down_proj.shape[1] == hidden_size
                else down_proj.shape[1]
            )
            normalized["down_proj"] = self._coerce_linear_weight_layout(
                down_proj,
                expected_shape=(intermediate_size, hidden_size),
            )
        return normalized

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
        positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[Tuple[object, ...]] = None,
        attention_wrapper=None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2MLAResidentCache, ...]]:
        del positions
        if kv_caches is not None:
            return self.model.forward_with_attention_wrapper(
                hidden_states=hidden_states,
                projection_weights=projection_weights,
                mlp_weights=mlp_weights,
                kv_caches=kv_caches,
                attention_wrapper=attention_wrapper,
                caches=caches,
                softmax_scale=softmax_scale,
            )
        return self.model(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            caches=caches,
            softmax_scale=softmax_scale,
        )

    def load_weights(self, *args, **kwargs):
        if args and isinstance(args[0], Mapping):
            strict = kwargs.pop("strict", True)
            if kwargs:
                raise ValueError(
                    "Unsupported kwargs for scaffold state-dict loading: "
                    + ", ".join(sorted(kwargs.keys()))
                )
            self.load_scaffold_state_dict(args[0], strict=strict)
            return
        if args and isinstance(args[0], (str, os.PathLike)):
            strict = kwargs.pop("strict", True)
            cache_dir = args[1] if len(args) > 1 else None
            load_format = args[2] if len(args) > 2 else "auto"
            revision = args[3] if len(args) > 3 else None
            if kwargs:
                raise ValueError(
                    "Unsupported kwargs for scaffold checkpoint loading: "
                    + ", ".join(sorted(kwargs.keys()))
                )
            state_dict = self._load_scaffold_state_dict_from_path(
                os.fspath(args[0]),
                cache_dir=cache_dir,
                load_format=load_format,
                revision=revision,
            )
            self.load_scaffold_state_dict(state_dict, strict=strict)
            return
        raise NotImplementedError(
            "DeepSeek-V2 weight loading is not implemented yet. "
            "The MLA attention/model path still needs to be added."
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise ValueError("lm_head is only available on the last pipeline stage")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.config.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        return self.lm_head(hidden_states)

    @staticmethod
    def _validate_token_ids(token_ids: torch.Tensor) -> None:
        if token_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("token_ids must be an integer tensor")
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape [tokens]")
        if token_ids.numel() == 0:
            raise ValueError("token_ids must contain at least one token")

    @staticmethod
    def _caches_are_layer_caches(caches) -> bool:
        if not isinstance(caches, tuple):
            return False
        return all(isinstance(cache, DeepseekV2LayerCache) for cache in caches)

    @staticmethod
    def _set_single_batch_wrapper_metadata(
        attention_wrapper,
        *,
        prompt_len: int = 0,
        cache_len: Optional[int] = None,
    ) -> None:
        if attention_wrapper is None or not hasattr(attention_wrapper, "set_mla_runtime_metadata"):
            return
        if prompt_len > 0:
            attention_wrapper.set_mla_runtime_metadata(
                prefill_query_lens=[prompt_len],
                prefill_cache_lens=[0 if cache_len is None else cache_len],
                batch_index=[0],
                batch_index_gen=[],
            )
            return
        attention_wrapper.set_mla_runtime_metadata(
            prefill_query_lens=[],
            prefill_cache_lens=[],
            decode_cache_lens=[] if cache_len is None else [cache_len],
            batch_index=[],
            batch_index_gen=[] if cache_len is None else [0],
        )

    def forward_logits(
        self,
        hidden_states: torch.Tensor,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2MLAResidentCache, ...]]:
        hidden_states, caches = self.forward(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            caches=caches,
            softmax_scale=softmax_scale,
        )
        return self.compute_logits(hidden_states), caches

    def forward_with_attention_wrapper(
        self,
        hidden_states: torch.Tensor,
        kv_caches: Tuple[object, ...],
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        attention_wrapper=None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
        ) -> Tuple[torch.Tensor, Tuple[DeepseekV2LayerCache, ...]]:
        return self.model.forward_with_attention_wrapper(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            kv_caches=kv_caches,
            attention_wrapper=attention_wrapper,
            caches=caches,
            softmax_scale=softmax_scale,
        )

    def forward_logits_with_attention_wrapper(
        self,
        hidden_states: torch.Tensor,
        kv_caches: Tuple[object, ...],
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        attention_wrapper=None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2LayerCache, ...]]:
        hidden_states, caches = self.forward_with_attention_wrapper(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            kv_caches=kv_caches,
            attention_wrapper=attention_wrapper,
            caches=caches,
            softmax_scale=softmax_scale,
        )
        return self.compute_logits(hidden_states), caches

    def prefill_tokens(
        self,
        token_ids: torch.Tensor,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        softmax_scale: Optional[float] = None,
        kv_caches: Optional[Tuple[object, ...]] = None,
        attention_wrapper=None,
    ):
        self._validate_token_ids(token_ids)
        if kv_caches is not None:
            return self.forward_logits_with_attention_wrapper(
                hidden_states=token_ids,
                kv_caches=kv_caches,
                projection_weights=projection_weights,
                mlp_weights=mlp_weights,
                attention_wrapper=attention_wrapper,
                softmax_scale=softmax_scale,
            )
        return self.forward_logits(
            hidden_states=token_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            softmax_scale=softmax_scale,
        )

    def decode_tokens(
        self,
        token_ids: torch.Tensor,
        caches,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        softmax_scale: Optional[float] = None,
        kv_caches: Optional[Tuple[object, ...]] = None,
        attention_wrapper=None,
    ):
        self._validate_token_ids(token_ids)
        if caches is None:
            raise ValueError("caches must be provided for decode_tokens")
        if kv_caches is not None:
            resident_caches = caches
            runtime_kv_caches = kv_caches
            if self._caches_are_layer_caches(caches):
                runtime_kv_caches = caches
                resident_caches = None
            return self.forward_logits_with_attention_wrapper(
                hidden_states=token_ids,
                kv_caches=runtime_kv_caches,
                projection_weights=projection_weights,
                mlp_weights=mlp_weights,
                attention_wrapper=attention_wrapper,
                caches=resident_caches,
                softmax_scale=softmax_scale,
        )
        return self.forward_logits(
            hidden_states=token_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            caches=caches,
            softmax_scale=softmax_scale,
        )

    def generate_greedy(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        softmax_scale: Optional[float] = None,
        kv_caches: Optional[Tuple[object, ...]] = None,
        attention_wrapper=None,
    ):
        self._validate_token_ids(token_ids)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if kv_caches is not None:
            self._set_single_batch_wrapper_metadata(
                attention_wrapper,
                prompt_len=token_ids.numel(),
                cache_len=0,
            )

        logits, caches = self.prefill_tokens(
            token_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            softmax_scale=softmax_scale,
            kv_caches=kv_caches,
            attention_wrapper=attention_wrapper,
        )
        if max_new_tokens == 0:
            empty = token_ids.new_empty((0,))
            return empty, logits, caches

        generated_tokens = []
        next_token = torch.argmax(logits[-1], dim=-1).to(dtype=token_ids.dtype).view(1)
        generated_tokens.append(next_token)
        current_logits = logits[-1:].clone()
        current_context_len = token_ids.numel()

        for _ in range(max_new_tokens - 1):
            if kv_caches is not None:
                self._set_single_batch_wrapper_metadata(
                    attention_wrapper,
                    prompt_len=0,
                    cache_len=current_context_len,
                )
            current_logits, caches = self.decode_tokens(
                next_token,
                caches=caches,
                projection_weights=projection_weights,
                mlp_weights=mlp_weights,
                softmax_scale=softmax_scale,
                kv_caches=kv_caches,
                attention_wrapper=attention_wrapper,
            )
            next_token = torch.argmax(current_logits[-1], dim=-1).to(dtype=token_ids.dtype).view(1)
            generated_tokens.append(next_token)
            current_context_len += 1

        return torch.cat(generated_tokens, dim=0), current_logits, caches

    def make_runtime_mla_kv_caches(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[DeepseekV2ComponentMLAKVCache, ...]:
        return self.model.make_runtime_mla_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
