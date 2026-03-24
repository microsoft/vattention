from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import torch
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
    q_proj: torch.Tensor
    kv_latent_proj: torch.Tensor
    k_rope_proj: torch.Tensor
    kv_up_proj: torch.Tensor
    o_proj: torch.Tensor


@dataclass(frozen=True)
class DeepseekV2MLPWeights:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


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
    q_proj: torch.Tensor,
    kv_latent_proj: torch.Tensor,
    k_rope_proj: torch.Tensor,
    kv_up_proj: torch.Tensor,
    o_proj: torch.Tensor,
    mla_dims: DeepseekV2MLADims,
) -> DeepseekV2MLAProjectionWeights:
    expected_q_proj = (mla_dims.hidden_size, mla_dims.q_proj_output_dim_local)
    expected_kv_latent_proj = (mla_dims.hidden_size, mla_dims.kv_lora_rank)
    expected_k_rope_proj = (
        mla_dims.hidden_size,
        mla_dims.num_heads * mla_dims.qk_rope_head_dim,
    )
    expected_kv_up_proj = (
        mla_dims.kv_lora_rank,
        mla_dims.kv_up_proj_output_dim_local,
    )
    expected_o_proj = (mla_dims.o_proj_input_dim_local, mla_dims.hidden_size)
    if tuple(q_proj.shape) != expected_q_proj:
        raise ValueError("q_proj shape does not match local MLA query projection size")
    if tuple(kv_latent_proj.shape) != expected_kv_latent_proj:
        raise ValueError("kv_latent_proj shape does not match local MLA latent projection size")
    if tuple(k_rope_proj.shape) != expected_k_rope_proj:
        raise ValueError("k_rope_proj shape does not match local MLA rope projection size")
    if tuple(kv_up_proj.shape) != expected_kv_up_proj:
        raise ValueError("kv_up_proj shape does not match local MLA up-projection size")
    if tuple(o_proj.shape) != expected_o_proj:
        raise ValueError("o_proj shape does not match local MLA output projection size")
    return DeepseekV2MLAProjectionWeights(
        q_proj=q_proj,
        kv_latent_proj=kv_latent_proj,
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

    query_states = hidden_states @ projection_weights.q_proj
    kv_latent = hidden_states @ projection_weights.kv_latent_proj
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
        q_proj: torch.Tensor,
        kv_latent_proj: torch.Tensor,
        k_rope_proj: torch.Tensor,
        kv_up_proj: torch.Tensor,
        o_proj: torch.Tensor,
    ) -> DeepseekV2MLAProjectionWeights:
        return make_projection_weights(
            q_proj=q_proj,
            kv_latent_proj=kv_latent_proj,
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
        # MoE and RMSNorm execution are still pending.
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
        mlp_weights: Optional[DeepseekV2MLPWeights] = None,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2MLAResidentCache]:
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
        return hidden_states, cache

    def forward_with_attention_wrapper(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
        kv_cache,
        mlp_weights: Optional[DeepseekV2MLPWeights] = None,
        attention_wrapper=None,
        cache: Optional[DeepseekV2MLAResidentCache] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, DeepseekV2LayerCache]:
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
        self.norm = nn.Identity() if self.is_pipeline_last_stage else None
        self.layer_projection_weights: Optional[
            Tuple[DeepseekV2MLAProjectionWeights, ...]
        ] = None
        self.layer_mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None

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
    ) -> None:
        if len(projection_weights) != self.num_layers:
            raise ValueError("projection_weights must provide one entry per local layer")
        if mlp_weights is None:
            mlp_weights = tuple(None for _ in range(self.num_layers))
        if len(mlp_weights) != self.num_layers:
            raise ValueError("mlp_weights must provide one entry per local layer")
        self.layer_projection_weights = tuple(projection_weights)
        self.layer_mlp_weights = tuple(mlp_weights)

    def _resolve_scaffold_weights(
        self,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]],
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]],
    ) -> Tuple[
        Tuple[DeepseekV2MLAProjectionWeights, ...],
        Tuple[Optional[DeepseekV2MLPWeights], ...],
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
        return tuple(projection_weights), tuple(mlp_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: Optional[Tuple[DeepseekV2MLAProjectionWeights, ...]] = None,
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2MLAResidentCache, ...]]:
        projection_weights, mlp_weights = self._resolve_scaffold_weights(
            projection_weights,
            mlp_weights,
        )
        if caches is None:
            caches = tuple(None for _ in range(self.num_layers))
        if len(caches) != self.num_layers:
            raise ValueError("caches must provide one entry per local layer")

        hidden_states = self._prepare_hidden_states(hidden_states)
        next_caches = []
        for layer, layer_projection_weights, layer_mlp_weights, layer_cache in zip(
            self.layers,
            projection_weights,
            mlp_weights,
            caches,
        ):
            hidden_states, next_cache = layer(
                hidden_states=hidden_states,
                projection_weights=layer_projection_weights,
                mlp_weights=layer_mlp_weights,
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
        attention_wrapper=None,
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2LayerCache, ...]]:
        projection_weights, mlp_weights = self._resolve_scaffold_weights(
            projection_weights,
            mlp_weights,
        )
        if len(kv_caches) != self.num_layers:
            raise ValueError("kv_caches must provide one entry per local layer")
        if caches is None:
            caches = tuple(None for _ in range(self.num_layers))
        if len(caches) != self.num_layers:
            raise ValueError("caches must provide one entry per local layer")

        hidden_states = self._prepare_hidden_states(hidden_states)
        next_caches = []
        for layer, layer_projection_weights, layer_mlp_weights, layer_kv_cache, layer_cache in zip(
            self.layers,
            projection_weights,
            mlp_weights,
            kv_caches,
            caches,
        ):
            hidden_states, next_cache = layer.forward_with_attention_wrapper(
                hidden_states=hidden_states,
                projection_weights=layer_projection_weights,
                mlp_weights=layer_mlp_weights,
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

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)
        self.mla_dims = DeepseekV2MLADims.from_config(config)
        self.lm_head = None
        if self.model.is_pipeline_last_stage and getattr(config, "vocab_size", None) is not None:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def set_scaffold_weights(
        self,
        projection_weights: Tuple[DeepseekV2MLAProjectionWeights, ...],
        mlp_weights: Optional[Tuple[Optional[DeepseekV2MLPWeights], ...]] = None,
    ) -> None:
        self.model.set_scaffold_weights(
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
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

    def load_scaffold_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        local_projection_weights = []
        local_mlp_weights = []
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
            local_prefix = f"model.layers.{layer_idx}"
            global_prefix = f"model.layers.{self.model.layer_offset + layer_idx}"
            projection_prefixes = (
                f"{local_prefix}.self_attn",
                f"{global_prefix}.self_attn",
            )
            projection_tensors = {
                "q_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.q_proj.weight" for prefix in projection_prefixes),
                ),
                "kv_latent_proj": self._get_scaffold_tensor(
                    state_dict,
                    *(f"{prefix}.kv_latent_proj.weight" for prefix in projection_prefixes),
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
            missing_projection_keys = [
                name for name, tensor in projection_tensors.items() if tensor is None
            ]
            if missing_projection_keys:
                raise KeyError(
                    "Missing scaffold weights for layer "
                    f"{layer_idx}: {', '.join(missing_projection_keys)}"
                )
            local_projection_weights.append(
                make_projection_weights(
                    q_proj=projection_tensors["q_proj"],
                    kv_latent_proj=projection_tensors["kv_latent_proj"],
                    k_rope_proj=projection_tensors["k_rope_proj"],
                    kv_up_proj=projection_tensors["kv_up_proj"],
                    o_proj=projection_tensors["o_proj"],
                    mla_dims=layer.self_attn.mla_dims,
                )
            )

            mlp_prefixes = (
                f"{local_prefix}.mlp",
                f"{global_prefix}.mlp",
            )
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
            present_mlp_keys = [
                name for name, tensor in mlp_tensors.items() if tensor is not None
            ]
            if present_mlp_keys:
                if len(present_mlp_keys) != len(mlp_tensors):
                    if strict:
                        missing = sorted(set(mlp_tensors) - set(present_mlp_keys))
                        raise KeyError(
                            "Missing scaffold MLP weights for layer "
                            f"{layer_idx}: {', '.join(missing)}"
                        )
                    local_mlp_weights.append(None)
                else:
                    local_mlp_weights.append(
                        make_mlp_weights(
                            gate_proj=mlp_tensors["gate_proj"],
                            up_proj=mlp_tensors["up_proj"],
                            down_proj=mlp_tensors["down_proj"],
                            hidden_size=hidden_size,
                        )
                    )
            else:
                local_mlp_weights.append(None)

        self.set_scaffold_weights(
            projection_weights=tuple(local_projection_weights),
            mlp_weights=tuple(local_mlp_weights),
        )

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
