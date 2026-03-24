from dataclasses import dataclass
from typing import Optional, Tuple

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: DeepseekV2MLAProjectionWeights,
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
        return hidden_states, cache


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: Tuple[DeepseekV2MLAProjectionWeights, ...],
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2MLAResidentCache, ...]]:
        if len(projection_weights) != self.num_layers:
            raise ValueError("projection_weights must provide one entry per local layer")
        if caches is None:
            caches = tuple(None for _ in range(self.num_layers))
        if len(caches) != self.num_layers:
            raise ValueError("caches must provide one entry per local layer")

        next_caches = []
        for layer, layer_projection_weights, layer_cache in zip(
            self.layers,
            projection_weights,
            caches,
        ):
            hidden_states, next_cache = layer(
                hidden_states=hidden_states,
                projection_weights=layer_projection_weights,
                cache=layer_cache,
                softmax_scale=softmax_scale,
            )
            next_caches.append(next_cache)
        return hidden_states, tuple(next_caches)


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)
        self.mla_dims = DeepseekV2MLADims.from_config(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        projection_weights: Tuple[DeepseekV2MLAProjectionWeights, ...],
        caches: Optional[Tuple[Optional[DeepseekV2MLAResidentCache], ...]] = None,
        softmax_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[DeepseekV2MLAResidentCache, ...]]:
        return self.model(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            caches=caches,
            softmax_scale=softmax_scale,
        )

    def load_weights(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 weight loading is not implemented yet. "
            "The MLA attention/model path still needs to be added."
        )
