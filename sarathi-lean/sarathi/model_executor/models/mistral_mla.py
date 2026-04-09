from __future__ import annotations

from typing import Mapping, Optional

import torch

from sarathi.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from sarathi.model_executor.models.mistral import MistralForCausalLM


class MistralMLAForCausalLM(DeepseekV2ForCausalLM):
    """Experimental Mistral->MLA scaffold for fragmentation studies.

    This conversion is intentionally shape-correct rather than quality-preserving.
    The goal is to exercise an MLA cache layout using Mistral-Nemo backbone weights
    so we can compare allocator behavior and fragmentation end to end.
    """

    @staticmethod
    def _normalize_source_state_dict(
        state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        normalized: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            normalized_name = MistralForCausalLM._normalize_weight_name(name)
            is_modern_name = name.startswith("model.") or name == "lm_head.weight"
            if normalized_name not in normalized or is_modern_name:
                normalized[normalized_name] = tensor
        return normalized

    @staticmethod
    def _require_tensor(
        state_dict: Mapping[str, torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        tensor = state_dict.get(name)
        if tensor is None:
            raise KeyError(f"Missing source weight: {name}")
        return tensor

    def _coerce_weight(
        self,
        tensor: torch.Tensor,
        *,
        expected_shape: tuple[int, int],
        name: str,
    ) -> torch.Tensor:
        coerced = self._coerce_linear_weight_layout(tensor, expected_shape=expected_shape)
        if tuple(coerced.shape) != expected_shape:
            raise ValueError(
                f"{name} has incompatible shape {tuple(tensor.shape)}; "
                f"expected {expected_shape} after layout coercion"
            )
        return coerced.contiguous()

    @staticmethod
    def _resize_linear_output(
        tensor: torch.Tensor,
        *,
        target_output_dim: int,
    ) -> torch.Tensor:
        current_output_dim = tensor.shape[1]
        if current_output_dim == target_output_dim:
            return tensor.contiguous()
        if current_output_dim > target_output_dim:
            return tensor[:, :target_output_dim].contiguous()

        pad = torch.zeros(
            (tensor.shape[0], target_output_dim - current_output_dim),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=1).contiguous()

    @staticmethod
    def _resize_linear_input(
        tensor: torch.Tensor,
        *,
        target_input_dim: int,
    ) -> torch.Tensor:
        current_input_dim = tensor.shape[0]
        if current_input_dim == target_input_dim:
            return tensor.contiguous()
        if current_input_dim > target_input_dim:
            return tensor[:target_input_dim, :].contiguous()

        pad = torch.zeros(
            (target_input_dim - current_input_dim, tensor.shape[1]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=0).contiguous()

    def _build_mla_attention_scaffold(
        self,
        normalized_state_dict: Mapping[str, torch.Tensor],
        *,
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        hidden_size = self.config.hidden_size
        total_q_heads = self.config.num_attention_heads
        total_kv_heads = self.config.num_key_value_heads
        head_dim = getattr(self.config, "head_dim", hidden_size // total_q_heads)
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        q_proj_global_dim = total_q_heads * q_head_dim
        kv_up_proj_global_dim = total_q_heads * (
            self.config.qk_nope_head_dim + self.config.v_head_dim
        )
        o_proj_input_global_dim = total_q_heads * self.config.v_head_dim

        q_proj = self._coerce_weight(
            self._require_tensor(
                normalized_state_dict,
                f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            ),
            expected_shape=(hidden_size, total_q_heads * head_dim),
            name=f"layer {layer_idx} q_proj",
        )
        q_proj = self._resize_linear_output(q_proj, target_output_dim=q_proj_global_dim)

        k_proj = self._coerce_weight(
            self._require_tensor(
                normalized_state_dict,
                f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            ),
            expected_shape=(hidden_size, total_kv_heads * head_dim),
            name=f"layer {layer_idx} k_proj",
        )
        v_proj = self._coerce_weight(
            self._require_tensor(
                normalized_state_dict,
                f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            ),
            expected_shape=(hidden_size, total_kv_heads * head_dim),
            name=f"layer {layer_idx} v_proj",
        )

        k_proj_heads = k_proj.view(hidden_size, total_kv_heads, head_dim)
        v_proj_heads = v_proj.view(hidden_size, total_kv_heads, head_dim)

        # Compress per-token state to a smaller resident cache that favors a
        # larger tokens-per-page count for the fragmentation experiment.
        k_rope_proj = (
            k_proj_heads[:, :, : self.config.qk_rope_head_dim]
            .mean(dim=1)
            .contiguous()
        )
        kv_latent_proj = v_proj_heads.mean(dim=1).contiguous()
        kv_latent_proj = self._resize_linear_output(
            kv_latent_proj,
            target_output_dim=self.config.kv_lora_rank,
        )

        kv_up_proj = torch.zeros(
            (
                self.config.kv_lora_rank,
                kv_up_proj_global_dim,
            ),
            dtype=q_proj.dtype,
        )
        k_nope_width = min(self.config.qk_nope_head_dim, self.config.kv_lora_rank)
        value_width = min(self.config.v_head_dim, self.config.kv_lora_rank)
        for head_idx in range(total_q_heads):
            head_offset = head_idx * (
                self.config.qk_nope_head_dim + self.config.v_head_dim
            )
            if k_nope_width > 0:
                kv_up_proj[
                    :k_nope_width,
                    head_offset : head_offset + k_nope_width,
                ] = torch.eye(k_nope_width, dtype=kv_up_proj.dtype)
            value_offset = head_offset + self.config.qk_nope_head_dim
            kv_up_proj[
                :value_width,
                value_offset : value_offset + value_width,
            ] = torch.eye(value_width, dtype=kv_up_proj.dtype)

        o_proj = self._coerce_weight(
            self._require_tensor(
                normalized_state_dict,
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            ),
            expected_shape=(total_q_heads * head_dim, hidden_size),
            name=f"layer {layer_idx} o_proj",
        )
        o_proj = self._resize_linear_input(o_proj, target_input_dim=o_proj_input_global_dim)

        return {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": q_proj,
            f"model.layers.{layer_idx}.self_attn.kv_latent_proj.weight": kv_latent_proj,
            f"model.layers.{layer_idx}.self_attn.k_rope_proj.weight": k_rope_proj,
            f"model.layers.{layer_idx}.self_attn.kv_up_proj.weight": kv_up_proj,
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": o_proj,
        }

    def _build_scaffold_state_dict(
        self,
        normalized_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        scaffold: dict[str, torch.Tensor] = {}

        passthrough_keys = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        for key in passthrough_keys:
            tensor = normalized_state_dict.get(key)
            if tensor is not None:
                scaffold[key] = tensor

        for layer_idx in range(self.config.num_hidden_layers):
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ):
                key = f"model.layers.{layer_idx}.{suffix}"
                scaffold[key] = self._require_tensor(normalized_state_dict, key)

            scaffold.update(
                self._build_mla_attention_scaffold(
                    normalized_state_dict,
                    layer_idx=layer_idx,
                )
            )

        return scaffold

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
        strict: bool = True,
    ):
        source_model_name = getattr(self.config, "source_model_name", model_name_or_path)
        source_state_dict = self._load_scaffold_state_dict_from_path(
            source_model_name,
            cache_dir=cache_dir,
            load_format=load_format,
            revision=revision,
        )
        normalized_state_dict = self._normalize_source_state_dict(source_state_dict)
        scaffold_state_dict = self._build_scaffold_state_dict(normalized_state_dict)
        self.load_scaffold_state_dict(scaffold_state_dict, strict=strict)
