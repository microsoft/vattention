#!/usr/bin/env python3

import argparse
import json
import os
import re
from types import SimpleNamespace

from sarathi.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from sarathi.model_executor.weight_utils import convert_pyslice_to_tensor, hf_model_weights_iterator


def _load_weight_names_and_config(checkpoint_path):
    config = None
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        state_dict = {}
        for name, tensor in hf_model_weights_iterator(checkpoint_path, load_format="auto"):
            state_dict[name] = convert_pyslice_to_tensor(tensor)
        return state_dict, config

    state_dict = DeepseekV2ForCausalLM._load_state_dict_file(checkpoint_path)
    return state_dict, config


def _extract_layer_indices(names):
    layer_indices = set()
    pattern = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")
    for name in names:
        match = pattern.search(name)
        if match is not None:
            layer_indices.add(int(match.group(1)))
    return tuple(sorted(layer_indices))


def _has_name(names, suffix):
    return any(name.endswith(suffix) for name in names)


def _layer_has_name(names, layer_idx, suffix):
    return any(
        name.endswith(f"layers.{layer_idx}.{suffix}")
        or name.endswith(f"model.layers.{layer_idx}.{suffix}")
        for name in names
    )


def inspect_deepseek_checkpoint(checkpoint_path):
    state_dict, config = _load_weight_names_and_config(checkpoint_path)
    names = tuple(sorted(state_dict.keys()))
    layer_indices = _extract_layer_indices(names)

    has_q_proj = _has_name(names, ".self_attn.q_proj.weight")
    has_q_lora = all(
        _has_name(names, suffix)
        for suffix in (
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_a_layernorm.weight",
            ".self_attn.q_b_proj.weight",
        )
    )
    has_combined_kv = _has_name(names, ".self_attn.kv_a_proj_with_mqa.weight")
    has_kv_a_layernorm = _has_name(names, ".self_attn.kv_a_layernorm.weight")
    has_kv_b_proj = _has_name(names, ".self_attn.kv_b_proj.weight")
    has_dense_mlp = all(
        _has_name(names, suffix)
        for suffix in (
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        )
    )
    has_moe = any(
        ".mlp.gate.weight" in name
        or ".mlp.shared_experts." in name
        or ".mlp.experts." in name
        for name in names
    )
    moe_layer_indices = tuple(
        layer_idx
        for layer_idx in layer_indices
        if _layer_has_name(names, layer_idx, "mlp.gate.weight")
    )
    config_first_k_dense_replace = None if config is None else config.get("first_k_dense_replace")
    config_n_routed_experts = None if config is None else config.get("n_routed_experts")
    config_n_shared_experts = None if config is None else config.get("n_shared_experts")
    config_tensor_parallel_world_size = (
        None if config is None else config.get("tensor_parallel_world_size")
    )

    status = "supported_non_moe_surface"
    blockers = []
    if not (has_q_proj or has_q_lora):
        status = "blocked"
        blockers.append("missing_query_projection_surface")
    if not has_combined_kv or not has_kv_b_proj:
        status = "blocked"
        blockers.append("missing_kv_projection_surface")
    if has_moe:
        if config_first_k_dense_replace is None or not config_n_routed_experts:
            status = "blocked"
            blockers.append("missing_moe_config")
        else:
            for layer_idx in moe_layer_indices:
                if layer_idx < config_first_k_dense_replace:
                    status = "blocked"
                    blockers.append("moe_before_first_k_dense_replace")
                    break
                if not _layer_has_name(names, layer_idx, "mlp.gate.weight"):
                    status = "blocked"
                    blockers.append("missing_moe_gate")
                    break
                if config_n_shared_experts:
                    for suffix in (
                        "mlp.shared_experts.gate_proj.weight",
                        "mlp.shared_experts.up_proj.weight",
                        "mlp.shared_experts.down_proj.weight",
                    ):
                        if not _layer_has_name(names, layer_idx, suffix):
                            status = "blocked"
                            blockers.append("missing_shared_expert_weights")
                            break
                    if blockers:
                        break
                for expert_idx in range(config_n_routed_experts):
                    for suffix in (
                        f"mlp.experts.{expert_idx}.gate_proj.weight",
                        f"mlp.experts.{expert_idx}.up_proj.weight",
                        f"mlp.experts.{expert_idx}.down_proj.weight",
                    ):
                        if not _layer_has_name(names, layer_idx, suffix):
                            status = "blocked"
                            blockers.append("missing_routed_expert_weights")
                            break
                    if blockers:
                        break
                if blockers:
                    break
        if not blockers:
            status = "supported_bounded_moe_surface"

    loadable_scaffold_surface = None
    load_error = None
    if config is not None and not blockers:
        try:
            model = DeepseekV2ForCausalLM(
                SimpleNamespace(**config),
                tensor_parallel_world_size=config.get("tensor_parallel_world_size", 1),
                pipeline_parallel_world_size=config.get("pipeline_parallel_world_size", 1),
                pipeline_parallel_rank=config.get("pipeline_parallel_rank", 0),
            )
            model.load_weights(state_dict)
            loadable_scaffold_surface = True
        except Exception as exc:
            status = "blocked"
            blockers.append("scaffold_load_failed")
            loadable_scaffold_surface = False
            load_error = str(exc)

    return {
        "status": status,
        "checkpoint_path": checkpoint_path,
        "config_model_type": None if config is None else config.get("model_type"),
        "config_q_lora_rank": None if config is None else config.get("q_lora_rank"),
        "config_first_k_dense_replace": config_first_k_dense_replace,
        "config_n_routed_experts": config_n_routed_experts,
        "config_n_shared_experts": config_n_shared_experts,
        "config_tensor_parallel_world_size": config_tensor_parallel_world_size,
        "has_q_proj": has_q_proj,
        "has_q_lora": has_q_lora,
        "has_combined_kv": has_combined_kv,
        "has_kv_a_layernorm": has_kv_a_layernorm,
        "has_kv_b_proj": has_kv_b_proj,
        "has_dense_mlp": has_dense_mlp,
        "has_moe": has_moe,
        "moe_layer_indices": list(moe_layer_indices),
        "loadable_scaffold_surface": loadable_scaffold_surface,
        "load_error": load_error,
        "blockers": blockers,
        "num_tensors": len(names),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()
    print(json.dumps(inspect_deepseek_checkpoint(args.checkpoint_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
