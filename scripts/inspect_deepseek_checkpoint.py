#!/usr/bin/env python3

import argparse
import json
import os

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
        return tuple(sorted(state_dict.keys())), config

    state_dict = DeepseekV2ForCausalLM._load_state_dict_file(checkpoint_path)
    return tuple(sorted(state_dict.keys())), config


def inspect_deepseek_checkpoint(checkpoint_path):
    names, config = _load_weight_names_and_config(checkpoint_path)

    has_q_proj = any(name.endswith(".self_attn.q_proj.weight") for name in names)
    has_q_lora = all(
        any(name.endswith(suffix) for name in names)
        for suffix in (
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_a_layernorm.weight",
            ".self_attn.q_b_proj.weight",
        )
    )
    has_combined_kv = any(name.endswith(".self_attn.kv_a_proj_with_mqa.weight") for name in names)
    has_kv_a_layernorm = any(name.endswith(".self_attn.kv_a_layernorm.weight") for name in names)
    has_kv_b_proj = any(name.endswith(".self_attn.kv_b_proj.weight") for name in names)
    has_dense_mlp = all(
        any(name.endswith(suffix) for name in names)
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

    status = "supported_non_moe_surface"
    blockers = []
    if not (has_q_proj or has_q_lora):
        status = "blocked"
        blockers.append("missing_query_projection_surface")
    if not has_combined_kv or not has_kv_b_proj:
        status = "blocked"
        blockers.append("missing_kv_projection_surface")
    if has_moe:
        status = "blocked"
        blockers.append("moe_not_supported")

    return {
        "status": status,
        "checkpoint_path": checkpoint_path,
        "config_model_type": None if config is None else config.get("model_type"),
        "config_q_lora_rank": None if config is None else config.get("q_lora_rank"),
        "config_n_routed_experts": None if config is None else config.get("n_routed_experts"),
        "has_q_proj": has_q_proj,
        "has_q_lora": has_q_lora,
        "has_combined_kv": has_combined_kv,
        "has_kv_a_layernorm": has_kv_a_layernorm,
        "has_kv_b_proj": has_kv_b_proj,
        "has_dense_mlp": has_dense_mlp,
        "has_moe": has_moe,
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
