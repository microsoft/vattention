#!/usr/bin/env python3

import argparse
import json
from types import SimpleNamespace

import torch


def build_config():
    return SimpleNamespace(
        vocab_size=16,
        hidden_size=6,
        num_attention_heads=4,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        q_lora_rank=None,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
    )


def make_projection_weights(deepseek_module, dims):
    return deepseek_module.make_projection_weights(
        q_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        ),
        kv_latent_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        k_rope_proj=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        kv_up_proj=torch.tensor(
            [
                [1.0, 0.0, 10.0, 20.0, 2.0, 0.0, 30.0, 40.0],
                [0.0, 1.0, 11.0, 21.0, 0.0, 2.0, 31.0, 41.0],
                [1.0, 1.0, 12.0, 22.0, 2.0, 2.0, 32.0, 42.0],
            ]
        ),
        o_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        mla_dims=dims,
    )


def make_mlp_weights(deepseek_module, hidden_size):
    return deepseek_module.make_mlp_weights(
        gate_proj=torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 0.0],
                [0.5, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.5, 0.5],
            ]
        ),
        up_proj=torch.tensor(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.5, 0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.5, 0.0],
            ]
        ),
        down_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        hidden_size=hidden_size,
    )


def build_scaffold_state_dict(model, projection_weights, mlp_weights):
    config = model.config
    state_dict = {
        "model.embed_tokens.weight": torch.arange(
            config.vocab_size * config.hidden_size,
            dtype=torch.float32,
        ).view(config.vocab_size, config.hidden_size)
        / 1000.0,
        "lm_head.weight": torch.arange(
            config.vocab_size * config.hidden_size,
            dtype=torch.float32,
        ).view(config.vocab_size, config.hidden_size)
        / 1000.0,
        "model.norm.weight": torch.ones(config.hidden_size),
    }
    for layer_idx, layer_projection_weights in enumerate(projection_weights):
        state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = torch.ones(
            config.hidden_size
        )
        state_dict[
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        ] = torch.ones(config.hidden_size)
        prefix = f"model.layers.{layer_idx}.self_attn"
        state_dict[f"{prefix}.q_proj.weight"] = layer_projection_weights.q_proj
        state_dict[f"{prefix}.kv_latent_proj.weight"] = layer_projection_weights.kv_latent_proj
        state_dict[f"{prefix}.k_rope_proj.weight"] = layer_projection_weights.k_rope_proj
        state_dict[f"{prefix}.kv_up_proj.weight"] = layer_projection_weights.kv_up_proj
        state_dict[f"{prefix}.o_proj.weight"] = layer_projection_weights.o_proj
    for layer_idx, layer_mlp_weights in enumerate(mlp_weights):
        prefix = f"model.layers.{layer_idx}.mlp"
        state_dict[f"{prefix}.gate_proj.weight"] = layer_mlp_weights.gate_proj
        state_dict[f"{prefix}.up_proj.weight"] = layer_mlp_weights.up_proj
        state_dict[f"{prefix}.down_proj.weight"] = layer_mlp_weights.down_proj
    return state_dict


def _run_scaffold_smoke_artifacts(mode="contiguous", prompt_token_ids=(1, 3), max_new_tokens=3):
    from sarathi.model_executor.parallel_utils.parallel_state import (
        set_pipeline_model_parallel_rank,
        set_pipeline_model_parallel_world_size,
        set_tensor_model_parallel_world_size,
    )
    from sarathi.model_executor.models.deepseek_v2 import (
        DeepseekV2ForCausalLM,
        DeepseekV2MLADims,
    )
    import sarathi.model_executor.models.deepseek_v2 as deepseek_module

    config = build_config()
    set_tensor_model_parallel_world_size(2)
    set_pipeline_model_parallel_world_size(1)
    set_pipeline_model_parallel_rank(0)

    model = DeepseekV2ForCausalLM(
        config,
        tensor_parallel_world_size=2,
        pipeline_parallel_world_size=1,
        pipeline_parallel_rank=0,
    )
    dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
    projection_weights = tuple(
        make_projection_weights(deepseek_module, dims)
        for _ in range(model.model.num_layers)
    )
    mlp_weights = tuple(
        make_mlp_weights(deepseek_module, config.hidden_size)
        for _ in range(model.model.num_layers)
    )
    model.load_weights(build_scaffold_state_dict(model, projection_weights, mlp_weights))

    prompt_token_ids = torch.tensor(prompt_token_ids, dtype=torch.long)
    generate_kwargs = {}
    if mode == "paged":
        from sarathi.model_executor.attention.vattention_flashattention_wrapper import (
            VAttentionFlashAttentionWrapper,
        )

        wrapper = VAttentionFlashAttentionWrapper()
        wrapper.device = torch.device("cpu")
        wrapper.is_metadata_initialized = True
        wrapper.is_profiling_iteration = False
        generate_kwargs["kv_caches"] = model.make_runtime_mla_kv_caches(
            batch_size=1,
            max_seq_len=prompt_token_ids.numel() + max_new_tokens + 1,
            device=torch.device("cpu"),
        )
        generate_kwargs["attention_wrapper"] = wrapper
    elif mode != "contiguous":
        raise ValueError(f"Unsupported smoke mode: {mode}")

    generated_token_ids, final_logits, final_caches = model.generate_greedy(
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        **generate_kwargs,
    )
    return generated_token_ids, final_logits, final_caches


def run_scaffold_smoke(mode="contiguous", prompt_token_ids=(1, 3), max_new_tokens=3):
    generated_token_ids, final_logits, final_caches = _run_scaffold_smoke_artifacts(
        mode=mode,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
    )
    return {
        "mode": mode,
        "prompt_token_ids": list(prompt_token_ids),
        "generated_token_ids": generated_token_ids.tolist(),
        "final_logits_shape": list(final_logits.shape),
        "cache_token_counts": [
            cache.num_tokens if hasattr(cache, "num_tokens") else cache.resident_cache.num_tokens
            for cache in final_caches
        ],
    }


def compare_scaffold_smoke(prompt_token_ids=(1, 3), max_new_tokens=3):
    contiguous_tokens, contiguous_logits, contiguous_caches = _run_scaffold_smoke_artifacts(
        mode="contiguous",
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
    )
    paged_tokens, paged_logits, paged_caches = _run_scaffold_smoke_artifacts(
        mode="paged",
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
    )

    contiguous_cache_counts = [cache.num_tokens for cache in contiguous_caches]
    paged_cache_counts = [cache.resident_cache.num_tokens for cache in paged_caches]
    return {
        "mode": "compare",
        "prompt_token_ids": list(prompt_token_ids),
        "generated_token_ids": contiguous_tokens.tolist(),
        "paged_generated_token_ids": paged_tokens.tolist(),
        "generated_tokens_match": torch.equal(contiguous_tokens, paged_tokens),
        "final_logits_match": torch.allclose(
            contiguous_logits,
            paged_logits,
            atol=1e-6,
            rtol=1e-6,
        ),
        "contiguous_cache_token_counts": contiguous_cache_counts,
        "paged_cache_token_counts": paged_cache_counts,
        "cache_token_counts_match": contiguous_cache_counts == paged_cache_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("contiguous", "paged", "compare"),
        default="contiguous",
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "compare":
        output = compare_scaffold_smoke(max_new_tokens=args.max_new_tokens)
    else:
        output = run_scaffold_smoke(
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
        )
    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
