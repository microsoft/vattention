#!/usr/bin/env python3

import argparse
import json
import sys
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


def make_projection_weights(deepseek_module, dims, *, device, dtype):
    return deepseek_module.make_projection_weights(
        q_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        ),
        kv_latent_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        ),
        k_rope_proj=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        ),
        kv_up_proj=torch.tensor(
            [
                [1.0, 0.0, 10.0, 20.0, 2.0, 0.0, 30.0, 40.0],
                [0.0, 1.0, 11.0, 21.0, 0.0, 2.0, 31.0, 41.0],
                [1.0, 1.0, 12.0, 22.0, 2.0, 2.0, 32.0, 42.0],
            ],
            device=device,
            dtype=dtype,
        ),
        o_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        ),
        mla_dims=dims,
    )


def make_mlp_weights(deepseek_module, hidden_size, *, device, dtype):
    return deepseek_module.make_mlp_weights(
        gate_proj=torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 0.0],
                [0.5, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.5, 0.5],
            ],
            device=device,
            dtype=dtype,
        ),
        up_proj=torch.tensor(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.5, 0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.5, 0.0],
            ],
            device=device,
            dtype=dtype,
        ),
        down_proj=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        ),
        hidden_size=hidden_size,
    )


def build_scaffold_state_dict(model, projection_weights, mlp_weights, *, device, dtype):
    config = model.config
    state_dict = {
        "embed_tokens.weight": torch.arange(
            config.vocab_size * config.hidden_size,
            dtype=dtype,
            device=device,
        ).view(config.vocab_size, config.hidden_size)
        / 1000.0,
        "lm_head.weight": torch.arange(
            config.vocab_size * config.hidden_size,
            dtype=dtype,
            device=device,
        ).view(config.vocab_size, config.hidden_size)
        / 1000.0,
        "norm.weight": torch.ones(config.hidden_size, device=device, dtype=dtype),
    }
    for layer_idx, layer_projection_weights in enumerate(projection_weights):
        state_dict[f"layers.{layer_idx}.input_layernorm.weight"] = torch.ones(
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        state_dict[f"layers.{layer_idx}.post_attention_layernorm.weight"] = torch.ones(
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        prefix = f"layers.{layer_idx}.self_attn"
        kv_a_proj_with_mqa = torch.cat(
            [
                layer_projection_weights.kv_latent_proj,
                layer_projection_weights.k_rope_proj,
            ],
            dim=1,
        )
        state_dict[f"{prefix}.q_proj.weight"] = layer_projection_weights.q_proj
        state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = kv_a_proj_with_mqa
        state_dict[f"{prefix}.kv_b_proj.weight"] = layer_projection_weights.kv_up_proj
        state_dict[f"{prefix}.o_proj.weight"] = layer_projection_weights.o_proj
    for layer_idx, layer_mlp_weights in enumerate(mlp_weights):
        prefix = f"layers.{layer_idx}.mlp"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    set_tensor_model_parallel_world_size(2)
    set_pipeline_model_parallel_world_size(1)
    set_pipeline_model_parallel_rank(0)

    model = DeepseekV2ForCausalLM(
        config,
        tensor_parallel_world_size=2,
        pipeline_parallel_world_size=1,
        pipeline_parallel_rank=0,
    )
    model = model.to(device=device, dtype=dtype)
    dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
    projection_weights = tuple(
        make_projection_weights(deepseek_module, dims, device=device, dtype=dtype)
        for _ in range(model.model.num_layers)
    )
    mlp_weights = tuple(
        make_mlp_weights(
            deepseek_module,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        for _ in range(model.model.num_layers)
    )
    model.load_weights(
        build_scaffold_state_dict(
            model,
            projection_weights,
            mlp_weights,
            device=device,
            dtype=dtype,
        )
    )

    prompt_token_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device)
    generate_kwargs = {}
    if mode == "paged":
        from sarathi.model_executor.attention.vattention_flashattention_wrapper import (
            VAttentionFlashAttentionWrapper,
        )

        wrapper = VAttentionFlashAttentionWrapper()
        wrapper.device = device
        wrapper.is_metadata_initialized = True
        wrapper.is_profiling_iteration = False
        generate_kwargs["kv_caches"] = model.make_runtime_mla_kv_caches(
            batch_size=1,
            max_seq_len=prompt_token_ids.numel() + max_new_tokens + 1,
            device=device,
            dtype=dtype,
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
    try:
        paged_tokens, paged_logits, paged_caches = _run_scaffold_smoke_artifacts(
            mode="paged",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
        )
    except RuntimeError as exc:
        return {
            "mode": "compare",
            "status": "blocked",
            "prompt_token_ids": list(prompt_token_ids),
            "generated_token_ids": contiguous_tokens.tolist(),
            "error": str(exc),
        }

    contiguous_cache_counts = [cache.num_tokens for cache in contiguous_caches]
    paged_cache_counts = [cache.resident_cache.num_tokens for cache in paged_caches]
    return {
        "mode": "compare",
        "status": "ok",
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


def validate_scaffold_smoke_compare(prompt_token_ids=(1, 3), max_new_tokens=3):
    result = compare_scaffold_smoke(
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
    )
    if result.get("status") == "blocked":
        raise RuntimeError(f"scaffold smoke compare blocked: {result['error']}")
    if not result["generated_tokens_match"]:
        raise RuntimeError("contiguous and paged generated tokens do not match")
    if not result["final_logits_match"]:
        raise RuntimeError("contiguous and paged final logits do not match")
    if not result["cache_token_counts_match"]:
        raise RuntimeError("contiguous and paged cache token counts do not match")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("contiguous", "paged", "compare"),
        default="contiguous",
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="fail with a non-zero exit code if compare mode detects a mismatch",
    )
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
    if args.mode == "compare" and args.require_match:
        if output.get("status") == "blocked":
            print(
                f"scaffold smoke compare blocked: {output['error']}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not output["generated_tokens_match"]:
            print("scaffold smoke compare failed: generated tokens differ", file=sys.stderr)
            raise SystemExit(1)
        if not output["final_logits_match"]:
            print("scaffold smoke compare failed: final logits differ", file=sys.stderr)
            raise SystemExit(1)
        if not output["cache_token_counts_match"]:
            print("scaffold smoke compare failed: cache token counts differ", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
