#!/usr/bin/env python3

import argparse
import json
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch


def build_config(query_mode="direct", mlp_mode="dense"):
    q_lora_rank = None
    if query_mode == "q_lora":
        q_lora_rank = 2
    elif query_mode != "direct":
        raise ValueError(f"Unsupported query mode: {query_mode}")
    config = SimpleNamespace(
        vocab_size=16,
        hidden_size=6,
        num_attention_heads=4,
        num_hidden_layers=4,
        rms_norm_eps=1e-6,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
    )
    if mlp_mode == "moe":
        config.first_k_dense_replace = 1
        config.n_routed_experts = 4
        config.n_shared_experts = 1
        config.num_experts_per_tok = 1
        config.norm_topk_prob = True
    elif mlp_mode != "dense":
        raise ValueError(f"Unsupported mlp mode: {mlp_mode}")
    return config


def make_projection_weights(deepseek_module, dims, *, device, dtype, query_mode="direct"):
    q_proj = None
    q_a_proj = None
    q_a_layernorm_weight = None
    q_b_proj = None
    if query_mode == "direct":
        q_proj = torch.tensor(
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
        )
    elif query_mode == "q_lora":
        q_a_proj = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        )
        q_a_layernorm_weight = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        q_b_proj = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported query mode: {query_mode}")
    return deepseek_module.make_projection_weights(
        q_proj=q_proj,
        q_a_proj=q_a_proj,
        q_a_layernorm_weight=q_a_layernorm_weight,
        q_b_proj=q_b_proj,
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


def make_moe_weights(deepseek_module, hidden_size, *, device, dtype, num_experts):
    experts = []
    for expert_idx in range(num_experts):
        experts.append(
            deepseek_module.make_mlp_weights(
                gate_proj=torch.full(
                    (hidden_size, 4),
                    1.0 + expert_idx,
                    device=device,
                    dtype=dtype,
                ),
                up_proj=torch.full(
                    (hidden_size, 4),
                    2.0 + expert_idx,
                    device=device,
                    dtype=dtype,
                ),
                down_proj=torch.full(
                    (4, hidden_size),
                    3.0 + expert_idx,
                    device=device,
                    dtype=dtype,
                ),
                hidden_size=hidden_size,
            )
        )
    return deepseek_module.make_moe_weights(
        gate=torch.arange(
            num_experts * hidden_size,
            device=device,
            dtype=dtype,
        ).view(num_experts, hidden_size)
        / 100.0,
        experts=tuple(experts),
        shared_experts=deepseek_module.make_mlp_weights(
            gate_proj=torch.full((hidden_size, 4), 0.5, device=device, dtype=dtype),
            up_proj=torch.full((hidden_size, 4), 0.75, device=device, dtype=dtype),
            down_proj=torch.full((4, hidden_size), 1.25, device=device, dtype=dtype),
            hidden_size=hidden_size,
        ),
        top_k=1,
        norm_topk_prob=True,
        hidden_size=hidden_size,
    )


def build_scaffold_state_dict(
    model,
    projection_weights,
    mlp_weights,
    *,
    device,
    dtype,
    moe_weights=None,
):
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
        if layer_projection_weights.q_proj is not None:
            state_dict[f"{prefix}.q_proj.weight"] = layer_projection_weights.q_proj
        else:
            state_dict[f"{prefix}.q_a_proj.weight"] = layer_projection_weights.q_a_proj
            state_dict[f"{prefix}.q_a_layernorm.weight"] = (
                layer_projection_weights.q_a_layernorm_weight
            )
            state_dict[f"{prefix}.q_b_proj.weight"] = layer_projection_weights.q_b_proj
        state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = kv_a_proj_with_mqa
        state_dict[f"{prefix}.kv_b_proj.weight"] = layer_projection_weights.kv_up_proj
        state_dict[f"{prefix}.o_proj.weight"] = layer_projection_weights.o_proj
    if moe_weights is None:
        moe_weights = tuple(None for _ in mlp_weights)
    for layer_idx, (layer_mlp_weights, layer_moe_weights) in enumerate(
        zip(mlp_weights, moe_weights)
    ):
        prefix = f"layers.{layer_idx}.mlp"
        if layer_mlp_weights is not None:
            state_dict[f"{prefix}.gate_proj.weight"] = layer_mlp_weights.gate_proj
            state_dict[f"{prefix}.up_proj.weight"] = layer_mlp_weights.up_proj
            state_dict[f"{prefix}.down_proj.weight"] = layer_mlp_weights.down_proj
        if layer_moe_weights is not None:
            state_dict[f"{prefix}.gate.weight"] = layer_moe_weights.gate
            if layer_moe_weights.shared_experts is not None:
                state_dict[f"{prefix}.shared_experts.gate_proj.weight"] = (
                    layer_moe_weights.shared_experts.gate_proj
                )
                state_dict[f"{prefix}.shared_experts.up_proj.weight"] = (
                    layer_moe_weights.shared_experts.up_proj
                )
                state_dict[f"{prefix}.shared_experts.down_proj.weight"] = (
                    layer_moe_weights.shared_experts.down_proj
                )
            for expert_idx, expert_weights in enumerate(layer_moe_weights.experts):
                state_dict[f"{prefix}.experts.{expert_idx}.gate_proj.weight"] = (
                    expert_weights.gate_proj
                )
                state_dict[f"{prefix}.experts.{expert_idx}.up_proj.weight"] = (
                    expert_weights.up_proj
                )
                state_dict[f"{prefix}.experts.{expert_idx}.down_proj.weight"] = (
                    expert_weights.down_proj
                )
    return state_dict


def write_scaffold_checkpoint(
    model,
    projection_weights,
    mlp_weights,
    *,
    device,
    dtype,
    output_dir,
    checkpoint_format="pt",
    moe_weights=None,
):
    state_dict = build_scaffold_state_dict(
        model,
        projection_weights,
        mlp_weights,
        device=device,
        dtype=dtype,
        moe_weights=moe_weights,
    )
    if checkpoint_format == "pt":
        checkpoint_path = f"{output_dir}/deepseek_scaffold.pt"
        torch.save(state_dict, checkpoint_path)
        return checkpoint_path
    if checkpoint_format == "safetensors":
        from safetensors.torch import save_file

        checkpoint_path = f"{output_dir}/deepseek_scaffold.safetensors"
        cpu_state_dict = {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in state_dict.items()
        }
        save_file(cpu_state_dict, checkpoint_path)
        return checkpoint_path
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")


def write_scaffold_hf_directory(
    model,
    projection_weights,
    mlp_weights,
    *,
    device,
    dtype,
    output_dir,
    checkpoint_format="safetensors",
    num_shards=2,
    moe_weights=None,
):
    state_dict = build_scaffold_state_dict(
        model,
        projection_weights,
        mlp_weights,
        device=device,
        dtype=dtype,
        moe_weights=moe_weights,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v2",
                "vocab_size": model.config.vocab_size,
                "hidden_size": model.config.hidden_size,
                "num_attention_heads": model.config.num_attention_heads,
                "num_hidden_layers": model.config.num_hidden_layers,
                "q_lora_rank": model.config.q_lora_rank,
                "kv_lora_rank": model.config.kv_lora_rank,
                "qk_nope_head_dim": model.config.qk_nope_head_dim,
                "qk_rope_head_dim": model.config.qk_rope_head_dim,
                "v_head_dim": model.config.v_head_dim,
                "first_k_dense_replace": getattr(model.config, "first_k_dense_replace", None),
                "n_routed_experts": getattr(model.config, "n_routed_experts", None),
                "n_shared_experts": getattr(model.config, "n_shared_experts", None),
                "num_experts_per_tok": getattr(model.config, "num_experts_per_tok", None),
                "norm_topk_prob": getattr(model.config, "norm_topk_prob", None),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if checkpoint_format != "safetensors":
        raise ValueError("HF directory scaffold writing currently supports only safetensors")

    from safetensors.torch import save_file

    state_items = list(state_dict.items())
    shard_states = [dict() for _ in range(num_shards)]
    weight_map = {}
    total_size = 0
    for index, (name, tensor) in enumerate(state_items):
        shard_name = f"model-{index % num_shards + 1:05d}-of-{num_shards:05d}.safetensors"
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        shard_states[index % num_shards][name] = cpu_tensor
        weight_map[name] = shard_name
        total_size += cpu_tensor.numel() * cpu_tensor.element_size()

    for shard_idx, shard_state in enumerate(shard_states, start=1):
        shard_path = output_path / f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        save_file(shard_state, shard_path)

    index_path = output_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return str(output_path)


def _run_scaffold_smoke_artifacts(
    mode="contiguous",
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
):
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

    config = build_config(query_mode=query_mode, mlp_mode=mlp_mode)
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
        make_projection_weights(
            deepseek_module,
            dims,
            device=device,
            dtype=dtype,
            query_mode=query_mode,
        )
        for _ in range(model.model.num_layers)
    )
    mlp_weights = tuple(
        (
            make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
            if (
                mlp_mode != "moe"
                or layer_idx < getattr(config, "first_k_dense_replace", model.model.num_layers)
            )
            else None
        )
        for layer_idx in range(model.model.num_layers)
    )
    if mlp_mode == "moe":
        moe_weights = tuple(
            (
                None
                if layer_idx < config.first_k_dense_replace
                else make_moe_weights(
                    deepseek_module,
                    config.hidden_size,
                    device=device,
                    dtype=dtype,
                    num_experts=config.n_routed_experts,
                )
            )
            for layer_idx in range(model.model.num_layers)
        )
    else:
        moe_weights = tuple(None for _ in range(model.model.num_layers))
    with tempfile.TemporaryDirectory() as tmpdir:
        if checkpoint_layout == "single_file":
            checkpoint_path = write_scaffold_checkpoint(
                model,
                projection_weights,
                mlp_weights,
                device=device,
                dtype=dtype,
                output_dir=tmpdir,
                checkpoint_format=checkpoint_format,
                moe_weights=moe_weights,
            )
        elif checkpoint_layout == "hf_dir":
            checkpoint_path = write_scaffold_hf_directory(
                model,
                projection_weights,
                mlp_weights,
                device=device,
                dtype=dtype,
                output_dir=tmpdir,
                checkpoint_format="safetensors",
                moe_weights=moe_weights,
            )
        else:
            raise ValueError(f"Unsupported checkpoint layout: {checkpoint_layout}")
        model.load_weights(checkpoint_path)

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


def run_scaffold_smoke(
    mode="contiguous",
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
):
    generated_token_ids, final_logits, final_caches = _run_scaffold_smoke_artifacts(
        mode=mode,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
    )
    return {
        "mode": mode,
        "checkpoint_format": checkpoint_format,
        "query_mode": query_mode,
        "checkpoint_layout": checkpoint_layout,
        "mlp_mode": mlp_mode,
        "prompt_token_ids": list(prompt_token_ids),
        "generated_token_ids": generated_token_ids.tolist(),
        "final_logits_shape": list(final_logits.shape),
        "cache_token_counts": [
            cache.num_tokens if hasattr(cache, "num_tokens") else cache.resident_cache.num_tokens
            for cache in final_caches
        ],
    }


def compare_scaffold_smoke(
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
):
    contiguous_tokens, contiguous_logits, contiguous_caches = _run_scaffold_smoke_artifacts(
        mode="contiguous",
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
    )
    try:
        paged_tokens, paged_logits, paged_caches = _run_scaffold_smoke_artifacts(
            mode="paged",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            checkpoint_format=checkpoint_format,
            query_mode=query_mode,
            checkpoint_layout=checkpoint_layout,
            mlp_mode=mlp_mode,
        )
    except RuntimeError as exc:
        return {
            "mode": "compare",
            "checkpoint_format": checkpoint_format,
            "query_mode": query_mode,
            "checkpoint_layout": checkpoint_layout,
            "mlp_mode": mlp_mode,
            "status": "blocked",
            "prompt_token_ids": list(prompt_token_ids),
            "generated_token_ids": contiguous_tokens.tolist(),
            "error": str(exc),
        }

    contiguous_cache_counts = [cache.num_tokens for cache in contiguous_caches]
    paged_cache_counts = [cache.resident_cache.num_tokens for cache in paged_caches]
    return {
        "mode": "compare",
        "checkpoint_format": checkpoint_format,
        "query_mode": query_mode,
        "checkpoint_layout": checkpoint_layout,
        "mlp_mode": mlp_mode,
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


def validate_scaffold_smoke_compare(
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
):
    result = compare_scaffold_smoke(
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
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
        "--checkpoint-format",
        choices=("pt", "safetensors"),
        default="pt",
    )
    parser.add_argument(
        "--checkpoint-layout",
        choices=("single_file", "hf_dir"),
        default="single_file",
    )
    parser.add_argument(
        "--query-mode",
        choices=("direct", "q_lora"),
        default="direct",
    )
    parser.add_argument(
        "--mlp-mode",
        choices=("dense", "moe"),
        default="dense",
    )
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="fail with a non-zero exit code if compare mode detects a mismatch",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        output = compare_scaffold_smoke(
            max_new_tokens=args.max_new_tokens,
            checkpoint_format=args.checkpoint_format,
            query_mode=args.query_mode,
            checkpoint_layout=args.checkpoint_layout,
            mlp_mode=args.mlp_mode,
        )
    else:
        output = run_scaffold_smoke(
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            checkpoint_format=args.checkpoint_format,
            query_mode=args.query_mode,
            checkpoint_layout=args.checkpoint_layout,
            mlp_mode=args.mlp_mode,
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
