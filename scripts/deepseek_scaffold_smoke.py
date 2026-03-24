#!/usr/bin/env python3

import argparse
import json
import os
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
        intermediate_size=8,
        moe_intermediate_size=8,
        num_attention_heads=4,
        num_hidden_layers=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        scoring_func="softmax",
        architectures=["DeepseekV2ForCausalLM"],
        tie_word_embeddings=False,
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
        kv_a_layernorm_weight=torch.tensor(
            [1.0, 0.5, 2.0],
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
    namespace="local",
):
    if namespace not in ("local", "hf"):
        raise ValueError(f"Unsupported scaffold namespace: {namespace}")

    tp_world_size = model.model.tensor_parallel_world_size if namespace == "hf" else 1

    def expand_tp_shard(tensor, *, shard_dim):
        if tensor is None or tp_world_size == 1:
            return tensor
        return torch.cat([tensor] * tp_world_size, dim=shard_dim)

    embed_key = "model.embed_tokens.weight" if namespace == "hf" else "embed_tokens.weight"
    lm_head_key = "lm_head.weight"
    norm_key = "model.norm.weight" if namespace == "hf" else "norm.weight"

    config = model.config
    state_dict = {
        embed_key: torch.arange(
            config.vocab_size * config.hidden_size,
            dtype=dtype,
            device=device,
        ).view(config.vocab_size, config.hidden_size)
        / 1000.0,
        lm_head_key: torch.arange(
            config.vocab_size * config.hidden_size,
            dtype=dtype,
            device=device,
        ).view(config.vocab_size, config.hidden_size)
        / 1000.0,
        norm_key: torch.ones(config.hidden_size, device=device, dtype=dtype),
    }
    for layer_idx, layer_projection_weights in enumerate(projection_weights):
        layer_prefix = (
            f"model.layers.{layer_idx}" if namespace == "hf" else f"layers.{layer_idx}"
        )
        state_dict[f"{layer_prefix}.input_layernorm.weight"] = torch.ones(
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        state_dict[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.ones(
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        prefix = f"{layer_prefix}.self_attn"
        kv_a_proj_with_mqa = torch.cat(
            [
                layer_projection_weights.kv_latent_proj,
                expand_tp_shard(layer_projection_weights.k_rope_proj, shard_dim=1),
            ],
            dim=1,
        )
        if layer_projection_weights.q_proj is not None:
            state_dict[f"{prefix}.q_proj.weight"] = expand_tp_shard(
                layer_projection_weights.q_proj,
                shard_dim=1,
            )
        else:
            state_dict[f"{prefix}.q_a_proj.weight"] = layer_projection_weights.q_a_proj
            state_dict[f"{prefix}.q_a_layernorm.weight"] = (
                layer_projection_weights.q_a_layernorm_weight
            )
            state_dict[f"{prefix}.q_b_proj.weight"] = expand_tp_shard(
                layer_projection_weights.q_b_proj,
                shard_dim=1,
            )
        state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = kv_a_proj_with_mqa
        if layer_projection_weights.kv_a_layernorm_weight is not None:
            state_dict[f"{prefix}.kv_a_layernorm.weight"] = (
                layer_projection_weights.kv_a_layernorm_weight
            )
        state_dict[f"{prefix}.kv_b_proj.weight"] = expand_tp_shard(
            layer_projection_weights.kv_up_proj,
            shard_dim=1,
        )
        state_dict[f"{prefix}.o_proj.weight"] = expand_tp_shard(
            layer_projection_weights.o_proj,
            shard_dim=0,
        )
    if moe_weights is None:
        moe_weights = tuple(None for _ in mlp_weights)
    for layer_idx, (layer_mlp_weights, layer_moe_weights) in enumerate(
        zip(mlp_weights, moe_weights)
    ):
        layer_prefix = (
            f"model.layers.{layer_idx}" if namespace == "hf" else f"layers.{layer_idx}"
        )
        prefix = f"{layer_prefix}.mlp"
        if layer_mlp_weights is not None:
            state_dict[f"{prefix}.gate_proj.weight"] = expand_tp_shard(
                layer_mlp_weights.gate_proj,
                shard_dim=1,
            )
            state_dict[f"{prefix}.up_proj.weight"] = expand_tp_shard(
                layer_mlp_weights.up_proj,
                shard_dim=1,
            )
            state_dict[f"{prefix}.down_proj.weight"] = expand_tp_shard(
                layer_mlp_weights.down_proj,
                shard_dim=0,
            )
        if layer_moe_weights is not None:
            state_dict[f"{prefix}.gate.weight"] = layer_moe_weights.gate
            if layer_moe_weights.shared_experts is not None:
                state_dict[f"{prefix}.shared_experts.gate_proj.weight"] = (
                    expand_tp_shard(
                        layer_moe_weights.shared_experts.gate_proj,
                        shard_dim=1,
                    )
                )
                state_dict[f"{prefix}.shared_experts.up_proj.weight"] = (
                    expand_tp_shard(
                        layer_moe_weights.shared_experts.up_proj,
                        shard_dim=1,
                    )
                )
                state_dict[f"{prefix}.shared_experts.down_proj.weight"] = (
                    expand_tp_shard(
                        layer_moe_weights.shared_experts.down_proj,
                        shard_dim=0,
                    )
                )
            for expert_idx, expert_weights in enumerate(layer_moe_weights.experts):
                state_dict[f"{prefix}.experts.{expert_idx}.gate_proj.weight"] = (
                    expand_tp_shard(expert_weights.gate_proj, shard_dim=1)
                )
                state_dict[f"{prefix}.experts.{expert_idx}.up_proj.weight"] = (
                    expand_tp_shard(expert_weights.up_proj, shard_dim=1)
                )
                state_dict[f"{prefix}.experts.{expert_idx}.down_proj.weight"] = (
                    expand_tp_shard(expert_weights.down_proj, shard_dim=0)
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
        namespace="local",
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
        namespace="hf",
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "config.json"
    config_payload = {
        "model_type": "deepseek_v2",
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": getattr(model.config, "intermediate_size", None),
        "moe_intermediate_size": getattr(model.config, "moe_intermediate_size", None),
        "num_attention_heads": model.config.num_attention_heads,
        "num_hidden_layers": model.config.num_hidden_layers,
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "tensor_parallel_world_size": model.model.tensor_parallel_world_size,
        "pipeline_parallel_world_size": model.model.pipeline_parallel_world_size,
        "pipeline_parallel_rank": model.model.pipeline_parallel_rank,
        "rms_norm_eps": getattr(model.config, "rms_norm_eps", None),
        "rope_theta": getattr(model.config, "rope_theta", None),
        "attention_bias": getattr(model.config, "attention_bias", None),
        "q_lora_rank": model.config.q_lora_rank,
        "kv_lora_rank": model.config.kv_lora_rank,
        "qk_nope_head_dim": model.config.qk_nope_head_dim,
        "qk_rope_head_dim": model.config.qk_rope_head_dim,
        "v_head_dim": model.config.v_head_dim,
        "first_k_dense_replace": getattr(model.config, "first_k_dense_replace", None),
        "n_routed_experts": getattr(model.config, "n_routed_experts", None),
        "n_shared_experts": getattr(model.config, "n_shared_experts", None),
        "num_experts_per_tok": getattr(model.config, "num_experts_per_tok", None),
        "scoring_func": getattr(model.config, "scoring_func", None),
        "norm_topk_prob": getattr(model.config, "norm_topk_prob", None),
        "architectures": getattr(model.config, "architectures", None),
        "tie_word_embeddings": getattr(model.config, "tie_word_embeddings", None),
    }
    config_payload = {
        key: value for key, value in config_payload.items() if value is not None
    }
    config_path.write_text(
        json.dumps(
            config_payload,
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
    _write_minimal_tokenizer_assets(
        output_path,
        vocab_size=model.config.vocab_size,
    )
    return str(output_path)


def _write_minimal_tokenizer_assets(output_path: Path, *, vocab_size: int) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    if vocab_size < 4:
        raise ValueError("vocab_size must be at least 4 to emit scaffold tokenizer assets")

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }
    for token_id in range(4, vocab_size):
        vocab[f"tok{token_id}"] = token_id

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast_tokenizer.model_max_length = 4096
    fast_tokenizer.save_pretrained(str(output_path))


def resolve_checkpoint_format(checkpoint_format, checkpoint_layout):
    if checkpoint_layout == "hf_dir":
        return "safetensors"
    return checkpoint_format


def _make_loader_model_config(checkpoint_path, config, dtype):
    return SimpleNamespace(
        model=checkpoint_path,
        hf_config=SimpleNamespace(**vars(config)),
        dtype=dtype,
        load_format="auto",
        download_dir=None,
        revision=None,
    )


def _load_model_via_model_loader(checkpoint_path, config, dtype):
    from sarathi.model_executor.model_loader import get_model

    return get_model(_make_loader_model_config(checkpoint_path, config, dtype))


class _NoOpModelRunnerWrapper:
    def init(self, model_config, parallel_config, block_size, device):
        del model_config, parallel_config, block_size
        self.device = device

    def begin_forward(self, seq_metadata_list):
        del seq_metadata_list

    def end_forward(self):
        return None


class _ModelRunnerSmokeConfig:
    def __init__(self, checkpoint_path, config, dtype):
        self.model = checkpoint_path
        self.hf_config = SimpleNamespace(**vars(config))
        self.dtype = dtype
        self.load_format = "auto"
        self.download_dir = None
        self.revision = None
        self.attention_backend = None
        self.seed = 0

    def get_num_q_heads(self, parallel_config):
        return self.hf_config.num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_kv_heads(self, parallel_config):
        return self.get_num_q_heads(parallel_config)

    def get_head_size(self):
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads


class _NullCpuTimer:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _run_model_runner_generation(
    checkpoint_path,
    config,
    dtype,
    *,
    runtime_mode,
    prompt_token_ids,
    max_new_tokens,
):
    import sarathi.model_executor.model_runner as model_runner_module
    from sarathi.model_executor.model_runner import ModelRunner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if runtime_mode == "paged":
        from sarathi.model_executor.attention.vattention_flashattention_wrapper import (
            VAttentionFlashAttentionWrapper,
        )

        attention_wrapper = VAttentionFlashAttentionWrapper()
        attention_wrapper.device = device
        attention_wrapper.is_metadata_initialized = True
        attention_wrapper.is_profiling_iteration = False
    elif runtime_mode == "contiguous":
        attention_wrapper = _NoOpModelRunnerWrapper()
    else:
        raise ValueError(f"Unsupported model runner runtime mode: {runtime_mode}")

    original_get_attention_wrapper = model_runner_module.get_attention_wrapper
    original_cpu_timer = model_runner_module.CpuTimer
    model_runner_module.get_attention_wrapper = lambda: attention_wrapper
    model_runner_module.CpuTimer = _NullCpuTimer
    try:
        model_config = _ModelRunnerSmokeConfig(checkpoint_path, config, dtype)
        model_config.attention_backend = (
            "FA_VATTN" if runtime_mode == "paged" else "NO_OP"
        )
        runner = ModelRunner(
            model_config=model_config,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(),
            cache_config=SimpleNamespace(block_size=16),
            device=device,
            rank=0,
        )
        gpu_cache = None
        if runtime_mode == "paged":
            gpu_cache = runner.model.make_runtime_mla_kv_caches(
                batch_size=1,
                max_seq_len=len(prompt_token_ids) + max_new_tokens + 1,
                device=device,
                dtype=dtype,
            )
        token_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device)
        return runner.run_greedy_generation(
            token_ids,
            max_new_tokens,
            gpu_cache=gpu_cache,
        )
    finally:
        model_runner_module.get_attention_wrapper = original_get_attention_wrapper
        model_runner_module.CpuTimer = original_cpu_timer


def _run_scaffold_smoke_artifacts(
    mode="contiguous",
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
    output_dir=None,
    use_model_loader=False,
):
    from sarathi.model_executor.parallel_utils.parallel_state import (
        set_pipeline_model_parallel_rank,
        set_pipeline_model_parallel_world_size,
        set_tensor_model_parallel_rank,
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
    set_tensor_model_parallel_rank(0)
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
    if output_dir is None:
        tempdir_ctx = tempfile.TemporaryDirectory()
        output_dir = tempdir_ctx.__enter__()
    else:
        tempdir_ctx = None
        os.makedirs(output_dir, exist_ok=True)
    try:
        checkpoint_format = resolve_checkpoint_format(
            checkpoint_format,
            checkpoint_layout,
        )
        if checkpoint_layout == "single_file":
            checkpoint_path = write_scaffold_checkpoint(
                model,
                projection_weights,
                mlp_weights,
                device=device,
                dtype=dtype,
                output_dir=output_dir,
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
                output_dir=output_dir,
                checkpoint_format=checkpoint_format,
                moe_weights=moe_weights,
            )
        else:
            raise ValueError(f"Unsupported checkpoint layout: {checkpoint_layout}")
        if use_model_loader:
            model = _load_model_via_model_loader(checkpoint_path, config, dtype)
        else:
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
        return generated_token_ids, final_logits, final_caches, checkpoint_path
    finally:
        if tempdir_ctx is not None:
            tempdir_ctx.__exit__(None, None, None)


def run_scaffold_smoke(
    mode="contiguous",
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
    output_dir=None,
):
    checkpoint_format = resolve_checkpoint_format(checkpoint_format, checkpoint_layout)
    generated_token_ids, final_logits, final_caches, checkpoint_path = _run_scaffold_smoke_artifacts(
        mode=mode,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
        output_dir=output_dir,
    )
    return {
        "mode": mode,
        "checkpoint_format": checkpoint_format,
        "query_mode": query_mode,
        "checkpoint_layout": checkpoint_layout,
        "mlp_mode": mlp_mode,
        "checkpoint_path": checkpoint_path if output_dir is not None else None,
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
    output_dir=None,
):
    checkpoint_format = resolve_checkpoint_format(checkpoint_format, checkpoint_layout)
    contiguous_tokens, contiguous_logits, contiguous_caches, checkpoint_path = _run_scaffold_smoke_artifacts(
        mode="contiguous",
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
        output_dir=output_dir,
    )
    try:
        paged_tokens, paged_logits, paged_caches, _ = _run_scaffold_smoke_artifacts(
            mode="paged",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            checkpoint_format=checkpoint_format,
            query_mode=query_mode,
            checkpoint_layout=checkpoint_layout,
            mlp_mode=mlp_mode,
            output_dir=output_dir,
        )
    except RuntimeError as exc:
        return {
            "mode": "compare",
            "checkpoint_format": checkpoint_format,
            "query_mode": query_mode,
            "checkpoint_layout": checkpoint_layout,
            "mlp_mode": mlp_mode,
            "checkpoint_path": checkpoint_path if output_dir is not None else None,
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
        "checkpoint_path": checkpoint_path if output_dir is not None else None,
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


def compare_loader_scaffold_smoke(
    runtime_mode="contiguous",
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
    output_dir=None,
):
    checkpoint_format = resolve_checkpoint_format(checkpoint_format, checkpoint_layout)
    direct_tokens, direct_logits, direct_caches, checkpoint_path = _run_scaffold_smoke_artifacts(
        mode=runtime_mode,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
        output_dir=output_dir,
        use_model_loader=False,
    )
    loader_tokens, loader_logits, loader_caches, _ = _run_scaffold_smoke_artifacts(
        mode=runtime_mode,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
        output_dir=output_dir,
        use_model_loader=True,
    )
    direct_cache_counts = [
        cache.num_tokens if hasattr(cache, "num_tokens") else cache.resident_cache.num_tokens
        for cache in direct_caches
    ]
    loader_cache_counts = [
        cache.num_tokens if hasattr(cache, "num_tokens") else cache.resident_cache.num_tokens
        for cache in loader_caches
    ]
    return {
        "mode": "loader_compare",
        "runtime_mode": runtime_mode,
        "checkpoint_format": checkpoint_format,
        "checkpoint_layout": checkpoint_layout,
        "query_mode": query_mode,
        "mlp_mode": mlp_mode,
        "checkpoint_path": checkpoint_path if output_dir is not None else None,
        "status": "ok",
        "prompt_token_ids": list(prompt_token_ids),
        "generated_token_ids": direct_tokens.tolist(),
        "loader_generated_token_ids": loader_tokens.tolist(),
        "generated_tokens_match": torch.equal(direct_tokens, loader_tokens),
        "final_logits_match": torch.allclose(
            direct_logits,
            loader_logits,
            atol=1e-6,
            rtol=1e-6,
        ),
        "direct_cache_token_counts": direct_cache_counts,
        "loader_cache_token_counts": loader_cache_counts,
        "cache_token_counts_match": direct_cache_counts == loader_cache_counts,
    }


def compare_model_runner_scaffold_smoke(
    runtime_mode="contiguous",
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
    output_dir=None,
):
    checkpoint_format = resolve_checkpoint_format(checkpoint_format, checkpoint_layout)
    if output_dir is None:
        tempdir_ctx = tempfile.TemporaryDirectory()
        output_dir = tempdir_ctx.__enter__()
    else:
        tempdir_ctx = None
        os.makedirs(output_dir, exist_ok=True)
    try:
        direct_tokens, direct_logits, direct_caches, checkpoint_path = _run_scaffold_smoke_artifacts(
            mode=runtime_mode,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            checkpoint_format=checkpoint_format,
            query_mode=query_mode,
            checkpoint_layout=checkpoint_layout,
            mlp_mode=mlp_mode,
            output_dir=output_dir,
            use_model_loader=True,
        )
        (
            runner_tokens,
            runner_logits,
            runner_caches,
        ) = _run_model_runner_generation(
            checkpoint_path,
            build_config(query_mode=query_mode, mlp_mode=mlp_mode),
            direct_logits.dtype,
            runtime_mode=runtime_mode,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
        )
        direct_cache_counts = [
            cache.num_tokens if hasattr(cache, "num_tokens") else cache.resident_cache.num_tokens
            for cache in direct_caches
        ]
        runner_cache_counts = [
            cache.num_tokens if hasattr(cache, "num_tokens") else cache.resident_cache.num_tokens
            for cache in runner_caches
        ]
        return {
            "mode": "runner_compare",
            "runtime_mode": runtime_mode,
            "checkpoint_format": checkpoint_format,
            "checkpoint_layout": checkpoint_layout,
            "query_mode": query_mode,
            "mlp_mode": mlp_mode,
            "checkpoint_path": checkpoint_path if output_dir is not None else None,
            "status": "ok",
            "prompt_token_ids": list(prompt_token_ids),
            "generated_token_ids": direct_tokens.tolist(),
            "runner_generated_token_ids": runner_tokens.tolist(),
            "generated_tokens_match": torch.equal(direct_tokens, runner_tokens),
            "final_logits_match": torch.allclose(
                direct_logits,
                runner_logits,
                atol=1e-6,
                rtol=1e-6,
            ),
            "direct_cache_token_counts": direct_cache_counts,
            "runner_cache_token_counts": runner_cache_counts,
            "cache_token_counts_match": direct_cache_counts == runner_cache_counts,
        }
    finally:
        if tempdir_ctx is not None:
            tempdir_ctx.__exit__(None, None, None)


def validate_scaffold_smoke_compare(
    prompt_token_ids=(1, 3),
    max_new_tokens=3,
    checkpoint_format="pt",
    query_mode="direct",
    checkpoint_layout="single_file",
    mlp_mode="dense",
    output_dir=None,
):
    result = compare_scaffold_smoke(
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        checkpoint_format=checkpoint_format,
        query_mode=query_mode,
        checkpoint_layout=checkpoint_layout,
        mlp_mode=mlp_mode,
        output_dir=output_dir,
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
        choices=("contiguous", "paged", "compare", "loader_compare", "runner_compare"),
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
        "--output-dir",
        default=None,
        help="optional directory where the emitted scaffold checkpoint artifacts should be kept",
    )
    parser.add_argument(
        "--loader-runtime-mode",
        choices=("contiguous", "paged"),
        default="contiguous",
        help="runtime path to compare when using loader_compare mode",
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
            output_dir=args.output_dir,
        )
    elif args.mode == "loader_compare":
        output = compare_loader_scaffold_smoke(
            runtime_mode=args.loader_runtime_mode,
            max_new_tokens=args.max_new_tokens,
            checkpoint_format=args.checkpoint_format,
            query_mode=args.query_mode,
            checkpoint_layout=args.checkpoint_layout,
            mlp_mode=args.mlp_mode,
            output_dir=args.output_dir,
        )
    elif args.mode == "runner_compare":
        output = compare_model_runner_scaffold_smoke(
            runtime_mode=args.loader_runtime_mode,
            max_new_tokens=args.max_new_tokens,
            checkpoint_format=args.checkpoint_format,
            query_mode=args.query_mode,
            checkpoint_layout=args.checkpoint_layout,
            mlp_mode=args.mlp_mode,
            output_dir=args.output_dir,
        )
    else:
        output = run_scaffold_smoke(
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            checkpoint_format=args.checkpoint_format,
            query_mode=args.query_mode,
            checkpoint_layout=args.checkpoint_layout,
            mlp_mode=args.mlp_mode,
            output_dir=args.output_dir,
        )
    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )
    if args.mode in ("compare", "loader_compare", "runner_compare") and args.require_match:
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
