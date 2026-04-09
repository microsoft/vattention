"""Utilities for selecting and loading models."""

import contextlib
import os
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sarathi.config import ModelConfig
from sarathi.model_executor.models import *  # pylint: disable=wildcard-import
from sarathi.model_executor.weight_utils import initialize_dummy_weights

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "InternLMForCausalLM": InternLMForCausalLM,
    "MistralForCausalLM": MistralForCausalLM,
    "MistralMLAForCausalLM": MistralMLAForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "YiForCausalLM": YiForCausalLM,
}


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}"
    )


def get_model(model_config: ModelConfig) -> nn.Module:
    if (
        os.environ.get("VATTN_ENABLE_MISTRAL_MLA_CONVERSION") == "1"
        and getattr(model_config.hf_config, "model_type", None) == "mistral"
    ):
        model_config.hf_config.architectures = ["MistralMLAForCausalLM"]
        model_config.hf_config.source_model_name = model_config.model
        model_config.hf_config.q_lora_rank = None
        model_config.hf_config.kv_lora_rank = int(
            os.environ.get("VATTN_MISTRAL_MLA_KV_LORA_RANK", "128")
        )
        model_config.hf_config.qk_rope_head_dim = int(
            os.environ.get("VATTN_MISTRAL_MLA_QK_ROPE_HEAD_DIM", "64")
        )
        default_qk_nope = max(
            0,
            model_config.get_head_size() - model_config.hf_config.qk_rope_head_dim,
        )
        model_config.hf_config.qk_nope_head_dim = int(
            os.environ.get(
                "VATTN_MISTRAL_MLA_QK_NOPE_HEAD_DIM",
                str(default_qk_nope),
            )
        )
        model_config.hf_config.v_head_dim = int(
            os.environ.get(
                "VATTN_MISTRAL_MLA_V_HEAD_DIM",
                str(model_config.get_head_size()),
            )
        )

    model_class = _get_model_architecture(model_config.hf_config)
    if model_config.model == '01-ai/Yi-34B':
        model_config.hf_config.hidden_size = 8192
        model_config.hf_config.num_attention_heads = 64
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            model = model_class(model_config.hf_config)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(
                model_config.model,
                model_config.download_dir,
                model_config.load_format,
                model_config.revision,
            )
    return model.eval()
