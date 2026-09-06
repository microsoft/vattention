import unittest

import torch
from transformers import MistralConfig

from sarathi.model_executor.models.mistral_mla import MistralMLAForCausalLM


class MistralMLAConversionTests(unittest.TestCase):
    def test_build_scaffold_state_dict_produces_expected_attention_shapes(self):
        config = MistralConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        config.architectures = ["MistralMLAForCausalLM"]
        config.q_lora_rank = None
        config.kv_lora_rank = 24
        config.qk_nope_head_dim = 16
        config.qk_rope_head_dim = 16
        config.v_head_dim = 32

        model = MistralMLAForCausalLM(
            config,
            tensor_parallel_world_size=1,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )

        state_dict = {
            "model.embed_tokens.weight": torch.randn(256, 128),
            "model.norm.weight": torch.randn(128),
            "lm_head.weight": torch.randn(256, 128),
        }
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(128)
            state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(128)
            state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(128, 128)
            state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(64, 128)
            state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(64, 128)
            state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(128, 128)
            state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(256, 128)
            state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(256, 128)
            state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(128, 256)

        scaffold = model._build_scaffold_state_dict(state_dict)

        self.assertEqual(
            tuple(scaffold["model.layers.0.self_attn.q_proj.weight"].shape),
            (128, 128),
        )
        self.assertEqual(
            tuple(scaffold["model.layers.0.self_attn.kv_latent_proj.weight"].shape),
            (128, 24),
        )
        self.assertEqual(
            tuple(scaffold["model.layers.0.self_attn.k_rope_proj.weight"].shape),
            (128, 16),
        )
        self.assertEqual(
            tuple(scaffold["model.layers.0.self_attn.kv_up_proj.weight"].shape),
            (24, 192),
        )
        self.assertEqual(
            tuple(scaffold["model.layers.0.self_attn.o_proj.weight"].shape),
            (128, 128),
        )


if __name__ == "__main__":
    unittest.main()
