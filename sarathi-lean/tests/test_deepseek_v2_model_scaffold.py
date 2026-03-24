import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SARATHI_ROOT = REPO_ROOT / "sarathi-lean" / "sarathi"


def _ensure_package(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_deepseek_model_module():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.model_executor", SARATHI_ROOT / "model_executor")
    _ensure_package(
        "sarathi.model_executor.parallel_utils",
        SARATHI_ROOT / "model_executor" / "parallel_utils",
    )
    _load_module(
        "sarathi.model_executor.parallel_utils.parallel_state",
        SARATHI_ROOT / "model_executor" / "parallel_utils" / "parallel_state.py",
    )
    return _load_module(
        "sarathi.model_executor.models.deepseek_v2",
        SARATHI_ROOT / "model_executor" / "models" / "deepseek_v2.py",
    )


deepseek_module = _load_deepseek_model_module()
DeepseekV2MLADims = deepseek_module.DeepseekV2MLADims
DeepseekV2MLAAttention = deepseek_module.DeepseekV2MLAAttention
DeepseekV2Model = deepseek_module.DeepseekV2Model
DeepseekV2ForCausalLM = deepseek_module.DeepseekV2ForCausalLM
make_mlp_weights = deepseek_module.make_mlp_weights


class DeepseekV2ModelScaffoldTests(unittest.TestCase):
    def _make_config(self):
        return types.SimpleNamespace(
            vocab_size=32000,
            hidden_size=5120,
            num_attention_heads=128,
            num_hidden_layers=60,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

    def test_mla_dims_compute_local_tensor_parallel_shapes(self):
        dims = DeepseekV2MLADims.from_config(
            self._make_config(),
            tensor_parallel_world_size=4,
        )

        self.assertEqual(dims.tensor_parallel_world_size, 4)
        self.assertEqual(dims.total_num_heads, 128)
        self.assertEqual(dims.num_heads, 32)
        self.assertEqual(dims.q_head_dim, 192)
        self.assertEqual(dims.q_proj_output_dim_local, 32 * 192)
        self.assertEqual(dims.kv_up_proj_output_dim_local, 32 * (128 + 128))
        self.assertEqual(dims.o_proj_input_dim_local, 32 * 128)
        self.assertEqual(dims.resident_cache_dim, 512 + 64)

    def test_mla_dims_reject_non_divisible_tensor_parallel_heads(self):
        config = self._make_config()
        config.num_attention_heads = 130

        with self.assertRaises(ValueError):
            DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=4)

    def test_attention_module_captures_mla_dims(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=4,
        )

        self.assertEqual(attention.mla_dims.num_heads, 32)
        self.assertEqual(attention.mla_dims.kv_lora_rank, 512)
        self.assertEqual(attention.mla_dims.qk_rope_head_dim, 64)

    def test_model_partitions_layers_by_pipeline_rank(self):
        model = DeepseekV2Model(
            self._make_config(),
            tensor_parallel_world_size=4,
            pipeline_parallel_world_size=3,
            pipeline_parallel_rank=1,
        )

        self.assertEqual(model.tensor_parallel_world_size, 4)
        self.assertEqual(model.pipeline_parallel_world_size, 3)
        self.assertEqual(model.pipeline_parallel_rank, 1)
        self.assertEqual(model.num_layers, 20)
        self.assertEqual(model.layer_offset, 20)
        self.assertEqual(len(model.layers), 20)
        self.assertIsInstance(model.layers[0], deepseek_module.DeepseekV2DecoderLayer)

    def test_model_rejects_non_divisible_pipeline_partition(self):
        config = self._make_config()
        config.num_hidden_layers = 61

        with self.assertRaises(ValueError):
            DeepseekV2Model(
                config,
                tensor_parallel_world_size=4,
                pipeline_parallel_world_size=3,
                pipeline_parallel_rank=0,
            )

    def test_causal_lm_exposes_model_and_dims(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        set_tensor_model_parallel_world_size(1)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(self._make_config())

        self.assertIsInstance(model.model, DeepseekV2Model)
        self.assertEqual(model.mla_dims.num_heads, 128)
        self.assertEqual(model.model.num_layers, 60)
        self.assertIsNotNone(model.model.embed_tokens)
        self.assertIsNotNone(model.lm_head)

    def test_model_rejects_token_ids_without_first_stage_embeddings(self):
        config = self._make_config()
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=4,
            pipeline_parallel_world_size=2,
            pipeline_parallel_rank=1,
        )

        with self.assertRaises(ValueError):
            model(
                hidden_states=torch.tensor([1, 2], dtype=torch.long),
                projection_weights=tuple(
                    model.layers[0].self_attn.make_projection_weights(
                        q_proj=torch.zeros(config.hidden_size, model.layers[0].self_attn.mla_dims.q_proj_output_dim_local),
                        kv_latent_proj=torch.zeros(config.hidden_size, model.layers[0].self_attn.mla_dims.kv_lora_rank),
                        k_rope_proj=torch.zeros(config.hidden_size, model.layers[0].self_attn.mla_dims.num_heads * model.layers[0].self_attn.mla_dims.qk_rope_head_dim),
                        kv_up_proj=torch.zeros(model.layers[0].self_attn.mla_dims.kv_lora_rank, model.layers[0].self_attn.mla_dims.kv_up_proj_output_dim_local),
                        o_proj=torch.zeros(model.layers[0].self_attn.mla_dims.o_proj_input_dim_local, config.hidden_size),
                    )
                    for _ in range(model.num_layers)
                ),
            )

    def test_model_rejects_forward_without_projection_or_installed_weights(self):
        config = self._make_config()
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=4,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )

        with self.assertRaises(ValueError):
            model(hidden_states=torch.zeros(2, config.hidden_size))

    def test_causal_lm_scaffold_loader_rejects_missing_projection_weight(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        set_tensor_model_parallel_world_size(1)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(self._make_config())

        with self.assertRaises(KeyError):
            model.load_weights(
                {
                    "model.embed_tokens.weight": torch.zeros(
                        model.config.vocab_size, model.config.hidden_size
                    ),
                    "lm_head.weight": torch.zeros(
                        model.config.vocab_size, model.config.hidden_size
                    ),
                }
            )

    def test_make_mlp_weights_rejects_invalid_down_projection_shape(self):
        hidden_size = 8

        with self.assertRaises(ValueError):
            make_mlp_weights(
                gate_proj=torch.zeros(hidden_size, 4),
                up_proj=torch.zeros(hidden_size, 4),
                down_proj=torch.zeros(5, hidden_size),
                hidden_size=hidden_size,
            )


if __name__ == "__main__":
    unittest.main()
