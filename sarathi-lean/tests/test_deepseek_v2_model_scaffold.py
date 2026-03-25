import importlib.util
import sys
import tempfile
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
make_moe_weights = deepseek_module.make_moe_weights
make_projection_weights = deepseek_module.make_projection_weights


class DeepseekV2ModelScaffoldTests(unittest.TestCase):
    def _make_config(self):
        return types.SimpleNamespace(
            vocab_size=32000,
            hidden_size=5120,
            intermediate_size=12288,
            moe_intermediate_size=1408,
            num_attention_heads=128,
            num_hidden_layers=60,
            rms_norm_eps=1e-6,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

    def _make_small_config(self):
        return types.SimpleNamespace(
            vocab_size=16,
            hidden_size=6,
            intermediate_size=8,
            moe_intermediate_size=8,
            num_attention_heads=4,
            num_hidden_layers=4,
            rms_norm_eps=1e-6,
            q_lora_rank=None,
            kv_lora_rank=3,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
        )

    def _make_small_moe_config(self):
        config = self._make_small_config()
        config.first_k_dense_replace = 1
        config.n_routed_experts = 4
        config.n_shared_experts = 1
        return config

    def _make_small_multi_shared_moe_config(self):
        config = self._make_small_moe_config()
        config.moe_intermediate_size = 2
        config.n_shared_experts = 2
        return config

    def _make_projection_weights(self, dims):
        return make_projection_weights(
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
            kv_a_layernorm_weight=torch.ones(dims.kv_lora_rank),
            k_rope_proj=torch.tensor(
                [
                    [1.0],
                    [0.0],
                    [0.0],
                    [1.0],
                    [0.0],
                    [0.0],
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

    def test_causal_lm_scaffold_loader_accepts_global_layer_ids_for_last_stage(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(2)
        set_pipeline_model_parallel_rank(1)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        state_dict = {
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for global_layer_idx in range(model.model.layer_offset, model.model.layer_offset + model.model.num_layers):
            projection_weights = self._make_projection_weights(dims)
            prefix = f"model.layers.{global_layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + global_layer_idx
            state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                projection_weights.kv_latent_proj + global_layer_idx
            )
            state_dict[f"{prefix}.k_rope_proj.weight"] = (
                projection_weights.k_rope_proj + global_layer_idx
            )
            state_dict[f"{prefix}.kv_up_proj.weight"] = (
                projection_weights.kv_up_proj + global_layer_idx
            )
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + global_layer_idx

        model.load_weights(state_dict)

        self.assertIsNone(model.model.embed_tokens)
        self.assertIsNotNone(model.lm_head)
        self.assertEqual(len(model.model.layer_projection_weights), model.model.num_layers)
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].q_proj,
                self._make_projection_weights(dims).q_proj + model.model.layer_offset,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[1].q_proj,
                self._make_projection_weights(dims).q_proj + model.model.layer_offset + 1,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.norm.weight,
                torch.ones(config.hidden_size),
            )
        )

    def test_causal_lm_scaffold_loader_accepts_global_layer_ids_for_first_stage(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(2)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for global_layer_idx in range(model.model.layer_offset, model.model.layer_offset + model.model.num_layers):
            projection_weights = self._make_projection_weights(dims)
            state_dict[
                f"model.layers.{global_layer_idx}.input_layernorm.weight"
            ] = torch.full((config.hidden_size,), 2.0 + global_layer_idx)
            state_dict[
                f"model.layers.{global_layer_idx}.post_attention_layernorm.weight"
            ] = torch.full((config.hidden_size,), 3.0 + global_layer_idx)
            prefix = f"model.layers.{global_layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + global_layer_idx
            state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                projection_weights.kv_latent_proj + global_layer_idx
            )
            state_dict[f"{prefix}.k_rope_proj.weight"] = (
                projection_weights.k_rope_proj + global_layer_idx
            )
            state_dict[f"{prefix}.kv_up_proj.weight"] = (
                projection_weights.kv_up_proj + global_layer_idx
            )
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + global_layer_idx

        model.load_weights(state_dict)

        self.assertIsNotNone(model.model.embed_tokens)
        self.assertIsNone(model.lm_head)
        self.assertEqual(len(model.model.layer_projection_weights), model.model.num_layers)
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].q_proj,
                self._make_projection_weights(dims).q_proj,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layers[0].input_layernorm.weight,
                torch.full((config.hidden_size,), 2.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layers[0].post_attention_layernorm.weight,
                torch.full((config.hidden_size,), 3.0),
            )
        )

    def test_causal_lm_scaffold_loader_accepts_bare_layer_prefixes(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "norm.weight": torch.full((config.hidden_size,), 1.5),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            projection_weights = self._make_projection_weights(dims)
            state_dict[f"layers.{layer_idx}.input_layernorm.weight"] = torch.full(
                (config.hidden_size,),
                2.0 + layer_idx,
            )
            state_dict[f"layers.{layer_idx}.post_attention_layernorm.weight"] = torch.full(
                (config.hidden_size,),
                3.0 + layer_idx,
            )
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                projection_weights.kv_latent_proj + layer_idx
            )
            state_dict[f"{prefix}.k_rope_proj.weight"] = (
                projection_weights.k_rope_proj + layer_idx
            )
            state_dict[f"{prefix}.kv_up_proj.weight"] = (
                projection_weights.kv_up_proj + layer_idx
            )
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx
            mlp_prefix = f"layers.{layer_idx}.mlp"
            mlp_weights = make_mlp_weights(
                gate_proj=torch.full((config.hidden_size, 4), 1.0 + layer_idx),
                up_proj=torch.full((config.hidden_size, 4), 2.0 + layer_idx),
                down_proj=torch.full((4, config.hidden_size), 3.0 + layer_idx),
                hidden_size=config.hidden_size,
            )
            state_dict[f"{mlp_prefix}.gate_proj.weight"] = mlp_weights.gate_proj
            state_dict[f"{mlp_prefix}.up_proj.weight"] = mlp_weights.up_proj
            state_dict[f"{mlp_prefix}.down_proj.weight"] = mlp_weights.down_proj

        model.load_weights(state_dict)

        self.assertTrue(
            torch.allclose(
                model.model.norm.weight,
                torch.full((config.hidden_size,), 1.5),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layers[0].input_layernorm.weight,
                torch.full((config.hidden_size,), 2.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layers[1].post_attention_layernorm.weight,
                torch.full((config.hidden_size,), 4.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[1].q_proj,
                self._make_projection_weights(dims).q_proj + 1,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_mlp_weights[0].gate_proj,
                torch.full((config.hidden_size, 4), 1.0),
            )
        )

    def test_causal_lm_scaffold_loader_accepts_combined_deepseek_mla_projection_names(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        combined_kv_a = torch.cat(
            [
                projection_weights.kv_latent_proj,
                projection_weights.k_rope_proj,
            ],
            dim=1,
        )
        state_dict = {
            "model.embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "model.norm.weight": torch.ones(config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            prefix = f"model.layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = combined_kv_a + layer_idx
            state_dict[f"{prefix}.kv_b_proj.weight"] = projection_weights.kv_up_proj + layer_idx
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx

        model.load_weights(state_dict)

        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].kv_latent_proj,
                projection_weights.kv_latent_proj,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].k_rope_proj,
                projection_weights.k_rope_proj,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[1].kv_up_proj,
                projection_weights.kv_up_proj + 1,
            )
        )

    def test_causal_lm_scaffold_loader_accepts_kv_a_layernorm_alias(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                projection_weights.kv_latent_proj + layer_idx
            )
            state_dict[f"{prefix}.kv_a_layernorm.weight"] = torch.full(
                (dims.kv_lora_rank,),
                1.0 + layer_idx,
            )
            state_dict[f"{prefix}.k_rope_proj.weight"] = (
                projection_weights.k_rope_proj + layer_idx
            )
            state_dict[f"{prefix}.kv_up_proj.weight"] = (
                projection_weights.kv_up_proj + layer_idx
            )
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx

        model.load_weights(state_dict)

        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].kv_a_layernorm_weight,
                torch.full((dims.kv_lora_rank,), 1.0),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[1].kv_a_layernorm_weight,
                torch.full((dims.kv_lora_rank,), 2.0),
            )
        )

    def test_causal_lm_scaffold_loader_accepts_q_lora_query_aliases(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        config.q_lora_rank = 2
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        base_projection_weights = self._make_projection_weights(dims)
        q_a_proj = torch.full((config.hidden_size, config.q_lora_rank), 1.0)
        q_a_layernorm_weight = torch.tensor([1.0, 2.0])
        q_b_proj = torch.full((config.q_lora_rank, dims.q_proj_output_dim_local), 0.5)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_a_proj.weight"] = q_a_proj + layer_idx
            state_dict[f"{prefix}.q_a_layernorm.weight"] = q_a_layernorm_weight + layer_idx
            state_dict[f"{prefix}.q_b_proj.weight"] = q_b_proj + layer_idx
            state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                base_projection_weights.kv_latent_proj + layer_idx
            )
            state_dict[f"{prefix}.k_rope_proj.weight"] = (
                base_projection_weights.k_rope_proj + layer_idx
            )
            state_dict[f"{prefix}.kv_up_proj.weight"] = (
                base_projection_weights.kv_up_proj + layer_idx
            )
            state_dict[f"{prefix}.o_proj.weight"] = base_projection_weights.o_proj + layer_idx

        model.load_weights(state_dict)

        self.assertIsNone(model.model.layer_projection_weights[0].q_proj)
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].q_a_proj,
                q_a_proj,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[1].q_b_proj,
                q_b_proj + 1,
            )
        )

    def test_causal_lm_scaffold_loader_accepts_local_checkpoint_directory(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = torch.cat(
                [
                    projection_weights.kv_latent_proj + layer_idx,
                    projection_weights.k_rope_proj + layer_idx,
                ],
                dim=1,
            )
            state_dict[f"{prefix}.kv_b_proj.weight"] = projection_weights.kv_up_proj + layer_idx
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "weights.pt"
            torch.save(state_dict, checkpoint_path)
            model.load_weights(tmpdir)

        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].kv_latent_proj,
                projection_weights.kv_latent_proj,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[1].k_rope_proj,
                projection_weights.k_rope_proj + 1,
            )
        )

    def test_causal_lm_loader_accepts_standard_loader_signature_for_local_path(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = torch.cat(
                [
                    projection_weights.kv_latent_proj + layer_idx,
                    projection_weights.k_rope_proj + layer_idx,
                ],
                dim=1,
            )
            state_dict[f"{prefix}.kv_b_proj.weight"] = projection_weights.kv_up_proj + layer_idx
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "weights.pt"
            torch.save(state_dict, checkpoint_path)
            model.load_weights(tmpdir, None, "auto", None)

        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].q_proj,
                projection_weights.q_proj,
            )
        )

    def test_causal_lm_loader_accepts_hf_oriented_linear_weights(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_moe_config()
        config.num_experts_per_tok = 1
        config.norm_topk_prob = True
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        dense_mlp = make_mlp_weights(
            gate_proj=torch.ones(config.hidden_size, 4),
            up_proj=torch.ones(config.hidden_size, 4) * 2.0,
            down_proj=torch.ones(4, config.hidden_size) * 3.0,
            hidden_size=config.hidden_size,
        )
        state_dict = {
            "model.embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "model.norm.weight": torch.ones(config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            attn_prefix = f"model.layers.{layer_idx}.self_attn"
            state_dict[f"{attn_prefix}.q_proj.weight"] = (
                projection_weights.q_proj + layer_idx
            ).t().contiguous()
            state_dict[f"{attn_prefix}.kv_a_proj_with_mqa.weight"] = torch.cat(
                [
                    projection_weights.kv_latent_proj + layer_idx,
                    projection_weights.k_rope_proj + layer_idx,
                ],
                dim=1,
            ).t().contiguous()
            state_dict[f"{attn_prefix}.kv_b_proj.weight"] = (
                projection_weights.kv_up_proj + layer_idx
            ).t().contiguous()
            state_dict[f"{attn_prefix}.o_proj.weight"] = (
                projection_weights.o_proj + layer_idx
            ).t().contiguous()
            mlp_prefix = f"model.layers.{layer_idx}.mlp"
            if layer_idx < config.first_k_dense_replace:
                state_dict[f"{mlp_prefix}.gate_proj.weight"] = dense_mlp.gate_proj.t().contiguous()
                state_dict[f"{mlp_prefix}.up_proj.weight"] = dense_mlp.up_proj.t().contiguous()
                state_dict[f"{mlp_prefix}.down_proj.weight"] = dense_mlp.down_proj.t().contiguous()
            else:
                state_dict[f"{mlp_prefix}.gate.weight"] = torch.zeros(
                    config.n_routed_experts,
                    config.hidden_size,
                )
                state_dict[f"{mlp_prefix}.shared_experts.gate_proj.weight"] = torch.ones(
                    4,
                    config.hidden_size,
                )
                state_dict[f"{mlp_prefix}.shared_experts.up_proj.weight"] = torch.ones(
                    4,
                    config.hidden_size,
                ) * 2.0
                state_dict[f"{mlp_prefix}.shared_experts.down_proj.weight"] = torch.ones(
                    config.hidden_size,
                    4,
                ) * 3.0
                for expert_idx in range(config.n_routed_experts):
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.gate_proj.weight"] = torch.full(
                        (4, config.hidden_size),
                        1.0 + expert_idx,
                    )
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.up_proj.weight"] = torch.full(
                        (4, config.hidden_size),
                        2.0 + expert_idx,
                    )
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.down_proj.weight"] = torch.full(
                        (config.hidden_size, 4),
                        3.0 + expert_idx,
                    )

        model.load_weights(state_dict)

        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].kv_up_proj,
                projection_weights.kv_up_proj,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_projection_weights[0].o_proj,
                projection_weights.o_proj,
            )
        )
        self.assertTrue(
            torch.allclose(
                model.model.layer_mlp_weights[0].gate_proj,
                dense_mlp.gate_proj,
            )
        )
        self.assertIsNotNone(model.model.layer_moe_weights[1])
        self.assertEqual(
            model.model.layer_moe_weights[1].experts[0].gate_proj.shape,
            (config.hidden_size, 4),
        )

    def test_causal_lm_loader_slices_global_attention_weights_by_tensor_parallel_rank(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_rank,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        rank0_projection_weights = self._make_projection_weights(dims)
        rank1_projection_weights = make_projection_weights(
            q_proj=rank0_projection_weights.q_proj + 100.0,
            kv_latent_proj=rank0_projection_weights.kv_latent_proj + 200.0,
            kv_a_layernorm_weight=rank0_projection_weights.kv_a_layernorm_weight + 300.0,
            k_rope_proj=rank0_projection_weights.k_rope_proj + 400.0,
            kv_up_proj=rank0_projection_weights.kv_up_proj + 500.0,
            o_proj=rank0_projection_weights.o_proj + 600.0,
            mla_dims=dims,
        )
        combined_global_kv_a = torch.cat(
            [
                rank0_projection_weights.kv_latent_proj,
                rank0_projection_weights.k_rope_proj,
            ],
            dim=1,
        )
        state_dict = {
            "model.embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "model.norm.weight": torch.ones(config.hidden_size),
        }
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = torch.cat(
                [
                    rank0_projection_weights.q_proj,
                    rank1_projection_weights.q_proj,
                ],
                dim=1,
            )
            state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = combined_global_kv_a
            state_dict[f"{prefix}.kv_b_proj.weight"] = torch.cat(
                [
                    rank0_projection_weights.kv_up_proj,
                    rank1_projection_weights.kv_up_proj,
                ],
                dim=1,
            )
            state_dict[f"{prefix}.o_proj.weight"] = torch.cat(
                [
                    rank0_projection_weights.o_proj,
                    rank1_projection_weights.o_proj,
                ],
                dim=0,
            )

        for rank, expected_projection_weights in (
            (0, rank0_projection_weights),
            (1, rank1_projection_weights),
        ):
            set_tensor_model_parallel_rank(rank)
            model = DeepseekV2ForCausalLM(config)
            model.load_weights(state_dict)

            self.assertTrue(
                torch.allclose(
                    model.model.layer_projection_weights[0].q_proj,
                    expected_projection_weights.q_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_projection_weights[0].kv_latent_proj,
                    rank0_projection_weights.kv_latent_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_projection_weights[0].k_rope_proj,
                    rank0_projection_weights.k_rope_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_projection_weights[0].kv_up_proj,
                    expected_projection_weights.kv_up_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_projection_weights[0].o_proj,
                    expected_projection_weights.o_proj,
                )
            )

    def test_causal_lm_loader_slices_global_mlp_and_moe_weights_by_tensor_parallel_rank(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_rank,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_moe_config()
        config.num_experts_per_tok = 1
        config.norm_topk_prob = True
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        rank0_dense_mlp = make_mlp_weights(
            gate_proj=torch.full((config.hidden_size, 4), 1.0),
            up_proj=torch.full((config.hidden_size, 4), 2.0),
            down_proj=torch.full((4, config.hidden_size), 3.0),
            hidden_size=config.hidden_size,
        )
        rank1_dense_mlp = make_mlp_weights(
            gate_proj=rank0_dense_mlp.gate_proj + 10.0,
            up_proj=rank0_dense_mlp.up_proj + 20.0,
            down_proj=rank0_dense_mlp.down_proj + 30.0,
            hidden_size=config.hidden_size,
        )
        rank0_shared_mlp = make_mlp_weights(
            gate_proj=torch.full((config.hidden_size, 4), 4.0),
            up_proj=torch.full((config.hidden_size, 4), 5.0),
            down_proj=torch.full((4, config.hidden_size), 6.0),
            hidden_size=config.hidden_size,
        )
        rank1_shared_mlp = make_mlp_weights(
            gate_proj=rank0_shared_mlp.gate_proj + 10.0,
            up_proj=rank0_shared_mlp.up_proj + 20.0,
            down_proj=rank0_shared_mlp.down_proj + 30.0,
            hidden_size=config.hidden_size,
        )
        rank0_expert_mlps = tuple(
            make_mlp_weights(
                gate_proj=torch.full((config.hidden_size, 4), 10.0 + expert_idx),
                up_proj=torch.full((config.hidden_size, 4), 20.0 + expert_idx),
                down_proj=torch.full((4, config.hidden_size), 30.0 + expert_idx),
                hidden_size=config.hidden_size,
            )
            for expert_idx in range(config.n_routed_experts)
        )
        rank1_expert_mlps = tuple(
            make_mlp_weights(
                gate_proj=expert_mlp.gate_proj + 100.0,
                up_proj=expert_mlp.up_proj + 200.0,
                down_proj=expert_mlp.down_proj + 300.0,
                hidden_size=config.hidden_size,
            )
            for expert_mlp in rank0_expert_mlps
        )
        routed_gate = torch.arange(
            config.n_routed_experts * config.hidden_size,
            dtype=torch.float32,
        ).view(config.n_routed_experts, config.hidden_size)
        state_dict = {
            "model.embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "model.norm.weight": torch.ones(config.hidden_size),
        }
        for layer_idx in range(config.num_hidden_layers):
            attn_prefix = f"model.layers.{layer_idx}.self_attn"
            state_dict[f"{attn_prefix}.q_proj.weight"] = projection_weights.q_proj
            state_dict[f"{attn_prefix}.kv_a_proj_with_mqa.weight"] = torch.cat(
                [
                    projection_weights.kv_latent_proj,
                    projection_weights.k_rope_proj,
                ],
                dim=1,
            )
            state_dict[f"{attn_prefix}.kv_b_proj.weight"] = projection_weights.kv_up_proj
            state_dict[f"{attn_prefix}.o_proj.weight"] = projection_weights.o_proj
            mlp_prefix = f"model.layers.{layer_idx}.mlp"
            if layer_idx < config.first_k_dense_replace:
                state_dict[f"{mlp_prefix}.gate_proj.weight"] = torch.cat(
                    [rank0_dense_mlp.gate_proj, rank1_dense_mlp.gate_proj],
                    dim=1,
                )
                state_dict[f"{mlp_prefix}.up_proj.weight"] = torch.cat(
                    [rank0_dense_mlp.up_proj, rank1_dense_mlp.up_proj],
                    dim=1,
                )
                state_dict[f"{mlp_prefix}.down_proj.weight"] = torch.cat(
                    [rank0_dense_mlp.down_proj, rank1_dense_mlp.down_proj],
                    dim=0,
                )
            else:
                state_dict[f"{mlp_prefix}.gate.weight"] = routed_gate
                state_dict[f"{mlp_prefix}.shared_experts.gate_proj.weight"] = torch.cat(
                    [rank0_shared_mlp.gate_proj, rank1_shared_mlp.gate_proj],
                    dim=1,
                )
                state_dict[f"{mlp_prefix}.shared_experts.up_proj.weight"] = torch.cat(
                    [rank0_shared_mlp.up_proj, rank1_shared_mlp.up_proj],
                    dim=1,
                )
                state_dict[f"{mlp_prefix}.shared_experts.down_proj.weight"] = torch.cat(
                    [rank0_shared_mlp.down_proj, rank1_shared_mlp.down_proj],
                    dim=0,
                )
                for expert_idx in range(config.n_routed_experts):
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.gate_proj.weight"] = torch.cat(
                        [
                            rank0_expert_mlps[expert_idx].gate_proj,
                            rank1_expert_mlps[expert_idx].gate_proj,
                        ],
                        dim=1,
                    )
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.up_proj.weight"] = torch.cat(
                        [
                            rank0_expert_mlps[expert_idx].up_proj,
                            rank1_expert_mlps[expert_idx].up_proj,
                        ],
                        dim=1,
                    )
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.down_proj.weight"] = torch.cat(
                        [
                            rank0_expert_mlps[expert_idx].down_proj,
                            rank1_expert_mlps[expert_idx].down_proj,
                        ],
                        dim=0,
                    )

        for rank, expected_dense_mlp, expected_shared_mlp, expected_expert_mlps in (
            (0, rank0_dense_mlp, rank0_shared_mlp, rank0_expert_mlps),
            (1, rank1_dense_mlp, rank1_shared_mlp, rank1_expert_mlps),
        ):
            set_tensor_model_parallel_rank(rank)
            model = DeepseekV2ForCausalLM(config)
            model.load_weights(state_dict)

            self.assertTrue(
                torch.allclose(
                    model.model.layer_mlp_weights[0].gate_proj,
                    expected_dense_mlp.gate_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_mlp_weights[0].up_proj,
                    expected_dense_mlp.up_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_mlp_weights[0].down_proj,
                    expected_dense_mlp.down_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].gate,
                    routed_gate,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].shared_experts.gate_proj,
                    expected_shared_mlp.gate_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].shared_experts.down_proj,
                    expected_shared_mlp.down_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].experts[0].up_proj,
                    expected_expert_mlps[0].up_proj,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].experts[-1].down_proj,
                    expected_expert_mlps[-1].down_proj,
                )
            )

    def test_causal_lm_loader_rejects_incomplete_moe_layer_weights(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_moe_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        for layer_idx in range(model.model.num_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = torch.cat(
                [
                    projection_weights.kv_latent_proj + layer_idx,
                    projection_weights.k_rope_proj + layer_idx,
                ],
                dim=1,
            )
            state_dict[f"{prefix}.kv_b_proj.weight"] = projection_weights.kv_up_proj + layer_idx
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx
        state_dict["layers.1.mlp.gate.weight"] = torch.zeros(
            config.n_routed_experts,
            config.hidden_size,
        )
        state_dict["layers.1.mlp.shared_experts.gate_proj.weight"] = torch.zeros(
            config.hidden_size,
            4,
        )
        state_dict["layers.1.mlp.experts.0.gate_proj.weight"] = torch.zeros(
            config.hidden_size,
            4,
        )

        with self.assertRaises(KeyError):
            model.load_weights(state_dict)

    def test_causal_lm_loader_accepts_bounded_moe_layer_weights(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_moe_config()
        config.num_experts_per_tok = 1
        config.norm_topk_prob = True
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        dense_mlp = make_mlp_weights(
            gate_proj=torch.ones(config.hidden_size, 4),
            up_proj=torch.ones(config.hidden_size, 4),
            down_proj=torch.ones(4, config.hidden_size),
            hidden_size=config.hidden_size,
        )
        for layer_idx in range(model.model.num_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = torch.cat(
                [
                    projection_weights.kv_latent_proj + layer_idx,
                    projection_weights.k_rope_proj + layer_idx,
                ],
                dim=1,
            )
            state_dict[f"{prefix}.kv_b_proj.weight"] = projection_weights.kv_up_proj + layer_idx
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx
            mlp_prefix = f"layers.{layer_idx}.mlp"
            if layer_idx < config.first_k_dense_replace:
                state_dict[f"{mlp_prefix}.gate_proj.weight"] = dense_mlp.gate_proj
                state_dict[f"{mlp_prefix}.up_proj.weight"] = dense_mlp.up_proj
                state_dict[f"{mlp_prefix}.down_proj.weight"] = dense_mlp.down_proj
            else:
                state_dict[f"{mlp_prefix}.gate.weight"] = torch.zeros(
                    config.n_routed_experts,
                    config.hidden_size,
                )
                state_dict[f"{mlp_prefix}.shared_experts.gate_proj.weight"] = torch.ones(
                    config.hidden_size,
                    4,
                )
                state_dict[f"{mlp_prefix}.shared_experts.up_proj.weight"] = torch.ones(
                    config.hidden_size,
                    4,
                )
                state_dict[f"{mlp_prefix}.shared_experts.down_proj.weight"] = torch.ones(
                    4,
                    config.hidden_size,
                )
                for expert_idx in range(config.n_routed_experts):
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.gate_proj.weight"] = torch.full(
                        (config.hidden_size, 4),
                        1.0 + expert_idx,
                    )
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.up_proj.weight"] = torch.full(
                        (config.hidden_size, 4),
                        2.0 + expert_idx,
                    )
                    state_dict[f"{mlp_prefix}.experts.{expert_idx}.down_proj.weight"] = torch.full(
                        (4, config.hidden_size),
                        3.0 + expert_idx,
                    )

        model.load_weights(state_dict)

        self.assertIsNotNone(model.model.layer_mlp_weights[0])
        self.assertIsNone(model.model.layer_moe_weights[0])
        self.assertIsNone(model.model.layer_mlp_weights[1])
        self.assertIsNotNone(model.model.layer_moe_weights[1])
        self.assertEqual(
            model.model.layer_moe_weights[1].gate.shape,
            (config.n_routed_experts, config.hidden_size),
        )
        self.assertEqual(
            len(model.model.layer_moe_weights[1].experts),
            config.n_routed_experts,
        )

    def test_causal_lm_loader_slices_multi_shared_expert_width_by_tensor_parallel_rank(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_rank,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_small_multi_shared_moe_config()
        config.num_experts_per_tok = 1
        config.norm_topk_prob = True
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        shared_width = config.moe_intermediate_size * config.n_shared_experts
        local_shared_width = shared_width // 2
        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        state_dict = {
            "embed_tokens.weight": torch.zeros(config.vocab_size, config.hidden_size),
            "lm_head.weight": torch.zeros(config.vocab_size, config.hidden_size),
        }
        rank0_shared_gate = torch.full((config.hidden_size, local_shared_width), 4.0)
        rank1_shared_gate = torch.full((config.hidden_size, local_shared_width), 14.0)
        rank0_shared_up = torch.full((config.hidden_size, local_shared_width), 5.0)
        rank1_shared_up = torch.full((config.hidden_size, local_shared_width), 15.0)
        rank0_shared_down = torch.full((local_shared_width, config.hidden_size), 6.0)
        rank1_shared_down = torch.full((local_shared_width, config.hidden_size), 16.0)

        for layer_idx in range(model.model.num_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = projection_weights.q_proj + layer_idx
            state_dict[f"{prefix}.kv_a_proj_with_mqa.weight"] = torch.cat(
                [
                    projection_weights.kv_latent_proj + layer_idx,
                    projection_weights.k_rope_proj + layer_idx,
                ],
                dim=1,
            )
            state_dict[f"{prefix}.kv_b_proj.weight"] = projection_weights.kv_up_proj + layer_idx
            state_dict[f"{prefix}.o_proj.weight"] = projection_weights.o_proj + layer_idx

            mlp_prefix = f"layers.{layer_idx}.mlp"
            if layer_idx < config.first_k_dense_replace:
                state_dict[f"{mlp_prefix}.gate_proj.weight"] = torch.ones(
                    config.hidden_size,
                    4,
                )
                state_dict[f"{mlp_prefix}.up_proj.weight"] = torch.ones(
                    config.hidden_size,
                    4,
                )
                state_dict[f"{mlp_prefix}.down_proj.weight"] = torch.ones(
                    4,
                    config.hidden_size,
                )
                continue

            state_dict[f"{mlp_prefix}.gate.weight"] = torch.zeros(
                config.n_routed_experts,
                config.hidden_size,
            )
            state_dict[f"{mlp_prefix}.shared_experts.gate_proj.weight"] = torch.cat(
                [rank0_shared_gate, rank1_shared_gate],
                dim=1,
            ).t().contiguous()
            state_dict[f"{mlp_prefix}.shared_experts.up_proj.weight"] = torch.cat(
                [rank0_shared_up, rank1_shared_up],
                dim=1,
            ).t().contiguous()
            state_dict[f"{mlp_prefix}.shared_experts.down_proj.weight"] = torch.cat(
                [rank0_shared_down, rank1_shared_down],
                dim=0,
            ).t().contiguous()
            for expert_idx in range(config.n_routed_experts):
                state_dict[f"{mlp_prefix}.experts.{expert_idx}.gate_proj.weight"] = torch.full(
                    (config.hidden_size, config.moe_intermediate_size),
                    1.0 + expert_idx,
                )
                state_dict[f"{mlp_prefix}.experts.{expert_idx}.up_proj.weight"] = torch.full(
                    (config.hidden_size, config.moe_intermediate_size),
                    2.0 + expert_idx,
                )
                state_dict[f"{mlp_prefix}.experts.{expert_idx}.down_proj.weight"] = torch.full(
                    (config.moe_intermediate_size, config.hidden_size),
                    3.0 + expert_idx,
                )

        for rank, expected_gate, expected_up, expected_down in (
            (0, rank0_shared_gate, rank0_shared_up, rank0_shared_down),
            (1, rank1_shared_gate, rank1_shared_up, rank1_shared_down),
        ):
            set_tensor_model_parallel_rank(rank)
            model = DeepseekV2ForCausalLM(config)
            model.load_weights(state_dict)

            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].shared_experts.gate_proj,
                    expected_gate,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].shared_experts.up_proj,
                    expected_up,
                )
            )
            self.assertTrue(
                torch.allclose(
                    model.model.layer_moe_weights[1].shared_experts.down_proj,
                    expected_down,
                )
            )


if __name__ == "__main__":
    unittest.main()
