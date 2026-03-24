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
DeepseekV2DecoderLayer = deepseek_module.DeepseekV2DecoderLayer
DeepseekV2Model = deepseek_module.DeepseekV2Model
DeepseekV2ForCausalLM = deepseek_module.DeepseekV2ForCausalLM
make_mlp_weights = deepseek_module.make_mlp_weights
make_projection_weights = deepseek_module.make_projection_weights


class DeepseekV2ModelForwardTests(unittest.TestCase):
    def _make_config(self):
        return types.SimpleNamespace(
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

    def _make_hidden_states(self):
        return torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

    def _set_embedding_and_lm_head_weights(self, model):
        embedding_weight = torch.arange(
            model.config.vocab_size * model.config.hidden_size,
            dtype=torch.float32,
        ).view(model.config.vocab_size, model.config.hidden_size) / 1000.0
        model.model.embed_tokens.weight.data.copy_(embedding_weight)
        if model.lm_head is not None:
            lm_head_weight = torch.arange(
                model.config.vocab_size * model.config.hidden_size,
                dtype=torch.float32,
            ).view(model.config.vocab_size, model.config.hidden_size) / 1000.0
            model.lm_head.weight.data.copy_(lm_head_weight)

    def _make_scaffold_state_dict(
        self,
        model,
        projection_weights,
        mlp_weights=None,
        *,
        norm_scale=1.0,
    ):
        state_dict = {}
        if model.model.embed_tokens is not None:
            state_dict["model.embed_tokens.weight"] = (
                model.model.embed_tokens.weight.detach().clone()
            )
        if model.model.norm is not None:
            state_dict["model.norm.weight"] = torch.full(
                (model.config.hidden_size,),
                norm_scale,
            )
        if model.lm_head is not None:
            state_dict["lm_head.weight"] = model.lm_head.weight.detach().clone()
        for layer_idx, layer_projection_weights in enumerate(projection_weights):
            state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = torch.full(
                (model.config.hidden_size,),
                norm_scale + layer_idx,
            )
            state_dict[
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            ] = torch.full(
                (model.config.hidden_size,),
                norm_scale + layer_idx + 0.5,
            )
            prefix = f"model.layers.{layer_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = layer_projection_weights.q_proj.detach().clone()
            state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                layer_projection_weights.kv_latent_proj.detach().clone()
            )
            state_dict[f"{prefix}.k_rope_proj.weight"] = (
                layer_projection_weights.k_rope_proj.detach().clone()
            )
            state_dict[f"{prefix}.kv_up_proj.weight"] = (
                layer_projection_weights.kv_up_proj.detach().clone()
            )
            state_dict[f"{prefix}.o_proj.weight"] = layer_projection_weights.o_proj.detach().clone()
        if mlp_weights is not None:
            for layer_idx, layer_mlp_weights in enumerate(mlp_weights):
                if layer_mlp_weights is None:
                    continue
                prefix = f"model.layers.{layer_idx}.mlp"
                state_dict[f"{prefix}.gate_proj.weight"] = (
                    layer_mlp_weights.gate_proj.detach().clone()
                )
                state_dict[f"{prefix}.up_proj.weight"] = (
                    layer_mlp_weights.up_proj.detach().clone()
                )
                state_dict[f"{prefix}.down_proj.weight"] = (
                    layer_mlp_weights.down_proj.detach().clone()
                )
        return state_dict

    def _make_mlp_weights(self, hidden_size):
        return make_mlp_weights(
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

    def test_decoder_layer_runs_attention_only_reference_forward(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        layer = DeepseekV2DecoderLayer(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)

        output, cache = layer(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertEqual(cache.num_tokens, 2)
        self.assertTrue(torch.isfinite(output).all())

    def test_decoder_layer_applies_mlp_block_when_weights_are_provided(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        layer = DeepseekV2DecoderLayer(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        mlp_weights = self._make_mlp_weights(config.hidden_size)
        hidden_states = self._make_hidden_states()

        baseline_output, baseline_cache = layer(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
        )
        mlp_output, mlp_cache = layer(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )

        self.assertEqual(tuple(mlp_output.shape), (2, config.hidden_size))
        self.assertEqual(mlp_cache.num_tokens, 2)
        self.assertTrue(torch.isfinite(mlp_output).all())
        self.assertTrue(torch.allclose(baseline_cache.kv_latent, mlp_cache.kv_latent))
        self.assertFalse(torch.allclose(mlp_output, baseline_output))

    def test_model_forward_runs_all_local_layers_and_returns_cache_tuple(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=2,
            pipeline_parallel_rank=0,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.num_layers)
        )

        output, caches = model(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
        )

        self.assertEqual(model.num_layers, 2)
        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertEqual(len(caches), model.num_layers)
        self.assertTrue(all(cache.num_tokens == 2 for cache in caches))

    def test_model_forward_applies_layerwise_mlp_weights(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=2,
            pipeline_parallel_rank=0,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.num_layers)
        )

        baseline_output, baseline_caches = model(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
        )
        mlp_output, mlp_caches = model(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )

        self.assertEqual(tuple(mlp_output.shape), (2, config.hidden_size))
        self.assertEqual(len(mlp_caches), model.num_layers)
        self.assertTrue(all(cache.num_tokens == 2 for cache in mlp_caches))
        self.assertFalse(torch.allclose(mlp_output, baseline_output))
        self.assertTrue(
            all(
                baseline_cache.kv_latent.shape == mlp_cache.kv_latent.shape
                and baseline_cache.k_rope.shape == mlp_cache.k_rope.shape
                for baseline_cache, mlp_cache in zip(baseline_caches, mlp_caches)
            )
        )

    def test_model_forward_reuses_layer_caches_on_decode_step(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=2,
            pipeline_parallel_rank=0,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.num_layers)
        )
        hidden_states = self._make_hidden_states()

        first_output, caches = model(
            hidden_states=hidden_states[:1],
            projection_weights=projection_weights,
        )
        second_output, caches = model(
            hidden_states=hidden_states[1:],
            projection_weights=projection_weights,
            caches=caches,
        )

        self.assertEqual(tuple(first_output.shape), (1, config.hidden_size))
        self.assertEqual(tuple(second_output.shape), (1, config.hidden_size))
        self.assertTrue(all(cache.num_tokens == 2 for cache in caches))

    def test_model_forward_uses_installed_scaffold_weights(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=2,
            pipeline_parallel_rank=0,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.num_layers)
        )
        model.set_scaffold_weights(
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )

        output, caches = model(hidden_states=self._make_hidden_states())

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertEqual(len(caches), model.num_layers)
        self.assertTrue(all(cache.num_tokens == 2 for cache in caches))

    def test_model_forward_with_attention_wrapper_applies_mlp_weights(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=2,
            pipeline_parallel_rank=0,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.num_layers)
        )

        class _Wrapper:
            def forward(self, query, key, value, kv_cache, softmax_scale=1.0, layer_id=None):
                return value[-query.shape[0] :].clone()

        wrapper = _Wrapper()
        kv_caches = tuple(object() for _ in range(model.num_layers))
        baseline_output, baseline_caches = model.forward_with_attention_wrapper(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            kv_caches=kv_caches,
            attention_wrapper=wrapper,
        )
        mlp_output, mlp_caches = model.forward_with_attention_wrapper(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            kv_caches=kv_caches,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(mlp_output.shape), (2, config.hidden_size))
        self.assertEqual(len(mlp_caches), model.num_layers)
        self.assertTrue(all(cache.resident_cache.num_tokens == 2 for cache in mlp_caches))
        self.assertFalse(torch.allclose(mlp_output, baseline_output))
        self.assertTrue(
            all(
                baseline_cache.resident_cache.kv_latent.shape
                == mlp_cache.resident_cache.kv_latent.shape
                and baseline_cache.resident_cache.k_rope.shape
                == mlp_cache.resident_cache.k_rope.shape
                for baseline_cache, mlp_cache in zip(baseline_caches, mlp_caches)
            )
        )

    def test_causal_lm_forward_delegates_to_model(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(2)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.model.num_layers)
        )

        output, caches = model(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertEqual(len(caches), model.model.num_layers)

    def test_causal_lm_accepts_token_ids_via_embedding_path(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(2)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.model.num_layers)
        )
        input_ids = torch.tensor([1, 3], dtype=torch.long)

        embedded_hidden_states = model.model.embed_tokens(input_ids)
        baseline_output, baseline_caches = model(
            hidden_states=embedded_hidden_states,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )
        output, caches = model(
            hidden_states=input_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )

        self.assertTrue(torch.allclose(output, baseline_output))
        self.assertEqual(len(caches), len(baseline_caches))
        self.assertTrue(all(cache.num_tokens == 2 for cache in caches))

    def test_causal_lm_uses_installed_scaffold_weights_for_wrapper_style_forward(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model.set_scaffold_weights(
            projection_weights=tuple(
                self._make_projection_weights(dims) for _ in range(model.model.num_layers)
            ),
            mlp_weights=tuple(
                self._make_mlp_weights(config.hidden_size)
                for _ in range(model.model.num_layers)
            ),
        )

        class _Wrapper:
            def forward(self, query, key, value, kv_cache, softmax_scale=1.0, layer_id=None):
                return value[-query.shape[0] :].clone()

        wrapper = _Wrapper()
        input_ids = torch.tensor([1, 2], dtype=torch.long)
        kv_caches = tuple(object() for _ in range(model.model.num_layers))

        output, caches = model(
            hidden_states=input_ids,
            kv_caches=kv_caches,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertEqual(len(caches), model.model.num_layers)
        self.assertTrue(all(cache.resident_cache.num_tokens == 2 for cache in caches))

    def test_causal_lm_loads_scaffold_state_dict_for_token_forward(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        reference_model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(reference_model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = tuple(
            self._make_projection_weights(dims)
            for _ in range(reference_model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size)
            for _ in range(reference_model.model.num_layers)
        )
        reference_model.set_scaffold_weights(
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )
        scaffold_state_dict = self._make_scaffold_state_dict(
            reference_model,
            projection_weights,
            mlp_weights,
        )
        loaded_model = DeepseekV2ForCausalLM(config)
        loaded_model.load_weights(scaffold_state_dict)

        input_ids = torch.tensor([1, 4], dtype=torch.long)
        loaded_output, loaded_caches = loaded_model(hidden_states=input_ids)
        loaded_logits, _ = loaded_model.forward_logits(hidden_states=input_ids)

        self.assertTrue(
            torch.allclose(
                loaded_model.model.embed_tokens.weight,
                scaffold_state_dict["model.embed_tokens.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded_model.lm_head.weight,
                scaffold_state_dict["lm_head.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded_model.model.layer_projection_weights[0].q_proj,
                scaffold_state_dict["model.layers.0.self_attn.q_proj.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded_model.model.layer_mlp_weights[0].gate_proj,
                scaffold_state_dict["model.layers.0.mlp.gate_proj.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded_model.model.layers[0].input_layernorm.weight,
                scaffold_state_dict["model.layers.0.input_layernorm.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded_model.model.layers[0].post_attention_layernorm.weight,
                scaffold_state_dict["model.layers.0.post_attention_layernorm.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                loaded_model.model.norm.weight,
                scaffold_state_dict["model.norm.weight"],
            )
        )
        self.assertEqual(tuple(loaded_output.shape), (2, config.hidden_size))
        self.assertEqual(tuple(loaded_logits.shape), (2, config.vocab_size))
        self.assertTrue(all(cache.num_tokens == 2 for cache in loaded_caches))

    def test_loaded_norm_weights_change_scaffold_forward_output(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        base_model = DeepseekV2ForCausalLM(config)
        scaled_model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(base_model)
        self._set_embedding_and_lm_head_weights(scaled_model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = tuple(
            self._make_projection_weights(dims)
            for _ in range(base_model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size)
            for _ in range(base_model.model.num_layers)
        )
        base_model.load_weights(
            self._make_scaffold_state_dict(
                base_model,
                projection_weights,
                mlp_weights,
                norm_scale=1.0,
            )
        )
        scaled_model.load_weights(
            self._make_scaffold_state_dict(
                scaled_model,
                projection_weights,
                mlp_weights,
                norm_scale=2.0,
            )
        )

        input_ids = torch.tensor([1, 4], dtype=torch.long)
        base_output, _ = base_model(hidden_states=input_ids)
        scaled_output, _ = scaled_model(hidden_states=input_ids)

        self.assertEqual(tuple(base_output.shape), (2, config.hidden_size))
        self.assertEqual(tuple(scaled_output.shape), (2, config.hidden_size))
        self.assertFalse(torch.allclose(base_output, scaled_output))

    def test_causal_lm_forward_logits_projects_hidden_states_to_vocab(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.model.num_layers)
        )
        input_ids = torch.tensor([2, 4], dtype=torch.long)

        hidden_states, caches = model(
            hidden_states=input_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )
        logits, logits_caches = model.forward_logits(
            hidden_states=input_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
        )

        self.assertEqual(tuple(logits.shape), (2, config.vocab_size))
        self.assertEqual(tuple(model.compute_logits(hidden_states).shape), (2, config.vocab_size))
        self.assertEqual(len(logits_caches), len(caches))

    def test_causal_lm_prefill_and_decode_tokens_match_full_contiguous_logits(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model.set_scaffold_weights(
            projection_weights=tuple(
                self._make_projection_weights(dims) for _ in range(model.model.num_layers)
            ),
            mlp_weights=tuple(
                self._make_mlp_weights(config.hidden_size)
                for _ in range(model.model.num_layers)
            ),
        )

        prompt_token_ids = torch.tensor([1, 3], dtype=torch.long)
        decode_token_ids = torch.tensor([5], dtype=torch.long)
        full_token_ids = torch.tensor([1, 3, 5], dtype=torch.long)

        prefill_logits, caches = model.prefill_tokens(prompt_token_ids)
        decode_logits, decode_caches = model.decode_tokens(
            decode_token_ids,
            caches=caches,
        )
        full_logits, full_caches = model.forward_logits(hidden_states=full_token_ids)

        self.assertEqual(tuple(prefill_logits.shape), (2, config.vocab_size))
        self.assertEqual(tuple(decode_logits.shape), (1, config.vocab_size))
        self.assertTrue(torch.allclose(decode_logits[0], full_logits[-1], atol=1e-6, rtol=1e-6))
        self.assertTrue(all(cache.num_tokens == 2 for cache in caches))
        self.assertTrue(all(cache.num_tokens == 3 for cache in decode_caches))
        self.assertTrue(
            all(
                torch.equal(decode_cache.kv_latent, full_cache.kv_latent)
                and torch.equal(decode_cache.k_rope, full_cache.k_rope)
                for decode_cache, full_cache in zip(decode_caches, full_caches)
            )
        )

    def test_causal_lm_decode_tokens_requires_non_empty_integer_decode_input_and_caches(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)

        with self.assertRaises(ValueError):
            model.prefill_tokens(torch.tensor([[1]], dtype=torch.long))

        with self.assertRaises(ValueError):
            model.prefill_tokens(torch.tensor([], dtype=torch.long))

        with self.assertRaises(ValueError):
            model.decode_tokens(torch.tensor([1.0]), caches=(object(),))

        with self.assertRaises(ValueError):
            model.decode_tokens(torch.tensor([1], dtype=torch.long), caches=None)

    def test_causal_lm_generate_greedy_matches_manual_contiguous_decode_loop(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model.set_scaffold_weights(
            projection_weights=tuple(
                self._make_projection_weights(dims) for _ in range(model.model.num_layers)
            ),
            mlp_weights=tuple(
                self._make_mlp_weights(config.hidden_size)
                for _ in range(model.model.num_layers)
            ),
        )

        prompt_token_ids = torch.tensor([1, 3], dtype=torch.long)
        generated_tokens, final_logits, final_caches = model.generate_greedy(
            prompt_token_ids,
            max_new_tokens=3,
        )

        manual_logits, manual_caches = model.prefill_tokens(prompt_token_ids)
        manual_tokens = []
        next_token = torch.argmax(manual_logits[-1], dim=-1).to(dtype=prompt_token_ids.dtype).view(1)
        manual_tokens.append(next_token)
        for _ in range(2):
            manual_logits, manual_caches = model.decode_tokens(
                next_token,
                caches=manual_caches,
            )
            next_token = torch.argmax(manual_logits[-1], dim=-1).to(dtype=prompt_token_ids.dtype).view(1)
            manual_tokens.append(next_token)

        self.assertEqual(tuple(generated_tokens.shape), (3,))
        self.assertTrue(torch.equal(generated_tokens, torch.cat(manual_tokens, dim=0)))
        self.assertTrue(torch.allclose(final_logits, manual_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(all(cache.num_tokens == 4 for cache in final_caches))
        self.assertTrue(
            all(
                torch.equal(final_cache.kv_latent, manual_cache.kv_latent)
                and torch.equal(final_cache.k_rope, manual_cache.k_rope)
                for final_cache, manual_cache in zip(final_caches, manual_caches)
            )
        )

    def test_causal_lm_generate_greedy_validates_max_new_tokens(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)

        with self.assertRaises(ValueError):
            model.generate_greedy(torch.tensor([1], dtype=torch.long), max_new_tokens=-1)

    def test_causal_lm_forward_logits_with_attention_wrapper_accepts_token_ids(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = DeepseekV2ForCausalLM(config)
        self._set_embedding_and_lm_head_weights(model)
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.model.num_layers)
        )

        class _Wrapper:
            def forward(self, query, key, value, kv_cache, softmax_scale=1.0, layer_id=None):
                return value[-query.shape[0] :].clone()

        wrapper = _Wrapper()
        input_ids = torch.tensor([1, 5], dtype=torch.long)
        kv_caches = tuple(object() for _ in range(model.model.num_layers))

        hidden_states, layer_caches = model.forward_with_attention_wrapper(
            hidden_states=input_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            kv_caches=kv_caches,
            attention_wrapper=wrapper,
        )
        logits, logits_caches = model.forward_logits_with_attention_wrapper(
            hidden_states=input_ids,
            projection_weights=projection_weights,
            mlp_weights=mlp_weights,
            kv_caches=kv_caches,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(logits.shape), (2, config.vocab_size))
        self.assertEqual(tuple(model.compute_logits(hidden_states).shape), (2, config.vocab_size))
        self.assertEqual(len(logits_caches), len(layer_caches))
        self.assertTrue(all(cache.resident_cache.num_tokens == 2 for cache in logits_caches))


if __name__ == "__main__":
    unittest.main()
