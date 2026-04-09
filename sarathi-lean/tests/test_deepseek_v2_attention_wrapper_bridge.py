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
DeepseekV2DecoderLayer = deepseek_module.DeepseekV2DecoderLayer
DeepseekV2LayerCache = deepseek_module.DeepseekV2LayerCache
DeepseekV2MLAWrapperInputs = deepseek_module.DeepseekV2MLAWrapperInputs
DeepseekV2Model = deepseek_module.DeepseekV2Model
DeepseekV2ForCausalLM = deepseek_module.DeepseekV2ForCausalLM
make_layer_cache = deepseek_module.make_layer_cache
make_projection_weights = deepseek_module.make_projection_weights
prepare_mla_wrapper_inputs = deepseek_module.prepare_mla_wrapper_inputs


class _RecordingAttentionWrapper:
    def __init__(self):
        self.calls = []

    def forward(self, query, key, value, kv_cache, softmax_scale=1.0, layer_id=None):
        self.calls.append(
            {
                "query": query.clone(),
                "key": key.clone(),
                "value": value.clone(),
                "kv_cache": kv_cache,
                "softmax_scale": softmax_scale,
                "layer_id": layer_id,
            }
        )
        return value[-query.shape[0] :].clone()


class _RecordingMLAAttentionWrapper:
    def __init__(self):
        self.calls = []
        self.dense_forward_called = False

    def forward(self, *args, **kwargs):
        self.dense_forward_called = True
        raise AssertionError("dense fallback should not be used when forward_mla is available")

    def forward_mla(self, wrapper_inputs):
        self.calls.append(wrapper_inputs)
        full_cache = deepseek_module.append_resident_cache(
            wrapper_inputs.past_resident_cache,
            wrapper_inputs.new_resident_cache,
        )
        _, value = deepseek_module.reconstruct_dense_kv(
            full_cache,
            wrapper_inputs.kv_up_proj_weight,
            wrapper_inputs.mla_dims,
        )
        return value[-wrapper_inputs.query.shape[0] :].reshape(wrapper_inputs.query.shape[0], -1)


class DeepseekV2AttentionWrapperBridgeTests(unittest.TestCase):
    def _make_config(self):
        return types.SimpleNamespace(
            hidden_size=6,
            num_attention_heads=4,
            num_hidden_layers=4,
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

    def test_attention_wrapper_bridge_flattens_dense_local_qkv(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        attention = DeepseekV2MLAAttention(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        wrapper = _RecordingAttentionWrapper()
        kv_cache = object()

        output, cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            kv_cache=kv_cache,
            layer_id=7,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertIsInstance(cache, DeepseekV2LayerCache)
        self.assertIs(cache.kv_cache, kv_cache)
        self.assertEqual(cache.resident_cache.num_tokens, 2)
        self.assertEqual(len(wrapper.calls), 1)
        self.assertEqual(
            tuple(wrapper.calls[0]["query"].shape),
            (2, dims.num_heads * dims.q_head_dim),
        )
        self.assertEqual(
            tuple(wrapper.calls[0]["key"].shape),
            (2, dims.num_heads * dims.q_head_dim),
        )
        self.assertEqual(
            tuple(wrapper.calls[0]["value"].shape),
            (2, dims.o_proj_input_dim_local),
        )
        self.assertIs(wrapper.calls[0]["kv_cache"], kv_cache)
        self.assertEqual(wrapper.calls[0]["layer_id"], 7)

    def test_attention_wrapper_bridge_accepts_combined_layer_cache_for_decode(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        attention = DeepseekV2MLAAttention(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        wrapper = _RecordingAttentionWrapper()
        kv_cache = object()

        _, cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=self._make_hidden_states()[:1],
            projection_weights=projection_weights,
            kv_cache=kv_cache,
            layer_id=3,
            attention_wrapper=wrapper,
        )
        output, cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=self._make_hidden_states()[1:],
            projection_weights=projection_weights,
            kv_cache=cache,
            layer_id=3,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (1, config.hidden_size))
        self.assertIs(cache.kv_cache, kv_cache)
        self.assertEqual(cache.resident_cache.num_tokens, 2)
        self.assertEqual(len(wrapper.calls), 2)
        self.assertIs(wrapper.calls[1]["kv_cache"], kv_cache)

    def test_prepare_mla_wrapper_inputs_exposes_resident_cache_components(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        kv_cache = object()
        hidden_states = self._make_hidden_states()[:1]

        wrapper_inputs, cache = prepare_mla_wrapper_inputs(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mla_dims=dims,
            kv_cache=kv_cache,
            layer_id=5,
        )

        self.assertIsInstance(wrapper_inputs, DeepseekV2MLAWrapperInputs)
        self.assertEqual(tuple(wrapper_inputs.query.shape), (1, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(wrapper_inputs.new_resident_cache.kv_latent.shape), (1, dims.kv_lora_rank))
        self.assertEqual(
            tuple(wrapper_inputs.new_resident_cache.k_rope.shape),
            (1, dims.num_heads, dims.qk_rope_head_dim),
        )
        self.assertIsNone(wrapper_inputs.past_resident_cache)
        self.assertIs(wrapper_inputs.kv_cache, kv_cache)
        self.assertEqual(wrapper_inputs.layer_id, 5)
        self.assertEqual(cache.num_tokens, 1)

    def test_attention_wrapper_bridge_prefers_forward_mla_when_available(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        attention = DeepseekV2MLAAttention(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        wrapper = _RecordingMLAAttentionWrapper()
        kv_cache = object()

        output, layer_cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            kv_cache=kv_cache,
            layer_id=9,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertFalse(wrapper.dense_forward_called)
        self.assertEqual(len(wrapper.calls), 1)
        self.assertIs(wrapper.calls[0].kv_cache, kv_cache)
        self.assertEqual(wrapper.calls[0].layer_id, 9)
        self.assertEqual(layer_cache.resident_cache.num_tokens, 2)

    def test_attention_wrapper_bridge_passes_combined_layer_cache_to_forward_mla(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        attention = DeepseekV2MLAAttention(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        wrapper = _RecordingMLAAttentionWrapper()
        kv_cache = object()

        _, layer_cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=self._make_hidden_states()[:1],
            projection_weights=projection_weights,
            kv_cache=kv_cache,
            layer_id=13,
            attention_wrapper=wrapper,
        )
        output, next_layer_cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=self._make_hidden_states()[1:],
            projection_weights=projection_weights,
            kv_cache=layer_cache,
            layer_id=13,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (1, config.hidden_size))
        self.assertEqual(len(wrapper.calls), 2)
        self.assertIsInstance(wrapper.calls[1].kv_cache, DeepseekV2LayerCache)
        self.assertIs(wrapper.calls[1].kv_cache.kv_cache, kv_cache)
        self.assertEqual(wrapper.calls[1].kv_cache.resident_cache.num_tokens, 1)
        self.assertEqual(next_layer_cache.resident_cache.num_tokens, 2)

    def test_decoder_layer_threads_layer_id_into_attention_wrapper(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        layer = DeepseekV2DecoderLayer(
            config,
            layer_id=11,
            tensor_parallel_world_size=2,
        )
        projection_weights = self._make_projection_weights(dims)
        wrapper = _RecordingAttentionWrapper()
        kv_cache = object()

        output, cache = layer.forward_with_attention_wrapper(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            kv_cache=kv_cache,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertIsInstance(cache, DeepseekV2LayerCache)
        self.assertEqual(cache.resident_cache.num_tokens, 2)
        self.assertEqual(len(wrapper.calls), 1)
        self.assertEqual(wrapper.calls[0]["layer_id"], 11)
        self.assertIs(wrapper.calls[0]["kv_cache"], kv_cache)

    def test_model_wrapper_forward_uses_per_layer_kv_cache_and_layer_ids(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        model = DeepseekV2Model(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=2,
            pipeline_parallel_rank=1,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.num_layers)
        )
        kv_caches = (object(), object())
        wrapper = _RecordingAttentionWrapper()

        output, caches = model.forward_with_attention_wrapper(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            kv_caches=kv_caches,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertEqual(len(caches), model.num_layers)
        self.assertTrue(all(isinstance(cache, DeepseekV2LayerCache) for cache in caches))
        self.assertTrue(all(cache.resident_cache.num_tokens == 2 for cache in caches))
        self.assertEqual([call["layer_id"] for call in wrapper.calls], [2, 3])
        self.assertEqual([call["kv_cache"] for call in wrapper.calls], list(kv_caches))

    def test_causal_lm_wrapper_forward_reuses_combined_layer_caches_on_decode(self):
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
        kv_caches = (object(), object())
        wrapper = _RecordingAttentionWrapper()
        hidden_states = self._make_hidden_states()

        _, caches = model.forward_with_attention_wrapper(
            hidden_states=hidden_states[:1],
            projection_weights=projection_weights,
            kv_caches=kv_caches,
            attention_wrapper=wrapper,
        )
        output, caches = model.forward_with_attention_wrapper(
            hidden_states=hidden_states[1:],
            projection_weights=projection_weights,
            kv_caches=caches,
            attention_wrapper=wrapper,
        )

        self.assertEqual(tuple(output.shape), (1, config.hidden_size))
        self.assertTrue(all(isinstance(cache, DeepseekV2LayerCache) for cache in caches))
        self.assertTrue(all(cache.resident_cache.num_tokens == 2 for cache in caches))
        self.assertEqual(len(wrapper.calls), 4)

    def test_make_layer_cache_preserves_raw_kv_cache_identity(self):
        kv_cache = object()
        layer_cache = make_layer_cache(kv_cache)

        self.assertIs(layer_cache.kv_cache, kv_cache)
        self.assertIsNone(layer_cache.resident_cache)


if __name__ == "__main__":
    unittest.main()
