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
make_projection_weights = deepseek_module.make_projection_weights


class DeepseekV2ModelForwardTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
