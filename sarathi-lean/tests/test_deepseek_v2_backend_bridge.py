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
DeepseekV2MLAAttention = deepseek_module.DeepseekV2MLAAttention
DeepseekV2MLADims = deepseek_module.DeepseekV2MLADims
make_projection_weights = deepseek_module.make_projection_weights


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def __call__(self, query, key, value, cache, softmax_scale):
        self.calls.append(
            {
                "query": query.clone(),
                "key": key.clone(),
                "value": value.clone(),
                "cache": cache,
                "softmax_scale": softmax_scale,
            }
        )
        return value[-query.shape[0] :].reshape(query.shape[0], -1)


class DeepseekV2BackendBridgeTests(unittest.TestCase):
    def _make_config(self):
        return types.SimpleNamespace(
            hidden_size=6,
            num_attention_heads=8,
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

    def test_backend_bridge_passes_reconstructed_dense_tensors(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=4,
        )
        dims = DeepseekV2MLADims.from_config(self._make_config(), tensor_parallel_world_size=4)
        projection_weights = self._make_projection_weights(dims)
        backend = _RecordingBackend()
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        output, cache = attention.forward_hidden_states_with_backend(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            backend=backend,
        )

        self.assertEqual(tuple(output.shape), (2, dims.hidden_size))
        self.assertEqual(cache.num_tokens, 2)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(tuple(backend.calls[0]["query"].shape), (2, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(backend.calls[0]["key"].shape), (2, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(backend.calls[0]["value"].shape), (2, dims.num_heads, dims.v_head_dim))
        self.assertIsNone(backend.calls[0]["cache"])

    def test_backend_bridge_reuses_prior_resident_cache_for_decode(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=4,
        )
        dims = DeepseekV2MLADims.from_config(self._make_config(), tensor_parallel_world_size=4)
        projection_weights = self._make_projection_weights(dims)
        backend = _RecordingBackend()

        _, cache = attention.forward_hidden_states_with_backend(
            hidden_states=torch.tensor([[1.0, 2.0, 3.0, 0.0, 1.0, 0.0]]),
            projection_weights=projection_weights,
            backend=backend,
        )
        output, cache = attention.forward_hidden_states_with_backend(
            hidden_states=torch.tensor([[0.0, 1.0, 0.0, 2.0, 0.0, 1.0]]),
            projection_weights=projection_weights,
            backend=backend,
            cache=cache,
        )

        self.assertEqual(tuple(output.shape), (1, dims.hidden_size))
        self.assertEqual(cache.num_tokens, 2)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[1]["cache"].num_tokens, 1)
        self.assertEqual(tuple(backend.calls[1]["key"].shape), (2, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(backend.calls[1]["value"].shape), (2, dims.num_heads, dims.v_head_dim))


if __name__ == "__main__":
    unittest.main()
