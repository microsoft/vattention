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
append_resident_cache = deepseek_module.append_resident_cache
make_resident_cache = deepseek_module.make_resident_cache
reconstruct_dense_kv = deepseek_module.reconstruct_dense_kv
split_query_projection = deepseek_module.split_query_projection


class DeepseekV2MLAHelperTests(unittest.TestCase):
    def _make_config(self):
        return types.SimpleNamespace(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=6,
            q_lora_rank=None,
            kv_lora_rank=3,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
        )

    def _make_dims(self):
        return DeepseekV2MLADims.from_config(
            self._make_config(),
            tensor_parallel_world_size=2,
        )

    def test_split_query_projection_splits_nope_and_rope_components(self):
        dims = self._make_dims()
        query_states = torch.arange(12, dtype=torch.float32).view(2, 6)

        q_nope, q_rope = split_query_projection(query_states, dims)

        self.assertEqual(tuple(q_nope.shape), (2, 2, 2))
        self.assertEqual(tuple(q_rope.shape), (2, 2, 1))
        self.assertTrue(torch.equal(q_nope[0], torch.tensor([[0.0, 1.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(q_rope[0], torch.tensor([[2.0], [5.0]])))

    def test_make_resident_cache_validates_local_shapes(self):
        dims = self._make_dims()
        kv_latent = torch.randn(3, dims.kv_lora_rank)
        k_rope = torch.randn(3, dims.num_heads, dims.qk_rope_head_dim)

        cache = make_resident_cache(kv_latent, k_rope, dims)

        self.assertEqual(cache.num_tokens, 3)
        self.assertTrue(torch.equal(cache.kv_latent, kv_latent))
        self.assertTrue(torch.equal(cache.k_rope, k_rope))

    def test_append_resident_cache_concatenates_component_state(self):
        dims = self._make_dims()
        first = make_resident_cache(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[[10.0], [11.0]]]),
            dims,
        )
        second = make_resident_cache(
            torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            torch.tensor([[[12.0], [13.0]], [[14.0], [15.0]]]),
            dims,
        )

        merged = append_resident_cache(first, second)

        self.assertEqual(tuple(merged.kv_latent.shape), (3, 3))
        self.assertEqual(tuple(merged.k_rope.shape), (3, 2, 1))
        self.assertTrue(
            torch.equal(
                merged.kv_latent,
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
            )
        )

    def test_reconstruct_dense_kv_combines_latent_projection_and_rope_cache(self):
        dims = self._make_dims()
        cache = make_resident_cache(
            torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]),
            torch.tensor(
                [
                    [[100.0], [200.0]],
                    [[300.0], [400.0]],
                ]
            ),
            dims,
        )
        kv_up_proj_weight = torch.tensor(
            [
                [1.0, 0.0, 10.0, 20.0, 2.0, 0.0, 30.0, 40.0],
                [0.0, 1.0, 11.0, 21.0, 0.0, 2.0, 31.0, 41.0],
                [1.0, 1.0, 12.0, 22.0, 2.0, 2.0, 32.0, 42.0],
            ]
        )

        key, value = reconstruct_dense_kv(cache, kv_up_proj_weight, dims)

        self.assertEqual(tuple(key.shape), (2, 2, 3))
        self.assertEqual(tuple(value.shape), (2, 2, 2))
        expected_key_token0 = torch.tensor([[4.0, 5.0, 100.0], [8.0, 10.0, 200.0]])
        expected_value_token0 = torch.tensor([[68.0, 128.0], [188.0, 248.0]])
        self.assertTrue(torch.equal(key[0], expected_key_token0))
        self.assertTrue(torch.equal(value[0], expected_value_token0))

    def test_attention_helper_methods_wrap_shared_functions(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=2,
        )
        query_states = torch.arange(6, dtype=torch.float32).view(1, 6)
        kv_latent = torch.ones(1, 3)
        k_rope = torch.ones(1, 2, 1)
        kv_up_proj_weight = torch.ones(3, 8)

        q_nope, q_rope = attention.split_query_projection(query_states)
        cache = attention.make_resident_cache(kv_latent, k_rope)
        key, value = attention.reconstruct_dense_kv(cache, kv_up_proj_weight)

        self.assertEqual(tuple(q_nope.shape), (1, 2, 2))
        self.assertEqual(tuple(q_rope.shape), (1, 2, 1))
        self.assertEqual(tuple(key.shape), (1, 2, 3))
        self.assertEqual(tuple(value.shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
