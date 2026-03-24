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
contiguous_mla_attention_forward = deepseek_module.contiguous_mla_attention_forward
make_projection_weights = deepseek_module.make_projection_weights
project_mla_from_hidden_states = deepseek_module.project_mla_from_hidden_states


class DeepseekV2MLAProjectionTests(unittest.TestCase):
    def _make_config(self):
        return types.SimpleNamespace(
            hidden_size=6,
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
            mla_dims=dims,
        )

    def test_make_projection_weights_validates_shapes(self):
        dims = self._make_dims()

        with self.assertRaises(ValueError):
            make_projection_weights(
                q_proj=torch.zeros(1, 1),
                kv_latent_proj=torch.zeros(dims.hidden_size, dims.kv_lora_rank),
                k_rope_proj=torch.zeros(
                    dims.hidden_size, dims.num_heads * dims.qk_rope_head_dim
                ),
                kv_up_proj=torch.zeros(
                    dims.kv_lora_rank, dims.kv_up_proj_output_dim_local
                ),
                mla_dims=dims,
            )

    def test_project_from_hidden_states_returns_query_and_resident_cache(self):
        dims = self._make_dims()
        projection_weights = self._make_projection_weights(dims)
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        query_states, cache = project_mla_from_hidden_states(
            hidden_states,
            projection_weights,
            dims,
        )

        self.assertEqual(tuple(query_states.shape), (2, dims.q_proj_output_dim_local))
        self.assertEqual(tuple(cache.kv_latent.shape), (2, dims.kv_lora_rank))
        self.assertEqual(tuple(cache.k_rope.shape), (2, dims.num_heads, dims.qk_rope_head_dim))
        self.assertTrue(
            torch.equal(
                cache.kv_latent,
                torch.tensor([[1.0, 3.0, 3.0], [2.0, 1.0, 1.0]]),
            )
        )

    def test_hidden_state_contiguous_path_matches_manual_projection_path(self):
        dims = self._make_dims()
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=2,
        )
        projection_weights = self._make_projection_weights(dims)
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        query_states, new_cache = attention.project_from_hidden_states(
            hidden_states,
            projection_weights,
        )
        manual_output, manual_cache = contiguous_mla_attention_forward(
            query_states=query_states,
            new_kv_latent=new_cache.kv_latent,
            new_k_rope=new_cache.k_rope,
            kv_up_proj_weight=projection_weights.kv_up_proj,
            mla_dims=dims,
        )
        projected_output, projected_cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
        )

        self.assertTrue(torch.allclose(projected_output, manual_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(projected_cache.kv_latent, manual_cache.kv_latent))
        self.assertTrue(torch.equal(projected_cache.k_rope, manual_cache.k_rope))

    def test_hidden_state_decode_reuses_and_appends_cache(self):
        dims = self._make_dims()
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=2,
        )
        projection_weights = self._make_projection_weights(dims)
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        first_output, cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_states[:1],
            projection_weights=projection_weights,
        )
        second_output, cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_states[1:],
            projection_weights=projection_weights,
            cache=cache,
        )

        self.assertEqual(tuple(first_output.shape), (1, dims.o_proj_input_dim_local))
        self.assertEqual(tuple(second_output.shape), (1, dims.o_proj_input_dim_local))
        self.assertEqual(cache.num_tokens, 2)


if __name__ == "__main__":
    unittest.main()
