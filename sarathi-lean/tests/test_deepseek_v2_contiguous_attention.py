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


class DeepseekV2ContiguousAttentionTests(unittest.TestCase):
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

    def _make_test_inputs(self):
        dims = self._make_dims()
        query_states = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.5, 1.0, 0.0, 0.5],
            ]
        )
        kv_latent = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        k_rope = torch.tensor(
            [
                [[0.5], [1.5]],
                [[2.5], [3.5]],
            ]
        )
        kv_up_proj_weight = torch.tensor(
            [
                [1.0, 0.0, 10.0, 20.0, 2.0, 0.0, 30.0, 40.0],
                [0.0, 1.0, 11.0, 21.0, 0.0, 2.0, 31.0, 41.0],
                [1.0, 1.0, 12.0, 22.0, 2.0, 2.0, 32.0, 42.0],
            ]
        )
        return dims, query_states, kv_latent, k_rope, kv_up_proj_weight

    def test_prefill_contiguous_attention_returns_local_output_and_cache(self):
        dims, query_states, kv_latent, k_rope, kv_up_proj_weight = self._make_test_inputs()

        output, cache = contiguous_mla_attention_forward(
            query_states=query_states,
            new_kv_latent=kv_latent,
            new_k_rope=k_rope,
            kv_up_proj_weight=kv_up_proj_weight,
            mla_dims=dims,
        )

        self.assertEqual(tuple(output.shape), (2, dims.o_proj_input_dim_local))
        self.assertEqual(tuple(cache.kv_latent.shape), (2, dims.kv_lora_rank))
        self.assertEqual(tuple(cache.k_rope.shape), (2, dims.num_heads, dims.qk_rope_head_dim))
        self.assertTrue(torch.isfinite(output).all())

    def test_decode_attention_appends_cache_and_only_emits_new_token_output(self):
        dims, query_states, kv_latent, k_rope, kv_up_proj_weight = self._make_test_inputs()
        _, cache = contiguous_mla_attention_forward(
            query_states=query_states[:1],
            new_kv_latent=kv_latent[:1],
            new_k_rope=k_rope[:1],
            kv_up_proj_weight=kv_up_proj_weight,
            mla_dims=dims,
        )

        output, updated_cache = contiguous_mla_attention_forward(
            query_states=query_states[1:],
            new_kv_latent=kv_latent[1:],
            new_k_rope=k_rope[1:],
            kv_up_proj_weight=kv_up_proj_weight,
            mla_dims=dims,
            cache=cache,
        )

        self.assertEqual(tuple(output.shape), (1, dims.o_proj_input_dim_local))
        self.assertEqual(updated_cache.num_tokens, 2)
        self.assertTrue(torch.equal(updated_cache.kv_latent[0], kv_latent[0]))
        self.assertTrue(torch.equal(updated_cache.kv_latent[1], kv_latent[1]))

    def test_multistep_decode_matches_single_prefill_run(self):
        dims, query_states, kv_latent, k_rope, kv_up_proj_weight = self._make_test_inputs()
        full_output, full_cache = contiguous_mla_attention_forward(
            query_states=query_states,
            new_kv_latent=kv_latent,
            new_k_rope=k_rope,
            kv_up_proj_weight=kv_up_proj_weight,
            mla_dims=dims,
        )

        step0_output, cache = contiguous_mla_attention_forward(
            query_states=query_states[:1],
            new_kv_latent=kv_latent[:1],
            new_k_rope=k_rope[:1],
            kv_up_proj_weight=kv_up_proj_weight,
            mla_dims=dims,
        )
        step1_output, cache = contiguous_mla_attention_forward(
            query_states=query_states[1:],
            new_kv_latent=kv_latent[1:],
            new_k_rope=k_rope[1:],
            kv_up_proj_weight=kv_up_proj_weight,
            mla_dims=dims,
            cache=cache,
        )

        stitched_output = torch.cat([step0_output, step1_output], dim=0)
        self.assertTrue(torch.allclose(stitched_output, full_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(cache.kv_latent, full_cache.kv_latent))
        self.assertTrue(torch.equal(cache.k_rope, full_cache.k_rope))

    def test_attention_module_wraps_contiguous_reference_forward(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=2,
        )
        _, query_states, kv_latent, k_rope, kv_up_proj_weight = self._make_test_inputs()

        output, cache = attention.forward_contiguous(
            query_states=query_states,
            new_kv_latent=kv_latent,
            new_k_rope=k_rope,
            kv_up_proj_weight=kv_up_proj_weight,
        )

        self.assertEqual(tuple(output.shape), (2, attention.mla_dims.o_proj_input_dim_local))
        self.assertEqual(cache.num_tokens, 2)


if __name__ == "__main__":
    unittest.main()
