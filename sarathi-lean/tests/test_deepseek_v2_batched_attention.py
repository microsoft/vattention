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


class DeepseekV2BatchedAttentionTests(unittest.TestCase):
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

    def _make_hidden_state_batch(self):
        return (
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [2.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                ]
            ),
        )

    def test_batched_forward_matches_per_sequence_outputs_for_mixed_lengths(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=4,
        )
        dims = DeepseekV2MLADims.from_config(self._make_config(), tensor_parallel_world_size=4)
        projection_weights = self._make_projection_weights(dims)
        hidden_state_batch = self._make_hidden_state_batch()

        batch_outputs, batch_caches = attention.forward_hidden_states_contiguous_batched(
            hidden_states=hidden_state_batch,
            projection_weights=projection_weights,
        )
        seq0_output, seq0_cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_state_batch[0],
            projection_weights=projection_weights,
        )
        seq1_output, seq1_cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_state_batch[1],
            projection_weights=projection_weights,
        )

        self.assertEqual(len(batch_outputs), 2)
        self.assertTrue(torch.allclose(batch_outputs[0], seq0_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(batch_outputs[1], seq1_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(batch_caches[0].kv_latent, seq0_cache.kv_latent))
        self.assertTrue(torch.equal(batch_caches[1].kv_latent, seq1_cache.kv_latent))

    def test_batched_decode_reuses_per_sequence_caches(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=4,
        )
        dims = DeepseekV2MLADims.from_config(self._make_config(), tensor_parallel_world_size=4)
        projection_weights = self._make_projection_weights(dims)

        prefill_outputs, caches = attention.forward_hidden_states_contiguous_batched(
            hidden_states=(
                torch.tensor([[1.0, 2.0, 3.0, 0.0, 1.0, 0.0]]),
                torch.tensor([[2.0, 0.0, 1.0, 1.0, 0.0, 0.0]]),
            ),
            projection_weights=projection_weights,
        )
        decode_outputs, caches = attention.forward_hidden_states_contiguous_batched(
            hidden_states=(
                torch.tensor([[0.0, 1.0, 0.0, 2.0, 0.0, 1.0]]),
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 2.0]]),
            ),
            projection_weights=projection_weights,
            caches=caches,
        )

        self.assertEqual(tuple(prefill_outputs[0].shape), (1, dims.hidden_size))
        self.assertEqual(tuple(prefill_outputs[1].shape), (1, dims.hidden_size))
        self.assertEqual(tuple(decode_outputs[0].shape), (1, dims.hidden_size))
        self.assertEqual(tuple(decode_outputs[1].shape), (1, dims.hidden_size))
        self.assertEqual(caches[0].num_tokens, 2)
        self.assertEqual(caches[1].num_tokens, 2)

    def test_batched_forward_validates_batch_and_cache_lengths(self):
        attention = DeepseekV2MLAAttention(
            self._make_config(),
            tensor_parallel_world_size=4,
        )
        dims = DeepseekV2MLADims.from_config(self._make_config(), tensor_parallel_world_size=4)
        projection_weights = self._make_projection_weights(dims)

        with self.assertRaises(ValueError):
            attention.forward_hidden_states_contiguous_batched(
                hidden_states=self._make_hidden_state_batch(),
                projection_weights=projection_weights,
                caches=(None,),
            )


if __name__ == "__main__":
    unittest.main()
