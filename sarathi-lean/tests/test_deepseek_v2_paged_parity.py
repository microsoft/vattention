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


def _exact_flash_attn(query, key, value, causal=True, softmax_scale=1.0):
    scores = torch.einsum("bthd,bshd->bhts", query, key) * softmax_scale
    if causal:
        source_positions = torch.arange(key.shape[1], device=query.device)
        past_len = key.shape[1] - query.shape[1]
        target_positions = past_len + torch.arange(query.shape[1], device=query.device)
        causal_mask = source_positions.unsqueeze(0) <= target_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshv->bthv", attn_weights, value)


def _install_stubs():
    originals = {
        name: sys.modules.get(name)
        for name in [
            "flash_attn",
            "sarathi.config",
            "sarathi.core.datatypes.sequence",
            "sarathi.logger",
            "sarathi.metrics.constants",
            "sarathi.metrics.cuda_timer",
            "sarathi.cache_ops",
            "vattention",
        ]
    }

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.flash_attn_func = _exact_flash_attn
    flash_attn_module.flash_attn_with_kvcache = lambda *args, **kwargs: None
    sys.modules["flash_attn"] = flash_attn_module

    config_module = types.ModuleType("sarathi.config")
    config_module.ModelConfig = object
    config_module.ParallelConfig = object
    sys.modules["sarathi.config"] = config_module

    sequence_module = types.ModuleType("sarathi.core.datatypes.sequence")
    sequence_module.SequenceMetadata = object
    sys.modules["sarathi.core.datatypes.sequence"] = sequence_module

    logger_module = types.ModuleType("sarathi.logger")
    logger_module.init_logger = lambda name: types.SimpleNamespace(warning=lambda *args, **kwargs: None)
    sys.modules["sarathi.logger"] = logger_module

    constants_module = types.ModuleType("sarathi.metrics.constants")
    constants_module.OperationMetrics = object
    sys.modules["sarathi.metrics.constants"] = constants_module

    cuda_timer_module = types.ModuleType("sarathi.metrics.cuda_timer")

    class _DummyCudaTimer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    cuda_timer_module.CudaTimer = _DummyCudaTimer
    sys.modules["sarathi.metrics.cuda_timer"] = cuda_timer_module

    cache_ops_module = types.ModuleType("sarathi.cache_ops")
    cache_ops_module.cache_flat = lambda *args, **kwargs: None
    sys.modules["sarathi.cache_ops"] = cache_ops_module

    sys.modules["vattention"] = types.ModuleType("vattention")
    return originals


def _restore_stubs(originals):
    for module_name, original in originals.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def _load_modules():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.model_executor", SARATHI_ROOT / "model_executor")
    _ensure_package(
        "sarathi.model_executor.parallel_utils",
        SARATHI_ROOT / "model_executor" / "parallel_utils",
    )
    _ensure_package(
        "sarathi.model_executor.attention",
        SARATHI_ROOT / "model_executor" / "attention",
    )
    _ensure_package(
        "sarathi.model_executor.models",
        SARATHI_ROOT / "model_executor" / "models",
    )

    originals = _install_stubs()
    project_originals = {
        name: sys.modules.get(name)
        for name in [
            "sarathi.model_executor.parallel_utils.parallel_state",
            "sarathi.model_executor.attention.base_attention_wrapper",
            "sarathi.model_executor.models.deepseek_v2",
            "sarathi.model_executor.attention.vattention_flashattention_wrapper",
        ]
    }
    try:
        _load_module(
            "sarathi.model_executor.parallel_utils.parallel_state",
            SARATHI_ROOT / "model_executor" / "parallel_utils" / "parallel_state.py",
        )
        _load_module(
            "sarathi.model_executor.attention.base_attention_wrapper",
            SARATHI_ROOT / "model_executor" / "attention" / "base_attention_wrapper.py",
        )
        deepseek_module = _load_module(
            "sarathi.model_executor.models.deepseek_v2",
            SARATHI_ROOT / "model_executor" / "models" / "deepseek_v2.py",
        )
        wrapper_module = _load_module(
            "sarathi.model_executor.attention.vattention_flashattention_wrapper",
            SARATHI_ROOT / "model_executor" / "attention" / "vattention_flashattention_wrapper.py",
        )
    finally:
        _restore_stubs(originals)
        for module_name, original in project_originals.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
    return deepseek_module, wrapper_module


deepseek_module, wrapper_module = _load_modules()


class DeepseekV2PagedParityTests(unittest.TestCase):
    def setUp(self):
        self._original_deepseek_module = sys.modules.get(
            "sarathi.model_executor.models.deepseek_v2"
        )
        sys.modules["sarathi.model_executor.models.deepseek_v2"] = deepseek_module

    def tearDown(self):
        if self._original_deepseek_module is None:
            sys.modules.pop("sarathi.model_executor.models.deepseek_v2", None)
        else:
            sys.modules["sarathi.model_executor.models.deepseek_v2"] = (
                self._original_deepseek_module
            )

    def _make_config(self):
        return types.SimpleNamespace(
            hidden_size=6,
            num_attention_heads=4,
            num_hidden_layers=2,
            q_lora_rank=None,
            kv_lora_rank=3,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
        )

    def _make_projection_weights(self, dims):
        return deepseek_module.make_projection_weights(
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

    def _make_hidden_states(self):
        return torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

    def _make_wrapper(self):
        wrapper = wrapper_module.VAttentionFlashAttentionWrapper()
        wrapper.device = torch.device("cpu")
        wrapper.is_metadata_initialized = True
        wrapper.is_profiling_iteration = False
        return wrapper

    def test_prefill_wrapper_path_matches_contiguous_reference(self):
        config = self._make_config()
        attention = deepseek_module.DeepseekV2MLAAttention(
            config,
            tensor_parallel_world_size=2,
        )
        dims = attention.mla_dims
        projection_weights = self._make_projection_weights(dims)
        hidden_states = self._make_hidden_states()
        runtime_cache = deepseek_module.make_component_mla_kv_cache(
            batch_size=1,
            max_seq_len=4,
            mla_dims=dims,
        )
        wrapper = self._make_wrapper()
        wrapper.set_mla_runtime_metadata(
            prefill_query_lens=[2],
            prefill_cache_lens=[0],
            batch_index=[0],
            batch_index_gen=[],
        )

        contiguous_output, contiguous_cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
        )
        wrapper_output, wrapper_cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            kv_cache=runtime_cache,
            layer_id=0,
            attention_wrapper=wrapper,
        )

        self.assertTrue(torch.allclose(wrapper_output, contiguous_output, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(wrapper_cache.resident_cache.kv_latent, contiguous_cache.kv_latent))
        self.assertTrue(torch.equal(wrapper_cache.resident_cache.k_rope, contiguous_cache.k_rope))

    def test_decode_wrapper_path_matches_contiguous_reference(self):
        config = self._make_config()
        attention = deepseek_module.DeepseekV2MLAAttention(
            config,
            tensor_parallel_world_size=2,
        )
        dims = attention.mla_dims
        projection_weights = self._make_projection_weights(dims)
        hidden_states = self._make_hidden_states()
        runtime_cache = deepseek_module.make_component_mla_kv_cache(
            batch_size=1,
            max_seq_len=4,
            mla_dims=dims,
        )
        wrapper = self._make_wrapper()

        wrapper.set_mla_runtime_metadata(
            prefill_query_lens=[1],
            prefill_cache_lens=[0],
            batch_index=[0],
            batch_index_gen=[],
        )
        _, first_wrapper_cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=hidden_states[:1],
            projection_weights=projection_weights,
            kv_cache=runtime_cache,
            layer_id=0,
            attention_wrapper=wrapper,
        )
        _, first_contiguous_cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_states[:1],
            projection_weights=projection_weights,
        )

        wrapper.set_mla_runtime_metadata(
            prefill_query_lens=[],
            prefill_cache_lens=[],
            decode_cache_lens=[1],
            batch_index=[],
            batch_index_gen=[0],
        )
        wrapper_output, wrapper_cache = attention.forward_hidden_states_with_attention_wrapper(
            hidden_states=hidden_states[1:],
            projection_weights=projection_weights,
            kv_cache=first_wrapper_cache,
            layer_id=0,
            attention_wrapper=wrapper,
        )
        contiguous_output, contiguous_cache = attention.forward_hidden_states_contiguous(
            hidden_states=hidden_states[1:],
            projection_weights=projection_weights,
            cache=first_contiguous_cache,
        )

        self.assertTrue(torch.allclose(wrapper_output, contiguous_output, atol=1e-6, rtol=1e-6))
        self.assertEqual(first_wrapper_cache.resident_cache.num_tokens, 1)
        self.assertEqual(wrapper_cache.resident_cache.num_tokens, 2)
        self.assertTrue(torch.equal(wrapper_cache.resident_cache.kv_latent, contiguous_cache.kv_latent))
        self.assertTrue(torch.equal(wrapper_cache.resident_cache.k_rope, contiguous_cache.k_rope))


if __name__ == "__main__":
    unittest.main()
