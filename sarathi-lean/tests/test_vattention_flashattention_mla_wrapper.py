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


def _install_wrapper_stubs(call_log):
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

    def _flash_attn_func(query, key, value, causal=True, softmax_scale=1.0):
        call_log.append(
            {
                "query": query.clone(),
                "key": key.clone(),
                "value": value.clone(),
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
        )
        return value[:, -query.shape[1] :, :, :].clone()

    flash_attn_module.flash_attn_func = _flash_attn_func
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


def _restore_wrapper_stubs(originals):
    for module_name, original in originals.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def _load_modules(call_log):
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
    originals = _install_wrapper_stubs(call_log)
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
            SARATHI_ROOT
            / "model_executor"
            / "attention"
            / "vattention_flashattention_wrapper.py",
        )
    finally:
        _restore_wrapper_stubs(originals)
        for module_name, original in project_originals.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
    return deepseek_module, wrapper_module


class VAttentionFlashAttentionMLAWrapperTests(unittest.TestCase):
    def setUp(self):
        self.flash_calls = []
        deepseek_module, wrapper_module = _load_modules(self.flash_calls)
        self.deepseek_module = deepseek_module
        self.wrapper_module = wrapper_module

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
        return self.deepseek_module.make_projection_weights(
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

    def _make_wrapper(self):
        wrapper = self.wrapper_module.VAttentionFlashAttentionWrapper()
        wrapper.device = torch.device("cpu")
        wrapper.is_metadata_initialized = True
        wrapper.is_profiling_iteration = False
        return wrapper

    def test_forward_mla_runs_single_prefill_sequence_from_resident_cache(self):
        dims = self.deepseek_module.DeepseekV2MLADims.from_config(
            self._make_config(),
            tensor_parallel_world_size=2,
        )
        projection_weights = self._make_projection_weights(dims)
        wrapper_inputs, _ = self.deepseek_module.prepare_mla_wrapper_inputs(
            hidden_states=self._make_hidden_states(),
            projection_weights=projection_weights,
            mla_dims=dims,
            kv_cache=object(),
            layer_id=3,
        )
        wrapper = self._make_wrapper()
        wrapper.prefill_query_lens = [2]
        wrapper.prefill_cache_lens = [0]
        wrapper.decode_cache_lens = None

        output = wrapper.forward_mla(wrapper_inputs)

        self.assertEqual(tuple(output.shape), (2, dims.o_proj_input_dim_local))
        self.assertEqual(len(self.flash_calls), 1)
        self.assertEqual(tuple(self.flash_calls[0]["query"].shape), (1, 2, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(self.flash_calls[0]["key"].shape), (1, 2, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(self.flash_calls[0]["value"].shape), (1, 2, dims.num_heads, dims.v_head_dim))

    def test_forward_mla_reuses_past_resident_cache_for_decode(self):
        dims = self.deepseek_module.DeepseekV2MLADims.from_config(
            self._make_config(),
            tensor_parallel_world_size=2,
        )
        projection_weights = self._make_projection_weights(dims)
        hidden_states = self._make_hidden_states()
        _, past_cache = self.deepseek_module.prepare_mla_wrapper_inputs(
            hidden_states=hidden_states[:1],
            projection_weights=projection_weights,
            mla_dims=dims,
            kv_cache=object(),
            layer_id=4,
        )
        wrapper_inputs, _ = self.deepseek_module.prepare_mla_wrapper_inputs(
            hidden_states=hidden_states[1:],
            projection_weights=projection_weights,
            mla_dims=dims,
            kv_cache=object(),
            layer_id=4,
            cache=past_cache,
        )
        wrapper = self._make_wrapper()
        wrapper.prefill_query_lens = []
        wrapper.prefill_cache_lens = []
        wrapper.decode_cache_lens = torch.tensor([1], dtype=torch.int32)

        output = wrapper.forward_mla(wrapper_inputs)

        self.assertEqual(tuple(output.shape), (1, dims.o_proj_input_dim_local))
        self.assertEqual(len(self.flash_calls), 1)
        self.assertEqual(tuple(self.flash_calls[0]["query"].shape), (1, 1, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(self.flash_calls[0]["key"].shape), (1, 2, dims.num_heads, dims.q_head_dim))
        self.assertEqual(tuple(self.flash_calls[0]["value"].shape), (1, 2, dims.num_heads, dims.v_head_dim))


if __name__ == "__main__":
    unittest.main()
