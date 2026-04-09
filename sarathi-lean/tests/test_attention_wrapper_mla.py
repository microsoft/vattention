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


def _install_attention_test_stubs():
    originals = {
        "sarathi.config": sys.modules.get("sarathi.config"),
        "sarathi.core.datatypes.sequence": sys.modules.get(
            "sarathi.core.datatypes.sequence"
        ),
        "sarathi.metrics.constants": sys.modules.get("sarathi.metrics.constants"),
        "sarathi.metrics.cuda_timer": sys.modules.get("sarathi.metrics.cuda_timer"),
    }

    config_module = types.ModuleType("sarathi.config")
    config_module.ModelConfig = object
    config_module.ParallelConfig = object
    sys.modules["sarathi.config"] = config_module

    sequence_module = types.ModuleType("sarathi.core.datatypes.sequence")
    sequence_module.SequenceMetadata = object
    sys.modules["sarathi.core.datatypes.sequence"] = sequence_module

    constants_module = types.ModuleType("sarathi.metrics.constants")
    constants_module.OperationMetrics = object
    sys.modules["sarathi.metrics.constants"] = constants_module

    cuda_timer_module = types.ModuleType("sarathi.metrics.cuda_timer")

    class _DummyCudaTimer:
        def __init__(self, *args, **kwargs):
            pass

    cuda_timer_module.CudaTimer = _DummyCudaTimer
    sys.modules["sarathi.metrics.cuda_timer"] = cuda_timer_module
    return originals


def _restore_attention_test_stubs(originals):
    for module_name, original_module in originals.items():
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


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
    originals = _install_attention_test_stubs()
    try:
        _load_module(
            "sarathi.model_executor.parallel_utils.parallel_state",
            SARATHI_ROOT / "model_executor" / "parallel_utils" / "parallel_state.py",
        )
        base_module = _load_module(
            "sarathi.model_executor.attention.base_attention_wrapper",
            SARATHI_ROOT / "model_executor" / "attention" / "base_attention_wrapper.py",
        )
        deepseek_module = _load_module(
            "sarathi.model_executor.models.deepseek_v2",
            SARATHI_ROOT / "model_executor" / "models" / "deepseek_v2.py",
        )
    finally:
        _restore_attention_test_stubs(originals)
    return base_module, deepseek_module


base_module, deepseek_module = _load_modules()
BaseAttentionWrapper = base_module.BaseAttentionWrapper
DeepseekV2MLADims = deepseek_module.DeepseekV2MLADims
make_projection_weights = deepseek_module.make_projection_weights
prepare_mla_wrapper_inputs = deepseek_module.prepare_mla_wrapper_inputs


class _RecordingWrapper(BaseAttentionWrapper):
    def __init__(self):
        self.calls = []

    def begin_forward(self, seq_metadata_list):
        pass

    def end_forward(self):
        pass

    def forward(
        self,
        query,
        key,
        value,
        kv_cache,
        softmax_scale=1.0,
        layer_id=None,
    ):
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


class BaseAttentionWrapperMLATests(unittest.TestCase):
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

    def test_forward_mla_reconstructs_dense_inputs_and_delegates_to_forward(self):
        config = self._make_config()
        dims = DeepseekV2MLADims.from_config(config, tensor_parallel_world_size=2)
        projection_weights = self._make_projection_weights(dims)
        wrapper = _RecordingWrapper()
        hidden_states = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )
        kv_cache = object()

        wrapper_inputs, _ = prepare_mla_wrapper_inputs(
            hidden_states=hidden_states,
            projection_weights=projection_weights,
            mla_dims=dims,
            kv_cache=kv_cache,
            layer_id=4,
        )
        output = wrapper.forward_mla(wrapper_inputs)

        self.assertEqual(len(wrapper.calls), 1)
        self.assertEqual(tuple(wrapper.calls[0]["query"].shape), (2, dims.num_heads * dims.q_head_dim))
        self.assertEqual(tuple(wrapper.calls[0]["key"].shape), (2, dims.num_heads * dims.q_head_dim))
        self.assertEqual(tuple(wrapper.calls[0]["value"].shape), (2, dims.o_proj_input_dim_local))
        self.assertIs(wrapper.calls[0]["kv_cache"], kv_cache)
        self.assertEqual(wrapper.calls[0]["layer_id"], 4)
        self.assertEqual(tuple(output.shape), (2, dims.o_proj_input_dim_local))

    def test_forward_mla_rejects_incomplete_wrapper_inputs(self):
        wrapper = _RecordingWrapper()

        with self.assertRaises(ValueError):
            wrapper.forward_mla(types.SimpleNamespace(query=torch.zeros(1, 1, 1)))


if __name__ == "__main__":
    unittest.main()
