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


def _install_runner_stubs():
    originals = {
        name: sys.modules.get(name)
        for name in [
            "sarathi.config",
            "sarathi.core.datatypes.sampling_params",
            "sarathi.core.datatypes.sequence",
            "sarathi.logger",
            "sarathi.metrics.constants",
            "sarathi.metrics.cpu_timer",
            "sarathi.metrics.cuda_timer",
            "sarathi.model_executor",
            "sarathi.model_executor.attention",
            "sarathi.model_executor.layers.sampler",
            "sarathi.model_executor.utils",
            "sarathi.utils",
            "sarathi.worker.cache_engine",
            "torch.distributed",
        ]
    }

    config_module = types.ModuleType("sarathi.config")
    config_module.BaseSchedulerConfig = object
    config_module.CacheConfig = object
    config_module.ModelConfig = object
    config_module.ParallelConfig = object
    config_module.SchedulerType = types.SimpleNamespace(
        SARATHI="SARATHI",
        SIMPLE_CHUNKING="SIMPLE_CHUNKING",
    )
    sys.modules["sarathi.config"] = config_module

    sampling_params_module = types.ModuleType("sarathi.core.datatypes.sampling_params")
    sampling_params_module.SamplingParams = object
    sys.modules["sarathi.core.datatypes.sampling_params"] = sampling_params_module

    sequence_module = types.ModuleType("sarathi.core.datatypes.sequence")
    sequence_module.Sequence = object
    sequence_module.SequenceMetadata = object
    sys.modules["sarathi.core.datatypes.sequence"] = sequence_module

    logger_module = types.ModuleType("sarathi.logger")
    logger_module.init_logger = lambda name: types.SimpleNamespace(error=lambda *args, **kwargs: None)
    sys.modules["sarathi.logger"] = logger_module

    constants_module = types.ModuleType("sarathi.metrics.constants")
    constants_module.CpuOperationMetrics = types.SimpleNamespace(
        PREPARE_INPUTS_E2E="prepare",
        SAMPLER_E2E="sampler",
        MODEL_EXECUTION_E2E="model",
    )
    constants_module.OperationMetrics = object
    sys.modules["sarathi.metrics.constants"] = constants_module

    cpu_timer_module = types.ModuleType("sarathi.metrics.cpu_timer")

    class _DummyCpuTimer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    cpu_timer_module.CpuTimer = _DummyCpuTimer
    sys.modules["sarathi.metrics.cpu_timer"] = cpu_timer_module

    cuda_timer_module = types.ModuleType("sarathi.metrics.cuda_timer")
    cuda_timer_module.CudaTimer = _DummyCpuTimer
    sys.modules["sarathi.metrics.cuda_timer"] = cuda_timer_module

    model_executor_module = types.ModuleType("sarathi.model_executor")
    model_executor_module.get_model = lambda config: None
    model_executor_module.set_random_seed = lambda seed: None
    sys.modules["sarathi.model_executor"] = model_executor_module

    attention_wrapper = object()
    attention_module = types.ModuleType("sarathi.model_executor.attention")
    attention_module.get_attention_wrapper = lambda: attention_wrapper
    attention_module.AttentionBackend = types.SimpleNamespace(
        is_vATTN=lambda backend: False
    )
    sys.modules["sarathi.model_executor.attention"] = attention_module

    sampler_module = types.ModuleType("sarathi.model_executor.layers.sampler")
    sampler_module.Sampler = object
    sys.modules["sarathi.model_executor.layers.sampler"] = sampler_module

    utils_module = types.ModuleType("sarathi.model_executor.utils")
    utils_module.pad_to_alignment = lambda values, multiple_of=8: values
    sys.modules["sarathi.model_executor.utils"] = utils_module

    general_utils_module = types.ModuleType("sarathi.utils")
    general_utils_module.get_gpu_memory = lambda: 0
    sys.modules["sarathi.utils"] = general_utils_module

    cache_engine_module = types.ModuleType("sarathi.worker.cache_engine")
    cache_engine_module.get_cache_engine = lambda backend: None
    sys.modules["sarathi.worker.cache_engine"] = cache_engine_module

    sys.modules["torch.distributed"] = types.ModuleType("torch.distributed")
    return originals, attention_wrapper


def _restore_runner_stubs(originals):
    for module_name, original in originals.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def _load_model_runner_module():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.model_executor", SARATHI_ROOT / "model_executor")

    originals, attention_wrapper = _install_runner_stubs()
    project_original = sys.modules.get("sarathi.model_executor.model_runner")
    try:
        model_runner_module = _load_module(
            "sarathi.model_executor.model_runner",
            SARATHI_ROOT / "model_executor" / "model_runner.py",
        )
    finally:
        _restore_runner_stubs(originals)
        if project_original is None:
            sys.modules.pop("sarathi.model_executor.model_runner", None)
        else:
            sys.modules["sarathi.model_executor.model_runner"] = project_original
    return model_runner_module, attention_wrapper


model_runner_module, ATTENTION_WRAPPER = _load_model_runner_module()
ModelRunner = model_runner_module.ModelRunner


class _RecordingModel:
    def __init__(self):
        self.calls = []
        self.wrapper_calls = []
        self.lm_head = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return "standard"

    def forward_with_attention_wrapper(self, **kwargs):
        self.wrapper_calls.append(kwargs)
        return "wrapper"


class ModelRunnerMLADispatchTests(unittest.TestCase):
    def test_execute_model_uses_standard_path_without_projection_weights(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = _RecordingModel()

        output = runner._execute_model(
            hidden_states=torch.tensor([1]),
            positions=torch.tensor([2]),
            kv_caches=("cache",),
        )

        self.assertEqual(output, "standard")
        self.assertEqual(len(runner.model.calls), 1)
        self.assertEqual(runner.model.calls[0]["positions"].tolist(), [2])
        self.assertEqual(runner.model.calls[0]["kv_caches"], ("cache",))

    def test_execute_model_uses_wrapper_path_for_projection_weight_execution(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = _RecordingModel()
        projection_weights = ("proj",)

        output = runner._execute_model(
            hidden_states=torch.tensor([1]),
            positions=torch.tensor([2]),
            kv_caches=("cache",),
            model_kwargs={
                "projection_weights": projection_weights,
                "mlp_weights": ("mlp",),
                "caches": ("resident",),
                "softmax_scale": 0.5,
            },
        )

        self.assertEqual(output, "wrapper")
        self.assertEqual(len(runner.model.wrapper_calls), 1)
        self.assertEqual(runner.model.wrapper_calls[0]["projection_weights"], projection_weights)
        self.assertEqual(runner.model.wrapper_calls[0]["mlp_weights"], ("mlp",))
        self.assertEqual(runner.model.wrapper_calls[0]["kv_caches"], ("cache",))
        self.assertEqual(runner.model.wrapper_calls[0]["attention_wrapper"], ATTENTION_WRAPPER)
        self.assertEqual(runner.model.wrapper_calls[0]["caches"], ("resident",))
        self.assertEqual(runner.model.wrapper_calls[0]["softmax_scale"], 0.5)

    def test_execute_model_uses_installed_scaffold_path_without_projection_weights(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = _RecordingModel()

        output = runner._execute_model(
            hidden_states=torch.tensor([1]),
            positions=torch.tensor([2]),
            kv_caches=("cache",),
            model_kwargs={
                "mlp_weights": ("mlp",),
                "caches": ("resident",),
                "softmax_scale": 0.25,
            },
        )

        self.assertEqual(output, "standard")
        self.assertEqual(len(runner.model.calls), 1)
        self.assertEqual(runner.model.calls[0]["positions"].tolist(), [2])
        self.assertEqual(runner.model.calls[0]["kv_caches"], ("cache",))
        self.assertEqual(runner.model.calls[0]["mlp_weights"], ("mlp",))
        self.assertEqual(runner.model.calls[0]["caches"], ("resident",))
        self.assertEqual(runner.model.calls[0]["softmax_scale"], 0.25)
        self.assertEqual(runner.model.calls[0]["attention_wrapper"], ATTENTION_WRAPPER)

    def test_execute_model_rejects_unknown_model_kwargs(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = _RecordingModel()

        with self.assertRaises(ValueError):
            runner._execute_model(
                hidden_states=torch.tensor([1]),
                positions=torch.tensor([2]),
                kv_caches=("cache",),
                model_kwargs={"unexpected": 1},
            )


if __name__ == "__main__":
    unittest.main()
