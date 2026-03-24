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

    class _DummyAttentionWrapper:
        def __init__(self):
            self.begin_calls = []
            self.end_call_count = 0

        def init(self, *args, **kwargs):
            return None

        def begin_forward(self, seq_metadata_list):
            self.begin_calls.append(seq_metadata_list)

        def end_forward(self):
            self.end_call_count += 1

        def forward(
            self, query, key, value, kv_cache, softmax_scale=1.0, layer_id=None
        ):
            return value[-query.shape[0] :].clone()

    attention_wrapper = _DummyAttentionWrapper()
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


def _load_deepseek_model_module():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.model_executor", SARATHI_ROOT / "model_executor")
    _ensure_package(
        "sarathi.model_executor.parallel_utils",
        SARATHI_ROOT / "model_executor" / "parallel_utils",
    )
    _ensure_package(
        "sarathi.model_executor.models",
        SARATHI_ROOT / "model_executor" / "models",
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
    def setUp(self):
        ATTENTION_WRAPPER.begin_calls.clear()
        ATTENTION_WRAPPER.end_call_count = 0

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

    def test_runner_can_execute_loaded_deepseek_scaffold_via_run(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        class _NullTimer:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        config = types.SimpleNamespace(
            vocab_size=16,
            hidden_size=6,
            num_attention_heads=4,
            num_hidden_layers=4,
            q_lora_rank=None,
            kv_lora_rank=3,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            v_head_dim=2,
        )
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        runner = ModelRunner.__new__(ModelRunner)
        runner.model = deepseek_module.DeepseekV2ForCausalLM(config)
        runner.sampler = None
        runner._prepare_inputs_e2e_timer = _NullTimer()
        runner._model_execution_e2e_timer = _NullTimer()
        runner._sampler_e2e_timer = _NullTimer()
        runner._prepare_inputs = lambda seq_metadata_list: (
            torch.tensor([1, 3], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        )

        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            deepseek_module.make_projection_weights(
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
            for _ in range(runner.model.model.num_layers)
        )
        mlp_weights = tuple(
            deepseek_module.make_mlp_weights(
                gate_proj=torch.tensor(
                    [
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.5, 1.0, 0.0],
                        [0.5, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.5, 0.5],
                    ]
                ),
                up_proj=torch.tensor(
                    [
                        [1.0, 0.0, 0.5, 0.0],
                        [0.0, 1.0, 0.0, 0.5],
                        [0.5, 0.0, 1.0, 0.0],
                        [0.0, 0.5, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.5],
                        [0.0, 1.0, 0.5, 0.0],
                    ]
                ),
                down_proj=torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                        [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
                        [0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                hidden_size=config.hidden_size,
            )
            for _ in range(runner.model.model.num_layers)
        )
        scaffold_state_dict = {
            "model.embed_tokens.weight": torch.arange(
                config.vocab_size * config.hidden_size, dtype=torch.float32
            ).view(config.vocab_size, config.hidden_size)
            / 1000.0,
            "lm_head.weight": torch.arange(
                config.vocab_size * config.hidden_size, dtype=torch.float32
            ).view(config.vocab_size, config.hidden_size)
            / 1000.0,
        }
        for layer_idx, layer_projection_weights in enumerate(projection_weights):
            prefix = f"model.layers.{layer_idx}.self_attn"
            scaffold_state_dict[f"{prefix}.q_proj.weight"] = layer_projection_weights.q_proj
            scaffold_state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                layer_projection_weights.kv_latent_proj
            )
            scaffold_state_dict[f"{prefix}.k_rope_proj.weight"] = (
                layer_projection_weights.k_rope_proj
            )
            scaffold_state_dict[f"{prefix}.kv_up_proj.weight"] = (
                layer_projection_weights.kv_up_proj
            )
            scaffold_state_dict[f"{prefix}.o_proj.weight"] = layer_projection_weights.o_proj
        for layer_idx, layer_mlp_weights in enumerate(mlp_weights):
            prefix = f"model.layers.{layer_idx}.mlp"
            scaffold_state_dict[f"{prefix}.gate_proj.weight"] = layer_mlp_weights.gate_proj
            scaffold_state_dict[f"{prefix}.up_proj.weight"] = layer_mlp_weights.up_proj
            scaffold_state_dict[f"{prefix}.down_proj.weight"] = layer_mlp_weights.down_proj

        runner.load_model_weights(scaffold_state_dict)

        output, caches = runner.run(
            seq_metadata_list=["seq-md"],
            gpu_cache=tuple(object() for _ in range(runner.model.model.num_layers)),
            model_kwargs={"softmax_scale": 0.5},
        )

        self.assertEqual(tuple(output.shape), (2, config.hidden_size))
        self.assertEqual(len(caches), runner.model.model.num_layers)
        self.assertEqual(len(ATTENTION_WRAPPER.begin_calls), 1)
        self.assertEqual(ATTENTION_WRAPPER.begin_calls[0], ["seq-md"])
        self.assertEqual(ATTENTION_WRAPPER.end_call_count, 1)


if __name__ == "__main__":
    unittest.main()
