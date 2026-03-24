import importlib.util
import sys
import types
import unittest
from enum import Enum
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


def _install_stubs(call_log):
    originals = {
        name: sys.modules.get(name)
        for name in [
            "flash_attn",
            "sarathi.config",
            "sarathi.core.datatypes.scheduler_output",
            "sarathi.core.datatypes.sequence",
            "sarathi.core.sequence_manager.worker_sequence_manager",
            "sarathi.logger",
            "sarathi.metrics.constants",
            "sarathi.metrics.cuda_timer",
            "sarathi.metrics.metrics_store",
            "sarathi.cache_ops",
            "sarathi.model_executor",
            "sarathi.model_executor.attention",
            "sarathi.model_executor.model_runner",
            "sarathi.utils",
            "sarathi.utils.threading_utils",
            "sarathi.worker.cache_engine",
            "sarathi.worker.cache_engine.base_cache_engine",
            "sarathi.worker.cache_engine.vattention_init",
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

    class CacheArchitecture(Enum):
        DENSE_KV = "dense_kv"
        MLA = "mla"

    config_module.BaseSchedulerConfig = object
    config_module.CacheArchitecture = CacheArchitecture
    config_module.CacheConfig = object
    config_module.MetricsConfig = object
    config_module.ModelConfig = object
    config_module.ParallelConfig = object
    sys.modules["sarathi.config"] = config_module

    scheduler_output_module = types.ModuleType("sarathi.core.datatypes.scheduler_output")
    scheduler_output_module.SchedulerOutputs = object
    sys.modules["sarathi.core.datatypes.scheduler_output"] = scheduler_output_module

    sequence_module = types.ModuleType("sarathi.core.datatypes.sequence")
    sequence_module.SamplerOutputs = object
    sequence_module.Sequence = object
    sequence_module.SequenceMetadata = object
    sys.modules["sarathi.core.datatypes.sequence"] = sequence_module

    seq_manager_module = types.ModuleType("sarathi.core.sequence_manager.worker_sequence_manager")
    seq_manager_module.WorkerSequenceManager = object
    sys.modules["sarathi.core.sequence_manager.worker_sequence_manager"] = seq_manager_module

    logger_module = types.ModuleType("sarathi.logger")
    logger_module.init_logger = lambda name: types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
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

    metrics_store_module = types.ModuleType("sarathi.metrics.metrics_store")
    metrics_store_module.MetricsStore = object
    sys.modules["sarathi.metrics.metrics_store"] = metrics_store_module

    cache_ops_module = types.ModuleType("sarathi.cache_ops")
    cache_ops_module.cache_flat = lambda *args, **kwargs: None
    sys.modules["sarathi.cache_ops"] = cache_ops_module

    model_executor_module = types.ModuleType("sarathi.model_executor")
    model_executor_module.set_random_seed = lambda seed: None
    sys.modules["sarathi.model_executor"] = model_executor_module

    attention_module = types.ModuleType("sarathi.model_executor.attention")
    attention_module.get_attention_wrapper = lambda: None
    attention_module.set_attention_backend = lambda backend: None
    sys.modules["sarathi.model_executor.attention"] = attention_module

    model_runner_module = types.ModuleType("sarathi.model_executor.model_runner")
    model_runner_module.ModelRunner = object
    sys.modules["sarathi.model_executor.model_runner"] = model_runner_module

    utils_module = types.ModuleType("sarathi.utils")
    utils_module.in_wsl = lambda: False
    sys.modules["sarathi.utils"] = utils_module

    threading_utils_module = types.ModuleType("sarathi.utils.threading_utils")
    threading_utils_module.synchronized = lambda fn: fn
    sys.modules["sarathi.utils.threading_utils"] = threading_utils_module

    worker_cache_engine_module = types.ModuleType("sarathi.worker.cache_engine")
    worker_cache_engine_module.get_cache_engine = lambda backend: None
    worker_cache_engine_module.get_cache_mem_alloc_backend = lambda backend: "noop"
    sys.modules["sarathi.worker.cache_engine"] = worker_cache_engine_module

    base_cache_engine_module = types.ModuleType("sarathi.worker.cache_engine.base_cache_engine")
    base_cache_engine_module.BaseCacheEngine = object
    sys.modules["sarathi.worker.cache_engine.base_cache_engine"] = base_cache_engine_module

    vattention_init_module = types.ModuleType("sarathi.worker.cache_engine.vattention_init")
    vattention_init_module.dispatch_init_kvcache = lambda backend, request: None
    sys.modules["sarathi.worker.cache_engine.vattention_init"] = vattention_init_module

    sys.modules["vattention"] = types.ModuleType("vattention")
    return originals, config_module.CacheArchitecture


def _restore_stubs(originals):
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
    _ensure_package("sarathi.worker", SARATHI_ROOT / "worker")
    _ensure_package("sarathi.worker.cache_engine", SARATHI_ROOT / "worker" / "cache_engine")

    originals, cache_architecture = _install_stubs(call_log)
    project_originals = {
        name: sys.modules.get(name)
        for name in [
            "sarathi.model_executor.attention.base_attention_wrapper",
            "sarathi.model_executor.attention.vattention_flashattention_wrapper",
            "sarathi.model_executor.models.deepseek_v2",
            "sarathi.worker.cache_engine.vATTN_cache_engine",
            "sarathi.worker.base_worker",
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
        wrapper_module = _load_module(
            "sarathi.model_executor.attention.vattention_flashattention_wrapper",
            SARATHI_ROOT / "model_executor" / "attention" / "vattention_flashattention_wrapper.py",
        )
        deepseek_module = _load_module(
            "sarathi.model_executor.models.deepseek_v2",
            SARATHI_ROOT / "model_executor" / "models" / "deepseek_v2.py",
        )
        cache_engine_module = _load_module(
            "sarathi.worker.cache_engine.vATTN_cache_engine",
            SARATHI_ROOT / "worker" / "cache_engine" / "vATTN_cache_engine.py",
        )
        worker_module = _load_module(
            "sarathi.worker.base_worker",
            SARATHI_ROOT / "worker" / "base_worker.py",
        )
    finally:
        _restore_stubs(originals)
        for module_name, original in project_originals.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
    return worker_module, deepseek_module, wrapper_module, cache_engine_module, cache_architecture


class _FakeBlockManager:
    def __init__(self):
        self.free_blocks = []

    def set_free_blocks(self, value):
        self.free_blocks.append(value)


class _FakeSeqManager:
    def __init__(self, seq_metadata_list):
        self.seq_metadata_list = seq_metadata_list
        self.block_manager = _FakeBlockManager()
        self.completed = []

    def on_schedule(self, scheduler_outputs):
        return None, self.seq_metadata_list

    def on_step_completed(self, scheduler_outputs, sampler_outputs):
        self.completed.append((scheduler_outputs, sampler_outputs))


class _SequencedFakeSeqManager(_FakeSeqManager):
    def __init__(self, seq_metadata_lists):
        super().__init__(seq_metadata_lists[0])
        self.seq_metadata_lists = list(seq_metadata_lists)
        self.schedule_index = 0

    def on_schedule(self, scheduler_outputs):
        del scheduler_outputs
        seq_metadata_list = self.seq_metadata_lists[self.schedule_index]
        if self.schedule_index < len(self.seq_metadata_lists) - 1:
            self.schedule_index += 1
        self.seq_metadata_list = seq_metadata_list
        return None, seq_metadata_list


class _FakeCacheEngine:
    def __init__(self, cache_usage_stats=None):
        self.steps = []
        self.completions = []
        self.free_blocks = 9
        self._cache_usage_stats = cache_usage_stats

    def num_free_blocks(self):
        return self.free_blocks

    def step(self, seq_metadata_list):
        self.steps.append(seq_metadata_list)

    def on_step_completion(self, seq_metadata_list):
        self.completions.append(seq_metadata_list)

    def preempt_requests(self, preempted_seq):
        pass

    def get_cache_usage_stats(self):
        return self._cache_usage_stats

    def get_cache_usage_history(self):
        return ()


class _SequencedFakeCacheEngine(_FakeCacheEngine):
    def __init__(self, cache_usage_history):
        super().__init__(cache_usage_stats=None)
        self.cache_usage_history = list(cache_usage_history)
        self.history_index = -1
        self.preempted = []

    def step(self, seq_metadata_list):
        super().step(seq_metadata_list)
        if self.history_index < len(self.cache_usage_history) - 1:
            self.history_index += 1

    def preempt_requests(self, preempted_seq):
        self.preempted.append(tuple(seq.seq_id for seq in preempted_seq))
        if self.history_index < len(self.cache_usage_history) - 1:
            self.history_index += 1

    def get_cache_usage_stats(self):
        if self.history_index < 0:
            return None
        return self.cache_usage_history[self.history_index]

    def get_cache_usage_history(self):
        if self.history_index < 0:
            return ()
        return tuple(self.cache_usage_history[: self.history_index + 1])


class _FakeMetricsStore:
    def __init__(self):
        self.calls = []

    def on_batch_stage_end(self, *args):
        self.calls.append(args)


class _FakeModelRunner:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def run(self, seq_metadata_list, gpu_cache, model_kwargs=None):
        self.calls.append(
            {
                "seq_metadata_list": seq_metadata_list,
                "gpu_cache": gpu_cache,
                "model_kwargs": model_kwargs,
            }
        )
        return self.output


class _WrapperExecutionModelRunner:
    def __init__(self, model, hidden_states, attention_wrapper):
        self.model = model
        self.hidden_states = hidden_states
        self.attention_wrapper = attention_wrapper
        self.calls = []

    def run(self, seq_metadata_list, gpu_cache, model_kwargs=None):
        self.calls.append(
            {
                "seq_metadata_list": seq_metadata_list,
                "gpu_cache": gpu_cache,
                "model_kwargs": model_kwargs,
            }
        )
        return self.model.forward_with_attention_wrapper(
            hidden_states=self.hidden_states,
            projection_weights=model_kwargs["projection_weights"],
            kv_caches=tuple(gpu_cache),
            attention_wrapper=self.attention_wrapper,
            caches=model_kwargs.get("caches"),
            softmax_scale=model_kwargs.get("softmax_scale"),
        )


class BaseWorkerMLARuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.flash_calls = []
        (
            self.worker_module,
            self.deepseek_module,
            self.wrapper_module,
            self.cache_engine_module,
            self.CacheArchitecture,
        ) = _load_modules(self.flash_calls)
        self.BaseWorker = self.worker_module.BaseWorker

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
        return torch.tensor([[1.0, 2.0, 3.0, 0.0, 1.0, 0.0]])

    def _make_wrapper(self):
        wrapper = self.wrapper_module.VAttentionFlashAttentionWrapper()
        wrapper.device = torch.device("cpu")
        wrapper.is_metadata_initialized = True
        wrapper.is_profiling_iteration = False
        wrapper.prefill_query_lens = [1]
        wrapper.prefill_cache_lens = [0]
        wrapper.decode_cache_lens = None
        wrapper.batch_index = torch.tensor([0], dtype=torch.int32)
        wrapper.batch_index_gen = torch.tensor([], dtype=torch.int32)
        return wrapper

    def _make_worker(self, model_runner, gpu_cache, cache_usage_stats=None, *, seq_manager=None, cache_engine=None):
        worker = self.BaseWorker.__new__(self.BaseWorker)
        worker.seq_manager = seq_manager or _FakeSeqManager(["seq-md"])
        worker.cache_engine = cache_engine or _FakeCacheEngine(cache_usage_stats=cache_usage_stats)
        worker.gpu_cache = gpu_cache
        worker.model_runner = model_runner
        worker.metrics_store = _FakeMetricsStore()
        worker.tensor_model_parallel_rank = 0
        worker.pipeline_model_parallel_rank = 0
        worker.preempt_requests = worker.cache_engine.preempt_requests
        return worker

    def test_worker_executes_mla_wrapper_path_with_component_runtime_cache(self):
        config = self._make_config()
        model = self.deepseek_module.DeepseekV2Model(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = self.deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.num_layers)
        )
        kv_latent = torch.zeros(1, 4, model.num_layers, dims.kv_lora_rank)
        k_rope = torch.zeros(
            1,
            4,
            model.num_layers,
            dims.num_heads * dims.qk_rope_head_dim,
        )
        cache_spec = types.SimpleNamespace(
            architecture=self.cache_engine_module.CacheArchitecture.MLA,
            num_layers=model.num_layers,
            num_heads=dims.num_heads,
            mla_qk_rope_head_dim=dims.qk_rope_head_dim,
        )
        gpu_cache = tuple(
            self.cache_engine_module.format_vattention_gpu_cache(
                cache_spec,
                (kv_latent, k_rope),
                torch.device("cpu"),
            )
        )
        model_runner = _WrapperExecutionModelRunner(
            model=model,
            hidden_states=self._make_hidden_states(),
            attention_wrapper=self._make_wrapper(),
        )
        cache_usage_stats = self.cache_engine_module.summarize_vattention_cache_usage(
            types.SimpleNamespace(
                architecture=self.cache_engine_module.CacheArchitecture.MLA,
                cached_token_bytes_local=model.num_layers
                * (dims.kv_lora_rank + dims.qk_rope_head_dim)
                * torch.tensor([], dtype=torch.float32).element_size(),
                page_buffer_token_bytes=(dims.kv_lora_rank + dims.qk_rope_head_dim)
                * torch.tensor([], dtype=torch.float32).element_size(),
                cache_components=(
                    types.SimpleNamespace(name="kv_latent"),
                    types.SimpleNamespace(name="k_rope"),
                ),
            ),
            [1],
        )
        worker = self._make_worker(
            model_runner=model_runner,
            gpu_cache=gpu_cache,
            cache_usage_stats=cache_usage_stats,
        )

        output, layer_caches = worker.execute_model_with_attention_wrapper(
            scheduler_outputs="scheduler",
            projection_weights=projection_weights,
            softmax_scale=0.5,
        )

        self.assertEqual(tuple(output.shape), (1, config.hidden_size))
        self.assertEqual(len(layer_caches), model.num_layers)
        self.assertTrue(all(cache.resident_cache.num_tokens == 1 for cache in layer_caches))
        self.assertEqual(worker.cache_engine.steps, [["seq-md"]])
        self.assertEqual(worker.cache_engine.completions, [["seq-md"]])
        self.assertEqual(worker.seq_manager.block_manager.free_blocks, [9])
        self.assertEqual(len(worker.metrics_store.calls), 1)
        self.assertEqual(
            model_runner.calls[0]["model_kwargs"]["projection_weights"],
            projection_weights,
        )
        self.assertEqual(model_runner.calls[0]["gpu_cache"], gpu_cache)
        self.assertEqual(
            worker.get_cache_usage_stats(),
            {
                "architecture": "mla",
                "persistent_tokens": 1,
                "persistent_bytes_per_token": 32,
                "persistent_bytes": 32,
                "page_buffer_token_bytes": 16,
                "cache_components": ("kv_latent", "k_rope"),
                "uses_component_resident_cache": True,
                "active_batch_indices": (0,),
                "active_request_count": 1,
                "free_blocks": None,
                "seq_to_batch_idx": None,
                "scheduled_batch_indices": None,
                "scheduled_prompt_batch_indices": None,
                "scheduled_decode_batch_indices": None,
            },
        )
        self.assertEqual(len(self.flash_calls), model.num_layers)
        self.assertTrue(torch.any(gpu_cache[0].kv_latent[0, 0] != 0))

    def test_worker_exposes_multi_step_mla_cache_history_across_prefill_decode_and_preemption(self):
        history = [
            {
                "event": "step",
                "architecture": "mla",
                "persistent_tokens": 2,
                "persistent_bytes_per_token": 32,
                "persistent_bytes": 64,
                "page_buffer_token_bytes": 16,
                "cache_components": ("kv_latent", "k_rope"),
                "uses_component_resident_cache": True,
                "active_batch_indices": (0,),
                "active_request_count": 1,
                "free_blocks": 8,
                "seq_to_batch_idx": {10: 0},
                "scheduled_batch_indices": (0,),
                "scheduled_prompt_batch_indices": (0,),
                "scheduled_decode_batch_indices": (),
            },
            {
                "event": "step",
                "architecture": "mla",
                "persistent_tokens": 3,
                "persistent_bytes_per_token": 32,
                "persistent_bytes": 96,
                "page_buffer_token_bytes": 16,
                "cache_components": ("kv_latent", "k_rope"),
                "uses_component_resident_cache": True,
                "active_batch_indices": (0,),
                "active_request_count": 1,
                "free_blocks": 7,
                "seq_to_batch_idx": {10: 0},
                "scheduled_batch_indices": (0,),
                "scheduled_prompt_batch_indices": (),
                "scheduled_decode_batch_indices": (0,),
            },
            {
                "event": "free_request",
                "architecture": "mla",
                "persistent_tokens": 0,
                "persistent_bytes_per_token": 32,
                "persistent_bytes": 0,
                "page_buffer_token_bytes": 16,
                "cache_components": ("kv_latent", "k_rope"),
                "uses_component_resident_cache": True,
                "active_batch_indices": (),
                "active_request_count": 0,
                "free_blocks": 9,
                "seq_to_batch_idx": {},
                "scheduled_batch_indices": (0,),
                "scheduled_prompt_batch_indices": (),
                "scheduled_decode_batch_indices": (0,),
            },
        ]
        seq_manager = _SequencedFakeSeqManager(
            [["prefill-md"], ["decode-md"], ["post-preempt-md"]]
        )
        cache_engine = _SequencedFakeCacheEngine(history)
        model_runner = _FakeModelRunner(output="sampler-output")
        worker = self._make_worker(
            model_runner=model_runner,
            gpu_cache=("gpu-cache",),
            seq_manager=seq_manager,
            cache_engine=cache_engine,
        )

        worker.execute_model(scheduler_outputs="prefill")
        first_stats = worker.get_cache_usage_stats()
        worker.execute_model(scheduler_outputs="decode")
        second_stats = worker.get_cache_usage_stats()
        worker.execute_model(
            scheduler_outputs="preempt",
            preempted_seq=[types.SimpleNamespace(seq_id=10)],
        )
        history_view = worker.get_cache_usage_history()

        self.assertEqual(first_stats["persistent_bytes"], 64)
        self.assertEqual(first_stats["scheduled_prompt_batch_indices"], (0,))
        self.assertEqual(second_stats["persistent_bytes"], 96)
        self.assertEqual(second_stats["scheduled_decode_batch_indices"], (0,))
        self.assertEqual(cache_engine.preempted, [(10,)])
        self.assertEqual([snapshot["event"] for snapshot in history_view], ["step", "step", "free_request"])
        self.assertEqual(history_view[-1]["persistent_bytes"], 0)
        self.assertEqual(history_view[-1]["free_blocks"], 9)
        self.assertEqual(history_view[-1]["seq_to_batch_idx"], {})


if __name__ == "__main__":
    unittest.main()
