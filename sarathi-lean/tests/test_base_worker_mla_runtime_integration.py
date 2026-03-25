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
    return originals, config_module.CacheArchitecture, attention_module


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

    originals, cache_architecture, attention_module = _install_stubs(call_log)
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
        sys.modules["sarathi.model_executor.attention"] = attention_module
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

    def get_cache_usage_transitions(self):
        return ()

    def get_cache_usage_summary(self):
        return None


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

    def get_cache_usage_transitions(self):
        history = self.get_cache_usage_history()
        return tuple(
            self.cache_usage_history[index + 1]
            | {
                "from_event": self.cache_usage_history[index]["event"],
                "to_event": self.cache_usage_history[index + 1]["event"],
                "persistent_token_delta": (
                    self.cache_usage_history[index + 1]["persistent_tokens"]
                    - self.cache_usage_history[index]["persistent_tokens"]
                ),
                "persistent_byte_delta": (
                    self.cache_usage_history[index + 1]["persistent_bytes"]
                    - self.cache_usage_history[index]["persistent_bytes"]
                ),
                "free_block_delta": (
                    self.cache_usage_history[index + 1]["free_blocks"]
                    - self.cache_usage_history[index]["free_blocks"]
                ),
                "active_request_delta": (
                    self.cache_usage_history[index + 1]["active_request_count"]
                    - self.cache_usage_history[index]["active_request_count"]
                ),
                "from_seq_to_batch_idx": self.cache_usage_history[index]["seq_to_batch_idx"],
                "to_seq_to_batch_idx": self.cache_usage_history[index + 1]["seq_to_batch_idx"],
                "from_active_batch_indices": self.cache_usage_history[index]["active_batch_indices"],
                "to_active_batch_indices": self.cache_usage_history[index + 1]["active_batch_indices"],
            }
            for index in range(len(history) - 1)
        )

    def get_cache_usage_summary(self):
        history = self.get_cache_usage_history()
        transitions = self.get_cache_usage_transitions()
        if not history:
            return None
        byte_deltas = [transition["persistent_byte_delta"] for transition in transitions]
        growth_deltas = [delta for delta in byte_deltas if delta > 0]
        reclaim_deltas = [-delta for delta in byte_deltas if delta < 0]
        return {
            "num_snapshots": len(history),
            "num_transitions": len(transitions),
            "peak_persistent_tokens": max(snapshot["persistent_tokens"] for snapshot in history),
            "peak_persistent_bytes": max(snapshot["persistent_bytes"] for snapshot in history),
            "final_persistent_tokens": history[-1]["persistent_tokens"],
            "final_persistent_bytes": history[-1]["persistent_bytes"],
            "min_free_blocks": min(snapshot["free_blocks"] for snapshot in history),
            "max_active_request_count": max(
                snapshot["active_request_count"] for snapshot in history
            ),
            "largest_growth_bytes": max(growth_deltas) if growth_deltas else 0,
            "largest_reclaim_bytes": max(reclaim_deltas) if reclaim_deltas else 0,
            "events": tuple(snapshot["event"] for snapshot in history),
        }


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


class _InstalledWrapperExecutionModelRunner:
    def __init__(self, model, hidden_states, attention_wrapper):
        self.model = model
        self.hidden_states = hidden_states
        self.attention_wrapper = attention_wrapper
        self.calls = []
        self.load_calls = []
        self.prefill_calls = []
        self.decode_calls = []
        self.generate_calls = []

    def load_model_weights(self, *args, **kwargs):
        self.load_calls.append((args, kwargs))
        return self.model.load_weights(*args, **kwargs)

    def run(self, seq_metadata_list, gpu_cache, model_kwargs=None):
        self.calls.append(
            {
                "seq_metadata_list": seq_metadata_list,
                "gpu_cache": gpu_cache,
                "model_kwargs": model_kwargs,
            }
        )
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        return self.model(
            hidden_states=self.hidden_states,
            kv_caches=tuple(gpu_cache),
            attention_wrapper=self.attention_wrapper,
            caches=model_kwargs.get("caches"),
            softmax_scale=model_kwargs.get("softmax_scale"),
            mlp_weights=model_kwargs.get("mlp_weights"),
        )

    def run_prefill_tokens(self, token_ids, gpu_cache, model_kwargs=None):
        self.prefill_calls.append(
            {
                "token_ids": token_ids,
                "gpu_cache": gpu_cache,
                "model_kwargs": model_kwargs,
            }
        )
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        return self.model.prefill_tokens(
            token_ids,
            kv_caches=tuple(gpu_cache),
            attention_wrapper=self.attention_wrapper,
            softmax_scale=model_kwargs.get("softmax_scale"),
            mlp_weights=model_kwargs.get("mlp_weights"),
        )

    def run_decode_tokens(self, token_ids, caches, gpu_cache, model_kwargs=None):
        self.decode_calls.append(
            {
                "token_ids": token_ids,
                "caches": caches,
                "gpu_cache": gpu_cache,
                "model_kwargs": model_kwargs,
            }
        )
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        return self.model.decode_tokens(
            token_ids,
            caches=caches,
            kv_caches=tuple(gpu_cache),
            attention_wrapper=self.attention_wrapper,
            softmax_scale=model_kwargs.get("softmax_scale"),
            mlp_weights=model_kwargs.get("mlp_weights"),
        )

    def run_greedy_generation(self, token_ids, max_new_tokens, gpu_cache, model_kwargs=None):
        self.generate_calls.append(
            {
                "token_ids": token_ids,
                "max_new_tokens": max_new_tokens,
                "gpu_cache": gpu_cache,
                "model_kwargs": model_kwargs,
            }
        )
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        return self.model.generate_greedy(
            token_ids,
            max_new_tokens=max_new_tokens,
            kv_caches=tuple(gpu_cache),
            attention_wrapper=self.attention_wrapper,
            softmax_scale=model_kwargs.get("softmax_scale"),
            mlp_weights=model_kwargs.get("mlp_weights"),
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
            vocab_size=16,
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
        return torch.tensor([[1.0, 2.0, 3.0, 0.0, 1.0, 0.0]])

    def _make_mlp_weights(self, hidden_size):
        return self.deepseek_module.make_mlp_weights(
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
            hidden_size=hidden_size,
        )

    def _make_scaffold_state_dict(
        self,
        config,
        projection_weights,
        mlp_weights,
        *,
        use_global_layer_ids=False,
        layer_offset=0,
        include_embed=True,
        include_lm_head=True,
    ):
        state_dict = {}
        if include_embed:
            state_dict["model.embed_tokens.weight"] = torch.arange(
                config.vocab_size * config.hidden_size, dtype=torch.float32
            ).view(config.vocab_size, config.hidden_size) / 1000.0
        if include_lm_head:
            state_dict["lm_head.weight"] = torch.arange(
                config.vocab_size * config.hidden_size, dtype=torch.float32
            ).view(config.vocab_size, config.hidden_size) / 1000.0
        for layer_idx, layer_projection_weights in enumerate(projection_weights):
            resolved_idx = layer_offset + layer_idx if use_global_layer_ids else layer_idx
            prefix = f"model.layers.{resolved_idx}.self_attn"
            state_dict[f"{prefix}.q_proj.weight"] = layer_projection_weights.q_proj
            state_dict[f"{prefix}.kv_latent_proj.weight"] = (
                layer_projection_weights.kv_latent_proj
            )
            state_dict[f"{prefix}.k_rope_proj.weight"] = layer_projection_weights.k_rope_proj
            state_dict[f"{prefix}.kv_up_proj.weight"] = layer_projection_weights.kv_up_proj
            state_dict[f"{prefix}.o_proj.weight"] = layer_projection_weights.o_proj
        for layer_idx, layer_mlp_weights in enumerate(mlp_weights):
            resolved_idx = layer_offset + layer_idx if use_global_layer_ids else layer_idx
            prefix = f"model.layers.{resolved_idx}.mlp"
            state_dict[f"{prefix}.gate_proj.weight"] = layer_mlp_weights.gate_proj
            state_dict[f"{prefix}.up_proj.weight"] = layer_mlp_weights.up_proj
            state_dict[f"{prefix}.down_proj.weight"] = layer_mlp_weights.down_proj
        return state_dict

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
            dims.qk_rope_head_dim,
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
        transitions = worker.get_cache_usage_transitions()

        self.assertEqual(first_stats["persistent_bytes"], 64)
        self.assertEqual(first_stats["scheduled_prompt_batch_indices"], (0,))
        self.assertEqual(second_stats["persistent_bytes"], 96)
        self.assertEqual(second_stats["scheduled_decode_batch_indices"], (0,))
        self.assertEqual(cache_engine.preempted, [(10,)])
        self.assertEqual([snapshot["event"] for snapshot in history_view], ["step", "step", "free_request"])
        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[0]["persistent_token_delta"], 1)
        self.assertEqual(transitions[0]["persistent_byte_delta"], 32)
        self.assertEqual(transitions[0]["free_block_delta"], -1)
        self.assertEqual(transitions[1]["persistent_token_delta"], -3)
        self.assertEqual(transitions[1]["persistent_byte_delta"], -96)
        self.assertEqual(transitions[1]["free_block_delta"], 2)
        self.assertEqual(history_view[-1]["persistent_bytes"], 0)
        self.assertEqual(history_view[-1]["free_blocks"], 9)
        self.assertEqual(history_view[-1]["seq_to_batch_idx"], {})
        self.assertEqual(
            worker.get_cache_usage_summary(),
            {
                "num_snapshots": 3,
                "num_transitions": 2,
                "peak_persistent_tokens": 3,
                "peak_persistent_bytes": 96,
                "final_persistent_tokens": 0,
                "final_persistent_bytes": 0,
                "min_free_blocks": 7,
                "max_active_request_count": 1,
                "largest_growth_bytes": 32,
                "largest_reclaim_bytes": 96,
                "events": ("step", "step", "free_request"),
            },
        )

    def test_worker_executes_partitioned_loaded_scaffold_with_installed_attention_wrapper(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(2)
        set_pipeline_model_parallel_rank(1)

        config = self._make_config()
        model = self.deepseek_module.DeepseekV2ForCausalLM(config)

        dims = self.deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.model.num_layers)
        )
        scaffold_state_dict = self._make_scaffold_state_dict(
            config,
            projection_weights,
            mlp_weights,
            use_global_layer_ids=True,
            layer_offset=model.model.layer_offset,
            include_embed=False,
            include_lm_head=True,
        )

        kv_latent = torch.zeros(1, 4, model.model.num_layers, dims.kv_lora_rank)
        k_rope = torch.zeros(
            1,
            4,
            model.model.num_layers,
            dims.qk_rope_head_dim,
        )
        cache_spec = types.SimpleNamespace(
            architecture=self.cache_engine_module.CacheArchitecture.MLA,
            num_layers=model.model.num_layers,
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
        model_runner = _InstalledWrapperExecutionModelRunner(
            model=model,
            hidden_states=self._make_hidden_states(),
            attention_wrapper=self._make_wrapper(),
        )
        cache_usage_stats = self.cache_engine_module.summarize_vattention_cache_usage(
            types.SimpleNamespace(
                architecture=self.cache_engine_module.CacheArchitecture.MLA,
                cached_token_bytes_local=model.model.num_layers
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
        worker.pipeline_model_parallel_rank = 1

        worker.load_model_weights(scaffold_state_dict)
        output, layer_caches = worker.execute_model_with_installed_attention_wrapper(
            scheduler_outputs="scheduler",
            softmax_scale=0.25,
        )

        self.assertEqual(worker.pipeline_model_parallel_rank, 1)
        self.assertEqual(model.model.layer_offset, 1)
        self.assertEqual(tuple(output.shape), (1, config.hidden_size))
        self.assertEqual(len(layer_caches), model.model.num_layers)
        self.assertEqual(len(model_runner.load_calls), 1)
        self.assertEqual(model_runner.calls[0]["model_kwargs"], {"softmax_scale": 0.25})
        self.assertTrue(all(cache.resident_cache.num_tokens == 1 for cache in layer_caches))
        self.assertEqual(worker.cache_engine.steps, [["seq-md"]])
        self.assertEqual(worker.cache_engine.completions, [["seq-md"]])
        self.assertEqual(len(self.flash_calls), model.model.num_layers)

    def test_worker_token_prefill_and_decode_match_full_loaded_paged_scaffold_logits(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = self.deepseek_module.DeepseekV2ForCausalLM(config)
        dims = self.deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.model.num_layers)
        )
        scaffold_state_dict = self._make_scaffold_state_dict(
            config,
            projection_weights,
            mlp_weights,
            include_embed=True,
            include_lm_head=True,
        )
        gpu_cache = model.make_runtime_mla_kv_caches(
            batch_size=1,
            max_seq_len=4,
            device=torch.device("cpu"),
        )
        wrapper = self._make_wrapper()
        model_runner = _InstalledWrapperExecutionModelRunner(
            model=model,
            hidden_states=self._make_hidden_states(),
            attention_wrapper=wrapper,
        )
        worker = self._make_worker(
            model_runner=model_runner,
            gpu_cache=gpu_cache,
            cache_usage_stats=None,
        )
        worker.load_model_weights(scaffold_state_dict)

        prompt_token_ids = torch.tensor([1, 3], dtype=torch.long)
        decode_token_ids = torch.tensor([5], dtype=torch.long)
        full_token_ids = torch.tensor([1, 3, 5], dtype=torch.long)

        wrapper.set_mla_runtime_metadata(
            prefill_query_lens=[2],
            prefill_cache_lens=[0],
            batch_index=[0],
            batch_index_gen=[],
        )
        prefill_logits, layer_caches = worker.prefill_tokens_with_installed_attention_wrapper(
            prompt_token_ids,
            softmax_scale=0.25,
        )

        wrapper.set_mla_runtime_metadata(
            prefill_query_lens=[],
            prefill_cache_lens=[],
            decode_cache_lens=[2],
            batch_index=[],
            batch_index_gen=[0],
        )
        decode_logits, next_layer_caches = worker.decode_tokens_with_installed_attention_wrapper(
            decode_token_ids,
            layer_caches,
            softmax_scale=0.25,
        )
        full_logits, full_caches = model.forward_logits(hidden_states=full_token_ids)

        self.assertEqual(tuple(prefill_logits.shape), (2, config.vocab_size))
        self.assertEqual(tuple(decode_logits.shape), (1, config.vocab_size))
        self.assertTrue(torch.allclose(decode_logits[0], full_logits[-1], atol=1e-6, rtol=1e-6))
        self.assertEqual(len(model_runner.prefill_calls), 1)
        self.assertEqual(len(model_runner.decode_calls), 1)
        self.assertEqual(model_runner.prefill_calls[0]["model_kwargs"], {"softmax_scale": 0.25})
        self.assertEqual(model_runner.decode_calls[0]["model_kwargs"], {"softmax_scale": 0.25})
        self.assertTrue(
            all(layer_cache.resident_cache.num_tokens == 3 for layer_cache in next_layer_caches)
        )
        self.assertTrue(
            all(
                layer_cache.resident_cache.kv_latent.shape == full_cache.kv_latent.shape
                and layer_cache.resident_cache.k_rope.shape == full_cache.k_rope.shape
                for layer_cache, full_cache in zip(next_layer_caches, full_caches)
            )
        )

    def test_worker_greedy_generation_matches_loaded_paged_scaffold_model_helper(self):
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_pipeline_model_parallel_rank,
            set_pipeline_model_parallel_world_size,
            set_tensor_model_parallel_world_size,
        )

        config = self._make_config()
        set_tensor_model_parallel_world_size(2)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        model = self.deepseek_module.DeepseekV2ForCausalLM(config)
        dims = self.deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self._make_projection_weights(dims) for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self._make_mlp_weights(config.hidden_size) for _ in range(model.model.num_layers)
        )
        scaffold_state_dict = self._make_scaffold_state_dict(
            config,
            projection_weights,
            mlp_weights,
            include_embed=True,
            include_lm_head=True,
        )
        worker_gpu_cache = model.make_runtime_mla_kv_caches(
            batch_size=1,
            max_seq_len=6,
            device=torch.device("cpu"),
        )
        direct_gpu_cache = model.make_runtime_mla_kv_caches(
            batch_size=1,
            max_seq_len=6,
            device=torch.device("cpu"),
        )
        wrapper = self._make_wrapper()
        model_runner = _InstalledWrapperExecutionModelRunner(
            model=model,
            hidden_states=self._make_hidden_states(),
            attention_wrapper=wrapper,
        )
        worker = self._make_worker(
            model_runner=model_runner,
            gpu_cache=worker_gpu_cache,
            cache_usage_stats=None,
        )
        worker.load_model_weights(scaffold_state_dict)

        prompt_token_ids = torch.tensor([1, 3], dtype=torch.long)
        generated_tokens, final_logits, final_caches = (
            worker.generate_greedy_with_installed_attention_wrapper(
                prompt_token_ids,
                max_new_tokens=3,
                softmax_scale=0.25,
            )
        )
        direct_generated_tokens, direct_final_logits, direct_final_caches = model.generate_greedy(
            prompt_token_ids,
            max_new_tokens=3,
            kv_caches=direct_gpu_cache,
            attention_wrapper=wrapper,
            softmax_scale=0.25,
        )

        self.assertTrue(torch.equal(generated_tokens, direct_generated_tokens))
        self.assertTrue(torch.allclose(final_logits, direct_final_logits, atol=1e-6, rtol=1e-6))
        self.assertEqual(len(model_runner.generate_calls), 1)
        self.assertEqual(model_runner.generate_calls[0]["model_kwargs"], {"softmax_scale": 0.25})
        self.assertTrue(
            all(layer_cache.resident_cache.num_tokens == 4 for layer_cache in final_caches)
        )
        self.assertTrue(
            all(
                layer_cache.resident_cache.kv_latent.shape == direct_cache.resident_cache.kv_latent.shape
                and layer_cache.resident_cache.k_rope.shape == direct_cache.resident_cache.k_rope.shape
                for layer_cache, direct_cache in zip(final_caches, direct_final_caches)
            )
        )

    def test_worker_can_compare_multiple_mla_runtime_patterns_via_sweep_summaries(self):
        patterns = [
            {
                "name": "single_seq_grow_then_free",
                "history": [
                    {
                        "event": "step",
                        "persistent_tokens": 2,
                        "persistent_bytes": 64,
                        "free_blocks": 8,
                        "active_request_count": 1,
                        "seq_to_batch_idx": {10: 0},
                        "active_batch_indices": (0,),
                    },
                    {
                        "event": "step",
                        "persistent_tokens": 3,
                        "persistent_bytes": 96,
                        "free_blocks": 7,
                        "active_request_count": 1,
                        "seq_to_batch_idx": {10: 0},
                        "active_batch_indices": (0,),
                    },
                    {
                        "event": "free_request",
                        "persistent_tokens": 0,
                        "persistent_bytes": 0,
                        "free_blocks": 9,
                        "active_request_count": 0,
                        "seq_to_batch_idx": {},
                        "active_batch_indices": (),
                    },
                ],
            },
            {
                "name": "overlap_two_reqs",
                "history": [
                    {
                        "event": "step",
                        "persistent_tokens": 2,
                        "persistent_bytes": 64,
                        "free_blocks": 8,
                        "active_request_count": 1,
                        "seq_to_batch_idx": {10: 0},
                        "active_batch_indices": (0,),
                    },
                    {
                        "event": "step",
                        "persistent_tokens": 5,
                        "persistent_bytes": 160,
                        "free_blocks": 5,
                        "active_request_count": 2,
                        "seq_to_batch_idx": {10: 0, 20: 1},
                        "active_batch_indices": (0, 1),
                    },
                    {
                        "event": "free_request",
                        "persistent_tokens": 1,
                        "persistent_bytes": 32,
                        "free_blocks": 7,
                        "active_request_count": 1,
                        "seq_to_batch_idx": {20: 1},
                        "active_batch_indices": (1,),
                    },
                ],
            },
        ]

        pattern_summaries = []
        for pattern in patterns:
            cache_engine = _SequencedFakeCacheEngine(pattern["history"])
            worker = self._make_worker(
                model_runner=_FakeModelRunner(output="sampler-output"),
                gpu_cache=("gpu-cache",),
                seq_manager=_SequencedFakeSeqManager([["a"], ["b"], ["c"]]),
                cache_engine=cache_engine,
            )
            worker.execute_model(scheduler_outputs="step-1")
            worker.execute_model(scheduler_outputs="step-2")
            worker.execute_model(
                scheduler_outputs="step-3",
                preempted_seq=[types.SimpleNamespace(seq_id=10)],
            )
            pattern_summaries.append(
                worker.get_cache_usage_summary() | {"pattern_name": pattern["name"]}
            )

        sweep_summary = self.cache_engine_module.summarize_vattention_cache_sweeps(
            pattern_summaries
        )

        self.assertEqual(sweep_summary["num_patterns"], 2)
        self.assertEqual(
            sweep_summary["pattern_names"],
            ("single_seq_grow_then_free", "overlap_two_reqs"),
        )
        self.assertEqual(sweep_summary["max_peak_persistent_bytes"], 160)
        self.assertEqual(sweep_summary["min_free_blocks_overall"], 5)
        self.assertEqual(sweep_summary["max_largest_growth_bytes"], 96)
        self.assertEqual(sweep_summary["max_largest_reclaim_bytes"], 128)
        self.assertEqual(
            sweep_summary["pattern_with_max_peak_bytes"],
            "overlap_two_reqs",
        )
        self.assertEqual(
            sweep_summary["pattern_with_min_free_blocks"],
            "overlap_two_reqs",
        )

    def test_worker_can_compare_mla_runtime_sweep_families(self):
        families = [
            {
                "family_name": "prompt_length_matrix",
                "patterns": [
                    {
                        "name": "short_prompt",
                        "history": [
                            {
                                "event": "step",
                                "persistent_tokens": 2,
                                "persistent_bytes": 64,
                                "free_blocks": 8,
                                "active_request_count": 1,
                                "seq_to_batch_idx": {10: 0},
                                "active_batch_indices": (0,),
                            },
                            {
                                "event": "free_request",
                                "persistent_tokens": 0,
                                "persistent_bytes": 0,
                                "free_blocks": 9,
                                "active_request_count": 0,
                                "seq_to_batch_idx": {},
                                "active_batch_indices": (),
                            },
                        ],
                    },
                    {
                        "name": "long_prompt",
                        "history": [
                            {
                                "event": "step",
                                "persistent_tokens": 4,
                                "persistent_bytes": 128,
                                "free_blocks": 6,
                                "active_request_count": 1,
                                "seq_to_batch_idx": {20: 0},
                                "active_batch_indices": (0,),
                            },
                            {
                                "event": "free_request",
                                "persistent_tokens": 0,
                                "persistent_bytes": 0,
                                "free_blocks": 9,
                                "active_request_count": 0,
                                "seq_to_batch_idx": {},
                                "active_batch_indices": (),
                            },
                        ],
                    },
                ],
            },
            {
                "family_name": "overlap_matrix",
                "patterns": [
                    {
                        "name": "single_req",
                        "history": [
                            {
                                "event": "step",
                                "persistent_tokens": 3,
                                "persistent_bytes": 96,
                                "free_blocks": 7,
                                "active_request_count": 1,
                                "seq_to_batch_idx": {30: 0},
                                "active_batch_indices": (0,),
                            },
                            {
                                "event": "free_request",
                                "persistent_tokens": 0,
                                "persistent_bytes": 0,
                                "free_blocks": 9,
                                "active_request_count": 0,
                                "seq_to_batch_idx": {},
                                "active_batch_indices": (),
                            },
                        ],
                    },
                    {
                        "name": "overlap_two_reqs",
                        "history": [
                            {
                                "event": "step",
                                "persistent_tokens": 2,
                                "persistent_bytes": 64,
                                "free_blocks": 8,
                                "active_request_count": 1,
                                "seq_to_batch_idx": {40: 0},
                                "active_batch_indices": (0,),
                            },
                            {
                                "event": "step",
                                "persistent_tokens": 5,
                                "persistent_bytes": 160,
                                "free_blocks": 5,
                                "active_request_count": 2,
                                "seq_to_batch_idx": {40: 0, 41: 1},
                                "active_batch_indices": (0, 1),
                            },
                            {
                                "event": "free_request",
                                "persistent_tokens": 1,
                                "persistent_bytes": 32,
                                "free_blocks": 7,
                                "active_request_count": 1,
                                "seq_to_batch_idx": {41: 1},
                                "active_batch_indices": (1,),
                            },
                        ],
                    },
                ],
            },
        ]

        family_summaries = []
        for family in families:
            pattern_summaries = []
            for pattern in family["patterns"]:
                cache_engine = _SequencedFakeCacheEngine(pattern["history"])
                seq_lists = [["step-1"], ["step-2"], ["step-3"]][: len(pattern["history"])]
                worker = self._make_worker(
                    model_runner=_FakeModelRunner(output="sampler-output"),
                    gpu_cache=("gpu-cache",),
                    seq_manager=_SequencedFakeSeqManager(seq_lists),
                    cache_engine=cache_engine,
                )
                worker.execute_model(scheduler_outputs="step-1")
                if len(pattern["history"]) > 1:
                    worker.execute_model(scheduler_outputs="step-2")
                if len(pattern["history"]) > 2:
                    worker.execute_model(
                        scheduler_outputs="step-3",
                        preempted_seq=[types.SimpleNamespace(seq_id=40)],
                    )
                pattern_summaries.append(
                    worker.get_cache_usage_summary() | {"pattern_name": pattern["name"]}
                )
            family_summaries.append(
                self.cache_engine_module.summarize_vattention_cache_sweep_family(
                    family["family_name"],
                    pattern_summaries,
                )
            )

        matrix_summary = self.cache_engine_module.summarize_vattention_cache_sweep_matrix(
            family_summaries
        )

        self.assertEqual(len(family_summaries), 2)
        self.assertEqual(family_summaries[0]["family_name"], "prompt_length_matrix")
        self.assertEqual(family_summaries[0]["max_peak_persistent_bytes"], 128)
        self.assertEqual(family_summaries[1]["family_name"], "overlap_matrix")
        self.assertEqual(family_summaries[1]["min_free_blocks_overall"], 5)
        self.assertEqual(matrix_summary["num_families"], 2)
        self.assertEqual(
            matrix_summary["family_names"],
            ("prompt_length_matrix", "overlap_matrix"),
        )
        self.assertEqual(matrix_summary["max_peak_persistent_bytes"], 160)
        self.assertEqual(matrix_summary["min_free_blocks_overall"], 5)
        self.assertEqual(matrix_summary["max_largest_growth_bytes"], 96)
        self.assertEqual(matrix_summary["max_largest_reclaim_bytes"], 128)
        self.assertEqual(matrix_summary["family_with_max_peak_bytes"], "overlap_matrix")
        self.assertEqual(matrix_summary["family_with_min_free_blocks"], "overlap_matrix")
        validation = self.cache_engine_module.validate_vattention_cache_sweep_matrix(
            matrix_summary,
            max_peak_persistent_bytes=160,
            min_free_blocks_overall=5,
            max_largest_growth_bytes=96,
            max_largest_reclaim_bytes=128,
        )
        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["violations"], ())

    def test_worker_can_validate_multiple_mla_runtime_matrices_as_one_suite(self):
        matrix_summaries = (
            {
                "matrix_name": "prompt_matrix",
                "max_peak_persistent_bytes": 128,
                "min_free_blocks_overall": 6,
                "max_largest_growth_bytes": 128,
                "max_largest_reclaim_bytes": 128,
            },
            {
                "matrix_name": "overlap_matrix",
                "max_peak_persistent_bytes": 160,
                "min_free_blocks_overall": 5,
                "max_largest_growth_bytes": 96,
                "max_largest_reclaim_bytes": 128,
            },
            {
                "matrix_name": "decode_pressure_matrix",
                "max_peak_persistent_bytes": 96,
                "min_free_blocks_overall": 7,
                "max_largest_growth_bytes": 32,
                "max_largest_reclaim_bytes": 96,
            },
        )

        suite_summary = self.cache_engine_module.summarize_vattention_cache_validation_suite(
            matrix_summaries
        )
        validation = self.cache_engine_module.validate_vattention_cache_validation_suite(
            suite_summary,
            max_peak_persistent_bytes=160,
            min_free_blocks_overall=5,
            max_largest_growth_bytes=128,
            max_largest_reclaim_bytes=128,
        )

        self.assertEqual(suite_summary["num_matrices"], 3)
        self.assertEqual(
            suite_summary["matrix_names"],
            ("prompt_matrix", "overlap_matrix", "decode_pressure_matrix"),
        )
        self.assertEqual(suite_summary["max_peak_persistent_bytes"], 160)
        self.assertEqual(suite_summary["min_free_blocks_overall"], 5)
        self.assertEqual(suite_summary["max_largest_growth_bytes"], 128)
        self.assertEqual(suite_summary["max_largest_reclaim_bytes"], 128)
        self.assertEqual(suite_summary["matrix_with_max_peak_bytes"], "overlap_matrix")
        self.assertEqual(suite_summary["matrix_with_min_free_blocks"], "overlap_matrix")
        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["violations"], ())
        profile_report = self.cache_engine_module.compare_vattention_cache_validation_suite_to_profile(
            suite_summary,
            {
                "profile_name": "bounded_mla_suite_v1",
                "max_peak_persistent_bytes": 160,
                "min_free_blocks_overall": 5,
                "max_largest_growth_bytes": 128,
                "max_largest_reclaim_bytes": 128,
            },
        )
        self.assertEqual(profile_report["profile_name"], "bounded_mla_suite_v1")
        self.assertTrue(profile_report["is_valid"])
        self.assertEqual(profile_report["violations"], ())
        worker = self._make_worker(
            model_runner=_FakeModelRunner(output="sampler-output"),
            gpu_cache=("gpu-cache",),
        )
        named_profile_report = worker.evaluate_cache_usage_suite_profile(
            suite_summary,
            "bounded_mla_suite_v1",
        )
        self.assertEqual(named_profile_report["profile_name"], "bounded_mla_suite_v1")
        self.assertTrue(named_profile_report["is_valid"])
        self.assertEqual(named_profile_report["violations"], ())
        multi_profile_reports = worker.evaluate_cache_usage_suite_profiles(suite_summary)
        self.assertEqual(len(multi_profile_reports), 2)
        self.assertEqual(multi_profile_reports[0]["profile_name"], "bounded_mla_suite_v1")
        selected_profile = worker.select_cache_usage_suite_profile(suite_summary)
        self.assertIsNotNone(selected_profile)
        self.assertEqual(selected_profile["profile_name"], "bounded_mla_suite_v1")

    def test_worker_can_select_relaxed_profile_when_strict_profile_fails(self):
        suite_summary = {
            "num_matrices": 3,
            "matrix_names": ("prompt_matrix", "overlap_matrix", "decode_pressure_matrix"),
            "max_peak_persistent_bytes": 176,
            "min_free_blocks_overall": 4,
            "max_largest_growth_bytes": 144,
            "max_largest_reclaim_bytes": 144,
            "matrix_with_max_peak_bytes": "overlap_matrix",
            "matrix_with_min_free_blocks": "overlap_matrix",
        }
        worker = self._make_worker(
            model_runner=_FakeModelRunner(output="sampler-output"),
            gpu_cache=("gpu-cache",),
        )

        selected_profile = worker.select_cache_usage_suite_profile(suite_summary)

        self.assertIsNotNone(selected_profile)
        self.assertEqual(selected_profile["profile_name"], "bounded_mla_suite_relaxed")

    def test_worker_can_recommend_cache_usage_profile_readiness(self):
        worker = self._make_worker(
            model_runner=_FakeModelRunner(output="sampler-output"),
            gpu_cache=("gpu-cache",),
        )

        ready = worker.recommend_cache_usage_suite_profile(
            {
                "max_peak_persistent_bytes": 160,
                "min_free_blocks_overall": 5,
                "max_largest_growth_bytes": 128,
                "max_largest_reclaim_bytes": 128,
            }
        )
        relaxed = worker.recommend_cache_usage_suite_profile(
            {
                "max_peak_persistent_bytes": 176,
                "min_free_blocks_overall": 4,
                "max_largest_growth_bytes": 144,
                "max_largest_reclaim_bytes": 144,
            }
        )
        blocked = worker.recommend_cache_usage_suite_profile(
            {
                "max_peak_persistent_bytes": 256,
                "min_free_blocks_overall": 3,
                "max_largest_growth_bytes": 192,
                "max_largest_reclaim_bytes": 192,
            }
        )

        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["selected_profile"], "bounded_mla_suite_v1")
        self.assertEqual(relaxed["status"], "relaxed_only")
        self.assertEqual(relaxed["selected_profile"], "bounded_mla_suite_relaxed")
        self.assertEqual(blocked["status"], "blocked")
        self.assertIsNone(blocked["selected_profile"])


if __name__ == "__main__":
    unittest.main()
