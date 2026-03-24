import importlib.util
import sys
import types
import unittest
from pathlib import Path


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


def _install_worker_stubs():
    originals = {
        name: sys.modules.get(name)
        for name in [
            "sarathi.config",
            "sarathi.core.datatypes.scheduler_output",
            "sarathi.core.datatypes.sequence",
            "sarathi.core.sequence_manager.worker_sequence_manager",
            "sarathi.logger",
            "sarathi.metrics.metrics_store",
            "sarathi.model_executor",
            "sarathi.model_executor.attention",
            "sarathi.model_executor.model_runner",
            "sarathi.model_executor.parallel_utils.parallel_state",
            "sarathi.utils.threading_utils",
            "sarathi.worker.cache_engine",
        ]
    }

    config_module = types.ModuleType("sarathi.config")
    config_module.BaseSchedulerConfig = object
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
    sys.modules["sarathi.core.datatypes.sequence"] = sequence_module

    seq_manager_module = types.ModuleType("sarathi.core.sequence_manager.worker_sequence_manager")
    seq_manager_module.WorkerSequenceManager = object
    sys.modules["sarathi.core.sequence_manager.worker_sequence_manager"] = seq_manager_module

    logger_module = types.ModuleType("sarathi.logger")
    logger_module.init_logger = lambda name: types.SimpleNamespace(info=lambda *args, **kwargs: None)
    sys.modules["sarathi.logger"] = logger_module

    metrics_store_module = types.ModuleType("sarathi.metrics.metrics_store")
    metrics_store_module.MetricsStore = object
    sys.modules["sarathi.metrics.metrics_store"] = metrics_store_module

    model_executor_module = types.ModuleType("sarathi.model_executor")
    model_executor_module.set_random_seed = lambda seed: None
    sys.modules["sarathi.model_executor"] = model_executor_module

    attention_module = types.ModuleType("sarathi.model_executor.attention")
    attention_module.set_attention_backend = lambda backend: None
    sys.modules["sarathi.model_executor.attention"] = attention_module

    model_runner_module = types.ModuleType("sarathi.model_executor.model_runner")
    model_runner_module.ModelRunner = object
    sys.modules["sarathi.model_executor.model_runner"] = model_runner_module

    parallel_state_module = types.ModuleType("sarathi.model_executor.parallel_utils.parallel_state")
    parallel_state_module.get_pipeline_model_parallel_rank = lambda: 0
    parallel_state_module.get_tensor_model_parallel_rank = lambda: 0
    parallel_state_module.initialize_model_parallel = lambda *args, **kwargs: None
    sys.modules["sarathi.model_executor.parallel_utils.parallel_state"] = parallel_state_module

    threading_utils_module = types.ModuleType("sarathi.utils.threading_utils")
    threading_utils_module.synchronized = lambda fn: fn
    sys.modules["sarathi.utils.threading_utils"] = threading_utils_module

    cache_engine_module = types.ModuleType("sarathi.worker.cache_engine")
    cache_engine_module.get_cache_engine = lambda backend: None
    cache_engine_module.get_cache_mem_alloc_backend = lambda backend: "noop"
    sys.modules["sarathi.worker.cache_engine"] = cache_engine_module

    return originals


def _restore_worker_stubs(originals):
    for module_name, original in originals.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def _load_worker_module():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.worker", SARATHI_ROOT / "worker")

    originals = _install_worker_stubs()
    project_original = sys.modules.get("sarathi.worker.base_worker")
    try:
        worker_module = _load_module(
            "sarathi.worker.base_worker",
            SARATHI_ROOT / "worker" / "base_worker.py",
        )
    finally:
        _restore_worker_stubs(originals)
        if project_original is None:
            sys.modules.pop("sarathi.worker.base_worker", None)
        else:
            sys.modules["sarathi.worker.base_worker"] = project_original
    return worker_module


worker_module = _load_worker_module()
BaseWorker = worker_module.BaseWorker


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


class _FakeCacheEngine:
    def __init__(self):
        self.steps = []
        self.completions = []
        self.free_blocks = 17

    def num_free_blocks(self):
        return self.free_blocks

    def step(self, seq_metadata_list):
        self.steps.append(seq_metadata_list)

    def on_step_completion(self, seq_metadata_list):
        self.completions.append(seq_metadata_list)

    def preempt_requests(self, preempted_seq):
        pass


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


class _FakeMetricsStore:
    def __init__(self):
        self.calls = []

    def on_batch_stage_end(self, *args):
        self.calls.append(args)


class BaseWorkerMLADispatchTests(unittest.TestCase):
    def _make_worker(self):
        seq_metadata_list = ["seq-md"]
        worker = BaseWorker.__new__(BaseWorker)
        worker.seq_manager = _FakeSeqManager(seq_metadata_list)
        worker.cache_engine = _FakeCacheEngine()
        worker.gpu_cache = ("gpu-cache",)
        worker.model_runner = _FakeModelRunner(output="sampler-output")
        worker.metrics_store = _FakeMetricsStore()
        worker.tensor_model_parallel_rank = 0
        worker.pipeline_model_parallel_rank = 0
        worker.preempt_requests = lambda preempted_seq: None
        return worker, seq_metadata_list

    def test_execute_model_forwards_model_runner_kwargs(self):
        worker, seq_metadata_list = self._make_worker()

        output = worker.execute_model(
            scheduler_outputs="scheduler",
            model_runner_kwargs={"projection_weights": ("proj",)},
        )

        self.assertEqual(output, "sampler-output")
        self.assertEqual(worker.cache_engine.steps, [seq_metadata_list])
        self.assertEqual(worker.cache_engine.completions, [seq_metadata_list])
        self.assertEqual(
            worker.model_runner.calls[0]["model_kwargs"],
            {"projection_weights": ("proj",)},
        )
        self.assertEqual(worker.model_runner.calls[0]["gpu_cache"], ("gpu-cache",))
        self.assertEqual(worker.seq_manager.block_manager.free_blocks, [17])
        self.assertEqual(worker.seq_manager.completed, [("scheduler", "sampler-output")])
        self.assertEqual(len(worker.metrics_store.calls), 1)

    def test_execute_model_defaults_model_runner_kwargs_to_none(self):
        worker, _ = self._make_worker()

        worker.execute_model(scheduler_outputs="scheduler")

        self.assertIsNone(worker.model_runner.calls[0]["model_kwargs"])

    def test_execute_model_with_attention_wrapper_packages_mla_kwargs(self):
        worker, _ = self._make_worker()

        worker.execute_model_with_attention_wrapper(
            scheduler_outputs="scheduler",
            projection_weights=("proj",),
            caches=("resident",),
            softmax_scale=0.25,
        )

        self.assertEqual(
            worker.model_runner.calls[0]["model_kwargs"],
            {
                "projection_weights": ("proj",),
                "caches": ("resident",),
                "softmax_scale": 0.25,
            },
        )


if __name__ == "__main__":
    unittest.main()
