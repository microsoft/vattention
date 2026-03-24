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


def _install_cache_engine_stubs():
    originals = {
        name: sys.modules.get(name)
        for name in [
            "sarathi.core.datatypes.sequence",
            "sarathi.config",
            "sarathi.logger",
            "sarathi.model_executor.attention",
            "sarathi.utils",
            "sarathi.worker.cache_engine.base_cache_engine",
            "sarathi.worker.cache_engine.vattention_init",
            "sarathi.model_executor.models.deepseek_v2",
            "vattention",
        ]
    }

    sequence_module = types.ModuleType("sarathi.core.datatypes.sequence")
    sequence_module.Sequence = object
    sequence_module.SequenceMetadata = object
    sys.modules["sarathi.core.datatypes.sequence"] = sequence_module

    config_module = types.ModuleType("sarathi.config")

    class CacheArchitecture(Enum):
        DENSE_KV = "dense_kv"
        MLA = "mla"

    config_module.CacheArchitecture = CacheArchitecture
    config_module.CacheConfig = object
    config_module.ModelConfig = object
    config_module.ParallelConfig = object
    sys.modules["sarathi.config"] = config_module

    logger_module = types.ModuleType("sarathi.logger")
    logger_module.init_logger = lambda name: types.SimpleNamespace()
    sys.modules["sarathi.logger"] = logger_module

    attention_module = types.ModuleType("sarathi.model_executor.attention")
    attention_module.get_attention_wrapper = lambda: None
    sys.modules["sarathi.model_executor.attention"] = attention_module

    utils_module = types.ModuleType("sarathi.utils")
    utils_module.in_wsl = lambda: False
    sys.modules["sarathi.utils"] = utils_module

    base_cache_engine_module = types.ModuleType(
        "sarathi.worker.cache_engine.base_cache_engine"
    )
    base_cache_engine_module.BaseCacheEngine = object
    sys.modules["sarathi.worker.cache_engine.base_cache_engine"] = (
        base_cache_engine_module
    )

    vattention_init_module = types.ModuleType(
        "sarathi.worker.cache_engine.vattention_init"
    )
    vattention_init_module.dispatch_init_kvcache = lambda backend, request: None
    sys.modules["sarathi.worker.cache_engine.vattention_init"] = vattention_init_module

    deepseek_module = types.ModuleType("sarathi.model_executor.models.deepseek_v2")

    class DeepseekV2ComponentMLAKVCache:
        def __init__(self, kv_latent, k_rope):
            self.kv_latent = kv_latent
            self.k_rope = k_rope

    deepseek_module.DeepseekV2ComponentMLAKVCache = DeepseekV2ComponentMLAKVCache
    sys.modules["sarathi.model_executor.models.deepseek_v2"] = deepseek_module

    sys.modules["vattention"] = types.ModuleType("vattention")
    return originals


def _restore_cache_engine_stubs(originals):
    for module_name, original in originals.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def _load_cache_engine_module():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.worker", SARATHI_ROOT / "worker")
    _ensure_package("sarathi.worker.cache_engine", SARATHI_ROOT / "worker" / "cache_engine")
    originals = _install_cache_engine_stubs()
    try:
        module = _load_module(
            "sarathi.worker.cache_engine.vATTN_cache_engine",
            SARATHI_ROOT / "worker" / "cache_engine" / "vATTN_cache_engine.py",
        )
        cache_architecture = sys.modules["sarathi.config"].CacheArchitecture
        deepseek_stub = sys.modules["sarathi.model_executor.models.deepseek_v2"]
    finally:
        _restore_cache_engine_stubs(originals)
    return module, cache_architecture, deepseek_stub


cache_engine_module, CacheArchitecture, DEEPSEEK_STUB = _load_cache_engine_module()
format_vattention_gpu_cache = cache_engine_module.format_vattention_gpu_cache
summarize_vattention_cache_usage = cache_engine_module.summarize_vattention_cache_usage


class VAttentionCacheEngineRuntimeCacheTests(unittest.TestCase):
    def setUp(self):
        self._original_deepseek_module = sys.modules.get(
            "sarathi.model_executor.models.deepseek_v2"
        )
        sys.modules["sarathi.model_executor.models.deepseek_v2"] = DEEPSEEK_STUB

    def tearDown(self):
        if self._original_deepseek_module is None:
            sys.modules.pop("sarathi.model_executor.models.deepseek_v2", None)
        else:
            sys.modules["sarathi.model_executor.models.deepseek_v2"] = (
                self._original_deepseek_module
            )

    def test_component_spec_mla_cache_formats_per_layer_component_cache_objects(self):
        batch_size = 2
        max_seq_len = 3
        num_layers = 2
        kv_lora_rank = 3
        num_heads = 2
        qk_rope_head_dim = 1
        kv_latent = torch.arange(
            batch_size * max_seq_len * num_layers * kv_lora_rank,
            dtype=torch.float32,
        ).view(batch_size, max_seq_len, num_layers, kv_lora_rank)
        k_rope = torch.arange(
            batch_size * max_seq_len * num_layers * num_heads * qk_rope_head_dim,
            dtype=torch.float32,
        ).view(batch_size, max_seq_len, num_layers, num_heads * qk_rope_head_dim)

        cache_spec = types.SimpleNamespace(
            architecture=CacheArchitecture.MLA,
            num_layers=num_layers,
            num_heads=num_heads,
            mla_qk_rope_head_dim=qk_rope_head_dim,
        )

        caches = format_vattention_gpu_cache(
            cache_spec,
            (kv_latent, k_rope),
            torch.device("cpu"),
        )

        self.assertEqual(len(caches), num_layers)
        self.assertEqual(tuple(caches[0].kv_latent.shape), (batch_size, max_seq_len, kv_lora_rank))
        self.assertEqual(
            tuple(caches[0].k_rope.shape),
            (batch_size, max_seq_len, num_heads, qk_rope_head_dim),
        )
        self.assertTrue(torch.equal(caches[1].kv_latent, kv_latent[:, :, 1, :]))
        self.assertTrue(
            torch.equal(
                caches[1].k_rope,
                k_rope[:, :, 1, :].view(batch_size, max_seq_len, num_heads, qk_rope_head_dim),
            )
        )

    def test_dense_megacache_formatting_is_unchanged(self):
        k_cache = torch.zeros(2, 4, 3, 5)
        v_cache = torch.zeros(2, 4, 3, 5)
        cache_spec = types.SimpleNamespace(
            architecture=CacheArchitecture.DENSE_KV,
            megacache=True,
            num_layers=3,
        )

        caches = format_vattention_gpu_cache(
            cache_spec,
            (k_cache, v_cache),
            torch.device("cpu"),
        )

        self.assertEqual(len(caches), 3)
        self.assertEqual(tuple(caches[0][0].shape), (2, 4, 5))
        self.assertEqual(tuple(caches[0][1].shape), (2, 4, 5))

    def test_mla_cache_usage_summary_counts_only_resident_component_bytes(self):
        cache_spec = types.SimpleNamespace(
            architecture=CacheArchitecture.MLA,
            cached_token_bytes_local=32,
            page_buffer_token_bytes=16,
            cache_components=(
                types.SimpleNamespace(name="kv_latent"),
                types.SimpleNamespace(name="k_rope"),
            ),
        )

        usage = summarize_vattention_cache_usage(
            cache_spec,
            [3, 0, 2],
            free_blocks=7,
            seq_to_batch_idx={11: 2, 10: 0},
        )

        self.assertEqual(usage["architecture"], "mla")
        self.assertEqual(usage["persistent_tokens"], 5)
        self.assertEqual(usage["persistent_bytes_per_token"], 32)
        self.assertEqual(usage["persistent_bytes"], 160)
        self.assertEqual(usage["page_buffer_token_bytes"], 16)
        self.assertEqual(usage["cache_components"], ("kv_latent", "k_rope"))
        self.assertTrue(usage["uses_component_resident_cache"])
        self.assertEqual(usage["active_batch_indices"], (0, 2))
        self.assertEqual(usage["active_request_count"], 2)
        self.assertEqual(usage["free_blocks"], 7)
        self.assertEqual(usage["seq_to_batch_idx"], {10: 0, 11: 2})

    def test_dense_cache_usage_summary_remains_non_mla(self):
        cache_spec = types.SimpleNamespace(
            architecture=CacheArchitecture.DENSE_KV,
            cached_token_bytes_local=64,
            page_buffer_token_bytes=32,
            cache_components=(
                types.SimpleNamespace(name="k"),
                types.SimpleNamespace(name="v"),
            ),
        )

        usage = summarize_vattention_cache_usage(cache_spec, [1, 2])

        self.assertEqual(usage["architecture"], "dense_kv")
        self.assertEqual(usage["persistent_tokens"], 3)
        self.assertEqual(usage["persistent_bytes"], 192)
        self.assertFalse(usage["uses_component_resident_cache"])
        self.assertEqual(usage["active_batch_indices"], (0, 1))
        self.assertEqual(usage["active_request_count"], 2)
        self.assertIsNone(usage["free_blocks"])
        self.assertIsNone(usage["seq_to_batch_idx"])

    def test_engine_cache_usage_stats_tracks_active_slots_and_free_blocks(self):
        engine = cache_engine_module.vATTNCacheEngine.__new__(
            cache_engine_module.vATTNCacheEngine
        )
        engine.cache_spec = types.SimpleNamespace(
            architecture=CacheArchitecture.MLA,
            cached_token_bytes_local=32,
            page_buffer_token_bytes=16,
            cache_components=(
                types.SimpleNamespace(name="kv_latent"),
                types.SimpleNamespace(name="k_rope"),
            ),
        )
        engine.curr_seq_lens = [3, 0, 2]
        engine.seq_to_batch_idx = {21: 2, 20: 0}
        engine.num_free_blocks = lambda: 5

        usage = engine.get_cache_usage_stats()

        self.assertEqual(usage["persistent_tokens"], 5)
        self.assertEqual(usage["free_blocks"], 5)
        self.assertEqual(usage["active_batch_indices"], (0, 2))
        self.assertEqual(usage["active_request_count"], 2)
        self.assertEqual(usage["seq_to_batch_idx"], {20: 0, 21: 2})

    def test_free_request_updates_runtime_state_for_accounting(self):
        freed_batch_indices = []
        cache_engine_module.vattention.free_batch_idx = freed_batch_indices.append

        engine = cache_engine_module.vATTNCacheEngine.__new__(
            cache_engine_module.vATTNCacheEngine
        )
        engine.curr_seq_lens = [4, 2, 0]
        engine.seq_to_batch_idx = {100: 0, 200: 1}

        engine.free_request(100)

        self.assertEqual(freed_batch_indices, [0])
        self.assertEqual(engine.curr_seq_lens, [0, 2, 0])
        self.assertEqual(engine.seq_to_batch_idx, {200: 1})


if __name__ == "__main__":
    unittest.main()
