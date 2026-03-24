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
summarize_vattention_cache_transition = (
    cache_engine_module.summarize_vattention_cache_transition
)


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
        self.assertIsNone(usage["scheduled_batch_indices"])
        self.assertIsNone(usage["scheduled_prompt_batch_indices"])
        self.assertIsNone(usage["scheduled_decode_batch_indices"])

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
        self.assertIsNone(usage["scheduled_batch_indices"])
        self.assertIsNone(usage["scheduled_prompt_batch_indices"])
        self.assertIsNone(usage["scheduled_decode_batch_indices"])

    def test_cache_usage_transition_summarizes_runtime_deltas(self):
        transition = summarize_vattention_cache_transition(
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
        )

        self.assertEqual(transition["from_event"], "step")
        self.assertEqual(transition["to_event"], "free_request")
        self.assertEqual(transition["persistent_token_delta"], -2)
        self.assertEqual(transition["persistent_byte_delta"], -64)
        self.assertEqual(transition["free_block_delta"], 1)
        self.assertEqual(transition["active_request_delta"], -1)
        self.assertEqual(transition["from_seq_to_batch_idx"], {10: 0})
        self.assertEqual(transition["to_seq_to_batch_idx"], {})

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
        self.assertIsNone(usage["scheduled_batch_indices"])

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

    def test_step_updates_scheduled_batch_state_and_runtime_accounting(self):
        class _FakeWrapper:
            def __init__(self):
                self.calls = []

            def set_batch_idx(self, batch_idx, batch_idx_gen):
                self.calls.append((batch_idx.clone(), batch_idx_gen.clone()))

        class _FakePromptSeq:
            def __init__(self, seq_id, processed_prompt_len, next_prompt_chunk_len):
                self.seq_id = seq_id
                self._processed_prompt_len = processed_prompt_len
                self._next_prompt_chunk_len = next_prompt_chunk_len

            def get_next_prompt_chunk_len(self, prompt_chunk_len):
                return min(prompt_chunk_len, self._next_prompt_chunk_len)

            def get_num_prompt_tokens_processed(self):
                return self._processed_prompt_len

        class _FakeDecodeSeq:
            def __init__(self, seq_id, seq_len):
                self.seq_id = seq_id
                self._seq_len = seq_len

            def get_len(self):
                return self._seq_len

        free_blocks_state = {"value": 6}
        next_batch_idx_state = {"value": 0}
        step_calls = []
        wrapper = _FakeWrapper()

        def _alloc_new_batch_idx(seq_len):
            del seq_len
            batch_idx = next_batch_idx_state["value"]
            next_batch_idx_state["value"] += 1
            return batch_idx

        cache_engine_module.vattention.alloc_new_batch_idx = _alloc_new_batch_idx
        cache_engine_module.vattention.step = lambda seq_lens, sync: step_calls.append(
            (list(seq_lens), sync)
        )
        cache_engine_module.get_attention_wrapper = lambda: wrapper

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
        engine.curr_seq_lens = [0, 0, 0, 0]
        engine.seq_to_batch_idx = {}
        engine.device = torch.device("cpu")
        engine.vattn_async = False
        engine.num_free_blocks = lambda: free_blocks_state["value"]
        engine.prompt_batch_indices = ()
        engine.decode_batch_indices = ()
        engine.curr_batch_idx = None

        seq_metadata_list = [
            types.SimpleNamespace(
                is_prompt=True,
                prompt_chunk_len=3,
                seq=_FakePromptSeq(seq_id=100, processed_prompt_len=2, next_prompt_chunk_len=3),
            ),
            types.SimpleNamespace(
                is_prompt=False,
                seq=_FakeDecodeSeq(seq_id=200, seq_len=5),
            ),
        ]

        engine.step(seq_metadata_list)

        usage = engine.get_cache_usage_stats()

        self.assertEqual(step_calls, [([5, 5, 0, 0], True)])
        self.assertEqual(tuple(engine.curr_batch_idx.tolist()), (0, 1))
        self.assertEqual(engine.prompt_batch_indices, (0,))
        self.assertEqual(engine.decode_batch_indices, (1,))
        self.assertEqual(wrapper.calls[0][0].tolist(), [0, 1])
        self.assertEqual(wrapper.calls[0][1].tolist(), [1])
        self.assertEqual(usage["persistent_tokens"], 10)
        self.assertEqual(usage["persistent_bytes"], 320)
        self.assertEqual(usage["active_batch_indices"], (0, 1))
        self.assertEqual(usage["scheduled_batch_indices"], (0, 1))
        self.assertEqual(usage["scheduled_prompt_batch_indices"], (0,))
        self.assertEqual(usage["scheduled_decode_batch_indices"], (1,))
        self.assertEqual(usage["seq_to_batch_idx"], {100: 0, 200: 1})
        self.assertEqual(usage["free_blocks"], 6)

    def test_cache_usage_history_records_step_and_free_transitions(self):
        class _FakeWrapper:
            def set_batch_idx(self, batch_idx, batch_idx_gen):
                del batch_idx, batch_idx_gen

        class _FakePromptSeq:
            def __init__(self, seq_id, processed_prompt_len, next_prompt_chunk_len):
                self.seq_id = seq_id
                self._processed_prompt_len = processed_prompt_len
                self._next_prompt_chunk_len = next_prompt_chunk_len

            def get_next_prompt_chunk_len(self, prompt_chunk_len):
                return min(prompt_chunk_len, self._next_prompt_chunk_len)

            def get_num_prompt_tokens_processed(self):
                return self._processed_prompt_len

        free_blocks_state = {"value": 8}
        next_batch_idx_state = {"value": 0}
        freed_batch_indices = []

        cache_engine_module.vattention.alloc_new_batch_idx = lambda seq_len: (
            next_batch_idx_state.__setitem__("value", next_batch_idx_state["value"] + 1)
            or next_batch_idx_state["value"] - 1
        )
        cache_engine_module.vattention.step = lambda seq_lens, sync: None
        cache_engine_module.vattention.free_batch_idx = freed_batch_indices.append
        cache_engine_module.get_attention_wrapper = lambda: _FakeWrapper()

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
        engine.curr_seq_lens = [0, 0, 0]
        engine.seq_to_batch_idx = {}
        engine.device = torch.device("cpu")
        engine.vattn_async = False
        engine.num_free_blocks = lambda: free_blocks_state["value"]
        engine.prompt_batch_indices = ()
        engine.decode_batch_indices = ()
        engine.curr_batch_idx = None
        engine.cache_usage_history = []

        engine.step(
            [
                types.SimpleNamespace(
                    is_prompt=True,
                    prompt_chunk_len=2,
                    seq=_FakePromptSeq(
                        seq_id=301,
                        processed_prompt_len=0,
                        next_prompt_chunk_len=2,
                    ),
                )
            ]
        )
        free_blocks_state["value"] = 7
        engine.free_request(301)

        history = engine.get_cache_usage_history()
        transitions = engine.get_cache_usage_transitions()

        self.assertEqual([snapshot["event"] for snapshot in history], ["step", "free_request"])
        self.assertEqual(history[0]["persistent_tokens"], 2)
        self.assertEqual(history[0]["scheduled_prompt_batch_indices"], (0,))
        self.assertEqual(history[1]["persistent_tokens"], 0)
        self.assertEqual(history[1]["free_blocks"], 7)
        self.assertEqual(history[1]["seq_to_batch_idx"], {})
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["persistent_token_delta"], -2)
        self.assertEqual(transitions[0]["persistent_byte_delta"], -64)
        self.assertEqual(transitions[0]["free_block_delta"], -1)
        self.assertEqual(transitions[0]["active_request_delta"], -1)
        self.assertEqual(freed_batch_indices, [0])


if __name__ == "__main__":
    unittest.main()
