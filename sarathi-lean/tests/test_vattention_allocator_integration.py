import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SARATHI_ROOT = REPO_ROOT / "sarathi-lean" / "sarathi"
VATTENTION_ROOT = REPO_ROOT / "vattention"


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


try:
    import torch
except ModuleNotFoundError:
    torch = None


def _install_transformers_stub():
    if "transformers" in sys.modules:
        return

    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        pass

    transformers.PretrainedConfig = PretrainedConfig
    sys.modules["transformers"] = transformers


def _load_config_module():
    if torch is None:
        return None

    _install_transformers_stub()
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.utils", SARATHI_ROOT / "utils")
    _ensure_package("sarathi.transformers_utils", SARATHI_ROOT / "transformers_utils")

    _load_module("sarathi.logger", SARATHI_ROOT / "logger.py")
    _load_module("sarathi.utils.base_int_enum", SARATHI_ROOT / "utils" / "base_int_enum.py")

    transformers_config = types.ModuleType("sarathi.transformers_utils.config")
    transformers_config.get_config = lambda *args, **kwargs: None
    sys.modules["sarathi.transformers_utils.config"] = transformers_config

    return _load_module("sarathi.config", SARATHI_ROOT / "config.py")


def _load_vattention():
    if torch is None:
        return None

    if str(VATTENTION_ROOT) not in sys.path:
        sys.path.insert(0, str(VATTENTION_ROOT))

    try:
        return importlib.import_module("vattention")
    except ModuleNotFoundError:
        return None


config_module = _load_config_module()
vattention = _load_vattention()


@unittest.skipUnless(torch is not None, "torch is required for allocator integration tests")
@unittest.skipUnless(config_module is not None, "sarathi config module could not be loaded")
@unittest.skipUnless(vattention is not None, "built vattention extension is required")
@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required for allocator integration tests")
class VAttentionAllocatorIntegrationTests(unittest.TestCase):
    def _make_model_config(self, *, hf_config, dtype=torch.float16):
        model_config = config_module.ModelConfig.__new__(config_module.ModelConfig)
        model_config.hf_config = hf_config
        model_config.dtype = dtype
        return model_config

    def setUp(self):
        torch.empty(1, device="cuda")
        self._cleanup_vattention()

    def tearDown(self):
        self._cleanup_vattention()

    def _cleanup_vattention(self):
        try:
            vattention.cleanup()
        except Exception:
            pass

    def _assert_fragmentation_metrics_match_expected(
        self,
        *,
        seq_len,
        mapped_blocks,
        tokens_per_page,
        pages_per_kvblock,
        page_size,
        cached_token_bytes_local,
    ):
        metrics = vattention.debug_fragmentation_metrics(seq_len, mapped_blocks)
        mapped_token_capacity = mapped_blocks * tokens_per_page
        resident_tokens = min(seq_len, mapped_token_capacity)
        slack_tokens = mapped_token_capacity - resident_tokens
        useful_payload_bytes = resident_tokens * cached_token_bytes_local
        mapped_physical_bytes = mapped_blocks * pages_per_kvblock * page_size
        token_fill_pct = (
            100.0 * resident_tokens / mapped_token_capacity
            if mapped_token_capacity
            else 0.0
        )
        token_frag_pct = (
            100.0 * slack_tokens / mapped_token_capacity
            if mapped_token_capacity
            else 0.0
        )
        payload_util_pct = (
            100.0 * useful_payload_bytes / mapped_physical_bytes
            if mapped_physical_bytes
            else 0.0
        )
        payload_overhead_pct = 100.0 - payload_util_pct if mapped_physical_bytes else 0.0

        self.assertEqual(metrics["seq_len"], seq_len)
        self.assertEqual(metrics["mapped_blocks"], mapped_blocks)
        self.assertEqual(metrics["pages_per_kvblock"], pages_per_kvblock)
        self.assertEqual(metrics["tokens_per_page"], tokens_per_page)
        self.assertEqual(metrics["mapped_token_capacity"], mapped_token_capacity)
        self.assertEqual(metrics["resident_tokens"], resident_tokens)
        self.assertEqual(metrics["slack_tokens"], slack_tokens)
        self.assertEqual(metrics["useful_payload_bytes"], useful_payload_bytes)
        self.assertEqual(metrics["mapped_physical_bytes"], mapped_physical_bytes)
        self.assertAlmostEqual(metrics["token_fill_pct"], token_fill_pct, places=6)
        self.assertAlmostEqual(metrics["token_frag_pct"], token_frag_pct, places=6)
        self.assertAlmostEqual(metrics["payload_util_pct"], payload_util_pct, places=6)
        self.assertAlmostEqual(
            metrics["payload_overhead_pct"],
            payload_overhead_pct,
            places=6,
        )

    def test_dense_allocator_debug_info_matches_python_spec(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = config_module.ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )
        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=False,
            max_batch_size=8,
            max_context_length=128,
            device_idx=0,
        )

        tensors = vattention.init_kvcache(*init_spec.to_legacy_init_kvcache_args())
        debug_info = vattention.get_allocator_debug_info()

        self.assertEqual(len(tensors), 24)
        self.assertEqual(debug_info["tokens_per_page"], init_spec.cache_spec.tokens_per_page)
        self.assertEqual(
            debug_info["page_buffer_token_bytes"],
            init_spec.cache_spec.page_buffer_token_bytes,
        )
        self.assertEqual(
            debug_info["cached_token_bytes_local"],
            init_spec.cache_spec.cached_token_bytes_local,
        )
        self.assertEqual(
            debug_info["pages_per_kvblock"],
            len(init_spec.cache_spec.cache_components) * init_spec.cache_spec.num_layers,
        )
        self.assertFalse(debug_info["component_spec_enabled"])

    def test_component_spec_allocator_debug_info_matches_python_spec(self):
        hf_config = types.SimpleNamespace(
            model_type="deepseek_v2",
            hidden_size=5120,
            num_attention_heads=128,
            num_hidden_layers=60,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = config_module.ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )
        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=True,
            max_batch_size=4,
            max_context_length=256,
            device_idx=0,
        )

        tensors = vattention.init_kvcache_component_spec(
            init_spec.get_extension_init_request()["payload"]
        )
        debug_info = vattention.get_allocator_debug_info()

        self.assertEqual(len(tensors), 2)
        self.assertEqual(
            list(tensors[0].shape),
            [
                init_spec.max_batch_size,
                init_spec.max_context_length,
                init_spec.cache_spec.num_layers,
                init_spec.cache_spec.cache_components[0].token_dim,
            ],
        )
        self.assertEqual(
            list(tensors[1].shape),
            [
                init_spec.max_batch_size,
                init_spec.max_context_length,
                init_spec.cache_spec.num_layers,
                init_spec.cache_spec.cache_components[1].token_dim,
            ],
        )
        self.assertEqual(debug_info["tokens_per_page"], init_spec.cache_spec.tokens_per_page)
        self.assertEqual(
            debug_info["page_buffer_token_bytes"],
            init_spec.cache_spec.page_buffer_token_bytes,
        )
        self.assertEqual(
            debug_info["cached_token_bytes_local"],
            init_spec.cache_spec.cached_token_bytes_local,
        )
        self.assertEqual(
            debug_info["pages_per_kvblock"],
            len(init_spec.cache_spec.cache_components),
        )
        self.assertTrue(debug_info["component_spec_enabled"])

    def test_component_spec_page_growth_boundaries_match_python_tokens_per_page(self):
        hf_config = types.SimpleNamespace(
            model_type="deepseek_v2",
            hidden_size=5120,
            num_attention_heads=128,
            num_hidden_layers=60,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = config_module.ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )
        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=True,
            max_batch_size=4,
            max_context_length=256,
            device_idx=0,
        )

        vattention.init_kvcache_component_spec(
            init_spec.get_extension_init_request()["payload"]
        )

        tokens_per_page = init_spec.cache_spec.tokens_per_page
        self.assertEqual(vattention.debug_tokens_to_pages(1), 1)
        self.assertEqual(vattention.debug_tokens_to_pages(tokens_per_page), 1)
        self.assertEqual(vattention.debug_tokens_to_pages(tokens_per_page + 1), 2)

    def test_component_spec_virtual_storage_covers_ceiling_page_reservation(self):
        hf_config = types.SimpleNamespace(
            model_type="deepseek_v2",
            hidden_size=5120,
            num_attention_heads=128,
            num_hidden_layers=60,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = config_module.ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )
        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=False,
            max_batch_size=1,
            max_context_length=32768,
            device_idx=0,
        )

        tensors = vattention.init_kvcache_component_spec(
            init_spec.get_extension_init_request()["payload"]
        )
        debug_info = vattention.get_allocator_debug_info()
        expected_pages = (
            init_spec.max_context_length + init_spec.cache_spec.tokens_per_page - 1
        ) // init_spec.cache_spec.tokens_per_page
        expected_reserved_bytes = expected_pages * init_spec.cache_spec.page_size

        self.assertEqual(debug_info["max_pages_per_req"], expected_pages)
        self.assertEqual(debug_info["virt_buff_size_per_req"], expected_reserved_bytes)
        for tensor in tensors:
            self.assertGreaterEqual(
                tensor.storage().nbytes(),
                expected_reserved_bytes,
            )

    def test_dense_fragmentation_metrics_match_token_and_byte_accounting(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = config_module.ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )
        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=False,
            max_batch_size=8,
            max_context_length=128,
            device_idx=0,
        )

        vattention.init_kvcache(*init_spec.to_legacy_init_kvcache_args())
        self._assert_fragmentation_metrics_match_expected(
            seq_len=3000,
            mapped_blocks=2,
            tokens_per_page=init_spec.cache_spec.tokens_per_page,
            pages_per_kvblock=(
                len(init_spec.cache_spec.cache_components) * init_spec.cache_spec.num_layers
            ),
            page_size=init_spec.cache_spec.page_size,
            cached_token_bytes_local=init_spec.cache_spec.cached_token_bytes_local,
        )

    def test_mla_fragmentation_metrics_match_token_and_byte_accounting(self):
        hf_config = types.SimpleNamespace(
            model_type="deepseek_v2",
            hidden_size=5120,
            num_attention_heads=128,
            num_hidden_layers=60,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = config_module.ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )
        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=False,
            max_batch_size=1,
            max_context_length=32768,
            device_idx=0,
        )

        vattention.init_kvcache_component_spec(
            init_spec.get_extension_init_request()["payload"]
        )
        self._assert_fragmentation_metrics_match_expected(
            seq_len=5444,
            mapped_blocks=3,
            tokens_per_page=init_spec.cache_spec.tokens_per_page,
            pages_per_kvblock=(
                len(init_spec.cache_spec.cache_components) * init_spec.cache_spec.num_layers
            ),
            page_size=init_spec.cache_spec.page_size,
            cached_token_bytes_local=init_spec.cache_spec.cached_token_bytes_local,
        )


if __name__ == "__main__":
    unittest.main()
