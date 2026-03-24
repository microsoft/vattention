import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SARATHI_ROOT = REPO_ROOT / "sarathi-lean" / "sarathi"


def _install_transformers_stub():
    if "transformers" in sys.modules:
        return

    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        pass

    transformers.PretrainedConfig = PretrainedConfig
    sys.modules["transformers"] = transformers


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


def _load_config_module():
    _install_transformers_stub()

    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.utils", SARATHI_ROOT / "utils")
    _ensure_package("sarathi.transformers_utils", SARATHI_ROOT / "transformers_utils")

    _load_module("sarathi.logger", SARATHI_ROOT / "logger.py")
    _load_module("sarathi.utils.base_int_enum", SARATHI_ROOT / "utils" / "base_int_enum.py")

    transformers_config = types.ModuleType("sarathi.transformers_utils.config")
    transformers_config.get_config = lambda *args, **kwargs: None
    sys.modules["sarathi.transformers_utils.config"] = transformers_config

    config_module = _load_module("sarathi.config", SARATHI_ROOT / "config.py")
    return config_module


config_module = _load_config_module()
CacheArchitecture = config_module.CacheArchitecture
CacheComponentSpec = config_module.CacheComponentSpec
CacheLayout = config_module.CacheLayout
MLAAttentionSpec = config_module.MLAAttentionSpec
ModelConfig = config_module.ModelConfig
ParallelConfig = config_module.ParallelConfig
VAttentionCacheSpec = config_module.VAttentionCacheSpec
VAttentionInitSpec = config_module.VAttentionInitSpec


class ModelConfigCacheArchitectureTests(unittest.TestCase):
    def _make_model_config(self, *, hf_config, dtype=torch.float16):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = hf_config
        model_config.dtype = dtype
        return model_config

    def test_dense_kv_models_are_not_detected_as_mla(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        model_config = self._make_model_config(hf_config=hf_config)

        self.assertFalse(model_config.is_mla_model())
        self.assertEqual(
            model_config.get_cache_architecture(), CacheArchitecture.DENSE_KV
        )

    def test_mla_models_are_detected_from_config_fields(self):
        hf_config = types.SimpleNamespace(
            model_type="deepseek_v2",
            hidden_size=5120,
            num_attention_heads=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model_config = self._make_model_config(hf_config=hf_config)

        self.assertTrue(model_config.is_mla_model())
        self.assertEqual(model_config.get_cache_architecture(), CacheArchitecture.MLA)
        self.assertIsNone(model_config.get_mla_q_lora_rank())
        self.assertEqual(model_config.get_mla_kv_lora_rank(), 512)
        self.assertEqual(model_config.get_mla_qk_nope_head_dim(), 128)
        self.assertEqual(model_config.get_mla_qk_rope_head_dim(), 64)
        self.assertEqual(model_config.get_mla_v_head_dim(), 128)
        self.assertEqual(model_config.get_mla_q_head_dim(), 192)
        self.assertEqual(model_config.get_mla_resident_cache_dim(), 576)

    def test_mla_attention_spec_packages_deepseek_dimensions(self):
        hf_config = types.SimpleNamespace(
            model_type="deepseek_v2",
            hidden_size=5120,
            num_attention_heads=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model_config = self._make_model_config(hf_config=hf_config)

        spec = model_config.get_mla_attention_spec()

        self.assertIsInstance(spec, MLAAttentionSpec)
        self.assertIsNone(spec.q_lora_rank)
        self.assertEqual(spec.kv_lora_rank, 512)
        self.assertEqual(spec.qk_nope_head_dim, 128)
        self.assertEqual(spec.qk_rope_head_dim, 64)
        self.assertEqual(spec.v_head_dim, 128)
        self.assertEqual(spec.q_head_dim, 192)
        self.assertEqual(spec.resident_cache_dim, 576)

    def test_mla_cache_component_specs_are_latent_and_rope(self):
        hf_config = types.SimpleNamespace(
            model_type="deepseek_v2",
            hidden_size=5120,
            num_attention_heads=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )

        components = model_config.get_cache_component_specs(parallel_config)

        self.assertEqual(len(components), 2)
        self.assertIsInstance(components[0], CacheComponentSpec)
        self.assertEqual(components[0].name, "kv_latent")
        self.assertEqual(components[1].name, "k_rope")
        self.assertEqual(components[0].token_dim, 512)
        self.assertEqual(components[1].token_dim, 64)
        self.assertEqual(
            model_config.get_resident_cache_token_dim(parallel_config),
            576,
        )

    def test_dense_models_do_not_expose_mla_attention_spec(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        model_config = self._make_model_config(hf_config=hf_config)

        with self.assertRaises(ValueError):
            model_config.get_mla_attention_spec()

    def test_dense_kv_cached_bytes_per_layer_uses_local_kv_heads(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )

        expected = 2 * (2 * 4 * 128)
        self.assertEqual(
            model_config.get_cached_token_bytes_per_layer(parallel_config),
            expected,
        )

    def test_dense_kv_cache_component_specs_are_k_and_v(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )

        components = model_config.get_cache_component_specs(parallel_config)

        self.assertEqual(len(components), 2)
        self.assertIsInstance(components[0], CacheComponentSpec)
        self.assertEqual(components[0].name, "k")
        self.assertEqual(components[1].name, "v")
        self.assertEqual(components[0].token_dim, 4 * 128)
        self.assertEqual(components[1].token_dim, 4 * 128)
        self.assertEqual(
            model_config.get_resident_cache_token_dim(parallel_config),
            1024,
        )

    def test_dense_kv_cached_bytes_local_multiplies_by_local_layers(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )

        per_layer = model_config.get_cached_token_bytes_per_layer(parallel_config)
        self.assertEqual(
            model_config.get_cached_token_bytes_local(parallel_config),
            12 * per_layer,
        )

    def test_dense_kv_page_buffer_bytes_match_single_side_storage(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )

        dtype_size = torch.tensor([], dtype=torch.float16).element_size()
        expected_non_mega = dtype_size * (4 * 128)
        expected_mega = 12 * expected_non_mega

        self.assertEqual(
            model_config.get_page_buffer_token_bytes(parallel_config),
            expected_non_mega,
        )
        self.assertEqual(
            model_config.get_page_buffer_token_bytes(
                parallel_config, megacache=True
            ),
            expected_mega,
        )

    def test_dense_kv_tokens_per_page_match_existing_vattention_semantics(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )
        page_size = 2 * 1024 * 1024

        self.assertEqual(
            model_config.get_num_cached_tokens_per_page(page_size, parallel_config),
            page_size // (2 * 4 * 128),
        )
        self.assertEqual(
            model_config.get_num_cached_tokens_per_page(
                page_size, parallel_config, megacache=True
            ),
            page_size // (12 * 2 * 4 * 128),
        )

    def test_dense_kv_cache_block_size_bytes_scale_with_block_size(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )

        per_token_local = 12 * (2 * 4 * 128 * 2)
        self.assertEqual(
            model_config.get_cache_block_size_bytes(16, parallel_config),
            16 * per_token_local,
        )

    def test_dense_kv_cache_layout_packages_all_derived_fields(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )
        page_size = 2 * 1024 * 1024

        layout = model_config.get_cache_layout(page_size, parallel_config)

        self.assertIsInstance(layout, CacheLayout)
        self.assertEqual(layout.architecture, CacheArchitecture.DENSE_KV)
        self.assertFalse(layout.megacache)
        self.assertEqual(layout.cached_token_bytes_per_layer, 2 * (2 * 4 * 128))
        self.assertEqual(layout.cached_token_bytes_local, 12 * (2 * 2 * 4 * 128))
        self.assertEqual(layout.page_buffer_token_bytes, 2 * 4 * 128)
        self.assertEqual(layout.tokens_per_page, page_size // (2 * 4 * 128))

    def test_dense_kv_vattention_cache_spec_contains_allocator_inputs(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )
        page_size = 2 * 1024 * 1024

        spec = model_config.get_vattention_cache_spec(page_size, parallel_config)

        self.assertIsInstance(spec, VAttentionCacheSpec)
        self.assertEqual(spec.architecture, CacheArchitecture.DENSE_KV)
        self.assertFalse(spec.megacache)
        self.assertEqual(spec.page_size, page_size)
        self.assertEqual(spec.tokens_per_page, page_size // (2 * 4 * 128))
        self.assertEqual(spec.cached_token_bytes_per_layer, 2 * (2 * 4 * 128))
        self.assertEqual(spec.cached_token_bytes_local, 12 * (2 * 2 * 4 * 128))
        self.assertEqual(spec.page_buffer_token_bytes, 2 * 4 * 128)
        self.assertEqual(spec.dtype_size, 2)
        self.assertEqual(spec.num_layers, 12)
        self.assertEqual(spec.num_kv_heads, 4)
        self.assertEqual(spec.head_size, 128)
        self.assertIsNone(spec.mla_kv_lora_rank)
        self.assertIsNone(spec.mla_qk_rope_head_dim)

    def test_dense_kv_vattention_init_spec_projects_to_legacy_args(self):
        hf_config = types.SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=24,
        )
        model_config = self._make_model_config(hf_config=hf_config)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
        )

        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=False,
            max_batch_size=128,
            max_context_length=8192,
            device_idx=0,
        )

        self.assertIsInstance(init_spec, VAttentionInitSpec)
        self.assertEqual(init_spec.max_batch_size, 128)
        self.assertEqual(init_spec.max_context_length, 8192)
        self.assertEqual(init_spec.device_idx, 0)
        self.assertEqual(init_spec.dtype, torch.float16)
        self.assertEqual(
            init_spec.to_legacy_init_kvcache_args(),
            (12, 4, 128, 128, 8192, 0, torch.float16, 2 * 1024 * 1024, False),
        )

    def test_mla_cached_bytes_per_layer_uses_resident_payload_only(self):
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
        parallel_config = ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )

        dtype_size = torch.tensor([], dtype=torch.float16).element_size()
        expected_per_layer = dtype_size * (512 + 64)
        self.assertEqual(
            model_config.get_cached_token_bytes_per_layer(parallel_config),
            expected_per_layer,
        )
        self.assertEqual(
            model_config.get_cached_token_bytes_local(parallel_config),
            20 * expected_per_layer,
        )

    def test_mla_tokens_per_page_use_resident_payload_formula(self):
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
        parallel_config = ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )
        page_size = 2 * 1024 * 1024

        expected_per_layer = 2 * (512 + 64)
        self.assertEqual(
            model_config.get_page_buffer_token_bytes(parallel_config),
            expected_per_layer,
        )
        self.assertEqual(
            model_config.get_page_buffer_token_bytes(
                parallel_config, megacache=True
            ),
            20 * expected_per_layer,
        )
        self.assertEqual(
            model_config.get_num_cached_tokens_per_page(page_size, parallel_config),
            page_size // expected_per_layer,
        )

    def test_mla_cache_block_size_bytes_use_resident_payload_formula(self):
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
        parallel_config = ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )

        per_token_local = 20 * (2 * (512 + 64))
        self.assertEqual(
            model_config.get_cache_block_size_bytes(32, parallel_config),
            32 * per_token_local,
        )

    def test_mla_cache_layout_packages_all_derived_fields(self):
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
        parallel_config = ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )
        page_size = 2 * 1024 * 1024

        layout = model_config.get_cache_layout(
            page_size,
            parallel_config,
            megacache=True,
        )

        expected_per_layer = 2 * (512 + 64)
        self.assertIsInstance(layout, CacheLayout)
        self.assertEqual(layout.architecture, CacheArchitecture.MLA)
        self.assertTrue(layout.megacache)
        self.assertEqual(layout.cached_token_bytes_per_layer, expected_per_layer)
        self.assertEqual(layout.cached_token_bytes_local, 20 * expected_per_layer)
        self.assertEqual(layout.page_buffer_token_bytes, 20 * expected_per_layer)
        self.assertEqual(
            layout.tokens_per_page,
            page_size // (20 * expected_per_layer),
        )

    def test_mla_vattention_cache_spec_contains_allocator_inputs(self):
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
        parallel_config = ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )
        page_size = 2 * 1024 * 1024

        spec = model_config.get_vattention_cache_spec(
            page_size,
            parallel_config,
            megacache=True,
        )

        expected_per_layer = 2 * (512 + 64)
        self.assertIsInstance(spec, VAttentionCacheSpec)
        self.assertEqual(spec.architecture, CacheArchitecture.MLA)
        self.assertTrue(spec.megacache)
        self.assertEqual(spec.page_size, page_size)
        self.assertEqual(spec.tokens_per_page, page_size // (20 * expected_per_layer))
        self.assertEqual(spec.cached_token_bytes_per_layer, expected_per_layer)
        self.assertEqual(spec.cached_token_bytes_local, 20 * expected_per_layer)
        self.assertEqual(spec.page_buffer_token_bytes, 20 * expected_per_layer)
        self.assertEqual(spec.dtype_size, 2)
        self.assertEqual(spec.num_layers, 20)
        self.assertEqual(spec.num_kv_heads, 32)
        self.assertEqual(spec.head_size, 40)
        self.assertEqual(spec.mla_kv_lora_rank, 512)
        self.assertEqual(spec.mla_qk_rope_head_dim, 64)

    def test_mla_vattention_init_spec_projects_to_legacy_args(self):
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
        parallel_config = ParallelConfig(
            pipeline_parallel_size=3,
            tensor_parallel_size=4,
        )

        init_spec = model_config.get_vattention_init_spec(
            page_size=2 * 1024 * 1024,
            parallel_config=parallel_config,
            megacache=True,
            max_batch_size=64,
            max_context_length=16384,
            device_idx=2,
        )

        self.assertIsInstance(init_spec, VAttentionInitSpec)
        self.assertEqual(init_spec.max_batch_size, 64)
        self.assertEqual(init_spec.max_context_length, 16384)
        self.assertEqual(init_spec.device_idx, 2)
        self.assertEqual(init_spec.dtype, torch.float16)
        self.assertEqual(
            init_spec.to_legacy_init_kvcache_args(),
            (20, 32, 40, 64, 16384, 2, torch.float16, 2 * 1024 * 1024, True),
        )


if __name__ == "__main__":
    unittest.main()
