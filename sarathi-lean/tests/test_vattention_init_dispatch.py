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


def _load_dispatch_module():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.worker", SARATHI_ROOT / "worker")
    _ensure_package(
        "sarathi.worker.cache_engine",
        SARATHI_ROOT / "worker" / "cache_engine",
    )
    return _load_module(
        "sarathi.worker.cache_engine.vattention_init",
        SARATHI_ROOT / "worker" / "cache_engine" / "vattention_init.py",
    )


dispatch_module = _load_dispatch_module()
dispatch_init_kvcache = dispatch_module.dispatch_init_kvcache
validate_component_spec_payload = dispatch_module.validate_component_spec_payload


class _LegacyBackend:
    def __init__(self):
        self.calls = []

    def init_kvcache(self, *args):
        self.calls.append(("init_kvcache", args))
        return "legacy-result"


class _ComponentBackend:
    def __init__(self):
        self.calls = []

    def init_kvcache_component_spec(self, payload):
        self.calls.append(("init_kvcache_component_spec", payload))
        return "component-result"


class VAttentionInitDispatchTests(unittest.TestCase):
    def _make_component_payload(self):
        return {
            "init_mode": "component_spec",
            "cache_spec": {
                "architecture": "mla",
                "megacache": True,
                "page_size": 2 * 1024 * 1024,
                "tokens_per_page": 91,
                "cached_token_bytes_per_layer": 1152,
                "cached_token_bytes_local": 23040,
                "page_buffer_token_bytes": 23040,
                "dtype_size": 2,
                "num_layers": 20,
                "num_kv_heads": 32,
                "head_size": 40,
                "tp_attention": {
                    "tensor_parallel_size": 4,
                    "num_q_heads_global": 128,
                    "num_q_heads_local": 32,
                    "num_kv_heads_global": 128,
                    "num_kv_heads_local": 32,
                    "head_size": 40,
                },
                "cache_components": [
                    {"name": "kv_latent", "token_dim": 512},
                    {"name": "k_rope", "token_dim": 64},
                ],
                "mla_kv_lora_rank": 512,
                "mla_qk_rope_head_dim": 64,
            },
            "max_batch_size": 64,
            "max_context_length": 16384,
            "device_idx": 2,
            "dtype": "float16",
        }

    def test_dispatch_init_kvcache_uses_legacy_backend_for_dense_request(self):
        backend = _LegacyBackend()
        request = {
            "init_mode": "legacy_dense_kv",
            "legacy_args": (1, 2, 3),
        }

        result = dispatch_init_kvcache(backend, request)

        self.assertEqual(result, "legacy-result")
        self.assertEqual(backend.calls, [("init_kvcache", (1, 2, 3))])

    def test_dispatch_init_kvcache_uses_component_backend_for_component_request(self):
        backend = _ComponentBackend()
        request = {
            "init_mode": "component_spec",
            "payload": self._make_component_payload(),
        }

        result = dispatch_init_kvcache(backend, request)

        self.assertEqual(result, "component-result")
        self.assertEqual(
            backend.calls,
            [("init_kvcache_component_spec", self._make_component_payload())],
        )

    def test_dispatch_init_kvcache_rejects_component_request_without_backend_support(self):
        backend = _LegacyBackend()
        request = {
            "init_mode": "component_spec",
            "payload": self._make_component_payload(),
        }

        with self.assertRaises(NotImplementedError):
            dispatch_init_kvcache(backend, request)

    def test_validate_component_spec_payload_accepts_valid_payload(self):
        validate_component_spec_payload(self._make_component_payload())

    def test_validate_component_spec_payload_rejects_missing_cache_keys(self):
        payload = self._make_component_payload()
        del payload["cache_spec"]["tokens_per_page"]

        with self.assertRaisesRegex(ValueError, "tokens_per_page"):
            validate_component_spec_payload(payload)

    def test_validate_component_spec_payload_rejects_mismatched_component_bytes(self):
        payload = self._make_component_payload()
        payload["cache_spec"]["cached_token_bytes_per_layer"] = 2048

        with self.assertRaisesRegex(ValueError, "cached_token_bytes_per_layer"):
            validate_component_spec_payload(payload)

    def test_validate_component_spec_payload_rejects_invalid_component_token_dim(self):
        payload = self._make_component_payload()
        payload["cache_spec"]["cache_components"][1]["token_dim"] = 0

        with self.assertRaisesRegex(ValueError, "positive integer token_dim"):
            validate_component_spec_payload(payload)

    def test_dispatch_init_kvcache_rejects_unknown_mode(self):
        backend = _LegacyBackend()
        request = {"init_mode": "unknown_mode"}

        with self.assertRaises(ValueError):
            dispatch_init_kvcache(backend, request)


if __name__ == "__main__":
    unittest.main()
