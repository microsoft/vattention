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
            "payload": {"cache_spec": {"architecture": "mla"}},
        }

        result = dispatch_init_kvcache(backend, request)

        self.assertEqual(result, "component-result")
        self.assertEqual(
            backend.calls,
            [("init_kvcache_component_spec", {"cache_spec": {"architecture": "mla"}})],
        )

    def test_dispatch_init_kvcache_rejects_component_request_without_backend_support(self):
        backend = _LegacyBackend()
        request = {
            "init_mode": "component_spec",
            "payload": {"cache_spec": {"architecture": "mla"}},
        }

        with self.assertRaises(NotImplementedError):
            dispatch_init_kvcache(backend, request)

    def test_dispatch_init_kvcache_rejects_unknown_mode(self):
        backend = _LegacyBackend()
        request = {"init_mode": "unknown_mode"}

        with self.assertRaises(ValueError):
            dispatch_init_kvcache(backend, request)


if __name__ == "__main__":
    unittest.main()
