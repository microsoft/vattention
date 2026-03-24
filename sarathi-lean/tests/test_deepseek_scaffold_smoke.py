import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SARATHI_ROOT = REPO_ROOT / "sarathi-lean" / "sarathi"
SCRIPTS_ROOT = REPO_ROOT / "scripts"


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


def _exact_flash_attn(query, key, value, causal=True, softmax_scale=1.0):
    scores = torch.einsum("bthd,bshd->bhts", query, key) * softmax_scale
    if causal:
        source_positions = torch.arange(key.shape[1], device=query.device)
        past_len = key.shape[1] - query.shape[1]
        target_positions = past_len + torch.arange(query.shape[1], device=query.device)
        causal_mask = source_positions.unsqueeze(0) <= target_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshv->bthv", attn_weights, value)


def _install_stubs():
    originals = {
        name: sys.modules.get(name)
        for name in [
            "flash_attn",
            "sarathi.config",
            "sarathi.core.datatypes.sequence",
            "sarathi.logger",
            "sarathi.metrics.constants",
            "sarathi.metrics.cuda_timer",
            "sarathi.cache_ops",
            "vattention",
        ]
    }

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.flash_attn_func = _exact_flash_attn
    flash_attn_module.flash_attn_with_kvcache = lambda *args, **kwargs: None
    sys.modules["flash_attn"] = flash_attn_module

    config_module = types.ModuleType("sarathi.config")
    config_module.ModelConfig = object
    config_module.ParallelConfig = object
    sys.modules["sarathi.config"] = config_module

    sequence_module = types.ModuleType("sarathi.core.datatypes.sequence")
    sequence_module.SequenceMetadata = object
    sys.modules["sarathi.core.datatypes.sequence"] = sequence_module

    logger_module = types.ModuleType("sarathi.logger")
    logger_module.init_logger = lambda name: types.SimpleNamespace(warning=lambda *args, **kwargs: None)
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

    cache_ops_module = types.ModuleType("sarathi.cache_ops")
    cache_ops_module.cache_flat = lambda *args, **kwargs: None
    sys.modules["sarathi.cache_ops"] = cache_ops_module

    sys.modules["vattention"] = types.ModuleType("vattention")
    return originals


def _restore_stubs(originals):
    for module_name, original in originals.items():
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def _load_smoke_module():
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

    _load_module(
        "sarathi.model_executor.parallel_utils.parallel_state",
        SARATHI_ROOT / "model_executor" / "parallel_utils" / "parallel_state.py",
    )
    _load_module(
        "sarathi.model_executor.attention.base_attention_wrapper",
        SARATHI_ROOT / "model_executor" / "attention" / "base_attention_wrapper.py",
    )
    _load_module(
        "sarathi.model_executor.models.deepseek_v2",
        SARATHI_ROOT / "model_executor" / "models" / "deepseek_v2.py",
    )
    _load_module(
        "sarathi.model_executor.attention.vattention_flashattention_wrapper",
        SARATHI_ROOT / "model_executor" / "attention" / "vattention_flashattention_wrapper.py",
    )
    return _load_module(
        "scripts.deepseek_scaffold_smoke",
        SCRIPTS_ROOT / "deepseek_scaffold_smoke.py",
    )


class DeepseekScaffoldSmokeTests(unittest.TestCase):
    def setUp(self):
        self.originals = _install_stubs()
        self.project_originals = {
            name: sys.modules.get(name)
            for name in [
                "sarathi.model_executor.parallel_utils.parallel_state",
                "sarathi.model_executor.attention.base_attention_wrapper",
                "sarathi.model_executor.models.deepseek_v2",
                "sarathi.model_executor.attention.vattention_flashattention_wrapper",
                "scripts.deepseek_scaffold_smoke",
            ]
        }
        self.smoke_module = _load_smoke_module()

    def tearDown(self):
        _restore_stubs(self.originals)
        for module_name, original in self.project_originals.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_run_scaffold_smoke_contiguous_executes_prompt_and_decode(self):
        result = self.smoke_module.run_scaffold_smoke(
            mode="contiguous",
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
        )

        self.assertEqual(result["mode"], "contiguous")
        self.assertEqual(result["prompt_token_ids"], [1, 3])
        self.assertEqual(len(result["generated_token_ids"]), 3)
        self.assertEqual(result["final_logits_shape"], [1, 16])
        self.assertTrue(all(token_count == 4 for token_count in result["cache_token_counts"]))

    def test_run_scaffold_smoke_paged_executes_prompt_and_decode(self):
        result = self.smoke_module.run_scaffold_smoke(
            mode="paged",
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
        )

        self.assertEqual(result["mode"], "paged")
        self.assertEqual(result["prompt_token_ids"], [1, 3])
        self.assertEqual(len(result["generated_token_ids"]), 3)
        self.assertEqual(result["final_logits_shape"], [1, 16])
        self.assertTrue(all(token_count == 4 for token_count in result["cache_token_counts"]))

    def test_compare_scaffold_smoke_matches_contiguous_and_paged_generation(self):
        result = self.smoke_module.compare_scaffold_smoke(
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
        )

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prompt_token_ids"], [1, 3])
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])
        self.assertEqual(len(result["generated_token_ids"]), 3)
        self.assertEqual(result["generated_token_ids"], result["paged_generated_token_ids"])
        self.assertTrue(
            all(token_count == 4 for token_count in result["contiguous_cache_token_counts"])
        )
        self.assertTrue(
            all(token_count == 4 for token_count in result["paged_cache_token_counts"])
        )

    def test_validate_scaffold_smoke_compare_returns_compare_result(self):
        result = self.smoke_module.validate_scaffold_smoke_compare(
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
        )

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])

    def test_compare_scaffold_smoke_reports_blocked_paged_runtime_errors(self):
        original = self.smoke_module._run_scaffold_smoke_artifacts

        def _fake_run(mode="contiguous", prompt_token_ids=(1, 3), max_new_tokens=3):
            del prompt_token_ids, max_new_tokens
            if mode == "paged":
                raise RuntimeError("real paged wrapper blocker")
            return (
                torch.tensor([1, 2, 3], dtype=torch.long),
                torch.zeros(1, 16),
                tuple(types.SimpleNamespace(num_tokens=4) for _ in range(2)),
            )

        try:
            self.smoke_module._run_scaffold_smoke_artifacts = _fake_run
            result = self.smoke_module.compare_scaffold_smoke()
        finally:
            self.smoke_module._run_scaffold_smoke_artifacts = original

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["prompt_token_ids"], [1, 3])
        self.assertEqual(result["generated_token_ids"], [1, 2, 3])
        self.assertIn("real paged wrapper blocker", result["error"])

    def test_validate_scaffold_smoke_compare_raises_for_blocked_runtime(self):
        original = self.smoke_module.compare_scaffold_smoke
        try:
            self.smoke_module.compare_scaffold_smoke = lambda **_: {
                "mode": "compare",
                "status": "blocked",
                "error": "real paged wrapper blocker",
            }
            with self.assertRaises(RuntimeError):
                self.smoke_module.validate_scaffold_smoke_compare()
        finally:
            self.smoke_module.compare_scaffold_smoke = original


if __name__ == "__main__":
    unittest.main()
