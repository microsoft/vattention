import importlib.util
import json
import sys
import tempfile
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


def _load_modules():
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.model_executor", SARATHI_ROOT / "model_executor")
    _ensure_package(
        "sarathi.model_executor.parallel_utils",
        SARATHI_ROOT / "model_executor" / "parallel_utils",
    )
    _ensure_package("sarathi.model_executor.models", SARATHI_ROOT / "model_executor" / "models")
    _load_module(
        "sarathi.model_executor.parallel_utils.parallel_state",
        SARATHI_ROOT / "model_executor" / "parallel_utils" / "parallel_state.py",
    )
    _load_module(
        "sarathi.model_executor.weight_utils",
        SARATHI_ROOT / "model_executor" / "weight_utils.py",
    )
    _load_module(
        "sarathi.model_executor.models.deepseek_v2",
        SARATHI_ROOT / "model_executor" / "models" / "deepseek_v2.py",
    )
    smoke = _load_module("scripts.deepseek_scaffold_smoke", SCRIPTS_ROOT / "deepseek_scaffold_smoke.py")
    inspect = _load_module("scripts.inspect_deepseek_checkpoint", SCRIPTS_ROOT / "inspect_deepseek_checkpoint.py")
    return smoke, inspect


class InspectDeepseekCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.smoke_module, self.inspect_module = _load_modules()
        self.deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]

    def _make_model_and_weights(self, *, query_mode="direct", mlp_mode="dense"):
        config = self.smoke_module.build_config(query_mode=query_mode, mlp_mode=mlp_mode)
        model = self.deepseek_module.DeepseekV2ForCausalLM(
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
            self.smoke_module.make_projection_weights(
                self.deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
                query_mode=query_mode,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            (
                self.smoke_module.make_mlp_weights(
                    self.deepseek_module,
                    config.hidden_size,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                if (
                    mlp_mode != "moe"
                    or layer_idx < getattr(config, "first_k_dense_replace", model.model.num_layers)
                )
                else None
            )
            for layer_idx in range(model.model.num_layers)
        )
        if mlp_mode == "moe":
            moe_weights = tuple(
                (
                    None
                    if layer_idx < config.first_k_dense_replace
                    else self.smoke_module.make_moe_weights(
                        self.deepseek_module,
                        config.hidden_size,
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                        num_experts=config.n_routed_experts,
                    )
                )
                for layer_idx in range(model.model.num_layers)
            )
        else:
            moe_weights = tuple(None for _ in range(model.model.num_layers))
        return model, projection_weights, mlp_weights, moe_weights

    def test_inspect_checkpoint_reports_supported_direct_hf_directory(self):
        model, projection_weights, mlp_weights, moe_weights = self._make_model_and_weights(
            query_mode="direct"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = self.smoke_module.write_scaffold_hf_directory(
                model,
                projection_weights,
                mlp_weights,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=tmpdir,
                moe_weights=moe_weights,
            )
            result = self.inspect_module.inspect_deepseek_checkpoint(checkpoint_dir)

        self.assertEqual(result["status"], "supported_non_moe_surface")
        self.assertTrue(result["has_q_proj"])
        self.assertTrue(result["has_combined_kv"])
        self.assertTrue(result["has_kv_b_proj"])
        self.assertFalse(result["has_moe"])
        self.assertEqual(result["config_model_type"], "deepseek_v2")
        self.assertEqual(result["config_tensor_parallel_world_size"], 2)
        self.assertTrue(result["loadable_scaffold_surface"])
        self.assertIsNone(result["load_error"])

    def test_inspect_checkpoint_reports_supported_q_lora_hf_directory(self):
        model, projection_weights, mlp_weights, moe_weights = self._make_model_and_weights(
            query_mode="q_lora"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = self.smoke_module.write_scaffold_hf_directory(
                model,
                projection_weights,
                mlp_weights,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=tmpdir,
                moe_weights=moe_weights,
            )
            result = self.inspect_module.inspect_deepseek_checkpoint(checkpoint_dir)

        self.assertEqual(result["status"], "supported_non_moe_surface")
        self.assertFalse(result["has_q_proj"])
        self.assertTrue(result["has_q_lora"])
        self.assertEqual(result["config_q_lora_rank"], 2)
        self.assertTrue(result["loadable_scaffold_surface"])

    def test_inspect_checkpoint_reports_supported_bounded_moe_surface(self):
        model, projection_weights, mlp_weights, moe_weights = self._make_model_and_weights(
            query_mode="direct",
            mlp_mode="moe",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = self.smoke_module.write_scaffold_hf_directory(
                model,
                projection_weights,
                mlp_weights,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=tmpdir,
                moe_weights=moe_weights,
            )
            result = self.inspect_module.inspect_deepseek_checkpoint(checkpoint_dir)

        self.assertEqual(result["status"], "supported_bounded_moe_surface")
        self.assertTrue(result["has_moe"])
        self.assertEqual(result["config_first_k_dense_replace"], 1)
        self.assertEqual(result["config_n_routed_experts"], 4)
        self.assertEqual(result["moe_layer_indices"], [1, 2, 3])
        self.assertTrue(result["loadable_scaffold_surface"])

    def test_inspect_checkpoint_reports_incomplete_moe_blocker(self):
        model, projection_weights, mlp_weights, moe_weights = self._make_model_and_weights(
            query_mode="direct",
            mlp_mode="moe",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = self.smoke_module.write_scaffold_hf_directory(
                model,
                projection_weights,
                mlp_weights,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=tmpdir,
                moe_weights=moe_weights,
            )
            shard_path = Path(checkpoint_dir) / "model-00001-of-00002.safetensors"
            from safetensors.torch import load_file, save_file

            shard_state = load_file(shard_path)
            del shard_state["layers.1.mlp.experts.0.down_proj.weight"]
            save_file(shard_state, shard_path)

            index_path = Path(checkpoint_dir) / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            del index["weight_map"]["layers.1.mlp.experts.0.down_proj.weight"]
            index_path.write_text(json.dumps(index, indent=2, sort_keys=True))

            result = self.inspect_module.inspect_deepseek_checkpoint(checkpoint_dir)

        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["has_moe"])
        self.assertIn("missing_routed_expert_weights", result["blockers"])
        self.assertIsNone(result["loadable_scaffold_surface"])


if __name__ == "__main__":
    unittest.main()
