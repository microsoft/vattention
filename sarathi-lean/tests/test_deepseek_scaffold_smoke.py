import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
import json

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
        self.assertEqual(result["query_mode"], "direct")
        self.assertEqual(result["checkpoint_layout"], "single_file")
        self.assertEqual(result["prompt_token_ids"], [1, 3])
        self.assertEqual(len(result["generated_token_ids"]), 3)
        self.assertEqual(result["final_logits_shape"], [1, 16])
        self.assertTrue(all(token_count == 4 for token_count in result["cache_token_counts"]))

    def test_build_scaffold_state_dict_uses_deepseek_style_projection_aliases(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config()
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )

        state_dict = self.smoke_module.build_scaffold_state_dict(
            model,
            projection_weights,
            mlp_weights,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertIn("embed_tokens.weight", state_dict)
        self.assertIn("norm.weight", state_dict)
        self.assertIn("layers.0.self_attn.kv_a_proj_with_mqa.weight", state_dict)
        self.assertIn("layers.0.self_attn.kv_a_layernorm.weight", state_dict)
        self.assertIn("layers.0.self_attn.kv_b_proj.weight", state_dict)
        self.assertNotIn("model.layers.0.self_attn.kv_latent_proj.weight", state_dict)
        self.assertNotIn("model.layers.0.self_attn.k_rope_proj.weight", state_dict)

    def test_build_scaffold_state_dict_uses_q_lora_query_aliases(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config(query_mode="q_lora")
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
                query_mode="q_lora",
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )

        state_dict = self.smoke_module.build_scaffold_state_dict(
            model,
            projection_weights,
            mlp_weights,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertIn("layers.0.self_attn.q_a_proj.weight", state_dict)
        self.assertIn("layers.0.self_attn.q_a_layernorm.weight", state_dict)
        self.assertIn("layers.0.self_attn.q_b_proj.weight", state_dict)
        self.assertNotIn("layers.0.self_attn.q_proj.weight", state_dict)

    def test_build_scaffold_state_dict_supports_hf_namespace(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config()
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )

        state_dict = self.smoke_module.build_scaffold_state_dict(
            model,
            projection_weights,
            mlp_weights,
            device=torch.device("cpu"),
            dtype=torch.float32,
            namespace="hf",
        )

        self.assertIn("model.embed_tokens.weight", state_dict)
        self.assertIn("model.norm.weight", state_dict)
        self.assertIn("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", state_dict)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", state_dict)
        self.assertNotIn("layers.0.self_attn.kv_a_proj_with_mqa.weight", state_dict)

    def test_build_scaffold_state_dict_emits_bounded_moe_weights(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config(mlp_mode="moe")
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            if layer_idx < config.first_k_dense_replace
            else None
            for layer_idx in range(model.model.num_layers)
        )
        moe_weights = tuple(
            None
            if layer_idx < config.first_k_dense_replace
            else self.smoke_module.make_moe_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
                num_experts=config.n_routed_experts,
            )
            for layer_idx in range(model.model.num_layers)
        )

        state_dict = self.smoke_module.build_scaffold_state_dict(
            model,
            projection_weights,
            mlp_weights,
            device=torch.device("cpu"),
            dtype=torch.float32,
            moe_weights=moe_weights,
        )

        self.assertIn("layers.0.mlp.gate_proj.weight", state_dict)
        self.assertIn("layers.1.mlp.gate.weight", state_dict)
        self.assertIn("layers.1.mlp.shared_experts.gate_proj.weight", state_dict)
        self.assertIn("layers.1.mlp.experts.0.gate_proj.weight", state_dict)
        self.assertNotIn("layers.1.mlp.gate_proj.weight", state_dict)

    def test_write_scaffold_checkpoint_emits_pt_checkpoint_file(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config()
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = self.smoke_module.write_scaffold_checkpoint(
                model,
                projection_weights,
                mlp_weights,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=tmpdir,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.assertTrue(str(checkpoint_path).endswith(".pt"))
        self.assertIn("embed_tokens.weight", checkpoint)
        self.assertIn("layers.0.self_attn.kv_a_proj_with_mqa.weight", checkpoint)
        self.assertIn("layers.0.self_attn.kv_a_layernorm.weight", checkpoint)
        self.assertEqual(checkpoint["layers.0.self_attn.q_proj.weight"].device.type, "cpu")

    def test_write_scaffold_checkpoint_emits_safetensors_checkpoint_file(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config()
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = self.smoke_module.write_scaffold_checkpoint(
                model,
                projection_weights,
                mlp_weights,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=tmpdir,
                checkpoint_format="safetensors",
            )
            checkpoint = deepseek_module.DeepseekV2ForCausalLM._load_state_dict_file(
                checkpoint_path
            )

        self.assertTrue(str(checkpoint_path).endswith(".safetensors"))
        self.assertIn("embed_tokens.weight", checkpoint)
        self.assertIn("layers.0.self_attn.kv_a_proj_with_mqa.weight", checkpoint)
        self.assertIn("layers.0.self_attn.kv_a_layernorm.weight", checkpoint)
        self.assertEqual(checkpoint["layers.0.self_attn.q_proj.weight"].device.type, "cpu")

    def test_write_scaffold_hf_directory_emits_sharded_safetensors_layout(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config()
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            for _ in range(model.model.num_layers)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = self.smoke_module.write_scaffold_hf_directory(
                model,
                projection_weights,
                mlp_weights,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=tmpdir,
            )
            index = json.loads((Path(checkpoint_dir) / "model.safetensors.index.json").read_text())
            config_json = json.loads((Path(checkpoint_dir) / "config.json").read_text())
            self.assertTrue((Path(checkpoint_dir) / "config.json").exists())
            self.assertTrue((Path(checkpoint_dir) / "model-00001-of-00002.safetensors").exists())
            self.assertTrue((Path(checkpoint_dir) / "model-00002-of-00002.safetensors").exists())
            self.assertIn("model.embed_tokens.weight", index["weight_map"])
            self.assertIn("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", index["weight_map"])
            self.assertIn("model.layers.0.self_attn.kv_a_layernorm.weight", index["weight_map"])
            self.assertEqual(config_json["tensor_parallel_world_size"], 2)
            self.assertEqual(config_json["intermediate_size"], 4)
            self.assertEqual(config_json["moe_intermediate_size"], 4)
            self.assertEqual(config_json["scoring_func"], "softmax")
            self.assertEqual(config_json["architectures"], ["DeepseekV2ForCausalLM"])
            self.assertFalse(config_json["tie_word_embeddings"])

    def test_write_scaffold_checkpoint_loads_back_into_runtime_device(self):
        deepseek_module = sys.modules["sarathi.model_executor.models.deepseek_v2"]
        config = self.smoke_module.build_config()
        model = deepseek_module.DeepseekV2ForCausalLM(
            config,
            tensor_parallel_world_size=2,
            pipeline_parallel_world_size=1,
            pipeline_parallel_rank=0,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = model.to(device=device, dtype=dtype)
        dims = deepseek_module.DeepseekV2MLADims.from_config(
            config,
            tensor_parallel_world_size=2,
        )
        projection_weights = tuple(
            self.smoke_module.make_projection_weights(
                deepseek_module,
                dims,
                device=device,
                dtype=dtype,
            )
            for _ in range(model.model.num_layers)
        )
        mlp_weights = tuple(
            self.smoke_module.make_mlp_weights(
                deepseek_module,
                config.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(model.model.num_layers)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = self.smoke_module.write_scaffold_checkpoint(
                model,
                projection_weights,
                mlp_weights,
                device=device,
                dtype=dtype,
                output_dir=tmpdir,
            )
            model.load_weights(checkpoint_path)

        installed_weights = model.model.layer_projection_weights
        self.assertIsNotNone(installed_weights)
        self.assertEqual(installed_weights[0].q_proj.device.type, device.type)
        self.assertEqual(installed_weights[0].q_proj.dtype, dtype)

    def test_run_scaffold_smoke_compare_supports_safetensors_checkpoints(self):
        result = self.smoke_module.compare_scaffold_smoke(
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
            checkpoint_format="safetensors",
        )

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["checkpoint_format"], "safetensors")
        self.assertEqual(result["query_mode"], "direct")
        self.assertEqual(result["checkpoint_layout"], "single_file")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])

    def test_run_scaffold_smoke_paged_executes_prompt_and_decode(self):
        result = self.smoke_module.run_scaffold_smoke(
            mode="paged",
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
        )

        self.assertEqual(result["mode"], "paged")
        self.assertEqual(result["checkpoint_format"], "pt")
        self.assertEqual(result["query_mode"], "direct")
        self.assertEqual(result["checkpoint_layout"], "single_file")
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
        self.assertEqual(result["checkpoint_format"], "pt")
        self.assertEqual(result["query_mode"], "direct")
        self.assertEqual(result["checkpoint_layout"], "single_file")
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
        self.assertEqual(result["checkpoint_format"], "pt")
        self.assertEqual(result["query_mode"], "direct")
        self.assertEqual(result["checkpoint_layout"], "single_file")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])

    def test_compare_scaffold_smoke_matches_q_lora_contiguous_and_paged_generation(self):
        result = self.smoke_module.compare_scaffold_smoke(
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
            query_mode="q_lora",
        )

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["query_mode"], "q_lora")
        self.assertEqual(result["checkpoint_layout"], "single_file")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])

    def test_compare_scaffold_smoke_matches_bounded_moe_generation(self):
        result = self.smoke_module.compare_scaffold_smoke(
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
            mlp_mode="moe",
        )

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["mlp_mode"], "moe")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])

    def test_compare_scaffold_smoke_matches_hf_directory_generation(self):
        result = self.smoke_module.compare_scaffold_smoke(
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
            checkpoint_layout="hf_dir",
        )

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["checkpoint_format"], "safetensors")
        self.assertEqual(result["checkpoint_layout"], "hf_dir")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])

    def test_compare_scaffold_smoke_matches_hf_directory_q_lora_moe_generation(self):
        result = self.smoke_module.compare_scaffold_smoke(
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
            checkpoint_layout="hf_dir",
            query_mode="q_lora",
            mlp_mode="moe",
        )

        self.assertEqual(result["mode"], "compare")
        self.assertEqual(result["checkpoint_format"], "safetensors")
        self.assertEqual(result["checkpoint_layout"], "hf_dir")
        self.assertEqual(result["query_mode"], "q_lora")
        self.assertEqual(result["mlp_mode"], "moe")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["generated_tokens_match"])
        self.assertTrue(result["final_logits_match"])
        self.assertTrue(result["cache_token_counts_match"])

    def test_compare_scaffold_smoke_reports_blocked_paged_runtime_errors(self):
        original = self.smoke_module._run_scaffold_smoke_artifacts

        def _fake_run(
            mode="contiguous",
            prompt_token_ids=(1, 3),
            max_new_tokens=3,
            checkpoint_format="pt",
            query_mode="direct",
            checkpoint_layout="single_file",
            mlp_mode="dense",
        ):
            del (
                prompt_token_ids,
                max_new_tokens,
                checkpoint_format,
                query_mode,
                checkpoint_layout,
                mlp_mode,
            )
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
        self.assertEqual(result["checkpoint_format"], "pt")
        self.assertEqual(result["query_mode"], "direct")
        self.assertEqual(result["checkpoint_layout"], "single_file")
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
