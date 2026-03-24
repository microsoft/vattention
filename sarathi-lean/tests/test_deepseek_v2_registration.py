import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
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


def _install_transformers_stub():
    transformers = types.ModuleType("transformers")
    transformers.__path__ = []

    class PretrainedConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class AutoConfig:
        from_pretrained_result = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls.from_pretrained_result

    transformers.PretrainedConfig = PretrainedConfig
    transformers.AutoConfig = AutoConfig
    configuration_utils = types.ModuleType("transformers.configuration_utils")
    configuration_utils.PretrainedConfig = PretrainedConfig
    transformers_utils = types.ModuleType("transformers.utils")

    class _Logging:
        @staticmethod
        def get_logger(_name):
            return types.SimpleNamespace(info=lambda *args, **kwargs: None)

    transformers_utils.logging = _Logging()
    sys.modules["transformers"] = transformers
    sys.modules["transformers.configuration_utils"] = configuration_utils
    sys.modules["transformers.utils"] = transformers_utils
    return transformers


def _install_torch_stub():
    import torch

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    return torch


def _load_config_modules():
    transformers = _install_transformers_stub()
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.transformers_utils", SARATHI_ROOT / "transformers_utils")
    _load_module(
        "sarathi.transformers_utils.configs.falcon",
        SARATHI_ROOT / "transformers_utils" / "configs" / "falcon.py",
    )
    _load_module(
        "sarathi.transformers_utils.configs.qwen",
        SARATHI_ROOT / "transformers_utils" / "configs" / "qwen.py",
    )
    _load_module(
        "sarathi.transformers_utils.configs.yi",
        SARATHI_ROOT / "transformers_utils" / "configs" / "yi.py",
    )
    deepseek_module = _load_module(
        "sarathi.transformers_utils.configs.deepseek_v2",
        SARATHI_ROOT / "transformers_utils" / "configs" / "deepseek_v2.py",
    )
    _load_module(
        "sarathi.transformers_utils.configs",
        SARATHI_ROOT / "transformers_utils" / "configs" / "__init__.py",
    )
    config_module = _load_module(
        "sarathi.transformers_utils.config",
        SARATHI_ROOT / "transformers_utils" / "config.py",
    )
    return transformers, deepseek_module, config_module


def _load_model_loader_module():
    _install_transformers_stub()
    _install_torch_stub()
    _ensure_package("sarathi", SARATHI_ROOT)
    _ensure_package("sarathi.model_executor", SARATHI_ROOT / "model_executor")

    sys.modules["sarathi.config"] = types.ModuleType("sarathi.config")
    sys.modules["sarathi.config"].ModelConfig = object

    weight_utils = types.ModuleType("sarathi.model_executor.weight_utils")
    weight_utils.initialize_dummy_weights = lambda _model: None
    sys.modules["sarathi.model_executor.weight_utils"] = weight_utils

    model_class_names = {
        "deepseek_v2": "DeepseekV2ForCausalLM",
        "falcon": "FalconForCausalLM",
        "internlm": "InternLMForCausalLM",
        "llama": "LlamaForCausalLM",
        "mistral": "MistralForCausalLM",
        "qwen": "QWenLMHeadModel",
        "yi": "YiForCausalLM",
    }
    for module_name, class_name in model_class_names.items():
        module = types.ModuleType(f"sarathi.model_executor.models.{module_name}")
        module.__dict__[class_name] = type(class_name, (), {})
        sys.modules[f"sarathi.model_executor.models.{module_name}"] = module

    _load_module(
        "sarathi.model_executor.models",
        SARATHI_ROOT / "model_executor" / "models" / "__init__.py",
    )
    return _load_module(
        "sarathi.model_executor.model_loader",
        SARATHI_ROOT / "model_executor" / "model_loader.py",
    )


@contextmanager
def _isolated_modules(prefixes):
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    for name in list(saved):
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


class DeepseekV2RegistrationTests(unittest.TestCase):
    def test_deepseek_v2_config_defaults_expose_mla_fields(self):
        with _isolated_modules(["sarathi.transformers_utils", "transformers"]):
            _transformers, deepseek_module, _config_module = _load_config_modules()

            config = deepseek_module.DeepseekV2Config()

            self.assertEqual(config.model_type, "deepseek_v2")
            self.assertEqual(config.architectures, ["DeepseekV2ForCausalLM"])
            self.assertEqual(config.kv_lora_rank, 512)
            self.assertEqual(config.qk_nope_head_dim, 128)
            self.assertEqual(config.qk_rope_head_dim, 64)
            self.assertEqual(config.v_head_dim, 128)
            self.assertEqual(config.num_attention_heads, 128)

    def test_get_config_uses_deepseek_registry_override(self):
        with _isolated_modules(["sarathi.transformers_utils", "transformers"]):
            transformers, deepseek_module, config_module = _load_config_modules()
            sentinel = deepseek_module.DeepseekV2Config(kv_lora_rank=256)
            recorded = {}

            class DummyAutoConfig:
                model_type = "deepseek_v2"

            transformers.AutoConfig.from_pretrained_result = DummyAutoConfig()
            original = deepseek_module.DeepseekV2Config.from_pretrained

            @classmethod
            def _fake_from_pretrained(cls, model, revision=None):
                recorded["model"] = model
                recorded["revision"] = revision
                return sentinel

            deepseek_module.DeepseekV2Config.from_pretrained = _fake_from_pretrained
            try:
                resolved = config_module.get_config(
                    "deepseek-ai/DeepSeek-V2-Lite",
                    trust_remote_code=True,
                    revision="main",
                )
            finally:
                deepseek_module.DeepseekV2Config.from_pretrained = original

            self.assertIs(resolved, sentinel)
            self.assertEqual(recorded["model"], "deepseek-ai/DeepSeek-V2-Lite")
            self.assertEqual(recorded["revision"], "main")

    def test_model_loader_resolves_deepseek_architecture(self):
        with _isolated_modules(
            ["sarathi.model_executor", "sarathi.config", "sarathi.transformers_utils", "transformers"]
        ):
            model_loader = _load_model_loader_module()

            model_class = model_loader._get_model_architecture(
                types.SimpleNamespace(architectures=["DeepseekV2ForCausalLM"])
            )

            self.assertEqual(model_class.__name__, "DeepseekV2ForCausalLM")

    def test_model_loader_rejects_unknown_deepseek_architecture_name(self):
        with _isolated_modules(
            ["sarathi.model_executor", "sarathi.config", "sarathi.transformers_utils", "transformers"]
        ):
            model_loader = _load_model_loader_module()

            with self.assertRaises(ValueError):
                model_loader._get_model_architecture(
                    types.SimpleNamespace(architectures=["DeepSeekV2ForCausalLM"])
                )


if __name__ == "__main__":
    unittest.main()
