import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SARATHI_ROOT = REPO_ROOT / "sarathi-lean" / "sarathi"


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TokenizerFallbackTests(unittest.TestCase):
    def setUp(self):
        self.original_modules = {
            name: sys.modules.get(name)
            for name in [
                "transformers",
                "sarathi",
                "sarathi.logger",
                "sarathi.transformers_utils",
                "sarathi.transformers_utils.tokenizer",
            ]
        }
        transformers = types.ModuleType("transformers")

        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                del args, kwargs
                raise KeyError("DeepseekV2Config")

        class PreTrainedTokenizer:
            pass

        class PreTrainedTokenizerFast:
            @classmethod
            def from_pretrained(cls, path, *args, **kwargs):
                del args, kwargs
                return {"loaded_from": path}

        transformers.AutoTokenizer = AutoTokenizer
        transformers.PreTrainedTokenizer = PreTrainedTokenizer
        transformers.PreTrainedTokenizerFast = PreTrainedTokenizerFast
        sys.modules["transformers"] = transformers

        sarathi = types.ModuleType("sarathi")
        sarathi.__path__ = [str(SARATHI_ROOT)]
        sys.modules["sarathi"] = sarathi

        logger_module = types.ModuleType("sarathi.logger")
        logger_module.init_logger = lambda name: types.SimpleNamespace(
            warning=lambda *args, **kwargs: None
        )
        sys.modules["sarathi.logger"] = logger_module

        transformers_utils = types.ModuleType("sarathi.transformers_utils")
        transformers_utils.__path__ = [str(SARATHI_ROOT / "transformers_utils")]
        sys.modules["sarathi.transformers_utils"] = transformers_utils

        self.tokenizer_module = _load_module(
            "sarathi.transformers_utils.tokenizer",
            SARATHI_ROOT / "transformers_utils" / "tokenizer.py",
        )

    def tearDown(self):
        for module_name, original in self.original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

    def test_get_tokenizer_falls_back_to_fast_tokenizer_for_local_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "tokenizer.json").write_text(json.dumps({"version": "1.0"}))

            tokenizer = self.tokenizer_module.get_tokenizer(tmpdir)

        self.assertEqual(tokenizer, {"loaded_from": tmpdir})


if __name__ == "__main__":
    unittest.main()
