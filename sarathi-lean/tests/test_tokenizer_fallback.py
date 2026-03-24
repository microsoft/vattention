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

        sys.modules.pop("sarathi.transformers_utils.tokenizer", None)
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

    def test_detokenize_incrementally_falls_back_when_fast_tokenizer_has_no_decoder(self):
        class _DecoderlessTokenizer:
            is_fast = True
            all_special_tokens = ["<pad>"]

            def get_added_vocab(self):
                return {}

            def convert_ids_to_tokens(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                mapping = {0: "<pad>", 1: "hello", 2: "world"}
                return [mapping[token_id] for token_id in token_ids]

            def convert_tokens_to_string(self, tokens):
                raise AttributeError("'NoneType' object has no attribute 'decode'")

            def __len__(self):
                return 3

        new_tokens, new_text, prefix_offset, read_offset = (
            self.tokenizer_module.detokenize_incrementally(
                _DecoderlessTokenizer(),
                [1, 2],
                prev_tokens=None,
            )
        )

        self.assertEqual(new_tokens, ["hello", "world"])
        self.assertEqual(new_text, " world")
        self.assertGreaterEqual(prefix_offset, 0)
        self.assertGreaterEqual(read_offset, 0)


if __name__ == "__main__":
    unittest.main()
