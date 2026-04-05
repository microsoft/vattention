import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "fragmentation_context_sweep.py"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sweep_module = _load_module("fragmentation_context_sweep", SCRIPT_PATH)


class _TokenizerWithSpecialTokenSupport:
    def encode(self, text, add_special_tokens=False):
        tokens = [len(piece) for piece in text.split()]
        if add_special_tokens:
            return [999] + tokens + [1000]
        return tokens


class _TokenizerWithoutSpecialTokenFlag:
    def encode(self, text):
        return [ord(char) for char in text]


class FragmentationContextSweepTests(unittest.TestCase):
    def test_encode_without_special_tokens_uses_flag_when_supported(self):
        tokenizer = _TokenizerWithSpecialTokenSupport()

        token_ids = sweep_module.encode_without_special_tokens(tokenizer, "alpha beta")

        self.assertEqual(token_ids, [5, 4])

    def test_encode_without_special_tokens_falls_back_for_simple_tokenizers(self):
        tokenizer = _TokenizerWithoutSpecialTokenFlag()

        token_ids = sweep_module.encode_without_special_tokens(tokenizer, "ab")

        self.assertEqual(token_ids, [97, 98])

    def test_build_exact_prompt_token_ids_tiles_and_truncates_pool(self):
        prompt_token_ids = sweep_module.build_exact_prompt_token_ids(8, [3, 5, 7])

        self.assertEqual(prompt_token_ids, [3, 5, 7, 3, 5, 7, 3, 5])

    def test_build_exact_prompt_token_ids_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            sweep_module.build_exact_prompt_token_ids(0, [1, 2, 3])

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            sweep_module.build_exact_prompt_token_ids(4, [])


if __name__ == "__main__":
    unittest.main()
