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
    def test_parse_context_lengths_uses_defaults_when_not_provided(self):
        lengths = sweep_module.parse_context_lengths(None)

        self.assertEqual(lengths, list(sweep_module.CONTEXT_LENGTHS))

    def test_parse_context_lengths_normalizes_and_sorts_values(self):
        lengths = sweep_module.parse_context_lengths("2048, 512,2048,1024")

        self.assertEqual(lengths, [512, 1024, 2048])

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

    def test_select_context_lengths_filters_by_server_limit(self):
        filtered = sweep_module.select_context_lengths(
            sweep_module.CONTEXT_LENGTHS,
            32768,
        )

        self.assertEqual(
            filtered,
            [
                128,
                512,
                1024,
                1536,
                1792,
                2048,
                2560,
                3072,
                3584,
                3840,
                4096,
                4352,
                4608,
                4864,
                5120,
                5632,
                6144,
                6656,
                7168,
                7680,
                8192,
                9216,
                10240,
                11264,
                12288,
                13312,
                14336,
                15360,
                16384,
                17408,
                18432,
                19456,
                20480,
                21504,
                22528,
                23552,
                24576,
                25600,
                26624,
                27648,
                28672,
                29696,
                30720,
                31744,
                32768,
            ],
        )

    def test_select_context_lengths_rejects_too_small_limit(self):
        with self.assertRaisesRegex(RuntimeError, "smaller than the smallest"):
            sweep_module.select_context_lengths(sweep_module.CONTEXT_LENGTHS, 64)


if __name__ == "__main__":
    unittest.main()
