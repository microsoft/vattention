import unittest

from sarathi.core.block_space_manager.vattention_block_space_manager import (
    vAttentionBlockSpaceManager,
)
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence import Sequence


class VAttentionBlockSpaceManagerTests(unittest.TestCase):
    def _make_sequence(self, *, prompt_len: int, block_size: int) -> Sequence:
        return Sequence(
            seq_id="req0",
            prompt=None,
            prompt_token_ids=[1] * prompt_len,
            block_size=block_size,
            eos_token_id=2,
            arrival_time=0.0,
            sampling_params=SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1),
        )

    def test_can_append_slot_allows_decode_within_existing_block(self):
        block_size = 262144
        manager = vAttentionBlockSpaceManager(
            block_size=block_size,
            num_gpu_blocks=1,
            max_model_len=128,
        )
        seq = self._make_sequence(prompt_len=2, block_size=block_size)
        manager.set_free_blocks(1)
        manager.allocate(seq)

        self.assertTrue(manager.can_append_slot(seq))
        manager.append_slot(seq)
        self.assertEqual(manager.promised_blocks, 1)

    def test_can_append_slot_requires_free_block_when_sequence_crosses_boundary(self):
        block_size = 2
        manager = vAttentionBlockSpaceManager(
            block_size=block_size,
            num_gpu_blocks=1,
            max_model_len=8,
        )
        seq = self._make_sequence(prompt_len=2, block_size=block_size)
        manager.set_free_blocks(1)
        manager.allocate(seq)

        self.assertFalse(manager.can_append_slot(seq))


if __name__ == "__main__":
    unittest.main()
