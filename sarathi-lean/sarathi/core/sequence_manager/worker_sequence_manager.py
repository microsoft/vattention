from typing import List

from sarathi.config import BaseSchedulerConfig, CacheConfig, ModelConfig
from sarathi.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from sarathi.core.block_space_manager.vattention_block_space_manager import vAttentionBlockSpaceManager
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from sarathi.model_executor.attention import AttentionBackend


class WorkerSequenceManager(BaseSequenceManager):

    def __init__(
        self,
        cache_config: CacheConfig,
        scheduler_config: BaseSchedulerConfig,
        model_config: ModelConfig,
    ):
        super().__init__()
        # we will have a clone of block manager here, it is supposed
        # to work in sync block manager in scheduler the idea is to avoid
        # sending block table every time to the worker
        if AttentionBackend.is_vATTN(model_config.attention_backend):
                self.block_manager = vAttentionBlockSpaceManager(
                    cache_config.block_size,
                    cache_config.num_gpu_blocks,
                    scheduler_config.max_model_len,
                )
        else:
            self.block_manager = BlockSpaceManagerRegistry.get(
                scheduler_config.type,
                cache_config.block_size,
                cache_config.num_gpu_blocks,
                scheduler_config.max_model_len,
            )

    def _free_seq(self, seq_id: int) -> None:
        # ignored sequences might not have been allocated
        assert seq_id in self.seq_map
        seq = self.seq_map[seq_id]
        if self.block_manager.is_allocated(seq):
            self.block_manager.free(seq)
        super()._free_seq(seq_id)

    def _preempt_seq(self, seq_id: int) -> None:
        super()._preempt_seq(seq_id)
        seq = self.seq_map[seq_id]
        self.block_manager.free(seq)

    def _on_seq_scheduled(self, seq_sched_metadata: SequenceScheduleMetadata) -> None:
        super()._on_seq_scheduled(seq_sched_metadata)
        seq = self.seq_map[seq_sched_metadata.seq_id]
        # print("is allocated", self.block_manager.is_allocated(seq), "Block manager type", self.block_manager.__class__.__name__)
        if self.block_manager.is_allocated(seq):
            self.block_manager.can_append_slot()
            self.block_manager.append_slot(seq)
        else:
            # lazily allocate memory when a seq
            # is allocated for the first time
            # assert self.block_manager.can_allocate(seq)
            self.block_manager.allocate(seq)

    def _on_append_token(self, seq: Sequence) -> None:
        # the engine performs detokenization at this point
        # but we don't need to do anything here on worker side
        pass

    def _get_block_table(self, seq: Sequence) -> List[int]:
        return self.block_manager.get_block_table(seq)
