from sarathi.core.datatypes.sequence import Sequence
import torch
from typing import Dict, List
import vattention
from sarathi.worker.cache_engine import get_cache_engine
from sarathi.model_executor.attention import get_attn_type
import math

class vAttentionBlockSpaceManager():

    def __init__(self, 
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size 
        self.num_total_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len
        self.promised_blocks = 0
        self.watermark = watermark
        assert watermark >= 0.0
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.active_requests: Dict[int, Sequence] = {}
        self.preemption_queue = []

    # def reset_free_blocks():
    #     self.free_blocks = 0

    def get_num_blocks(self, seq: Sequence) -> int:
        # print("seq.get_len(): ", seq.get_len(), " self.block_size: ", self.block_size)
        len_seq = seq.get_len()
        num_blocks = math.ceil(len_seq / self.block_size)
        return num_blocks

    def can_allocate(self, seq: Sequence) -> bool:
        # return True
        # if self.__getattribute__('free_blocks') is None:
        #     return True
        num_required_blocks = self.get_num_blocks(seq)
        num_free_gpu_blocks = self.free_blocks
        # print("num_free_gpu_blocks: ", num_free_gpu_blocks, " num_required_blocks: ", num_required_blocks, " self.promised_blocks: ", self.promised_blocks, " self.watermark_blocks: ", self.watermark_blocks)
        return num_free_gpu_blocks - self.promised_blocks - num_required_blocks >= self.watermark_blocks

    def set_free_blocks(self, free_blocks: int) -> None:
        self.free_blocks = free_blocks

    def allocate(self, seq: Sequence) -> None:
        self.active_requests[seq.seq_id] = seq
        self.promised_blocks += self.get_num_blocks(seq)
    
    def can_append_slot(self) -> bool:
        # num_free_gpu_blocks = self.free_blocks
        # return (num_free_gpu_blocks - self.promised_blocks) > 0
        # return True
        # return self.free_blocks > self.promised_blocks *1.1
        return self.free_blocks - self.promised_blocks > 0
        

    def append_slot(self, seq: Sequence) -> None:
        """Allocate a physical slot for a new token."""
        len_seq = seq.get_len()
        num_blocks_current = math.ceil(len_seq / self.block_size)
        num_blocks_new = math.ceil((len_seq + 1) / self.block_size)
        if num_blocks_new > num_blocks_current:
            self.promised_blocks += 1
        # pass

    def _get_physical_blocks(self, seq: Sequence):
        pass

    def _free_block_table(self, block_table) -> None:
        pass

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.active_requests:
            # Already freed or haven't been scheduled yet.
            return
        else:
            del self.active_requests[seq.seq_id]
            self.free_blocks += self.get_num_blocks(seq)

    def reset(self) -> None:
        self.active_requests = {}
        pass

    def clear_promised_blocks(self) -> None:
        self.promised_blocks = 0

    def get_block_table(self, seq: Sequence) -> List[int]:
        pass

    def is_allocated(self, seq: Sequence) -> bool:
        return seq.seq_id in self.active_requests
    
    def get_num_free_gpu_blocks(self, seq: Sequence) -> int:
        return self.free_blocks


# class BaseBlockSpaceManager(ABC):
#     """Manages the mapping between logical and physical token blocks."""

#     def __init__(
#         self,
#         block_size: int,
#         num_gpu_blocks: int,
#         max_model_len: int,
#         watermark: float = 0.01,
#     ) -> None:
#         self.block_size = block_size
#         self.num_total_gpu_blocks = num_gpu_blocks
#         self.max_model_len = max_model_len

#         self.watermark = watermark
#         assert watermark >= 0.0

#         self.watermark_blocks = int(watermark * num_gpu_blocks)
#         self.gpu_allocator = BlockAllocator(block_size, num_gpu_blocks)
#         # Mapping: seq_id -> BlockTable.
#         self.block_tables: Dict[int, BlockTable] = {}

#     @abstractmethod
#     def get_num_initial_blocks(self, seq: Sequence) -> int:
#         """Returns the number of blocks to allocate for a request initially."""
#         pass

#     def can_allocate(self, seq: Sequence) -> bool:
#         # FIXME(woosuk): Here we assume that all sequences in the group share
#         # the same prompt. This may not be true for preempted sequences.
#         num_required_blocks = self.get_num_initial_blocks(seq)
#         num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
#         # Use watermark to avoid frequent cache eviction.
#         return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

#     def allocate(self, seq: Sequence) -> None:
#         # Allocate new physical token blocks that will store the prompt tokens.
#         block_table: BlockTable = []
#         num_initial_blocks = self.get_num_initial_blocks(seq)
#         for _ in range(num_initial_blocks):
#             block = self.gpu_allocator.allocate()
#             block_table.append(block)

#         self.block_tables[seq.seq_id] = block_table

#     def can_append_slot(self) -> bool:
#         num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
#         return num_free_gpu_blocks > 0

#     def append_slot(self, seq: Sequence) -> None:
#         """Allocate a physical slot for a new token."""
#         logical_blocks = seq.logical_token_blocks
#         block_table = self.block_tables[seq.seq_id]

#         if len(block_table) < len(logical_blocks):
#             # The sequence has a new logical block.
#             # Allocate a new physical block.
#             block = self.gpu_allocator.allocate()
#             block_table.append(block)

#     def _get_physical_blocks(self, seq: Sequence) -> BlockTable:
#         assert seq.is_executing()
#         return self.block_tables[seq.seq_id]

#     def _free_block_table(self, block_table: BlockTable) -> None:
#         for block in set(block_table):
#             self.gpu_allocator.free(block)

#     def free(self, seq: Sequence) -> None:
#         if seq.seq_id not in self.block_tables:
#             # Already freed or haven't been scheduled yet.
#             return
#         block_table = self.block_tables[seq.seq_id]
#         self._free_block_table(block_table)
#         del self.block_tables[seq.seq_id]

#     def reset(self) -> None:
#         for block_table in self.block_tables.values():
#             self._free_block_table(block_table)
#         self.block_tables.clear()

#     def get_block_table(self, seq: Sequence) -> List[int]:
#         block_table = self.block_tables[seq.seq_id]
#         return [block.block_number for block in block_table]

#     def get_num_free_gpu_blocks(self) -> int:
#         return self.gpu_allocator.get_num_free_blocks()

#     def is_allocated(self, seq: Sequence) -> bool:
#         return seq.seq_id in self.block_tables
# 