"""CacheEngine class for managing the KV cache."""
import traceback
from typing import List, Tuple, Union
from sarathi.core.datatypes.sequence import Sequence
import torch
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.utils import in_wsl
from sarathi.worker.cache_engine.base_cache_engine import BaseCacheEngine
import vattention
from sarathi.model_executor.attention import get_attention_wrapper
logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

class vATTNCacheEngine(BaseCacheEngine):
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """
    _instance = None

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        mem_alloc_backend: str,
    ) -> None:
        self.max_batch_size = cache_config.max_batch_size
        self.device = torch.empty(1).cuda().device if not in_wsl() else torch.device("cuda")
        self.device_idx = int(str(self.device).split(":")[-1])
        self.max_model_seq_len = model_config.max_model_len
        self.curr_seq_lens = [0 for i in range(self.max_batch_size)]
        self.seq_to_batch_idx = {}
        self.page_size = cache_config.page_size
        self.vattn_async = True if mem_alloc_backend == "async" else False
        self.cache_mem_size = cache_config.memory_for_gpu
        super().__init__(cache_config, model_config, parallel_config)

    def num_free_blocks(self) -> int:
        return vattention.can_allocate_new_sequence()

    def allocate_gpu_cache(self) -> List[torch.Tensor]:
        kv_cache = vattention.init_kvcache(
                                    self.num_layers,
                                    self.num_heads,
                                    self.head_size,
                                    self.max_batch_size,
                                    self.max_model_seq_len,
                                    self.device_idx,
                                    self.dtype,
                                    self.page_size)

        k_cache = kv_cache[:self.num_layers]
        v_cache = kv_cache[self.num_layers:]
        for i in range(self.num_layers):
            assert k_cache[i].device == self.device, \
                        "k_cache device mismatch. expected: {}, got: {}".format(self.device, self.k_cache[i].device)
            assert v_cache[i].device == self.device, \
                        "v_cache device mismatch expected: {}, got: {}".format(self.device, self.v_cache[i].device)
        vattention.reserve_physical_pages(self.cache_mem_size)
        return list(zip(k_cache, v_cache))

    def get_k_cache(self, layer_idx: int) -> torch.Tensor:
        return self.gpu_cache[layer_idx][0]

    def get_v_cache(self, layer_idx: int) -> torch.Tensor:
        return self.gpu_cache[layer_idx][1]
    
    def step(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        b_idx_prompt = []
        b_idx_gen = []
        for seq_metadata in seq_metadata_list:
            
            if seq_metadata.is_prompt:
                seq_id = seq_metadata.seq.seq_id
                prompt_chunk_len = seq_metadata.prompt_chunk_len
                current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
                )
                processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

                context_len = processed_prompt_len + current_prompt_chunk_len
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len
                # b_idx.append(new_batch_idx)
                b_idx_prompt.append(new_batch_idx)
            
            else:
                context_len = seq_metadata.seq.get_len()
                seq_id = seq_metadata.seq.seq_id
                new_batch_idx = self.get_req_batch_idx(seq_id, context_len)
                self.curr_seq_lens[new_batch_idx] = context_len 
                # b_idx.append(new_batch_idx)
                b_idx_gen.append(new_batch_idx)

        if self.vattn_async:
            vattention.step_async(self.curr_seq_lens)
        else:
            vattention.step(self.curr_seq_lens, True)

        self.curr_batch_idx = torch.tensor(b_idx_prompt+b_idx_gen, dtype=torch.int32, device=self.device)
        get_attention_wrapper().set_batch_idx(self.curr_batch_idx, torch.tensor(b_idx_gen, dtype=torch.int32, device=self.device))

    def on_step_completion(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        for seq_metadata in seq_metadata_list:
            if seq_metadata.seq.is_finished():
                self.free_request(seq_metadata.seq.seq_id)

    def get_req_batch_idx(self, seq_id: int, seq_len: int) -> int:
        if seq_id in self.seq_to_batch_idx:
            return self.seq_to_batch_idx[seq_id]

        return self.alloc_new_batch_idx(seq_id, seq_len)

    def alloc_new_batch_idx(self, seq_id: int, seq_len: int) -> int:
        new_batch_idx = vattention.alloc_new_batch_idx(seq_len)
        if new_batch_idx == -1:
            print(self.curr_seq_lens)
        assert new_batch_idx != -1, "Failed to allocate new batch idx. This is not expected..."
        self.seq_to_batch_idx[seq_id] = new_batch_idx
        return new_batch_idx

    def free_request(self, seq_id: int) -> None:
        if seq_id in self.seq_to_batch_idx:
            batch_idx = self.seq_to_batch_idx[seq_id]
            vattention.free_batch_idx(batch_idx)
            self.seq_to_batch_idx.pop(seq_id)
            self.curr_seq_lens[batch_idx] = 0
            return
        raise Exception(f"seq_id {seq_id} not found in req_table")

    def reclaim_req_ids(self) -> None:
        for seq_id in list(self.seq_to_batch_idx.keys()):
            self.free_request(seq_id)

    def get_batch_idx(self) -> torch.Tensor:
        return self.curr_batch_idx

    def clear_batch_index(self) -> None:
        self.curr_batch_idx = None

    def release_kvcache_physical(self):
        vattention.release_kvcache_physical()

    def disable_deferred_reclamation(self):
        vattention.set_deferred_reclamation(False)

    def get_attention_context_lens(self):
        return self.attn_context_lens

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total

    def cleanup_kvcache(self):
        # this is required to ensure UVM module is not holding on to the memory
        vattention.cleanup()


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
