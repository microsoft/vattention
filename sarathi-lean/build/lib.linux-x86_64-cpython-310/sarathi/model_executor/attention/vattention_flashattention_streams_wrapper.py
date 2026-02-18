from typing import List, Optional, Tuple

import torch
from flash_attn import flash_attn_with_kvcache, flash_attn_func

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
import vattention
from sarathi.cache_ops import cache_flat

logger = init_logger(__name__)


class VAttentionFlashAttentionStreamsWrapper(BaseAttentionWrapper):
    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        super().init(model_config, parallel_config, block_size, device)
        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.prefill_query_lens: List[int] = None
        self.prefill_cache_lens: List[int] = []
        self.decode_cache_lens: torch.Tensor = None
        self.batch_index: List[int] = None
        self.batch_index_gen: List[int] = None
        self.current_total_len_device_lst: List[int] = []
        self.max_cache_len = 0
        self.decode_batch_size = 0
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        self.event = torch.cuda.Event(enable_timing=False)

    def get_cache_block(
        self, num_blocks: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        prefill_query_lens: List[int] = []
        decode_cache_lens: List[int] = []
        current_total_len_list: List[int] = []
       
        self.is_profiling_iteration = False
        self.is_metadata_initialized = True
        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue
      
            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
            )
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

            current_total_len = processed_prompt_len + current_prompt_chunk_len

            prefill_query_lens.append(current_prompt_chunk_len)
            self.prefill_cache_lens.append(processed_prompt_len)
            current_total_len_list.append(current_total_len)

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            context_len = seq_metadata.seq.get_len()
            decode_cache_lens.append(context_len - 1)

        self.prefill_query_lens = prefill_query_lens
        self.current_total_len_device_lst = [
            torch.tensor([total_len], dtype=torch.int32, device=self.device)
            for total_len in current_total_len_list
        ]
      
        if decode_cache_lens == []:
            return

        self.decode_batch_size = len(decode_cache_lens)
        self.decode_cache_lens = torch.tensor(
            decode_cache_lens, dtype=torch.int32, device=self.device
        )
        self.max_cache_len = max(decode_cache_lens) + 1

    def end_forward(self):
        self.is_metadata_initialized = False
        # self.is_profiling_iteration = False
        self.prefill_query_lens = None
        self.prefill_cache_lens = []
        self.prefill_block_tables = None
        self.decode_cache_lens = None
        self.decode_block_table = None
        self.batch_index = None
        self.batch_index_gen = None
        self.current_total_len = None
        self.max_cache_len = 0
        self.decode_batch_size = 0
    
    def set_batch_idx(self, batch_idx: torch.Tensor, batch_idx_gen: torch.Tensor) -> None:
        self.batch_index = batch_idx.to(torch.int32)
        self.batch_index_gen = batch_idx_gen.to(torch.int32)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)

        # record the event and make sure that attention kernels do not run
        # before the event is recorded
        self.event.record()
        self.event.wait(self.stream1)
        self.event.wait(self.stream2)

        token_offset = 0
        output = torch.empty_like(query, device=self.device)
        # first process the prefill attention
        idx = 0
        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            with torch.cuda.stream(self.stream1):
                for prefill_cache_len, query_len, current_len_device in zip(
                    self.prefill_cache_lens, self.prefill_query_lens, self.current_total_len_device_lst
                ):
                    index = self.batch_index[idx]
                    # pick cache up to current context length and reshape
                    with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                        seq_query = query[token_offset : token_offset + query_len].reshape(
                            1, -1, self.num_q_heads, self.head_dim
                        )
                        seq_key = key[token_offset : token_offset + query_len].reshape(
                            1, -1, self.num_kv_heads, self.head_dim
                        )
                        seq_value = value[token_offset : token_offset + query_len].reshape(
                            1, -1, self.num_kv_heads, self.head_dim
                        )

                        # no need to slice as [:prefill_cache_len+query_len] since we are now using the
                        # flash_attn_with_kvcache API
                        key_cache = kv_cache[0][index].reshape(1, -1, self.num_kv_heads, self.head_dim) 
                        value_cache = kv_cache[1][index].reshape(1, -1, self.num_kv_heads, self.head_dim)

                    with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                        cache_flat(seq_key.squeeze(0),
                                seq_value.squeeze(0),
                                key_cache.squeeze(0)[prefill_cache_len:],
                                value_cache.squeeze(0)[prefill_cache_len:],
                                "auto")

                    
                    seq_output = flash_attn_with_kvcache(
                        seq_query,
                        key_cache,
                        value_cache,
                        cache_seqlens=current_len_device,
                        causal=True,
                        softmax_scale=softmax_scale,
                        )

                    with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                        output[token_offset : token_offset + query_len].copy_(
                            seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                        )
                
                    token_offset += query_len
                    idx += 1
       
            if self.decode_batch_size == 0:
                self.stream1.synchronize()
                return output

            with torch.cuda.stream(self.stream2):
                with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                    decode_query = query[
                        token_offset : token_offset + self.decode_batch_size
                    ].reshape(-1, 1, self.num_q_heads, self.head_dim)
                    decode_key = key[token_offset : token_offset + self.decode_batch_size].reshape(
                        -1, 1, self.num_kv_heads, self.head_dim
                    )
                    decode_value = value[
                        token_offset : token_offset + self.decode_batch_size
                    ].reshape(-1, 1, self.num_kv_heads, self.head_dim)
                    # print(" kv cache shape", kv_cache[0].shape)

                try:
                    # print("kv_cache shape", kv_cache[0].shape)
                    decode_output = flash_attn_with_kvcache(
                        decode_query,
                        kv_cache[0][:, :self.max_cache_len],  # k_cache,
                        kv_cache[1][:, :self.max_cache_len],  # v_cache,
                        decode_key,
                        decode_value,
                        cache_seqlens=self.decode_cache_lens,
                        block_table=None,
                        softmax_scale=softmax_scale,
                        causal=True,
                        cache_batch_idx=self.batch_index_gen,
                    )
                except RuntimeError as e:
                    if (
                        "If key is supplied, it must have seqlen <= the seqlen of the KV cache"
                        in str(e)
                    ):
                        logger.warning(
                            "Ran into transient error with flash attention: Key length is greater than the cache length. Skipping the attention computation."
                        )
                        return output
                    else:
                        raise e

                with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                    # flatten the seq_output and copy it to the output tensor
                    output[token_offset : token_offset + self.decode_batch_size].copy_(
                        decode_output.reshape(-1, self.num_q_heads * self.head_dim)
                    )
            if len(self.prefill_cache_lens) > 0:
                self.stream1.synchronize()
            self.stream2.synchronize()
        return output