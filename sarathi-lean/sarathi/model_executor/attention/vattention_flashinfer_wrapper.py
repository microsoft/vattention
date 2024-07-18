from typing import List, Optional, Tuple

import torch
from flash_attn import flash_attn_with_kvcache
from flashinfer import single_prefill_with_kv_cache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
import vattention
from sarathi.cache_ops import cache_flat

logger = init_logger(__name__)


class VAttentionFlashInferWrapper(BaseAttentionWrapper):
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
        self.prefill_cache_lens: List[torch.Tensor] = None
        self.decode_cache_len: torch.Tensor = None
        self.batch_index: List[int] = None
        self.batch_index_gen: List[int] = None
        self.prefill_cache_lens_cpu: List[int] = []
        self.prefill_cache_lens_device: List[torch.Tensor] = None
        self.decode_cache_lens_cpu: List[int] = []
        self.decode_cache_lens_device: torch.Tensor = None
        self.prefill_query_lens_cpu: List[int] = []
        # self.prefill_block_tables: List[torch.Tensor] = None
        # self.decode_block_table: torch.Tensor = None

    def get_cache_block(
        self, num_blocks: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def set_batch_idx(self, batch_idx: torch.Tensor, batch_idx_gen: torch.Tensor) -> None:
        self.batch_index = batch_idx
        self.batch_index_gen = batch_idx_gen

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        prefill_query_lens: List[int] = []
        prefill_cache_lens: List[List[int]] = []
        decode_cache_len: List[int] = []
       
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
            prefill_cache_lens.append([processed_prompt_len])

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue
            context_len = seq_metadata.seq.get_len()
            decode_cache_len.append(context_len - 1)

        self.prefill_query_lens = prefill_query_lens
        self.prefill_cache_lens = [
            torch.tensor(cache_lens, dtype=torch.int32, device=self.device)
            for cache_lens in prefill_cache_lens
        ]
        self.prefill_cache_lens_cpu = [torch.tensor(cache_lens) for cache_lens in prefill_cache_lens]
        self.prefill_query_lens_cpu = torch.tensor(prefill_query_lens)

        if decode_cache_len == []:
            # no decode block table
            return

        self.decode_cache_len = torch.tensor(
            decode_cache_len, dtype=torch.int32, device=self.device
        )
        self.decode_cache_lens_cpu = torch.tensor(decode_cache_len)

       

    def end_forward(self):
        self.is_metadata_initialized = False
        # self.is_profiling_iteration = False
        self.prefill_query_lens = None
        self.prefill_cache_lens = None
        self.prefill_block_tables = None
        self.decode_cache_len = None
        self.decode_block_table = None
        self.batch_index = None
        self.batch_index_gen = None
        self.prefill_cache_lens_cpu = []
        self.prefill_cache_lens_device = None
        self.decode_cache_lens_cpu = []
        self.decode_cache_lens_device = None

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

        token_offset = 0

        output = torch.empty_like(query, device=self.device)
        # first process the prefill attention
        idx = 0
        for prefill_cache_len, query_len, prefill_cache_len_cpu, query_lens_cpu  in zip(
            self.prefill_cache_lens, self.prefill_query_lens, self.prefill_cache_lens_cpu, self.prefill_query_lens_cpu
        ):
            
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                seq_query = query[token_offset : token_offset + query_lens_cpu].reshape(
                    -1, self.num_q_heads, self.head_dim
                ).contiguous()
                seq_key = key[token_offset : token_offset + query_lens_cpu].reshape(
                    -1, self.num_kv_heads, self.head_dim
                ).contiguous()
                seq_value = value[token_offset : token_offset + query_lens_cpu].reshape(
                    -1, self.num_kv_heads, self.head_dim
                ).contiguous()
                index = self.batch_index[idx]
                # kv_cache[0][index][prefill_cache_len:prefill_cache_len+query_len].copy_(seq_key.squeeze(0))
                # kv_cache[1][index][prefill_cache_len:prefill_cache_len+query_len].copy_(seq_value.squeeze(0))
                key_cache = kv_cache[0][index][:prefill_cache_len_cpu+query_lens_cpu].reshape(-1, self.num_kv_heads, self.head_dim)
                value_cache = kv_cache[1][index][:prefill_cache_len_cpu+query_lens_cpu].reshape(-1, self.num_kv_heads, self.head_dim)
            
            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                cache_flat(seq_key.squeeze(0), 
                           seq_value.squeeze(0), 
                           key_cache.squeeze(0)[prefill_cache_len_cpu:], 
                           value_cache.squeeze(0)[prefill_cache_len_cpu:],
                           "auto")
            

            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                seq_output = single_prefill_with_kv_cache(
                                    seq_query,
                                    # seq_key,
                                    # seq_value,
                                    key_cache,
                                    value_cache,
                                    causal = True,
                                )

            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                output[token_offset : token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                )
            
            token_offset += query_len

            idx += 1
       

        if self.decode_cache_len is None:
            return output

        decode_batch_size = self.decode_cache_len.size(0)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            decode_query = query[
                token_offset : token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_q_heads, self.head_dim)
            decode_key = key[token_offset : token_offset + decode_batch_size].reshape(
                -1, 1, self.num_kv_heads, self.head_dim
            )
            decode_value = value[
                token_offset : token_offset + decode_batch_size
            ].reshape(-1, 1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            try:
                decode_output = flash_attn_with_kvcache(
                    decode_query,
                    kv_cache[0],  # k_cache,
                    kv_cache[1],  # v_cache,
                    decode_key,
                    decode_value,
                    cache_seqlens=self.decode_cache_len,
                    block_table=None, #self.decode_block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                    cache_batch_idx=self.batch_index_gen,
                )
            except RuntimeError as e:
                if (
                    "If key is supplied, it must have seqlen <= the seqlen of the KV cache"
                    in str(e)
                ):
                    return output
                else:
                    raise e


        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            # flatten the seq_output and copy it to the output tensor
            output[token_offset : token_offset + decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim)
            )

        return output