from typing import List, Optional, Tuple

import torch
from flash_attn import flash_attn_with_kvcache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.cache_ops import reshape_and_cache_flash

logger = init_logger(__name__)


class FlashAttentionWrapper(BaseAttentionWrapper):
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
        self.prefill_block_tables: List[torch.Tensor] = None
        self.decode_block_table: torch.Tensor = None
        self.prefix_plus_current_prompt_tokens_slot_mapping: torch.Tensor = None
        self.current_tokens_slot_mapping: torch.Tensor = None

    def get_cache_block(
        self, num_blocks: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_cache = torch.randn(
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )
        v_cache = torch.randn(
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

        return k_cache, v_cache

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        prefill_query_lens: List[int] = []
        prefill_cache_lens: List[List[int]] = []
        decode_cache_len: List[int] = []
        prefill_block_tables: List[List[int]] = []
        decode_block_table: List[List[int]] = []
        prefix_plus_current_prompt_tokens_slot_mapping: List[int] = []
        current_tokens_slot_mapping: List[int] = []


        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue
            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_len = seq_metadata.seq.get_next_prompt_chunk_len(
                prompt_chunk_len
            )
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

            current_total_len = processed_prompt_len + current_prompt_chunk_len

            prefill_query_lens.append(current_prompt_chunk_len)
            prefill_cache_lens.append([processed_prompt_len])

            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            # print("block_table", seq_metadata.block_table, "num_blocks_in_use", num_blocks_in_use)
            prefill_block_tables.append(seq_metadata.block_table[:num_blocks_in_use])
            seq_blc_table = seq_metadata.block_table[:num_blocks_in_use]
            context_end = processed_prompt_len + current_prompt_chunk_len
            context_start = 0
            # print("context_end", context_end, " processed_prompt_len", processed_prompt_len)
            for i in range(context_end):
                block_number = seq_blc_table[i // self.block_size]
                # print("block_number", block_number, "block_size", self.block_size, " seq_blc_table", seq_blc_table, " i", i)
                block_offset = i % self.block_size
                slot = (block_number) * self.block_size + block_offset
                # if i >= context_start:
                #     prefix_plus_current_prompt_tokens_slot_mapping.append(slot)
                if i >= processed_prompt_len:
                    # current_tokens_slot_mapping.append(slot)
                    prefix_plus_current_prompt_tokens_slot_mapping.append(slot)
                # print("slot", slot)
        # print("prefix_plus_current_prompt_tokens_slot_mapping", prefix_plus_current_prompt_tokens_slot_mapping)

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            context_len = seq_metadata.seq.get_len()
            decode_cache_len.append(context_len - 1)
            position = context_len - 1
            # Compute the kv page indices for the prompt tokens.
            decode_block_table.append(seq_metadata.block_table)
            gen_blc_table = seq_metadata.block_table
            block_number = gen_blc_table[position // self.block_size]
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            current_tokens_slot_mapping.append(slot)

        self.prefill_query_lens = prefill_query_lens
        self.prefill_cache_lens = [
            torch.tensor(cache_lens, dtype=torch.int32, device=self.device)
            for cache_lens in prefill_cache_lens
        ]
        self.prefill_block_tables = [
            torch.tensor(block_table, dtype=torch.int32, device=self.device).reshape(
                1, -1
            )
            for block_table in prefill_block_tables
        ]
        self.prefix_plus_current_prompt_tokens_slot_mapping = torch.tensor(
            prefix_plus_current_prompt_tokens_slot_mapping, dtype=torch.long, device=self.device
        )

        if decode_cache_len == []:
            # no decode block table
            return

        self.decode_cache_len = torch.tensor(
            decode_cache_len, dtype=torch.int32, device=self.device
        )

        max_decode_blocks = max(len(seq_block) for seq_block in decode_block_table)
        decode_block_table_padded = [
            seq_block + [-1] * (max_decode_blocks - len(seq_block))
            for seq_block in decode_block_table
        ]
        self.decode_block_table = torch.tensor(
            decode_block_table_padded, dtype=torch.int32, device=self.device
        )
        
        self.current_tokens_slot_mapping = torch.tensor(
            current_tokens_slot_mapping, dtype=torch.long, device=self.device
        )
        # print("self.prefix_plus_current_prompt_tokens_slot_mapping", self.prefix_plus_current_prompt_tokens_slot_mapping)

    def end_forward(self):
        self.is_metadata_initialized = False

        self.prefill_query_lens = None
        self.prefill_cache_lens = None
        self.prefill_block_tables = None
        self.decode_cache_len = None
        self.decode_block_table = None

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
        # print(" self.prefiix_plus_current_prompt_tokens_slot_mapping", self.prefix_plus_current_prompt_tokens_slot_mapping)
        # first process the prefill attention
        for prefill_cache_len, prefill_block_table, query_len in zip(
            self.prefill_cache_lens, self.prefill_block_tables, self.prefill_query_lens
        ):
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
            
            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                slot_mapping = self.prefix_plus_current_prompt_tokens_slot_mapping[token_offset: token_offset + query_len]
                # print(" slot_mapping", slot_mapping, slot_mapping.size(), " query_len+ prefill_cache_len", query_len+ prefill_cache_len)
                assert slot_mapping is not None
                # print(" slot_mapping", slot_mapping.type(), seq_key.type(), seq_value.type())
                reshape_and_cache_flash(seq_key.squeeze(0), 
                                        seq_value.squeeze(0), 
                                        kv_cache[0],
                                        kv_cache[1],
                                        slot_mapping,
                                        "auto",
                                        )


            # print(" cache_copy done")
            
            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                seq_output = flash_attn_with_kvcache(
                    seq_query,
                    kv_cache[0],  # k_cache,
                    kv_cache[1],  # v_cache,
                    # seq_key,
                    # seq_value,
                    cache_seqlens=prefill_cache_len+query_len,
                    block_table=prefill_block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                )

            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                output[token_offset : token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                )

            token_offset += query_len

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

        
            # try:
        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            slot_mapping = self.current_tokens_slot_mapping[token_offset: token_offset + decode_batch_size]
            # reshape_and_cache_flash(decode_key,
            #                         decode_value,
            #                         kv_cache[0],
            #                         kv_cache[1],
            #                         slot_mapping.flatten(),
            #                         "auto",
            #                         )
        # print("decode_key", decode_key.size(), "decode_value", decode_value.size(), "kv_cache", kv_cache[0].size(), kv_cache[1].size(), "slot_mapping", slot_mapping.size())
        # print(" block_table", self.decode_block_table)
        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            decode_output = flash_attn_with_kvcache(
                decode_query,
                kv_cache[0],  # k_cache,
                kv_cache[1],  # v_cache,
                decode_key,
                decode_value,
                cache_seqlens=self.decode_cache_len,
                block_table=self.decode_block_table,
                softmax_scale=softmax_scale,
                causal=True,
            )
    # except RuntimeError as e:
    #     if (
    #         "If key is supplied, it must have seqlen <= the seqlen of the KV cache"
    #         in str(e)
    #     ):
    #         logger.warning(
    #             "Ran into transient error with flash attention: Key length is greater than the cache length. Skipping the attention computation."
    #         )
    #         return output
    #     else:
    #         raise e

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            # flatten the seq_output and copy it to the output tensor
            output[token_offset : token_offset + decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim)
            )

        return output
