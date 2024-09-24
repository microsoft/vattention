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
import pod_ampere as fused

logger = init_logger(__name__)


class VAttentionFlashAttentionPODWrapper(BaseAttentionWrapper):
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
        self.fused_param = 11

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
            #self.fused_param = 11 if processed_prompt_len < current_prompt_chunk_len else 9
            current_total_len = processed_prompt_len + current_prompt_chunk_len
            prefill_query_lens.append(current_prompt_chunk_len)
            self.prefill_cache_lens.append(processed_prompt_len)
            current_total_len_list.append(current_total_len)

        if len(prefill_query_lens) > 1:
            raise ValueError("Batched prefills are not supported currently ...")

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
        if len(prefill_query_lens)>0 and processed_prompt_len > 10240:
            self.fused_param = 11
        else:
            self.fused_param = 9

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

        token_offset = 0
        output = torch.empty_like(query, device=self.device)
        # first process the prefill attention
        idx = 0

        p_query, p_kcache, p_vcache = None, None, None
        p_query_len, p_cache_len = 0, 0
        if len(self.prefill_cache_lens) == 1:
            p_cache_len = self.current_total_len_device_lst[0]
            p_query_len = self.prefill_query_lens[0]
            cache_idx_p = self.batch_index[idx]
            p_query = query[: self.prefill_query_lens[0]].reshape(1, -1, self.num_q_heads, self.head_dim)
            p_k = key[: self.prefill_query_lens[0]].reshape(1, -1, self.num_kv_heads, self.head_dim)
            p_v = value[: self.prefill_query_lens[0]].reshape(1, -1, self.num_kv_heads, self.head_dim)
            p_kcache = kv_cache[0][cache_idx_p].reshape(1, -1, self.num_kv_heads, self.head_dim)
            p_vcache = kv_cache[1][cache_idx_p].reshape(1, -1, self.num_kv_heads, self.head_dim)
            token_offset = self.prefill_query_lens[0]
            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                cache_flat(p_k.squeeze(0),
                        p_v.squeeze(0),
                        p_kcache.squeeze(0),
                        p_vcache.squeeze(0),
                        "auto")
        elif len(self.prefill_cache_lens) > 1:
            raise ValueError("Multiple prefill cache lengths not supported")

        d_query, d_k, d_v = None, None, None
        if self.decode_batch_size != 0:
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                d_query = query[
                    token_offset : token_offset + self.decode_batch_size
                ].reshape(-1, 1, self.num_q_heads, self.head_dim)
                d_k = key[token_offset : token_offset + self.decode_batch_size].reshape(
                    -1, 1, self.num_kv_heads, self.head_dim
                )
                d_v = value[
                    token_offset : token_offset + self.decode_batch_size
                ].reshape(-1, 1, self.num_kv_heads, self.head_dim)
                # print(" kv cache shape", kv_cache[0].shape)

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            output_p, output_d = fused.true_fused_attn_with_kvcache(
                p_query,
                p_kcache,
                p_vcache,
                d_query,
                kv_cache[0],
                kv_cache[1],
                d_k,
                d_v,
                causal=True,
                cache_seqlens_p=p_cache_len,
                cache_seqlens_d=self.decode_cache_lens,
                cache_batch_idx=self.batch_index_gen,
                fused_params=self.fused_param
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            if p_query is not None:
                output[:p_query_len].copy_(
                    output_p.reshape(-1, self.num_q_heads * self.head_dim)
                )
            if d_query is not None:
                output[p_query_len : p_query_len + self.decode_batch_size].copy_(
                    output_d.reshape(-1, self.num_q_heads * self.head_dim)
                )

        return output
