import torch
import time
import heapq
import math
import vattention
from typing import Dict, Tuple, List

import sys

class BaseKVCache:
    """
    A class which is the key-value buffer for the model.
    A loose analogy is that this buffer is like an L1 cache and the conventional
    KV-cache is like an L2 cache
    """
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_batch_size: int,
        max_model_seq_len: int
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.device = device
        self.dtype = dtype
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.max_batch_size = max_batch_size 
        self.req_table: Dict[int, int]  = {}
        self.max_model_seq_len =  max_model_seq_len
        self._free_stack = list(range(max_batch_size))
        self.curr_batch_idx = None
        self.kv_reserved = 0
        self.curr_seq_lens = [0 for i in range(max_batch_size)]
        self.attn_context_lens = None
        self.flashinfer_cl = 0
   
    def free_request(self, seq_id: int) -> None:
        b_idx = self.req_table[seq_id]
        self.req_table.pop(seq_id)
        self._free_stack.append(b_idx)

    def reserve_kv_cache(self):
        #elem_size = torch.tensor([1], dtype=self.dtype).element_size()
        #memory_per_token  = self.num_kv_heads * self.head_size * elem_size *2 
        #memory_per_batch = memory_per_token * self.max_batch_size
        free_mem, tot_memory = torch.cuda.mem_get_info()
        elem_size = torch.tensor([1], dtype=self.dtype).element_size()
        memory_per_token  = self.num_kv_heads * self.head_size * elem_size *2 
        memory_per_batch_per_layer = memory_per_token * self.max_batch_size 
        max_len_possible = ((free_mem*.9 )// memory_per_batch_per_layer) // self.num_layers

        # cl_avail = self.max_model_seq_len
        cl_avail = min(max_len_possible, self.max_model_seq_len)
        print("cl_avail: ", cl_avail)
        self.k_cache = [torch.zeros((self.max_batch_size, cl_avail, self.num_kv_heads, self.head_size),
                                        dtype=self.dtype, device=self.device) for i in range(self.num_layers)]
        self.v_cache = [torch.zeros(( self.max_batch_size, cl_avail, self.num_kv_heads, self.head_size),
                                        dtype=self.dtype, device=self.device) for i in range(self.num_layers)]

    def get_req_batch_idx(self, seq_id: int, seq_len: int) -> int:
        if seq_id in self.req_table:
            return self.req_table[seq_id]

        return self.alloc_new_batch_idx(seq_id, seq_len)

    def alloc_new_batch_idx(self, seq_id: int, seq_len: int) -> int:
        new_batch_idx = self._free_stack.pop()
        if new_batch_idx == -1:
            print(self.curr_seq_lens)
        assert new_batch_idx != -1, "Failed to allocate new batch idx. This is not expected..."
        self.req_table[seq_id] = new_batch_idx
        return new_batch_idx


    def model_step(self, input_metadata: InputMetadata, is_profiling_iteration: bool) -> None:
        gb_idx = 0
        for idx, cl in enumerate(input_metadata.current_prompt_chunk_lens):
            gb_idx+=1
            seq_id = input_metadata.gen_seq_ids[idx]
            new_batch_idx = self.get_req_batch_idx(seq_id, cl)
            self.curr_seq_lens[new_batch_idx] = cl

        attn_context_lens = [0 for i in range(len(input_metadata.context_lens))]
        for idx, cl in enumerate(input_metadata.context_lens):
            idx_md = idx + gb_idx
            # seq_id = input_metadata.prompt_seq_ids[idx]
            seq_id = input_metadata.gen_seq_ids[idx_md]
            seq_len = cl.item()
            new_batch_idx = self.get_req_batch_idx(seq_id, seq_len)
            self.curr_seq_lens[new_batch_idx] = seq_len
            attn_context_lens[idx] = seq_len - 1

        self.flashinfer_cl = attn_context_lens[0] if len(attn_context_lens) > 0 else 0
        self.attn_context_lens = torch.tensor(attn_context_lens, dtype=torch.int32, device=self.device)
        # gb_idx = 0
        # for idx, cl in enumerate(input_metadata.current_prompt_chunk_lens):
        #     seq_id = input_metadata.gen_seq_ids[idx]
        #     gb_idx+=1
        #     if seq_id not in self.req_table:
        #         if len(self._free_stack) == 0:
        #             raise Exception("No free slots available in the cache")
        #         self.req_table[seq_id] = self._free_stack.pop()

        # attn_context_lens = [0 for i in range(len(input_metadata.context_lens))] 
        # for idx, cl in enumerate(input_metadata.context_lens):
        #     idx = idx + gb_idx
        #     seq_id = input_metadata.gen_seq_ids[idx]
        #     if seq_id not in self.req_table:
        #         self.req_table[seq_id] = self._free_stack.pop()
        #     self.curr_seq_lens[self.req_table[seq_id]] = cl.item()
        #     attn_context_lens[idx] = cl.item()
        # self.flashinfer_cl = attn_context_lens[0] if len(attn_context_lens) > 0 else 0
        # self.attn_context_lens = torch.tensor(attn_context_lens, dtype=torch.int32, device=self.device)

    def add_request_with_kv(self,
                                seq_id: int,
                                buffer_len: int,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                layer_idx: int) -> None:
        req_batch_idx = self.req_table[seq_id]
        self.k_cache[layer_idx][req_batch_idx][:buffer_len].copy_(k)
        self.v_cache[layer_idx][req_batch_idx][:buffer_len].copy_(v)

    """
    def get_kv_cache(self, seq_id: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        b_idx = []
        for id in seq_id:
            b_idx.append(self.req_table[id])
            sums = self.k_cache[b_idx].sum(dim=(0, 2, 3))
            non_zero_indices = torch.nonzero(sums).squeeze()
        return self.k_cache[b_idx], self.v_cache[b_idx]
    """

    """
    def add_k_v_to_cache(self, seq_id, key, value):
        b_idx = self.req_table[seq_id]
        length = self.req_len[seq_id]
        self.k_cache[b_idx, length].copy_(key.squeeze(0))
        self.v_cache[b_idx, length].copy_(value.squeeze(0))
        self.req_len[seq_id] += 1
    """

    def get_batch_idx(self) -> torch.Tensor:
        return self.curr_batch_idx

    def create_batch_index(self, seq_ids: List[int]) -> torch.Tensor:
        try:
            b_idx = []
            for id in seq_ids:
                b_idx.append(self.req_table[id])
            self.curr_batch_idx = torch.tensor(b_idx, dtype=torch.int32, device=self.device)
        except Exception as e:
            print("Exception: ", e)

    def clear_batch_index(self) -> None:
        self.curr_batch_idx = None

    def get_k_cache(self, layer_idx: int) -> torch.Tensor:
        return self.k_cache[layer_idx]

    def get_v_cache(self, layer_idx: int) -> torch.Tensor:
        return self.v_cache[layer_idx]

    def disable_deferred_reclamation(self):
        pass

    def reclaim_req_ids(self) -> None:
        pass

    def set_attention_context_lens(self, InputMetadata):
        attn_context_lens = [0 for i in range(len(InputMetadata.context_lens))]
        for idx, cl in enumerate(InputMetadata.context_lens):
            # seq_id = InputMetadata.prompt_seq_ids[idx]
            attn_context_lens[idx] = cl.item()
        self.attn_context_lens = torch.tensor(attn_context_lens, dtype=torch.int32, device=self.device)

    def get_attention_context_lens(self):
        return BaseKVCache.attn_context_lens

    def get_attention_metadata(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx], self.attn_context_lens, self.curr_batch_idx

