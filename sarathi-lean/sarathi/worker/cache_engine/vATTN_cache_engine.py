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
from sarathi.worker.cache_engine.vattention_init import dispatch_init_kvcache
import vattention
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.config import CacheArchitecture
logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


def summarize_vattention_cache_usage(
    cache_spec,
    seq_lens,
    *,
    free_blocks=None,
    seq_to_batch_idx=None,
    scheduled_batch_indices=None,
    scheduled_prompt_batch_indices=None,
    scheduled_decode_batch_indices=None,
) -> dict:
    persistent_tokens = sum(max(int(seq_len), 0) for seq_len in seq_lens)
    active_batch_indices = tuple(
        batch_idx for batch_idx, seq_len in enumerate(seq_lens) if int(seq_len) > 0
    )
    architecture = (
        cache_spec.architecture.value
        if hasattr(cache_spec.architecture, "value")
        else str(cache_spec.architecture)
    )
    cache_components = tuple(
        component.name
        for component in getattr(cache_spec, "cache_components", ())
    )
    return {
        "architecture": architecture,
        "persistent_tokens": persistent_tokens,
        "persistent_bytes_per_token": cache_spec.cached_token_bytes_local,
        "persistent_bytes": persistent_tokens * cache_spec.cached_token_bytes_local,
        "page_buffer_token_bytes": cache_spec.page_buffer_token_bytes,
        "cache_components": cache_components,
        "uses_component_resident_cache": cache_spec.architecture == CacheArchitecture.MLA,
        "active_batch_indices": active_batch_indices,
        "active_request_count": len(active_batch_indices),
        "free_blocks": free_blocks,
        "seq_to_batch_idx": (
            dict(sorted(seq_to_batch_idx.items()))
            if seq_to_batch_idx is not None
            else None
        ),
        "scheduled_batch_indices": scheduled_batch_indices,
        "scheduled_prompt_batch_indices": scheduled_prompt_batch_indices,
        "scheduled_decode_batch_indices": scheduled_decode_batch_indices,
    }


def format_vattention_gpu_cache(cache_spec, kv_cache, device) -> List[object]:
    if cache_spec.architecture == CacheArchitecture.MLA:
        from sarathi.model_executor.models.deepseek_v2 import (
            DeepseekV2ComponentMLAKVCache,
        )

        kv_latent_cache, k_rope_cache = kv_cache
        assert kv_latent_cache.device == device, (
            "kv_latent cache device mismatch. expected: {}, got: {}".format(
                device, kv_latent_cache.device
            )
        )
        assert k_rope_cache.device == device, (
            "k_rope cache device mismatch expected: {}, got: {}".format(
                device, k_rope_cache.device
            )
        )
        return [
            DeepseekV2ComponentMLAKVCache(
                kv_latent=kv_latent_cache[:, :, layer_idx, :],
                k_rope=k_rope_cache[:, :, layer_idx, :].view(
                    kv_latent_cache.shape[0],
                    kv_latent_cache.shape[1],
                    cache_spec.num_heads,
                    cache_spec.mla_qk_rope_head_dim,
                ),
            )
            for layer_idx in range(cache_spec.num_layers)
        ]

    if cache_spec.megacache:
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]
        assert k_cache.device == device, \
                    "k_cache device mismatch. expected: {}, got: {}".format(device, k_cache.device)
        assert v_cache.device == device, \
                    "v_cache device mismatch expected: {}, got: {}".format(device, v_cache.device)

        return [(k_cache[:, :, i], v_cache[:, :, i]) for i in range(cache_spec.num_layers)]

    k_cache = kv_cache[:cache_spec.num_layers]
    v_cache = kv_cache[cache_spec.num_layers:]
    for i in range(cache_spec.num_layers):
        assert k_cache[i].device == device, \
                    "k_cache device mismatch. expected: {}, got: {}".format(device, k_cache[i].device)
        assert v_cache[i].device == device, \
                    "v_cache device mismatch expected: {}, got: {}".format(device, v_cache[i].device)
    return list(zip(k_cache, v_cache))

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
        self.curr_batch_idx = None
        self.prompt_batch_indices = ()
        self.decode_batch_indices = ()
        self.cache_usage_history = []
        self.page_size = cache_config.page_size
        self.vattn_async = True if mem_alloc_backend == "async" else False
        self.vattn_mega_cache = True if "megacache" in model_config.attention_backend.lower() else False
        self.cache_mem_size = cache_config.memory_for_gpu
        self.init_spec = model_config.get_vattention_init_spec(
            page_size=self.page_size,
            parallel_config=parallel_config,
            megacache=self.vattn_mega_cache,
            max_batch_size=self.max_batch_size,
            max_context_length=self.max_model_seq_len,
            device_idx=self.device_idx,
        )
        self.cache_spec = self.init_spec.cache_spec
        super().__init__(cache_config, model_config, parallel_config)

    def num_free_blocks(self) -> int:
        return vattention.num_free_kvblocks()

    def _init_kvcache_from_spec(self):
        return dispatch_init_kvcache(
            vattention,
            self.init_spec.get_extension_init_request(),
        )

    def allocate_gpu_cache(self) -> List[torch.Tensor]:
        print(f"\n[PYTHON TRACE] Initializing KV Cache:")
        print(f" > Architecture: {self.cache_spec.architecture.value}")
        print(f" > Layers: {self.num_layers}, Heads: {self.num_heads}, Head Size: {self.head_size}")
        print(f" > Max Batch: {self.max_batch_size}, Max Seq: {self.max_model_seq_len}")
        print(f" > MegaCache Enabled: {self.vattn_mega_cache}")
        print(f" > Tokens Per Page: {self.cache_spec.tokens_per_page}")
        print(f" > Page Buffer Token Bytes: {self.cache_spec.page_buffer_token_bytes}")
        
        kv_cache = self._init_kvcache_from_spec()
        cache_list = format_vattention_gpu_cache(self.cache_spec, kv_cache, self.device)
        vattention.reserve_physical_pages(self.cache_mem_size)
        
        print(f"[PYTHON TRACE] Reserving Physical Memory: {self.cache_mem_size / (1024**2):.2f} MB")
        vattention.reserve_physical_pages(self.cache_mem_size)
        return cache_list

    def preempt_requests(self, preempted_seq: List[int]) -> None:
        for seq in preempted_seq:
            self.free_request(seq.seq_id)

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

        self.prompt_batch_indices = tuple(b_idx_prompt)
        self.decode_batch_indices = tuple(b_idx_gen)
        self.curr_batch_idx = torch.tensor(b_idx_prompt+b_idx_gen, dtype=torch.int32, device=self.device)
        get_attention_wrapper().set_batch_idx(self.curr_batch_idx, torch.tensor(b_idx_gen, dtype=torch.int32, device=self.device))
        self._record_cache_usage_snapshot("step")

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
            self._record_cache_usage_snapshot("free_request")
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

    def get_cache_usage_stats(self) -> dict:
        return summarize_vattention_cache_usage(
            self.cache_spec,
            self.curr_seq_lens,
            free_blocks=self.num_free_blocks(),
            seq_to_batch_idx=self.seq_to_batch_idx,
            scheduled_batch_indices=(
                None
                if getattr(self, "curr_batch_idx", None) is None
                else tuple(self.curr_batch_idx.tolist())
            ),
            scheduled_prompt_batch_indices=getattr(self, "prompt_batch_indices", None),
            scheduled_decode_batch_indices=getattr(self, "decode_batch_indices", None),
        )

    def _record_cache_usage_snapshot(self, event: str) -> None:
        if not hasattr(self, "cache_spec") or not hasattr(self, "curr_seq_lens"):
            return
        snapshot = dict(self.get_cache_usage_stats())
        snapshot["event"] = event
        if not hasattr(self, "cache_usage_history"):
            self.cache_usage_history = []
        self.cache_usage_history.append(snapshot)

    def get_cache_usage_history(self):
        return tuple(getattr(self, "cache_usage_history", ()))

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        return model_config.get_cache_block_size_bytes(block_size, parallel_config)

    def cleanup_kvcache(self):
        # this is required to ensure UVM module is not holding on to the memory
        vattention.cleanup()
