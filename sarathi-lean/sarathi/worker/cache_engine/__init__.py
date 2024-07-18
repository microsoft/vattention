import torch
from typing import List, Tuple, Union
from sarathi.model_executor.attention import AttentionBackend


KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

def get_cache_engine(attn_backend: str):
    if AttentionBackend.is_vATTN(attn_backend):
        from sarathi.worker.cache_engine.vATTN_cache_engine import vATTNCacheEngine
        return vATTNCacheEngine
    elif AttentionBackend.is_vLLM(attn_backend):
        from sarathi.worker.cache_engine.vLLM_cache_engine import vLLMCacheEngine
        return vLLMCacheEngine
    else: 
        # from sarathi.worker.cache_engine.base_cache_engine import BaseCacheEngine
        # return BaseCacheEngine
        raise NotImplementedError(f"Cache engine for {attn_backend} is not implemented yet.")

def get_cache_mem_alloc_backend(attn_backend: str):
    if AttentionBackend.is_vATTN_SYNC(attn_backend):
        return "sync"
    elif AttentionBackend.is_vATTN(attn_backend):
        return "async"
    return "noop"
