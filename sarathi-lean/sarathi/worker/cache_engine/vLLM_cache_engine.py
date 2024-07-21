"""CacheEngine class for managing the KV cache."""

from typing import List, Tuple, Union

from ...core.datatypes.sequence import SequenceMetadata
import torch

from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.utils import in_wsl
from sarathi.worker.cache_engine.base_cache_engine import BaseCacheEngine
logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class vLLMCacheEngine(BaseCacheEngine):
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        mem_alloc_backend: str, # this is noop for this class
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config)   

    def allocate_gpu_cache(self) -> List[torch.Tensor]:
        gpu_cache: List[torch.Tensor] = []

        for _ in range(self.num_layers):
            gpu_blocks = get_attention_wrapper().get_cache_block(
                self.num_gpu_blocks, dtype=self.dtype, device="cuda"
            )
            gpu_cache.append(gpu_blocks)
        return gpu_cache

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

    def step(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        pass
    def on_step_completion(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        pass

    def on_step_completion(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        pass

    def num_free_blocks(self) -> int:
        return self.num_gpu_blocks 

    def cleanup_kvcache(self) -> None:
        pass


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
