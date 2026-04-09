"""CacheEngine class for managing the KV cache."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.config import CacheConfig, ModelConfig, ParallelConfig
from sarathi.logger import init_logger
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.utils import in_wsl

logger = init_logger(__name__)

KVCache = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseCacheEngine(ABC):
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU KV cache.
    """
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()

    @abstractmethod
    def allocate_gpu_cache(self) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def step(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        pass

    @abstractmethod
    def on_step_completion(self, seq_metadata_list: List[SequenceMetadata]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        pass



