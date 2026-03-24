from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer


class BaseAttentionWrapper(ABC):
    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        self.device = device
        self.num_q_heads = model_config.get_num_q_heads(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_dim = model_config.get_head_size()
        self.dtype = model_config.dtype
        self.block_size = block_size
        self._timers = {}

    """
    For a given model, all layers same the same AttentionWrapper instance.
    However, we cannot have a single timer for all layers because the same timer cannot be turned on/off dynamically.
    So, we have timers for each layer separately.
    """

    def get_timer(self, operation: OperationMetrics, layer_id: Optional[int] = None):
        if self._timers.get((operation, layer_id)) is None:
            self._timers[(operation, layer_id)] = CudaTimer(operation, layer_id)
        return self._timers.get((operation, layer_id))

    @abstractmethod
    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        pass

    @classmethod
    def get_instance(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    @abstractmethod
    def end_forward(self):
        pass

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        pass

    def forward_mla(self, wrapper_inputs) -> torch.Tensor:
        required_attrs = (
            "query",
            "kv_cache",
            "kv_up_proj_weight",
            "past_resident_cache",
            "new_resident_cache",
            "softmax_scale",
            "layer_id",
            "mla_dims",
        )
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(wrapper_inputs, attr)
        ]
        if missing_attrs:
            raise ValueError(
                "wrapper_inputs is missing required MLA fields: "
                + ", ".join(missing_attrs)
            )

        from sarathi.model_executor.models.deepseek_v2 import (
            append_resident_cache,
            get_layer_cache_kv_handle,
            reconstruct_dense_kv,
            resolve_layer_cache,
        )

        runtime_kv_cache, past_resident_cache = resolve_layer_cache(
            wrapper_inputs.kv_cache,
            wrapper_inputs.past_resident_cache,
        )
        full_cache = append_resident_cache(
            past_resident_cache,
            wrapper_inputs.new_resident_cache,
        )
        key, value = reconstruct_dense_kv(
            full_cache,
            wrapper_inputs.kv_up_proj_weight,
            wrapper_inputs.mla_dims,
        )
        return self.forward(
            wrapper_inputs.query.reshape(wrapper_inputs.query.shape[0], -1),
            key.reshape(key.shape[0], -1),
            value.reshape(value.shape[0], -1),
            get_layer_cache_kv_handle(runtime_kv_cache),
            wrapper_inputs.softmax_scale,
            wrapper_inputs.layer_id,
        )
