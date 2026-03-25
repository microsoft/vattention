from abc import ABC
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PretrainedConfig

from sarathi.logger import init_logger
from sarathi.transformers_utils.config import get_config
from sarathi.utils.base_int_enum import BaseIntEnum

logger = init_logger(__name__)


class CacheArchitecture(Enum):
    DENSE_KV = "dense_kv"
    MLA = "mla"


@dataclass(frozen=True)
class CacheLayout:
    architecture: CacheArchitecture
    megacache: bool
    cached_token_bytes_per_layer: int
    cached_token_bytes_local: int
    page_buffer_token_bytes: int
    tokens_per_page: int


@dataclass(frozen=True)
class MLAAttentionSpec:
    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_head_dim: int
    resident_cache_dim: int


@dataclass(frozen=True)
class TensorParallelAttentionSpec:
    tensor_parallel_size: int
    num_q_heads_global: int
    num_q_heads_local: int
    num_kv_heads_global: int
    num_kv_heads_local: int
    head_size: int


@dataclass(frozen=True)
class MLATensorParallelAttentionSpec:
    tp_attention: TensorParallelAttentionSpec
    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_head_dim: int
    resident_cache_dim: int


@dataclass(frozen=True)
class CacheComponentSpec:
    name: str
    token_dim: int

    def __post_init__(self):
        if not self.name:
            raise ValueError("Cache component name must be non-empty")
        if self.token_dim <= 0:
            raise ValueError("Cache component token_dim must be positive")


@dataclass(frozen=True)
class VAttentionCacheSpec:
    architecture: CacheArchitecture
    megacache: bool
    page_size: int
    tokens_per_page: int
    cached_token_bytes_per_layer: int
    cached_token_bytes_local: int
    page_buffer_token_bytes: int
    dtype_size: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    tp_attention: TensorParallelAttentionSpec
    cache_components: Tuple[CacheComponentSpec, ...]
    mla_kv_lora_rank: Optional[int]
    mla_qk_rope_head_dim: Optional[int]

    def __post_init__(self):
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if self.cached_token_bytes_per_layer <= 0:
            raise ValueError("cached_token_bytes_per_layer must be positive")
        if self.cached_token_bytes_local <= 0:
            raise ValueError("cached_token_bytes_local must be positive")
        if self.page_buffer_token_bytes <= 0:
            raise ValueError("page_buffer_token_bytes must be positive")
        if self.dtype_size <= 0:
            raise ValueError("dtype_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.head_size <= 0:
            raise ValueError("head_size must be positive")
        if not self.cache_components:
            raise ValueError("cache_components must be non-empty")

        component_token_dim = sum(
            component.token_dim for component in self.cache_components
        )
        if component_token_dim * self.dtype_size != self.cached_token_bytes_per_layer:
            raise ValueError(
                "cache_components do not match cached_token_bytes_per_layer"
            )
        if self.page_buffer_token_bytes > self.page_size:
            raise ValueError("page_buffer_token_bytes cannot exceed page_size")
        if self.tokens_per_page != self.page_size // self.page_buffer_token_bytes:
            raise ValueError("tokens_per_page does not match page_size and page_buffer_token_bytes")

        is_mla = self.architecture == CacheArchitecture.MLA
        if is_mla:
            if self.mla_kv_lora_rank is None or self.mla_qk_rope_head_dim is None:
                raise ValueError("MLA cache spec requires MLA dimensions")
        else:
            if self.mla_kv_lora_rank is not None or self.mla_qk_rope_head_dim is not None:
                raise ValueError("Dense KV cache spec cannot carry MLA dimensions")

    def to_extension_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "megacache": self.megacache,
            "page_size": self.page_size,
            "tokens_per_page": self.tokens_per_page,
            "cached_token_bytes_per_layer": self.cached_token_bytes_per_layer,
            "cached_token_bytes_local": self.cached_token_bytes_local,
            "page_buffer_token_bytes": self.page_buffer_token_bytes,
            "dtype_size": self.dtype_size,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "tp_attention": {
                "tensor_parallel_size": self.tp_attention.tensor_parallel_size,
                "num_q_heads_global": self.tp_attention.num_q_heads_global,
                "num_q_heads_local": self.tp_attention.num_q_heads_local,
                "num_kv_heads_global": self.tp_attention.num_kv_heads_global,
                "num_kv_heads_local": self.tp_attention.num_kv_heads_local,
                "head_size": self.tp_attention.head_size,
            },
            "cache_components": [
                {"name": component.name, "token_dim": component.token_dim}
                for component in self.cache_components
            ],
            "mla_kv_lora_rank": self.mla_kv_lora_rank,
            "mla_qk_rope_head_dim": self.mla_qk_rope_head_dim,
        }


@dataclass(frozen=True)
class VAttentionInitSpec:
    cache_spec: VAttentionCacheSpec
    max_batch_size: int
    max_context_length: int
    device_idx: int
    dtype: torch.dtype

    def __post_init__(self):
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if self.device_idx < 0:
            raise ValueError("device_idx must be non-negative")

    def get_extension_init_mode(self) -> str:
        if self.cache_spec.architecture == CacheArchitecture.MLA:
            return "component_spec"
        return "legacy_dense_kv"

    def get_extension_init_request(self) -> Dict[str, Any]:
        mode = self.get_extension_init_mode()
        request: Dict[str, Any] = {"init_mode": mode}
        if mode == "legacy_dense_kv":
            request["legacy_args"] = self.to_legacy_init_kvcache_args()
        else:
            request["payload"] = self.to_extension_dict()
        return request

    def to_legacy_init_kvcache_args(self) -> Tuple[int, int, int, int, int, int, torch.dtype, int, bool]:
        if self.get_extension_init_mode() != "legacy_dense_kv":
            raise ValueError(
                "Legacy init_kvcache args are only valid for dense KV cache specs"
            )
        return (
            self.cache_spec.num_layers,
            self.cache_spec.num_kv_heads,
            self.cache_spec.head_size,
            self.max_batch_size,
            self.max_context_length,
            self.device_idx,
            self.dtype,
            self.cache_spec.page_size,
            self.cache_spec.megacache,
        )

    def to_extension_dict(self) -> Dict[str, Any]:
        return {
            "init_mode": self.get_extension_init_mode(),
            "cache_spec": self.cache_spec.to_extension_dict(),
            "max_batch_size": self.max_batch_size,
            "max_context_length": self.max_context_length,
            "device_idx": self.device_idx,
            "dtype": str(self.dtype).replace("torch.", ""),
        }


class SchedulerType(BaseIntEnum):
    VLLM = 1
    ORCA = 2
    FASTER_TRANSFORMER = 3
    SARATHI = 4
    SIMPLE_CHUNKING = 5


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        download_dir: Optional[str],
        load_format: str,
        dtype: str,
        seed: int,
        revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        attention_backend: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.attention_backend = attention_backend

        self.hf_config = get_config(model, trust_remote_code, revision)

        # support fschat to load model which uses dynamic ntk (e.g Qwen)
        use_dynamic_ntk = getattr(self.hf_config, "use_dynamic_ntk", None)
        if use_dynamic_ntk is not None:
            self.hf_config.max_sequence_length = 16384

        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.hf_config.dtype = self.dtype
        self.max_model_len = _get_and_verify_max_len(self.hf_config, max_model_len)
        self._verify_load_format()
        self._verify_tokenizer_mode()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in ["auto", "pt", "safetensors", "npcache", "dummy"]:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'."
            )
        self.load_format = load_format

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'."
            )
        self.tokenizer_mode = tokenizer_mode

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size})."
            )

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_total_num_q_heads(self) -> int:
        if getattr(self.hf_config, "num_attention_heads", None) is not None:
            return self.hf_config.num_attention_heads
        raise ValueError("num_attention_heads is not defined in the model config")

    def get_total_num_kv_heads(self) -> int:
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            return 1
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return self.hf_config.n_head_kv
        if getattr(self.hf_config, "num_kv_heads", None) is not None:
            return self.hf_config.num_kv_heads
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return self.hf_config.num_key_value_heads
        return self.get_total_num_q_heads()

    def is_mla_model(self) -> bool:
        return (
            getattr(self.hf_config, "kv_lora_rank", None) is not None
            and getattr(self.hf_config, "qk_rope_head_dim", None) is not None
        )

    def get_cache_architecture(self) -> CacheArchitecture:
        if self.is_mla_model():
            return CacheArchitecture.MLA
        return CacheArchitecture.DENSE_KV

    def get_mla_kv_lora_rank(self) -> int:
        kv_lora_rank = getattr(self.hf_config, "kv_lora_rank", None)
        if kv_lora_rank is None:
            raise ValueError("kv_lora_rank is not defined for this model")
        return kv_lora_rank

    def get_mla_q_lora_rank(self) -> Optional[int]:
        return getattr(self.hf_config, "q_lora_rank", None)

    def get_mla_qk_nope_head_dim(self) -> int:
        qk_nope_head_dim = getattr(self.hf_config, "qk_nope_head_dim", None)
        if qk_nope_head_dim is None:
            raise ValueError("qk_nope_head_dim is not defined for this model")
        return qk_nope_head_dim

    def get_mla_qk_rope_head_dim(self) -> int:
        qk_rope_head_dim = getattr(self.hf_config, "qk_rope_head_dim", None)
        if qk_rope_head_dim is None:
            raise ValueError("qk_rope_head_dim is not defined for this model")
        return qk_rope_head_dim

    def get_mla_v_head_dim(self) -> int:
        v_head_dim = getattr(self.hf_config, "v_head_dim", None)
        if v_head_dim is None:
            raise ValueError("v_head_dim is not defined for this model")
        return v_head_dim

    def get_mla_q_head_dim(self) -> int:
        return self.get_mla_qk_nope_head_dim() + self.get_mla_qk_rope_head_dim()

    def get_mla_resident_cache_dim(self) -> int:
        return self.get_mla_kv_lora_rank() + self.get_mla_qk_rope_head_dim()

    def get_mla_attention_spec(self) -> MLAAttentionSpec:
        if not self.is_mla_model():
            raise ValueError("MLA attention spec is only defined for MLA models")

        return MLAAttentionSpec(
            q_lora_rank=self.get_mla_q_lora_rank(),
            kv_lora_rank=self.get_mla_kv_lora_rank(),
            qk_nope_head_dim=self.get_mla_qk_nope_head_dim(),
            qk_rope_head_dim=self.get_mla_qk_rope_head_dim(),
            v_head_dim=self.get_mla_v_head_dim(),
            q_head_dim=self.get_mla_q_head_dim(),
            resident_cache_dim=self.get_mla_resident_cache_dim(),
        )

    def get_tensor_parallel_attention_spec(
        self,
        parallel_config: "ParallelConfig",
    ) -> TensorParallelAttentionSpec:
        return TensorParallelAttentionSpec(
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            num_q_heads_global=self.get_total_num_q_heads(),
            num_q_heads_local=self.get_num_q_heads(parallel_config),
            num_kv_heads_global=self.get_total_num_kv_heads(),
            num_kv_heads_local=self.get_num_kv_heads(parallel_config),
            head_size=self.get_head_size(),
        )

    def get_mla_tensor_parallel_attention_spec(
        self,
        parallel_config: "ParallelConfig",
    ) -> MLATensorParallelAttentionSpec:
        if not self.is_mla_model():
            raise ValueError("MLA tensor-parallel spec is only defined for MLA models")

        mla_spec = self.get_mla_attention_spec()
        return MLATensorParallelAttentionSpec(
            tp_attention=self.get_tensor_parallel_attention_spec(parallel_config),
            q_lora_rank=mla_spec.q_lora_rank,
            kv_lora_rank=mla_spec.kv_lora_rank,
            qk_nope_head_dim=mla_spec.qk_nope_head_dim,
            qk_rope_head_dim=mla_spec.qk_rope_head_dim,
            v_head_dim=mla_spec.v_head_dim,
            q_head_dim=mla_spec.q_head_dim,
            resident_cache_dim=mla_spec.resident_cache_dim,
        )

    def get_cache_component_specs(
        self,
        parallel_config: "ParallelConfig",
    ) -> Tuple[CacheComponentSpec, ...]:
        if self.get_cache_architecture() == CacheArchitecture.MLA:
            mla_spec = self.get_mla_attention_spec()
            return (
                CacheComponentSpec(
                    name="kv_latent",
                    token_dim=mla_spec.kv_lora_rank,
                ),
                CacheComponentSpec(
                    name="k_rope",
                    token_dim=mla_spec.qk_rope_head_dim,
                ),
            )

        dense_token_dim = self.get_num_kv_heads(parallel_config) * self.get_head_size()
        return (
            CacheComponentSpec(name="k", token_dim=dense_token_dim),
            CacheComponentSpec(name="v", token_dim=dense_token_dim),
        )

    def get_resident_cache_token_dim(
        self,
        parallel_config: "ParallelConfig",
    ) -> int:
        return sum(
            component.token_dim
            for component in self.get_cache_component_specs(parallel_config)
        )

    def get_cached_token_bytes_per_layer(
        self,
        parallel_config: "ParallelConfig",
    ) -> int:
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        return dtype_size * self.get_resident_cache_token_dim(parallel_config)

    def get_cached_token_bytes_local(
        self,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> int:
        del megacache  # Reserved for call-site clarity; resident bytes are unchanged.
        num_layers = self.get_num_layers(parallel_config)
        return num_layers * self.get_cached_token_bytes_per_layer(parallel_config)

    def get_page_buffer_token_bytes(
        self,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> int:
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()

        if self.get_cache_architecture() == CacheArchitecture.MLA:
            per_layer_bytes = self.get_cached_token_bytes_per_layer(parallel_config)
            if megacache:
                return self.get_num_layers(parallel_config) * per_layer_bytes
            return per_layer_bytes

        per_layer_per_side_bytes = (
            self.get_num_kv_heads(parallel_config) * self.get_head_size() * dtype_size
        )
        if megacache:
            return self.get_num_layers(parallel_config) * per_layer_per_side_bytes
        return per_layer_per_side_bytes

    def get_num_cached_tokens_per_page(
        self,
        page_size: int,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> int:
        return page_size // self.get_page_buffer_token_bytes(
            parallel_config,
            megacache=megacache,
        )

    def get_cache_block_size_bytes(
        self,
        block_size: int,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> int:
        return block_size * self.get_cached_token_bytes_local(
            parallel_config,
            megacache=megacache,
        )

    def get_cache_layout(
        self,
        page_size: int,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> CacheLayout:
        return CacheLayout(
            architecture=self.get_cache_architecture(),
            megacache=megacache,
            cached_token_bytes_per_layer=self.get_cached_token_bytes_per_layer(
                parallel_config
            ),
            cached_token_bytes_local=self.get_cached_token_bytes_local(
                parallel_config,
                megacache=megacache,
            ),
            page_buffer_token_bytes=self.get_page_buffer_token_bytes(
                parallel_config,
                megacache=megacache,
            ),
            tokens_per_page=self.get_num_cached_tokens_per_page(
                page_size,
                parallel_config,
                megacache=megacache,
            ),
        )

    def get_vattention_cache_spec(
        self,
        page_size: int,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> VAttentionCacheSpec:
        layout = self.get_cache_layout(
            page_size,
            parallel_config,
            megacache=megacache,
        )
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        is_mla = self.get_cache_architecture() == CacheArchitecture.MLA
        mla_spec = self.get_mla_attention_spec() if is_mla else None
        cache_components = self.get_cache_component_specs(parallel_config)
        tp_attention = self.get_tensor_parallel_attention_spec(parallel_config)
        return VAttentionCacheSpec(
            architecture=layout.architecture,
            megacache=layout.megacache,
            page_size=page_size,
            tokens_per_page=layout.tokens_per_page,
            cached_token_bytes_per_layer=layout.cached_token_bytes_per_layer,
            cached_token_bytes_local=layout.cached_token_bytes_local,
            page_buffer_token_bytes=layout.page_buffer_token_bytes,
            dtype_size=dtype_size,
            num_layers=self.get_num_layers(parallel_config),
            num_kv_heads=self.get_num_kv_heads(parallel_config),
            head_size=self.get_head_size(),
            tp_attention=tp_attention,
            cache_components=cache_components,
            mla_kv_lora_rank=mla_spec.kv_lora_rank if is_mla else None,
            mla_qk_rope_head_dim=mla_spec.qk_rope_head_dim if is_mla else None,
        )

    def get_vattention_pages_per_kvblock(
        self,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> int:
        num_components = len(self.get_cache_component_specs(parallel_config))
        if megacache:
            return num_components
        return self.get_num_layers(parallel_config) * num_components

    def get_vattention_cache_block_size_bytes(
        self,
        page_size: int,
        parallel_config: "ParallelConfig",
        megacache: bool = False,
    ) -> int:
        return (
            page_size
            * self.get_vattention_pages_per_kvblock(
                parallel_config,
                megacache=megacache,
            )
        )

    def get_vattention_init_spec(
        self,
        *,
        page_size: int,
        parallel_config: "ParallelConfig",
        megacache: bool,
        max_batch_size: int,
        max_context_length: int,
        device_idx: int,
    ) -> VAttentionInitSpec:
        return VAttentionInitSpec(
            cache_spec=self.get_vattention_cache_spec(
                page_size,
                parallel_config,
                megacache=megacache,
            ),
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            device_idx=device_idx,
            dtype=self.dtype,
        )

    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        # For GPTBigCode & Falcon:
        # Note: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            return 1
        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return self.hf_config.n_head_kv // parallel_config.tensor_parallel_size
        # For Falcon-40b/Falcon-180b:
        if getattr(self.hf_config, "num_kv_heads", None) is not None:
            return self.hf_config.num_kv_heads // parallel_config.tensor_parallel_size
        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (
                self.hf_config.num_key_value_heads
                // parallel_config.tensor_parallel_size
            )
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_q_heads(self, parallel_config: "ParallelConfig") -> int:
        if getattr(self.hf_config, "num_attention_heads", None) is not None:
            return (
                self.hf_config.num_attention_heads
                // parallel_config.tensor_parallel_size
            )
        raise ValueError("num_attention_heads is not defined in the model config")

    def get_max_model_len(self) -> int:
        return self.max_model_len

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_total_num_layers(self) -> int:
        return self.hf_config.num_hidden_layers


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            Sarathi execution.
        max_batch_size: Maximum batch size for the model.
    """

    def __init__(
        self,
        block_size: int,
        page_size: int,
        gpu_memory_utilization: float,
        max_batch_size: int,
    ) -> None:
        self.block_size = block_size
        self.page_size = page_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self._verify_args()
        self.max_batch_size = max_batch_size

        # Will be set after profiling.
        self.num_gpu_blocks = None
        self.memory_for_gpu = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}."
            )


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        replica_resource_mapping: List[Tuple[str, int]] = [],
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size

        if not replica_resource_mapping:
            replica_resource_mapping = [
                (None, i) for i in range(pipeline_parallel_size * tensor_parallel_size)
            ]

        self.replica_resource_mapping = replica_resource_mapping

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self._verify_args()

    def _verify_args(self) -> None:
        pass


class BaseSchedulerConfig(ABC):
    """BaseScheduler configuration.

    Args:
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration. Aka batch size.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.num_pipeline_stages = num_pipeline_stages

    @property
    def max_num_batched_tokens(self):
        pass

    @property
    def type(self):
        pass


class VLLMSchedulerConfig(BaseSchedulerConfig):
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
            This only takes into account number of tokens
            moving from WAITING to RUNNING states.
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
        max_num_batched_tokens: int,
    ) -> None:
        super().__init__(max_num_seqs, max_model_len, num_pipeline_stages)
        self._max_num_batched_tokens = (
            max_num_batched_tokens if max_num_batched_tokens else max_model_len
        )
        # Requests with context length upto max_model_len must be schedulable.
        assert max_model_len <= self._max_num_batched_tokens

    @property
    def max_num_batched_tokens(self):
        return self._max_num_batched_tokens

    @property
    def type(self):
        return SchedulerType.VLLM


class SimpleChunkingSchedulerConfig(BaseSchedulerConfig):

    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
        chunk_size: Optional[int],
    ) -> None:
        super().__init__(max_num_seqs, max_model_len, num_pipeline_stages)
        self.chunk_size = chunk_size

    @property
    def max_num_batched_tokens(self):
        return self.chunk_size

    @property
    def type(self):
        return SchedulerType.SIMPLE_CHUNKING


class OrcaSchedulerConfig(BaseSchedulerConfig):

    @property
    def max_num_batched_tokens(self):
        return self.max_num_seqs * self.max_model_len

    @property
    def type(self):
        return SchedulerType.ORCA


class FasterTransformerSchedulerConfig(BaseSchedulerConfig):

    @property
    def max_num_batched_tokens(self):
        return self.max_num_seqs * self.max_model_len

    @property
    def type(self):
        return SchedulerType.FASTER_TRANSFORMER


class SarathiSchedulerConfig(BaseSchedulerConfig):

    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
        chunk_size: Optional[int],
        enable_dynamic_chunking_schedule: bool,
        low_chunk_size: Optional[int],
        high_chunk_size: Optional[int],
        chunk_schedule_max_tokens: Optional[int],
        chunk_schedule_stages: Optional[int],
    ) -> None:
        super().__init__(max_num_seqs, max_model_len, num_pipeline_stages)
        self.chunk_size = chunk_size
        self.enable_dynamic_chunking_schedule = enable_dynamic_chunking_schedule
        self.low_chunk_size = low_chunk_size
        self.high_chunk_size = high_chunk_size
        self.chunk_schedule_max_tokens = chunk_schedule_max_tokens
        self.chunk_schedule_stages = chunk_schedule_stages

    @property
    def max_num_batched_tokens(self):
        # Sarathi never schedules more than chunk_size tokens in one iteration.
        if self.enable_dynamic_chunking_schedule:
            return self.high_chunk_size
        else:
            return self.chunk_size

    @property
    def type(self):
        return SchedulerType.SARATHI


class MetricsConfig:
    """Metric configuration."""

    def __init__(
        self,
        replica_id: int,
        write_metrics: bool,
        output_dir: str,
        wandb_project: str,
        wandb_group: str,
        wandb_run_name: str,
        wandb_sweep_id: str,
        wandb_run_id: str,
        enable_op_level_metrics: bool,
        enable_cpu_op_level_metrics: bool,
        enable_chrome_trace: bool,
        enable_request_outputs: bool,
        keep_individual_batch_metrics: bool,
        model_num_layers: int,
    ) -> None:
        self.replica_id = replica_id
        self.write_metrics = write_metrics
        self.output_dir = output_dir
        self.wandb_project = wandb_project
        self.wandb_sweep_id = wandb_sweep_id
        self.wandb_run_id = wandb_run_id
        self.wandb_group = wandb_group
        self.wandb_run_name = wandb_run_name
        self.enable_op_level_metrics = enable_op_level_metrics
        self.enable_cpu_op_level_metrics = enable_cpu_op_level_metrics
        self.enable_chrome_trace = enable_chrome_trace
        self.enable_request_outputs = enable_request_outputs
        self.keep_individual_batch_metrics = keep_individual_batch_metrics
        self.model_num_layers = model_num_layers

    def __str__(self) -> str:
        return (
            f"MetricsConfig(replica_id={self.replica_id}, "
            f"write_metrics={self.write_metrics}, "
            f"output_dir={self.output_dir}, "
            f"wandb_project={self.wandb_project}, "
            f"wandb_group={self.wandb_group}, "
            f"wandb_run_name={self.wandb_run_name}, "
            f"enable_op_level_metrics={self.enable_op_level_metrics}, "
            f"enable_cpu_op_level_metrics={self.enable_cpu_op_level_metrics}, "
            f"enable_chrome_trace={self.enable_chrome_trace}, "
            f"enable_request_outputs={self.enable_request_outputs}, "
            f"keep_individual_batch_metrics="
            f"{self.keep_individual_batch_metrics})"
        )


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str,
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    dtype = dtype.lower()
    if dtype == "auto":
        if config_dtype == torch.float32:
            # Following the common practice, we use float16 for float32 models.
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}."
            )
    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if derived_max_model_len == float("inf"):
            # Default to a sane value if context length keys aren't found
            derived_max_model_len = 4096 
        
        # Relaxed check: default factor to 1.0 if missing
        scaling_factor = rope_scaling.get("factor", 1.0)
        
        if rope_scaling.get("type") == "yarn":
            derived_max_model_len = rope_scaling.get("original_max_position_embeddings", derived_max_model_len)
        
        derived_max_model_len *= scaling_factor

    if max_model_len is None:
        logger.info(f"Using the derived maximum model length: {derived_max_model_len}")
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        logger.info(
            f"Applying rope_scaling to the maximum model length: "
            f"{derived_max_model_len} -> {max_model_len}"
        )
        # force rope_scaling
        scaling_factor = max_model_len / derived_max_model_len
        rope_scaling = {"type": "linear", "factor": scaling_factor}
        hf_config.rope_scaling = rope_scaling

    return max_model_len
