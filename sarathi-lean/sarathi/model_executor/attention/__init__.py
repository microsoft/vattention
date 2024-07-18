from enum import Enum
from typing import Union

from sarathi.model_executor.attention.flash_attention_wrapper import (
    FlashAttentionWrapper,
)
from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
    FlashInferAttentionWrapper,
)
from sarathi.model_executor.attention.vattention_flashinfer_wrapper import (
    VAttentionFlashInferWrapper,
)
from sarathi.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,
)
from sarathi.model_executor.attention.vattention_flashattention_wrapper import (
    VAttentionFlashAttentionWrapper,
)
from sarathi.model_executor.attention.flashinfer_unpaged_attention_wrapper import (
    FlashinferUnpagedAttentionWrapper,
)

# FA: FLASHATTENTION
# FI: FLASHINFER
class AttentionBackend(Enum):
    FA_PAGED = "FA_PAGED"
    FI_PAGED = "FI_PAGED"
    FA_VATTN = "FA_VATTN"
    FI_VATTN = "FI_VATTN"
    FA_VATTN_SYNC = "FA_VATTN_SYNC"
    FI_VATTN_SYNC = "FI_VATTN_SYNC"
    #TODO(ashish): remove the following?
    FI_UNPAGED = "FI_UNPAGED"
    NO_OP = "NO_OP"

    def is_attn_contiguous(attn_cfg):
      
        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN.value,
            AttentionBackend.FI_VATTN.value,
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
        ]

    def is_vATTN(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN.value,
            AttentionBackend.FI_VATTN.value,
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
        ]

    def is_vATTN_SYNC(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
        ]

    def is_vLLM(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_PAGED.value,
            AttentionBackend.FI_PAGED.value,
            AttentionBackend.FI_UNPAGED.value,
        ]

ATTENTION_BACKEND = AttentionBackend.NO_OP

def get_attn_type():
    return ATTENTION_BACKEND.value  

    
def set_attention_backend(backend: Union[str, AttentionBackend]):
    if isinstance(backend, str):
        backend = backend.upper()
        if backend not in AttentionBackend.__members__:
            raise ValueError(f"Unsupported attention backend: {backend}")
        backend = AttentionBackend[backend]
    elif not isinstance(backend, AttentionBackend):
        raise ValueError(f"Unsupported attention backend: {backend}")

    global ATTENTION_BACKEND
    ATTENTION_BACKEND = backend


def get_attention_wrapper():
    if ATTENTION_BACKEND == AttentionBackend.FI_PAGED:
        return FlashInferAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_PAGED:
        return FlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.NO_OP:
        return NoOpAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_VATTN:
        return VAttentionFlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_VATTN_SYNC:
        return VAttentionFlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_VATTN:
        return VAttentionFlashInferWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_VATTN_SYNC:
        return VAttentionFlashInferWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_UNPAGED:
        return FlashinferUnpagedAttentionWrapper.get_instance()


    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")

#TODO(ashish): these functions are also defined above?
def is_vattention_backend():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_VATTN,
        AttentionBackend.FI_VATTN,
        AttentionBackend.FA_VATTN_SYNC,
        AttentionBackend.FI_VATTN_SYNC,
    ]

def is_vLLM_backend():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_PAGED,
        AttentionBackend.FI_PAGED,
        AttentionBackend.FI_UNPAGED,
    ]

def is_attn_contiguous():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_VATTN,
        AttentionBackend.FI_VATTN,
        AttentionBackend.FA_VATTN_SYNC,
        AttentionBackend.FI_VATTN_SYNC,
    ]