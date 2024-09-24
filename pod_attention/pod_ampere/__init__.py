__version__ = "2.6.1"

from pod_ampere.fused_attn_interface import (
    true_fused_attn_with_kvcache,
)

from pod_ampere.flash_attn_interface import (
    flash_attn_with_kvcache,
)