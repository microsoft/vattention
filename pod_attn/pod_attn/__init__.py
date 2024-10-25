__version__ = "1.0.0"

from pod_attn.fused_attn_interface import (
    true_fused_attn_with_kvcache,
)

from pod_attn.flash_attn_interface import (
    flash_attn_with_kvcache,
)