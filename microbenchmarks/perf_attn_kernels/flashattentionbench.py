import flash_attn as fa
import torch
import utils
import math
from einops import rearrange

@torch.inference_mode
def do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim):
    try:
        q = torch.randn(bs, cl, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        for _ in range(utils.warmup_steps):
            fa.flash_attn_func(q, k, v, causal=True)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            fa.flash_attn_func(q, k, v, causal=True)
        utils.end.record()
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    try:
        num_blocks = math.ceil(cl / block_size) * bs
        q = torch.randn(bs, cl, num_heads, head_dim, device=utils.device, dtype=torch.float16)
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        block_table = rearrange(torch.arange(num_blocks, dtype=torch.int32, device='cuda'), "(b nblocks) -> b nblocks", b=bs,)
        softmax_scale = 1.0 / math.sqrt(head_dim)
        seqlens = torch.tensor([cl-1] * bs,  device=utils.device, dtype=torch.int32)
        for _ in range(utils.warmup_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
        utils.end.record()
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_decode(bs, cl, num_heads, num_kv_heads, head_dim):
    try:
        q = torch.randn(bs, 1, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        for _ in range(utils.warmup_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=False)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            o = fa.flash_attn_with_kvcache(q, k, v, causal=False)
        utils.end.record()
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    try:
        num_blocks = math.ceil(cl/block_size) * bs
        seqlens = torch.tensor([cl] * bs,  device=utils.device, dtype=torch.int32)
        q = torch.randn(bs, 1, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        block_table = rearrange(torch.arange(num_blocks, dtype=torch.int32, device='cuda'), "(b nblocks) -> b nblocks", b=bs,)
        for _ in range(utils.warmup_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=False, block_table=block_table, cache_seqlens=seqlens)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=False, block_table=block_table, cache_seqlens=seqlens)
        utils.end.record()
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return None, -1

