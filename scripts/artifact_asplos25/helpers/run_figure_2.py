import torch
import flashinfer as fi
import flash_attn as fa
from einops import rearrange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

helpers = os.path.dirname(os.path.abspath(__file__))
src = os.path.dirname(helpers)
plots = os.path.join(src, "plots")

bs = 1
context_lens = [1024, 2048, 4096, 8192, 16384, 32768]

dtype, device = torch.float16, 'cuda'
warmup_steps, active_steps = 10, 100

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

attn_configs = {
    "llama-3-8b-tp1": {'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128},
}

# we use this to hide the cpu launch overhead from measurements
def launch_big_kernel():
    m, n, k = 4096, 4096, 4096
    a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    c = torch.matmul(a, b)
    return c

def calc_latency(start, end, steps):
    return round(start.elapsed_time(end) / steps, 3)

@torch.inference_mode
def do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim):
    try:
        q = torch.randn(bs, cl, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, cl, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(bs, cl, num_kv_heads, head_dim, device=device, dtype=dtype)
        for _ in range(warmup_steps):
            fa.flash_attn_func(q, k, v, causal=True)
        launch_big_kernel()
        start.record()
        for _ in range(active_steps):
            fa.flash_attn_func(q, k, v, causal=True)
        end.record()
        torch.cuda.synchronize()
        return calc_latency(start, end, active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    try:
        num_blocks = math.ceil(cl / block_size) * bs
        q = torch.randn(bs, cl, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
        block_table = rearrange(torch.arange(num_blocks, dtype=torch.int32, device='cuda'), "(b nblocks) -> b nblocks", b=bs,)
        softmax_scale = 1.0 / math.sqrt(head_dim)
        seqlens = torch.tensor([cl-1] * bs,  device=device, dtype=torch.int32)
        for _ in range(warmup_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
        launch_big_kernel()
        start.record()
        for _ in range(active_steps):
            fa.flash_attn_with_kvcache(q, k, v, causal=True, block_table=block_table, softmax_scale=softmax_scale)
        end.record()
        torch.cuda.synchronize()
        return calc_latency(start, end, active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_prefill_ragged(bs, cl, num_heads, num_kv_heads, head_dim):
    try:
        assert bs == 1, "batch size must be 1 for flashinfer prefill"
        q = torch.randn(bs*cl, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(bs*cl, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(bs*cl, num_kv_heads, head_dim, dtype=dtype, device=device)
        qo_idx_ptr = torch.tensor([i*cl for i in range(bs+1)], dtype=torch.int32, device=device)
        kv_idx_ptr = torch.tensor([i*cl for i in range(bs+1)], dtype=torch.int32, device=device)
        # allocate 16MB workspace buffer
        workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
        prefill_wrapper = fi.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        prefill_wrapper.begin_forward(
            qo_idx_ptr,
            kv_idx_ptr,
            num_heads,
            num_kv_heads,
            head_dim
        )
        for _ in range(warmup_steps):
            prefill_wrapper.forward(q, k, v, causal=True)
        launch_big_kernel()
        start.record()
        for _ in range(active_steps):
                prefill_wrapper.forward(q, k, v, causal=True)
        end.record()
        torch.cuda.synchronize()
        return calc_latency(start, end, active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size):
    try:
        assert block_size == 16, "block size must be 16 for flashinfer paged prefill"
        assert cl % block_size == 0, "context length must be divisible by block_size"
        max_num_pages = (cl // block_size) * bs
        workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
        prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        nnz_qo = cl * bs
        qo_iptr = [cl * i for i in range(bs)]
        qo_iptr.append(nnz_qo)
        qo_indptr = torch.tensor(qo_iptr, dtype=torch.int32, device=device)
        paged_kv_indices = torch.arange(max_num_pages).int().to(device)
        paged_kv_iptr = [(cl // block_size) * i for i in range(bs)]
        paged_kv_iptr.append(max_num_pages)
        paged_kv_indptr = torch.tensor(
            paged_kv_iptr, dtype=torch.int32, device=device
        )
        paged_kv_last_page_len= torch.tensor(
            [block_size] * bs, dtype=torch.int32, device=device
        )
        kv_data = torch.randn(
                max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size
        )
        q = torch.randn(cl, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(warmup_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        launch_big_kernel()
        start.record()
        for _ in range(active_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        end.record()
        torch.cuda.synchronize()
        prefill_wrapper.end_forward()
        latency = calc_latency(start, end, active_steps)
        return latency
    except Exception as e:
        print(e)
        return -1


print("model;cl;fa_latency;fa_paged_latency;fi_latency;fi_paged_latency")
latency_dict = {}
for model in attn_configs:
    num_heads = attn_configs[model]['num_heads']
    num_kv_heads = attn_configs[model]['num_kv_heads']
    head_dim = attn_configs[model]['head_dim']
    fa_latency, fa_paged_latency, fi_latency, fi_ragged_latency, fi_paged_latency = -1, -1, -1, -1, -1
    latency_dict[model] = {}
    for cl in context_lens:
        fa_latency = do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim)
        fa_paged_latency = do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
        fi_latency = do_flashinfer_prefill_ragged(bs, cl, num_heads, num_kv_heads, head_dim)
        fi_paged_latency = do_flashinfer_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16)
        print(f"{model};{cl};{fa_latency};{fa_paged_latency};{fi_latency};{fi_paged_latency}")
        latency_dict[model][cl] = {
            'fa_latency': fa_latency,
            'fa_paged_latency': fa_paged_latency,
            'fi_latency': fi_latency,
            'fi_paged_latency': fi_paged_latency
        }
    print()

df = pd.DataFrame(latency_dict['llama-3-8b-tp1']).transpose()
os.makedirs(os.path.join(src, "logs"), exist_ok=True)
df.to_csv(os.path.join(src, "logs/figure_2.csv"))


