import torch
import flash_attn as fa

warmup_steps, active_steps = 1, 10
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

@torch.inference_mode
def launch_big_kernel():
    m, n, k = 48000, 48000, 48000
    a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    c = torch.matmul(a, b)
    return c

@torch.inference_mode
def do_flashattention_prefill(q, k, v):
    try:
        output = None
        fa.flash_attn_with_kvcache(q, k, v, causal=True)
        launch_big_kernel()
        start.record()
        for _ in range(active_steps):
            output = fa.flash_attn_with_kvcache(q, k, v, causal=True)
        end.record()
        torch.cuda.synchronize()
        duration = round(start.elapsed_time(end) / active_steps, 3)
        return duration
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashattention_decode(q, k_cache, v_cache):
    fa.flash_attn_with_kvcache(q, k_cache, v_cache)
    launch_big_kernel()
    start.record()
    for _ in range(active_steps):
        o = fa.flash_attn_with_kvcache(q, k_cache, v_cache)
    end.record()
    torch.cuda.synchronize()
    duration = round(start.elapsed_time(end) / active_steps, 3)
    return duration