import torch

dtype = torch.float16
device = 'cuda'
warmup_steps, active_steps = 5, 10

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

attn_configs = {
    #'yi-6B-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_dim': 128},
    #'yi-6B-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_dim': 128},
    #'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_dim': 128},
    #'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_dim': 128},
    #'yi-34B-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_dim': 128},
    #'yi-34B-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_dim': 128},
    #'llama-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_dim': 128},
    #'llama-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_dim': 128},
    'llama-70B-tp4': {'num_heads': 16, 'num_kv_heads': 2, 'head_dim': 128},
    #'llama-70B-tp8': {'num_heads': 8, 'num_kv_heads': 1, 'head_dim': 128},
}

def launch_big_kernel():
    m, n, k = 48000, 48000, 48000
    a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    c = torch.matmul(a, b)
    return c

def calc_latency(start, end, steps):
    return round(start.elapsed_time(end) / steps, 3)