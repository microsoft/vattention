import torch
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
warmup_steps, active_steps = 10, 200
device, dtype = 'cuda', torch.float16
p_batch_size = 1

def launch_big_kernel():
    return
    # this leads to thermal throttling, affecting timing results.
    # better to repeat more iterations and take the average
    """
    m, n, k = 4000, 8000, 8000
    a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    c = torch.matmul(a, b)
    return c
    """

def calc_latency(start, end, steps):
    return round(start.elapsed_time(end) / steps, 3)

model_configs = {
    #MHA
    'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp4': {'num_heads': 8, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp8': {'num_heads': 4, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    # 'llama-13b-tp1': {'num_heads': 40, 'num_kv_heads': 40, 'head_size': 128, 'num_layers': 32},
    # 'llama-13b-tp2': {'num_heads': 20, 'num_kv_heads': 20, 'head_size': 128, 'num_layers': 32},
    #GQA
    'llama-3-8b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp2': {'num_heads': 16, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp4': {'num_heads': 8, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp8': {'num_heads': 4, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
    # 'llama-3-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    # 'llama-3-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp4': {'num_heads': 8, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
}

GB = (1024 * 1024 * 1024)
per_elem_size = 2

# NOTE: all calculations are per-GPU, assuming 80GB of memory
def get_per_token_kv_cache_size(model):
    num_kv_heads = model_configs[model]['num_kv_heads']
    head_size = model_configs[model]['head_size']
    num_layers = model_configs[model]['num_layers']
    return num_kv_heads * head_size * 2 * per_elem_size * num_layers

def get_model_size(model):
    return 12* GB if '6b' in model else \
            14 * GB if '7b' in model else \
            16 * GB if '8b' in model else \
            18 * GB if '9b' in model else \
            26 * GB if '13b' in model else \
            68 * GB if '34b' in model else \
            140 * GB if '70b' in model else -1
    raise ValueError(f"Unknown model {model}")

def get_tp_dim(model):
    return 1 if 'tp1' in model else \
            2 if 'tp2' in model else \
            4 if 'tp4' in model else \
            8 if 'tp8' in model else -1

def get_model_weight_footprint(model):
    model_size = get_model_size(model)
    tp_dim = get_tp_dim(model)
    if model_size == -1 or tp_dim == -1:
        raise ValueError(f"Unknown model config: {model}")

    return model_size // tp_dim

def get_available_kvcache_memory(model):
    available_memory = 80 * GB
    model_size = get_model_weight_footprint(model)
    # return 90% of available memory after model weights
    return 0.9 * (available_memory - model_size)

def calc_max_batch_size(model, cl):
    free_memory = get_available_kvcache_memory(model)
    per_reqest_memory = get_per_token_kv_cache_size(model) * cl
    return int(free_memory // per_reqest_memory)

def get_min_batch_size(model):
    return 8 if 'tp1' in model else 16

def get_high_decode_batch_sizes(model, cl):
    max_bs = calc_max_batch_size(model, cl)
    return [int(max_bs * 0.9), int(max_bs * 0.95), max_bs]

def get_context_lens(model):
    if model == 'llama-7b-tp1':
        return [2048, 4096]
    if model == 'llama-7b-tp2':
        return [2048, 4096, 8192]
    if model == 'llama-3-8b-tp1' or model == 'llama-3-8b-tp2':
        return [4096, 8192, 16384]

def get_d_batch_sizes(model, cl):
    batch_sizes = [8, 12, 16, 32, 48, 64, 80, 96, 112, 128]
    max_batch_size = calc_max_batch_size(model, cl)
    min_batch_size = get_min_batch_size(model)
    return [bs for bs in batch_sizes if bs <= max_batch_size and bs >= min_batch_size]

def get_chunk_sizes(model, cl):
    num_heads = model_configs[model]['num_heads']
    num_kv_heads = model_configs[model]['num_kv_heads']
    if num_heads == num_kv_heads:
        return [cs for cs in [1024, 2048, 4096] if cs <= cl]

    return [cs for cs in [2048, 4096, 8192] if cs <= cl]
