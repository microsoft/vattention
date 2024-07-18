import torch
import vattention
import sys
import time
import random
import argparse

import utils

KB = 1024
MB = (1024 * KB)
GB = (1024 * MB)
KVCACHE_SIZE = (48 * GB)

parser = argparse.ArgumentParser(description='Flash Attention Pagesize Benchmark')
parser.add_argument('--page_size', type=str, default='2MB', help='page size')
parser.add_argument('--phase', type=str, default='prefill', help='prefill or decode')
args = parser.parse_args()

assert args.page_size in ["64KB", "2MB"], f"Invalid page size {args.page_size}..."

num_layers, max_batch_size, max_context_len = 1, 512, 65536

prefill_batch_sizes = [1]
prefill_context_lens = [1024 * (2**i) for i in range(7)]

decode_batch_sizes = [2**i for i in range(7)]
decode_context_lens = [1024 * (2**i) for i in range(7)]

models = {
    #'yi-6B-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128},
    #'yi-6B-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128},
    #'llama-2-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128},
    #'llama-2-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128},
    #'yi-34B-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_size': 128},
    #'yi-34B-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_size': 128},
    #'llama-3-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_size': 128},
    #'llama-3-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128},
    'llama-3-70B-tp4': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128},
    'llama-3-70B-tp8': {'num_heads': 8, 'num_kv_heads': 1, 'head_size': 128},
}

def get_model_params(model):
    assert model in models, f"Model {model} not found..."
    num_heads = models[model]['num_heads']
    num_kv_heads = models[model]['num_kv_heads']
    head_size = models[model]['head_size']
    return num_heads, num_kv_heads, head_size


# we need this to initialize CUDA context for vAttention
utils.launch_big_kernel()

# enable this to see vattention logs
# vattention.set_verbose(True)
def config_kvcache(num_kv_heads, head_dim, page_size):
    use_uvm_backend = False if page_size == "2MB" else True
    kv_cache = vattention.init_kvcache(num_layers, num_kv_heads, head_dim,
                                        max_batch_size, max_context_len, 0,
                                        torch.float16, use_uvm_backend)
    vattention.reserve_physical_pages(KVCACHE_SIZE)
    return kv_cache

def cleanup_kvcache():
    vattention.cleanup()

def profile_prefill_attention(model, page_size, k_cache, v_cache):
    num_heads, num_kv_heads, head_size = get_model_params(model)
    for batch_size in prefill_batch_sizes:
        for context_len in prefill_context_lens:
            q = torch.randn(batch_size, context_len, num_heads, head_size, device='cuda', dtype=torch.float16)
            k, v = k_cache[:batch_size, :context_len], v_cache[:batch_size, :context_len]
            seqlens = [context_len for i in range(batch_size)]
            vattention.step_async(seqlens)
            fa_latency = utils.do_flashattention_prefill(q, k, v)
            print(f"{model};{args.phase};{batch_size};{context_len};{page_size};{fa_latency}")

def profile_decode_attention(model, page_size, k_cache, v_cache):
    num_heads, num_kv_heads, head_size = get_model_params(model)
    for batch_size in decode_batch_sizes:
        for context_len in decode_context_lens:
            q = torch.randn(batch_size, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
            k, v = k_cache[:batch_size, :context_len], v_cache[:batch_size, :context_len]
            seqlens = [context_len for i in range(batch_size)]
            vattention.step_async(seqlens)
            fa_latency = utils.do_flashattention_decode(q, k, v)
            print(f"{model};{args.phase};{batch_size};{context_len};{page_size};{fa_latency}")

def profile_attention():
    for model in models:
        num_heads = models[model]['num_heads']
        num_kv_heads = models[model]['num_kv_heads']
        head_size = models[model]['head_size']
        page_size = args.page_size
        kv_cache = config_kvcache(num_kv_heads, head_size, page_size)
        assert len(kv_cache) == 2 * num_layers, "kv_cache size mismatch..."
        k_cache, v_cache = kv_cache[0], kv_cache[1]
        if args.phase == 'prefill':
            profile_prefill_attention(model, page_size, k_cache, v_cache)
        if args.phase == 'decode':
            profile_decode_attention(model, page_size, k_cache, v_cache)
        cleanup_kvcache()

profile_attention()
#show_results()
