import torch
import random
import vattention

MB = (1024 * 1024)
GB = (1024 * MB)
# reserve memory for the kv cache
GPU_MEM_RESERVE = (70*GB)
PAGE_SIZE = (2 * MB)
USE_UVM = False

NUM_LAYERS=32
NUM_KV_HEADS=32
HEAD_DIM=128
MAX_BATCH_SIZE=100
MAX_CONTEXT_LEN=32768

INIT_SEQ_LEN = 1024
INCR_SEQ_LEN = 250

a = torch.randn(1024, 4096, dtype=torch.float16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

def do_matmul():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(50):
        c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end), 3)


def init_kvcache():
    kv_cache = vattention.init_kvcache(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_BATCH_SIZE, MAX_CONTEXT_LEN, 0, torch.float16, USE_UVM)
    print(f"number of virtual tensors: {len(kv_cache)}")
    vattention.reserve_physical_pages(GPU_MEM_RESERVE)
    vattention.set_verbose(False)
    #vattention.show_kvcache_config()
    return kv_cache

def access_kv_cache(kv_cache, seqlens):
    for req_id in range(MAX_BATCH_SIZE):
        if seqlens[req_id] == 0:
            continue
        for l in range(2 * NUM_LAYERS):
            kv_buff = kv_cache[l][req_id]
            kv_buff[:seqlens[req_id]].fill_(1.0)

def cleanup_kvcache_manager():
    # this will join the background thread of the memory manager
    # vattention.step_begin(seqlens)
    vattention.cleanup()

def get_new_req_seq_len():
    return random.randint(1, 1024)