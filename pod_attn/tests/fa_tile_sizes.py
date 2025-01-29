import torch
import pod_attn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tile', type=int, default=0, help='0 = (64, 128), 1 = (128, 64), 2 = (32, 64), 3 = (16, 32)')
parser.add_argument('--d_bs', type=int, default=1, help='decode batch size')
parser.add_argument('--d_cl', type=int, default=4096, help='decode context length')
args = parser.parse_args()

d_cl = args.d_cl
d_bs = args.d_bs
tile = args.tile

model_configs = {
    'llama-2-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
}

num_heads = model_configs['llama-2-7b-tp1']['num_heads']
num_kv_heads = model_configs['llama-2-7b-tp1']['num_kv_heads']
head_size = model_configs['llama-2-7b-tp1']['head_size']

active_steps = 1

def calc_latency(start, end, steps):
    return round(start.elapsed_time(end) / steps, 3)

@torch.inference_mode
def do_fa_decode(q_d, k_d, v_d, k_new=None, v_new=None, seq_lens_k=None, cache_batch_idx=None, splits=0, fused_params=0):
    try:
        for _ in range(active_steps):
            output = pod_attn.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, causal=True, \
                cache_seqlens=None, cache_batch_idx=cache_batch_idx, num_splits=1, fused_params=fused_params)
        torch.cuda.synchronize()
        return output
    except Exception as e:
        print(e)
        return None

q_d = torch.randn(d_bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
k_d = torch.randn(d_bs, d_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
v_d = torch.randn(d_bs, d_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)

do_fa_decode(q_d, k_d, v_d, fused_params=tile)