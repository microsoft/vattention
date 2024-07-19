import torch
import flashinferbench as fibench
import flashattentionbench as fabench
#import vllmbench
import sys
import utils

context_lens = [16384]

def get_batch_sizes(model, num_heads, num_kv_heads):
    batch_sizes = []
    return [1, 2, 4, 8, 16] if num_heads == num_kv_heads else [1, 2, 4, 8, 16, 32, 64, 128, 256]

print("model;num_heads;num_kv_heads;head_dim;bs;cl;fa_latency;fa_paged_latency;fi_latency;fi_paged_latency")
for model in utils.attn_configs:
    num_heads = utils.attn_configs[model]['num_heads']
    num_kv_heads = utils.attn_configs[model]['num_kv_heads']
    head_dim = utils.attn_configs[model]['head_dim']
    batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
    fa_latency, fa_paged_latency, fi_latency, fi_paged_latency = -1, -1, -1, -1
    for bs in batch_sizes:
        for cl in context_lens:
            fa_latency = fabench.do_flashattention_decode(bs, cl, num_heads, num_kv_heads, head_dim)
            fa_paged_latency = fabench.do_flashattention_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
            fi_latency = fibench.do_flashinfer_decode(bs, cl, num_heads, num_kv_heads, head_dim)
            fi_paged_latency = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16)
            print(f"{model};{num_heads};{num_kv_heads};{head_dim};{bs};{cl};" +
                  f"{fa_latency};{fa_paged_latency};{fi_latency};{fi_paged_latency}")
    print()