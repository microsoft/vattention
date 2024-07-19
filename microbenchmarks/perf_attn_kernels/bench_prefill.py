import torch
import flashinferbench as fibench
import flashattentionbench as fabench
#import vllmbench
import sys
import utils

# prefills do not batch well together. hence profiling bs=1 is enough
bs = 1
context_lens = [1024, 2048, 4096, 8192, 16384]

print("model;num_heads;num_kv_heads;head_dim;bs;cl;fa_latency;fa_paged_latency;fi_latency;fi_ragged_latency;fi_paged_latency")
for model in utils.attn_configs:
    num_heads = utils.attn_configs[model]['num_heads']
    num_kv_heads = utils.attn_configs[model]['num_kv_heads']
    head_dim = utils.attn_configs[model]['head_dim']
    fa_latency, fa_paged_latency, fi_latency, fi_ragged_latency, fi_paged_latency = -1, -1, -1, -1, -1
    for cl in context_lens:
        fa_latency = fabench.do_flashattention_prefill(bs, cl, num_heads, num_kv_heads, head_dim)
        fa_paged_latency = fabench.do_flashattention_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
        fi_latency = fibench.do_flashinfer_prefill(bs, cl, num_heads, num_kv_heads, head_dim)
        fi_ragged_latency = fibench.do_flashinfer_prefill_ragged(bs, cl, num_heads, num_kv_heads, head_dim)
        fi_paged_latency = fibench.do_flashinfer_prefill_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16)
        print(f"{model};{num_heads};{num_kv_heads};{head_dim};{bs};{cl};{fa_latency};{fa_paged_latency};{fi_latency};{fi_ragged_latency};{fi_paged_latency}")
    print()