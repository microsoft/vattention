import torch
import fabench
import sys
p_batch_size = 1
model_configs = {
    #GQA
    'llama-3-8b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
}
context_lens = [8192]
d_batch_sizes = [32, 64, 96, 128, 192]

results = {}
for bs in d_batch_sizes:
    results[bs] = {}

#print("model;num_heads;num_kv_heads;head_size;bs;cl;fu_eq;fu_prop;")
for model in model_configs:
    num_heads = model_configs[model]['num_heads']
    num_kv_heads = model_configs[model]['num_kv_heads']
    head_size = model_configs[model]['head_size']
    for cl in context_lens:
        for bs in d_batch_sizes:
            # Parameters for running tests
            q_p = torch.randn(p_batch_size, cl, num_heads, head_size, device='cuda', dtype=torch.float16)
            k_p = torch.randn(p_batch_size, cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
            v_p = torch.randn(p_batch_size, cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
            q_d = torch.randn(bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
            k_d = torch.randn(bs, cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
            v_d = torch.randn(bs, cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
            #k_new = torch.randn(bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
            #v_new = torch.randn(bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
            k_new, v_new = None, None
            cache_seqlens_p = torch.tensor([cl] * p_batch_size, dtype=torch.int32, device='cuda')
            # leave space for the new k and v tokens
            cache_seqlens_d = torch.tensor([cl-1] * bs, dtype=torch.int32, device='cuda')
            cache_batch_idx = None#torch.randperm(bs, dtype=torch.int32, device='cuda')

            # Conventional FlashAttention-2 operations
            fa_prefill, fa_time = fabench.do_fa_prefill(q_p, k_p, v_p, seq_lens_k=cache_seqlens_p)
            #print(f"Prefill: {model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{fa_time};")
            fa_decode, fa_time = fabench.do_fa_decode(q_d, k_d, v_d, k_new=k_new, v_new=v_new, seq_lens_k=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
            #print(f"Decode: {model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{fa_time};")
            #fa_prefill2, fa_decode2, fa_time = fabench.do_fa_prefill_decode(q_p, k_p, v_p, q_d, k_d, v_d, k_new=k_new, v_new=v_new,
            #    seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
            #print(f"PrefillDecode: {model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{fa_time};")

            lat = {}
            for fused_op in [8, 10, 9, 11]:
                # Our shiny new FusedAttention operation
                fu_prefill, fu_decode, tot_time = fabench.do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_op, k_new=k_new, v_new=v_new,
                    seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
                lat[fused_op] = tot_time
                #print(f"Fused-{fused_op}: {model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{tot_time};{speedup:.2f}")
                if fu_prefill is not None:
                    assert torch.allclose(fa_prefill, fu_prefill, atol=1e-3), "prefill output mismatch"
                if fu_decode is not None:
                    assert torch.allclose(fa_decode, fu_decode, atol=1e-3), "decode output mismatch"
            cfg_suf = "-llama" if model == 'llama-3-8b-tp1' else "-yi"
            
            results[bs]["equal" + cfg_suf] = min(lat[8], lat[10])
            results[bs]["proportional" + cfg_suf] = min(lat[9], lat[11])
            #print(f"{model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{min(lat[8], lat[10])};{min(lat[9], lat[11])}")

for i, bs in enumerate(d_batch_sizes):
    if(i == 0):
        print("bs", end = "\t")
        for cfg in results[bs]:
            print(f"{cfg}\t", end = "")
        print()
    print(bs, end = "\t")
    for cfg in results[bs]:
        print(f"{results[bs][cfg]:.2f}\t", end = "")
    print()
