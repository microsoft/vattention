import torch
import fabench
import sys
p_batch_size = 1
model_configs = {
     'llama-3-8b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
}

print("model;num_heads;num_kv_heads;head_size;bs;cl;fa_p;fa_d;fu_2cta;fu_4cta;")
for model in model_configs:
    num_heads = model_configs[model]['num_heads']
    num_kv_heads = model_configs[model]['num_kv_heads']
    head_size = model_configs[model]['head_size']
    context_lens = [1024, 2048, 4096, 8192, 16384]#get_context_lens(model)
    for cl in context_lens:
        d_batch_sizes = [8, 16, 32, 64, 128] #get_d_batch_sizes(model, cl)
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
            fa_prefill, fa_p_time = fabench.do_fa_prefill(q_p, k_p, v_p, seq_lens_k=cache_seqlens_p)
            #print(f"Prefill: {model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{fa_time};")
            fa_decode, fa_d_time = fabench.do_fa_decode(q_d, k_d, v_d, k_new=k_new, v_new=v_new, seq_lens_k=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
            #print(f"Decode: {model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{fa_time};")
            #fa_prefill2, fa_decode2, fa_time = fabench.do_fa_prefill_decode(q_p, k_p, v_p, q_d, k_d, v_d, k_new=k_new, v_new=v_new,
            #    seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
            #print(f"PrefillDecode: {model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{fa_time};")
            fused_lat = {}
            #speedup = {}
            for fused_op in [9, 11]:
                # Our shiny new FusedAttention operation
                fu_prefill, fu_decode, tot_time = fabench.do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_op, k_new=k_new, v_new=v_new,
                    seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
                
                fused_lat[fused_op] = tot_time
                #speedup[fused_op] = (fa_p_time + fa_d_time) / fused_lat[fused_op]
                if fu_prefill is not None:
                    assert torch.allclose(fa_prefill, fu_prefill, atol=1e-3), "prefill output mismatch"
                if fu_decode is not None:
                    assert torch.allclose(fa_decode, fu_decode, atol=1e-3), "decode output mismatch"
            print(f"{model};{num_heads};{num_kv_heads};{head_size};{bs};{cl};{fa_p_time};{fa_d_time};{fused_lat[9]};{fused_lat[11]};")