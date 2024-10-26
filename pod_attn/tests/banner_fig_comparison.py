import torch
import fabench
import fibench
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p_bs', type=int, default=1, help='prefill batch size')
parser.add_argument('--p_cs', type=int, default=16384, help='prefill chunk size')
parser.add_argument('--p_cl', type=int, default=16384, help='prefill context length')
parser.add_argument('--d_bs', type=int, default=250, help='decode batch size')
parser.add_argument('--d_cl', type=int, default=12288, help='decode context length')
parser.add_argument('--fused', type=int, default=9, help='fused params')
args = parser.parse_args()

p_cl = args.p_cl
p_cs = args.p_cs
p_bs = args.p_bs
d_cl = args.d_cl
d_bs = args.d_bs
fused_params=args.fused

model_configs = {
    'llama-3-8b-tp2': {'num_heads': 16, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
}

for model in model_configs:
    num_heads = model_configs[model]['num_heads']
    num_kv_heads = model_configs[model]['num_kv_heads']
    head_size = model_configs[model]['head_size']
    print("model;num_heads;num_kv_heads;head_size;p_cs;p_cl;d_bs;d_cl")
    print(f"{model};{num_heads};{num_kv_heads};{head_size};{p_cs};{p_cl};{d_bs};{d_cl}")
    # Parameters for running tests
    q_p = torch.randn(p_bs, p_cs, num_heads, head_size, device='cuda', dtype=torch.float16)
    k_p = torch.randn(p_bs, p_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
    v_p = torch.randn(p_bs, p_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
    q_d = torch.randn(d_bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
    k_d = torch.randn(d_bs, d_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
    v_d = torch.randn(d_bs, d_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
    #k_new = torch.randn(bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
    #v_new = torch.randn(bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
    k_new, v_new = None, None
    cache_seqlens_p = torch.tensor([p_cl] * p_bs, dtype=torch.int32, device='cuda')
    # leave space for the new k and v tokens
    cache_seqlens_d = torch.tensor([d_cl-1] * d_bs, dtype=torch.int32, device='cuda')
    cache_batch_idx = None#torch.randperm(bs, dtype=torch.int32, device='cuda')

    # Conventional FlashAttention-2 operations
    fa_prefill, fa_p_time = fabench.do_fa_prefill(q_p, k_p, v_p, seq_lens_k=cache_seqlens_p)
    #print(f"fa_p: {model};{num_heads};{num_kv_heads};{head_size};{p_bs};{p_cl};{fa_p_time};")
    fa_decode, fa_d_time = fabench.do_fa_decode(q_d, k_d, v_d, k_new=k_new, v_new=v_new, seq_lens_k=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
    #print(f"fa_d: {model};{num_heads};{num_kv_heads};{head_size};{d_bs};{d_cl};{fa_d_time};")

    # FlashInfer operations
    fi_prefill, fi_p_time = fibench.do_flashinfer_prefill_paged(1, p_cs, p_cl, num_heads, num_kv_heads, head_size, 16)
    #if fi_prefill is not None:
    #    assert torch.allclose(fa_prefill, fi_prefill, atol=1e-2), "fi_prefill output mismatch"
    
    fi_prefill2, fi_p2_time = fibench.do_flashinfer_prefill_ragged_input( \
        q_p.view(p_cs, num_heads, head_size), k_p.view(p_cl, num_kv_heads, head_size), v_p.view(p_cl, num_kv_heads, head_size))
    if fi_prefill2 is not None:
        assert torch.allclose(fa_prefill, fi_prefill2, atol=1e-2), "fi_prefill2 output mismatch" + str(torch.allclose(fa_prefill, fi_prefill2, atol=1e-2))

    fi_prefill3, fi_p3_time = fibench.do_flashinfer_prefill_input( \
        q_p.view(p_cs, num_heads, head_size), k_p.view(p_cl, num_kv_heads, head_size), v_p.view(p_cl, num_kv_heads, head_size))
    if fi_prefill3 is not None:
        assert torch.allclose(fa_prefill, fi_prefill3, atol=1e-2), "fi_prefill3 output mismatch" + str(torch.isclose(fa_prefill, fi_prefill3, atol=1e-2))
    #print(f"fi_p: {model};{num_heads};{num_kv_heads};{head_size};{p_bs};{p_cl};{fi_p_time};")
    fi_d_time = fibench.do_flashinfer_decode_paged(d_bs, d_cl, num_heads, num_kv_heads, head_size, 16)
    #print(f"fi_d: {model};{num_heads};{num_kv_heads};{head_size};{d_bs};{d_cl};{fi_d_time};")
    fi_fused_time = fibench.do_flashinfer_fused_paged(p_cs, p_cl, d_bs, d_cl, num_heads, num_kv_heads, head_size, 16)
    #print(f"fi_fused: {model};{num_heads};{num_kv_heads};{head_size};{d_bs};{d_cl};{fi_fused_time};")

    fa_pod = 10000
    for fused_op in [9, 11]:
        # Our shiny new FusedAttention operation
        fu_prefill, fu_decode, tot_time = fabench.do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_op, k_new=k_new, v_new=v_new,
            seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
        speedup=(fa_p_time + fa_d_time) / tot_time
        fa_pod = min(fa_pod, tot_time)
        #print(f"Fused-{fused_op}: {model};{num_heads};{num_kv_heads};{head_size};{d_bs};{p_cl};{tot_time};{speedup:.2f}")
        if fu_prefill is not None:
            assert torch.allclose(fa_prefill, fu_prefill, atol=1e-3), "prefill output mismatch"
        if fu_decode is not None:
            assert torch.allclose(fa_decode, fu_decode, atol=1e-3), "decode output mismatch"
    print(";fa_p;fa_d;fi_p;fi_d;fi_batched;pod")
    print(f";{fa_p_time:.2f};{fa_d_time:.2f};{fi_p3_time:.2f};{fi_d_time:.2f};{fi_fused_time:.2f};{fa_pod:.2f}")
    print()