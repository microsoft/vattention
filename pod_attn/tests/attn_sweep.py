import sys
import torch
import fabench
import utils
import fibench
#import linearbench

model_configs = {
    #MHA
    #'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
    #'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128, 'num_layers': 32},
    #'llama-7b-tp4': {'num_heads': 8, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    #'llama-7b-tp8': {'num_heads': 4, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
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

chunk_sizes = [512, 1024, 2048]
print(f"model;cl;cs;kv_len;chunk_id;bs;fa_p;fa_d;fa_serial;fa_stream;" +
        f"fi_p;fi_d;fi_serial;fi_batched;HFuse;" +
        f"fa_fused;best_fused_op;speedup_fa_serial;")
for model in model_configs:
    num_heads = model_configs[model]['num_heads']
    num_kv_heads = model_configs[model]['num_kv_heads']
    head_size = model_configs[model]['head_size']
    for chunk_size in chunk_sizes:
        #context_lens = [chunk_size * i for i in range(1, 20)]
        # limit to within 24k
        #context_lens = [cl for cl in context_lens if cl <= 4*4096]
        context_lens = [4096, 8192, 12288, 16384, 20480]
        for cl in context_lens:
            if cl < chunk_size:
                continue
            nr_chunks = cl // chunk_size
            curr_chunk_sizes = [chunk_size for i in range(nr_chunks)]
            # d_batch_sizes = get_d_batch_sizes(model, cl)
            d_batch_sizes = utils.get_high_decode_batch_sizes(model, cl)
            for bs in d_batch_sizes:
                all_chunks_baseline, all_chunks_opt = 0, 0
                for chunk_idx, cs in enumerate(curr_chunk_sizes):
                    # Parameters for running tests
                    cache_seqlen = sum(curr_chunk_sizes[:chunk_idx+1])
                    q_p = torch.randn(1, cs, num_heads, head_size, device='cuda', dtype=torch.float16)
                    k_p = torch.randn(1, cache_seqlen, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
                    v_p = torch.randn(1, cache_seqlen, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
                    q_d = torch.randn(bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
                    k_d = torch.randn(bs, cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
                    v_d = torch.randn(bs, cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
                    k_new, v_new = None, None
                    cache_seqlens_p = torch.tensor([cache_seqlen], dtype=torch.int32, device='cuda')
                    cache_seqlens_d = torch.tensor([cl-1] * bs, dtype=torch.int32, device='cuda')
                    cache_batch_idx = None
                    fa_prefill, fa_p_latency = fabench.do_fa_prefill(q_p, k_p, v_p, seq_lens_k=cache_seqlens_p)
                    fa_decode, fa_d_latency = fabench.do_fa_decode(q_d, k_d, v_d, k_new=k_new, v_new=v_new, seq_lens_k=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
                    fa_stream_p, fa_stream_d, fa_stream_latency = fabench.do_fa_prefill_decode_streams(q_p, k_p, v_p, q_d, k_d, v_d, k_new=k_new, v_new=v_new, seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
                    fi_p_latency = fibench.do_flashinfer_prefill_paged(1, cs, cache_seqlen, num_heads, num_kv_heads, head_size, 16)
                    fi_d_latency = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_size, 16)
                    fi_fused_latency = fibench.do_flashinfer_fused_paged(cs, cache_seqlen, bs, cl, num_heads, num_kv_heads, head_size, 16)
                    fa_fused_latency, best_fused_op = 99999, -1
                    for fused_op in [9, 9, 11, 15]:
                        # Our shiny new FusedAttention operation
                        fu_prefill, fu_decode, fused_latency = fabench.do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_op, k_new=k_new, v_new=v_new,
                            seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
                        #print(fused_latency, end=" ")
                        if fu_prefill is not None:
                            assert torch.allclose(fa_prefill, fu_prefill, atol=1e-3), "prefill output mismatch"
                        if fu_decode is not None:
                            assert torch.allclose(fa_decode, fu_decode, atol=1e-3), "decode output mismatch"
                        fa_fused_latency = round(min(fa_fused_latency, fused_latency), 3)
                        if fa_fused_latency == fused_latency:
                            best_fused_op = fused_op
                    hfuse_latency = 0
                    for fused_op in [64]:
                        # HFuse
                        fu_prefill, fu_decode, hfuse_latency = fabench.do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_op, k_new=k_new, v_new=v_new,
                            seq_lens_k_p=cache_seqlens_p, seq_lens_k_d=cache_seqlens_d, cache_batch_idx=cache_batch_idx)
                        #print(fused_latency, end=" ")
                        if fu_prefill is not None:
                            assert torch.allclose(fa_prefill, fu_prefill, atol=1e-3), "prefill output mismatch"
                        if fu_decode is not None:
                            assert torch.allclose(fa_decode, fu_decode, atol=1e-3), "decode output mismatch"
                    #print()
                    #quit()
                    fa_serial_latency = round(fa_p_latency + fa_d_latency, 3)
                    fi_serial_latency = round(fi_p_latency + fi_d_latency, 3)
                    speedup_fa_serial = round(fa_serial_latency / fa_fused_latency, 2)
                    #speedup_fi_serial = round(fi_serial_latency / fa_fused_latency, 2)
                    #speedup_fi_fused = round(fi_fused_latency / fa_fused_latency, 2)
                    '''
                    linear_latency = linearbench.do_model_linear(model, cl + bs)
                    attn_ratio = round(fa_serial_latency / (linear_latency + fa_serial_latency), 2)
                    baseline_latency = round(fa_serial_latency + linear_latency, 2)
                    opt_latency = round(fa_fused_latency + linear_latency, 2)
                    expected_speedup = round(baseline_latency / opt_latency, 2)
                    all_chunks_baseline += baseline_latency
                    all_chunks_opt += opt_latency
                    avg_speedup_chunks = round(all_chunks_baseline / all_chunks_opt, 2)
                    '''
                    print(f"{model};{cl};{cs};{cache_seqlen};{chunk_idx};{bs};{fa_p_latency};{fa_d_latency};{fa_serial_latency};{fa_stream_latency};" +
                            f"{fi_p_latency};{fi_d_latency};{fi_serial_latency};{fi_fused_latency};{hfuse_latency};" +
                          f"{fa_fused_latency};{best_fused_op};{speedup_fa_serial};")
        print()