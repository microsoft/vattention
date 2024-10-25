import torch
import pod_attn
import argparse
import fibench

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, default='prefill', help='')
parser.add_argument('--p_bs', type=int, default=1, help='prefill batch size')
parser.add_argument('--p_cl', type=int, default=1024, help='prefill context length')
parser.add_argument('--p_cs', type=int, default=1024, help='prefill chunk size')
parser.add_argument('--d_bs', type=int, default=1, help='decode batch size')
parser.add_argument('--d_cl', type=int, default=4096, help='decode context length')
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

num_heads = model_configs['llama-3-8b-tp2']['num_heads']
num_kv_heads = model_configs['llama-3-8b-tp2']['num_kv_heads']
head_size = model_configs['llama-3-8b-tp2']['head_size']

active_steps = 1

def calc_latency(start, end, steps):
    return round(start.elapsed_time(end) / steps, 3)

@torch.inference_mode
def do_fa_prefill(q, k, v, seq_lens_k=None):
    try:
        for _ in range(active_steps):
            output = pod_attn.flash_attn_with_kvcache(q, k, v, causal=True, cache_seqlens=seq_lens_k)
        torch.cuda.synchronize()
        return output
    except Exception as e:
        print(e)
        return None

@torch.inference_mode
def do_fa_decode_lean(q_d, k_d, v_d, k_new=None, v_new=None, seq_lens_k=None, cache_batch_idx=None, splits=0, fused_params=0):
    try:
        for _ in range(active_steps):
            output = pod_attn.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, causal=True, \
                cache_seqlens=None, cache_batch_idx=cache_batch_idx, num_splits=1, fused_params=3) # 3 is for lean tiles
            print(output.shape)
        torch.cuda.synchronize()
        return output
    except Exception as e:
        print(e)
        return None

@torch.inference_mode
def do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_params=9, k_new=None, v_new=None, seq_lens_k_p=None, seq_lens_k_d=None, cache_batch_idx=None):
    try:
        for _ in range(active_steps):
            output_pref, output_dec = pod_attn.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=True, \
                cache_seqlens_p=seq_lens_k_p, cache_seqlens_d=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params)
        torch.cuda.synchronize()
        return output_pref, output_dec
    except Exception as e:
        print(e)
        return None, None

@torch.inference_mode
def do_fi_prefill_paged(bs, cl, cache_seqlen, num_heads, num_kv_heads, head_dim, block_size=16):
    import flashinfer as fi
    try:
        assert block_size == 16, "block size must be 16 for flashinfer paged prefill"
        assert cl % block_size == 0, "context length must be divisible by block_size"
        assert cache_seqlen % block_size == 0, "cache_seqlen must be divisible by block_size"
        max_num_pages = (cache_seqlen // block_size) * bs
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
        prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        nnz_qo = cl * bs
        qo_iptr = [cl * i for i in range(bs)]
        qo_iptr.append(nnz_qo)
        qo_indptr = torch.tensor(qo_iptr, dtype=torch.int32, device='cuda')
        paged_kv_indices = torch.arange(max_num_pages).int().to('cuda')
        paged_kv_iptr = [(cache_seqlen // block_size) * i for i in range(bs)]
        paged_kv_iptr.append(max_num_pages)
        paged_kv_indptr = torch.tensor(
            paged_kv_iptr, dtype=torch.int32, device='cuda'
        )
        paged_kv_last_page_len= torch.tensor(
            [block_size] * bs, dtype=torch.int32, device='cuda'
        )
        kv_data = torch.randn(
                max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda'
        )
        prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size
        )
        q = torch.randn(cl, num_heads, head_dim, dtype=torch.float16, device='cuda')
        for _ in range(active_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        torch.cuda.synchronize()
        prefill_wrapper.end_forward()
        return 0
    except Exception as e:
        print(e)
        return -1
    
@torch.inference_mode
def do_fi_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size=16):
    import flashinfer as fi
    import math
    try:
        q = torch.randn(bs, num_heads, head_dim, dtype=torch.float16, device='cuda')
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device='cuda')
        decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=True)
        num_pages_per_req = math.ceil(cl / block_size)
        max_num_pages = num_pages_per_req * bs
        kv_page_indices = torch.arange(max_num_pages).int().to('cuda')
        kv_page_indptr = torch.arange(0, bs + 1).int().to('cuda') * num_pages_per_req
        kv_last_page_len = torch.full((bs,), (cl  - 1) % block_size + 1, dtype=torch.int32).to('cuda')
        kv_data = torch.randn(max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        decode_wrapper.begin_forward(
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
        )
        for _ in range(active_steps):
            decode_wrapper.forward(q, kv_data)
        torch.cuda.synchronize()
        return 0
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_fi_fused_paged(chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size=16):
    import flashinfer as fi
    try:
        assert block_size == 16, f"block size {block_size} must be 16 for flashinfer paged prefill"
        assert chunk_size % block_size == 0, f"chunk size {chunk_size} must be divisible by block_size {block_size}"
        assert p_cache_seqlen % block_size == 0, f"prefill cache_seqlen {p_cache_seqlen} must be divisible by block_size {block_size}"
        assert d_cache_seqlen % block_size == 0, f"decode context length {d_cache_seqlen} must be divisible by block_size {block_size}"
        max_num_pages = (p_cache_seqlen + (d_cache_seqlen * d_bs)) // block_size
        workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
        prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        nnz_qo = chunk_size + d_bs
        # add prefill chunk tokens first
        qo_iptr = [0, chunk_size]
        for i in range(1, d_bs):
            qo_iptr.append(chunk_size + i)
        qo_iptr.append(nnz_qo)
        qo_indptr = torch.tensor(qo_iptr, dtype=torch.int32, device='cuda')

        paged_kv_indices = torch.arange(max_num_pages).int().to('cuda')
        # add prefill chunk cache first
        paged_kv_iptr = [0, p_cache_seqlen // block_size]
        for _ in range(d_bs - 1):
            paged_kv_iptr.append((d_cache_seqlen // block_size) + paged_kv_iptr[-1])
            # paged_kv_iptr.append((cl // block_size) + paged_kv_iptr[-1] if len(paged_kv_iptr) > 0 else cl // block_size)
        paged_kv_iptr.append(max_num_pages)
        paged_kv_indptr = torch.tensor(
            paged_kv_iptr, dtype=torch.int32, device='cuda'
        )
        paged_kv_last_page_len= torch.tensor(
            [block_size-1] * (d_bs + 1), dtype=torch.int32, device='cuda'
        )
        kv_data = torch.randn(
                max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda'
        )
        # print("qo_indptr", qo_indptr)
        # print("paged_kv_indptr", paged_kv_indptr)
        # print("paged_kv_indices", paged_kv_indices)
        # print("paged_kv_last_page_len", paged_kv_last_page_len)
        # print("num_heads", num_heads)
        prefill_wrapper.begin_forward(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size
        )
        q = torch.randn(chunk_size + d_bs, num_heads, head_dim, dtype=torch.float16, device='cuda')
        for _ in range(active_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        torch.cuda.synchronize()
        prefill_wrapper.end_forward()
        return 0
    except Exception as e:
        print(e)
        return -1

q_p = torch.randn(p_bs, p_cl, num_heads, head_size, device='cuda', dtype=torch.float16)
k_p = torch.randn(p_bs, p_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
v_p = torch.randn(p_bs, p_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
q_d = torch.randn(d_bs, 1, num_heads, head_size, device='cuda', dtype=torch.float16)
k_d = torch.randn(d_bs, d_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
v_d = torch.randn(d_bs, d_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)

if args.stage == 'prefill':
    fa_prefill = do_fa_prefill(q_p, k_p, v_p)
    exit()

if args.stage == 'decode':
    fa_decode = do_fa_decode_lean(q_d, k_d, v_d)
    exit()

if args.stage == 'fused':
    q_p = torch.randn(p_bs, p_cs, num_heads, head_size, device='cuda', dtype=torch.float16)
    fa_prefill, fa_decode = do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_params=fused_params)
    exit()

if args.stage == 'fi_prefill':
    fi_p_latency = do_fi_prefill_paged(1, p_cs, p_cl, num_heads, num_kv_heads, head_size, 16)
    exit()

if args.stage == 'fi_decode':
    fi_d_latency = do_fi_decode_paged(d_bs, d_cl, num_heads, num_kv_heads, head_size, 16)
    exit()

if args.stage == 'fi_batch':
    fi_fused_latency = do_fi_fused_paged(p_cs, p_cl, d_bs, d_cl, num_heads, num_kv_heads, head_size, 16)
    exit()
