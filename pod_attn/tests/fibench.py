import torch
import flashinfer as fi
import utils
import math

@torch.inference_mode
def do_flashinfer_prefill(bs, cl, num_heads, num_kv_heads, head_dim):
    try:
        assert bs == 1, "batch size must be 1 for flashinfer prefill"
        q = torch.randn(cl, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        k = torch.randn(cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        return do_flashinfer_prefill_input(q, k, v)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_prefill_input(q, k, v):
    try:
        #assert bs == 1, "batch size must be 1 for flashinfer prefill"
        for _ in range(utils.warmup_steps):
            fi.single_prefill_with_kv_cache(q, k, v, causal=True)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            output = fi.single_prefill_with_kv_cache(q, k, v, causal=True)
        utils.end.record()
        torch.cuda.synchronize()
        latency = utils.calc_latency(utils.start, utils.end, utils.active_steps)
        return output, latency
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_prefill_ragged(bs, cl, cache_seqlen, num_heads, num_kv_heads, head_dim):
    try:
        assert bs == 1, "batch size must be 1 for flashinfer prefill"
        q = torch.randn(bs*cl, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        k = torch.randn(bs*cache_seqlen, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        v = torch.randn(bs*cache_seqlen, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        return do_flashinfer_prefill_ragged_input(q, k, v)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_prefill_ragged_input(q, k, v):
    try:
        #assert bs == 1, "batch size must be 1 for flashinfer prefill"
        bs = 1
        assert k.size(2) == q.size(2), "head_dim must be the same for q and k"
        cl = q.size(0)
        num_heads = q.size(1)
        cache_seqlen = k.size(0)
        num_kv_heads = k.size(1)
        head_dim = k.size(2)
        qo_idx_ptr = torch.tensor([i*cl for i in range(bs+1)], dtype=torch.int32, device=utils.device)
        kv_idx_ptr = torch.tensor([i*cache_seqlen for i in range(bs+1)], dtype=torch.int32, device=utils.device)
        # allocate 16MB workspace buffer
        workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=utils.device)
        prefill_wrapper = fi.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        prefill_wrapper.begin_forward(
            qo_idx_ptr,
            kv_idx_ptr,
            num_heads,
            num_kv_heads,
            head_dim
        )
        output = prefill_wrapper.forward(q, k, v, causal=True)
        for _ in range(utils.warmup_steps):
            prefill_wrapper.forward(q, k, v, causal=True)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            prefill_wrapper.forward(q, k, v, causal=True)
        utils.end.record()
        torch.cuda.synchronize()
        return output, utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_pod(q_p, k_p, v_p, bs, cl, num_heads, num_kv_heads, head_dim, use_tensor = True, block_size = 16):
    try:
        q_d = torch.randn(bs, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=utils.device)
        pod_wrapper = fi.PODWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=use_tensor)
        num_pages_per_req = math.ceil(cl / block_size)
        max_num_pages = num_pages_per_req * bs
        kv_page_indices = torch.arange(max_num_pages).int().to(utils.device)
        kv_page_indptr = torch.arange(0, bs + 1).int().to(utils.device) * num_pages_per_req
        kv_last_page_len = torch.full((bs,), (cl  - 1) % block_size + 1, dtype=torch.int32).to(utils.device)
        kv_data = torch.randn(max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        pod_wrapper.begin_forward(
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            data_type=torch.float16,
            q_data_type=torch.float16,
        )
        for _ in range(utils.warmup_steps):
            pod_wrapper.run(q_p, k_p, v_p, q_d, kv_data, 
                causal_p = True)
        utils.start.record()
        for _ in range(utils.active_steps):
            o_p, o_d = pod_wrapper.run(q_p, k_p, v_p, q_d, kv_data, 
                causal_p = True)
        utils.end.record()
        torch.cuda.synchronize()
        latency = utils.calc_latency(utils.start, utils.end, utils.active_steps)
        return o_p, o_d, latency
    except Exception as e:
        print(e)
        return -1, -1, -1

@torch.inference_mode
def do_flashinfer_prefill_paged(bs, cl, cache_seqlen, num_heads, num_kv_heads, head_dim, block_size):
    try:
        assert block_size == 16, "block size must be 16 for flashinfer paged prefill"
        assert cl % block_size == 0, "context length must be divisible by block_size"
        assert cache_seqlen % block_size == 0, "cache_seqlen must be divisible by block_size"
        max_num_pages = (cache_seqlen // block_size) * bs
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=utils.device)
        prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        nnz_qo = cl * bs
        qo_iptr = [cl * i for i in range(bs)]
        qo_iptr.append(nnz_qo)
        qo_indptr = torch.tensor(qo_iptr, dtype=torch.int32, device=utils.device)
        paged_kv_indices = torch.arange(max_num_pages).int().to(utils.device)
        paged_kv_iptr = [(cache_seqlen // block_size) * i for i in range(bs)]
        paged_kv_iptr.append(max_num_pages)
        paged_kv_indptr = torch.tensor(
            paged_kv_iptr, dtype=torch.int32, device=utils.device
        )
        paged_kv_last_page_len= torch.tensor(
            [block_size] * bs, dtype=torch.int32, device=utils.device
        )
        kv_data = torch.randn(
                max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device
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
        q = torch.randn(cl, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        output = prefill_wrapper.forward(q, kv_data, causal=True)
        for _ in range(utils.warmup_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        utils.start.record()
        for _ in range(utils.active_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        utils.end.record()
        torch.cuda.synchronize()
        prefill_wrapper.end_forward()
        return output, utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_decode(bs, cl, num_heads, num_kv_heads, head_dim):
    try:
        q = torch.randn(bs, num_heads, head_dim, device=utils.device, dtype=utils.dtype)
        k = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        v = torch.randn(bs, cl, num_kv_heads, head_dim, device=utils.device, dtype=utils.dtype)
        for _ in range(utils.warmup_steps):
            fi.batch_decode_with_padded_kv_cache(q, k, v)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            o = fi.batch_decode_with_padded_kv_cache(q, k, v)
        utils.end.record()
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, block_size = 16):
    try:
        q = torch.randn(bs, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=utils.device)
        decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=True)
        num_pages_per_req = math.ceil(cl / block_size)
        max_num_pages = num_pages_per_req * bs
        kv_page_indices = torch.arange(max_num_pages).int().to(utils.device)
        kv_page_indptr = torch.arange(0, bs + 1).int().to(utils.device) * num_pages_per_req
        kv_last_page_len = torch.full((bs,), (cl  - 1) % block_size + 1, dtype=torch.int32).to(utils.device)
        kv_data = torch.randn(max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device)
        decode_wrapper.begin_forward(
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
        )
        for _ in range(utils.warmup_steps):
            decode_wrapper.forward(q, kv_data)
        utils.start.record()
        for _ in range(utils.active_steps):
            decode_wrapper.forward(q, kv_data)
        utils.end.record()
        torch.cuda.synchronize()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1


@torch.inference_mode
def do_flashinfer_fused_paged(chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size):
    try:
        assert block_size == 16, f"block size {block_size} must be 16 for flashinfer paged prefill"
        assert chunk_size % block_size == 0, f"chunk size {chunk_size} must be divisible by block_size {block_size}"
        assert p_cache_seqlen % block_size == 0, f"prefill cache_seqlen {p_cache_seqlen} must be divisible by block_size {block_size}"
        assert d_cache_seqlen % block_size == 0, f"decode context length {d_cache_seqlen} must be divisible by block_size {block_size}"
        max_num_pages = (p_cache_seqlen + (d_cache_seqlen * d_bs)) // block_size
        workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=utils.device)
        prefill_wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        nnz_qo = chunk_size + d_bs
        # add prefill chunk tokens first
        qo_iptr = [0, chunk_size]
        for i in range(1, d_bs):
            qo_iptr.append(chunk_size + i)
        qo_iptr.append(nnz_qo)
        qo_indptr = torch.tensor(qo_iptr, dtype=torch.int32, device=utils.device)

        paged_kv_indices = torch.arange(max_num_pages).int().to(utils.device)
        # add prefill chunk cache first
        paged_kv_iptr = [0, p_cache_seqlen // block_size]
        for _ in range(d_bs - 1):
            paged_kv_iptr.append((d_cache_seqlen // block_size) + paged_kv_iptr[-1])
            # paged_kv_iptr.append((cl // block_size) + paged_kv_iptr[-1] if len(paged_kv_iptr) > 0 else cl // block_size)
        paged_kv_iptr.append(max_num_pages)
        paged_kv_indptr = torch.tensor(
            paged_kv_iptr, dtype=torch.int32, device=utils.device
        )
        paged_kv_last_page_len= torch.tensor(
            [block_size-1] * (d_bs + 1), dtype=torch.int32, device=utils.device
        )
        kv_data = torch.randn(
                max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=utils.dtype, device=utils.device
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
        q = torch.randn(chunk_size + d_bs, num_heads, head_dim, dtype=utils.dtype, device=utils.device)
        for _ in range(utils.warmup_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        utils.launch_big_kernel()
        utils.start.record()
        for _ in range(utils.active_steps):
            prefill_wrapper.forward(q, kv_data, causal=True)
        utils.end.record()
        torch.cuda.synchronize()
        prefill_wrapper.end_forward()
        return utils.calc_latency(utils.start, utils.end, utils.active_steps)
    except Exception as e:
        print(e)
        return -1
