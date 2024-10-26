#include "flash.h"
#include "static_switch.h"
// fused_api_setup needs flash.h to be included prior!
#include "fused_api_setup.h"

void run_fused_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    /*FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_fused_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_fused_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });*/
}

void run_true_fused_mha_fwd(Flash_fwd_params &params_prefill, Flash_fwd_params &params_decode, cudaStream_t stream) { 
    HEADDIM_SWITCH(params_prefill.d, [&] {
        BOOL_SWITCH(params_prefill.is_causal, Is_causal, [&] {
            BOOL_SWITCH(params_decode.num_splits > 1, DecodeSplit, [&] {
                if(params_prefill.fused_params == 15) {
                    auto dprops = at::cuda::getCurrentDeviceProperties();
                    Flash_fwd_params &pp = params_prefill;
                    Flash_fwd_params &pd = params_decode;
                    int dim_p = 128;
                    if(pp.num_splits > 1)
                        dim_p = 64;
                    // Pick the most suitable param for this scenario
                    int prefill_work = pp.b * pp.h * ((pp.seqlen_q + dim_p - 1) / dim_p) * pp.num_splits;
                    int decode_work = pd.b * pd.h * ((pd.seqlen_q + 64) / 64) * pd.num_splits;
                    if(pp.num_splits > 1 || pd.num_splits > 1) {
                        // Prefill/Decode is split. 2 CTAs per SM by default.
                        run_true_fused_mha_fwd_<cutlass::half_t, kHeadDim, Is_causal, 9, DecodeSplit>(pp, pd, stream);
                    } else if (prefill_work < dprops->multiProcessorCount * 2 && decode_work < dprops->multiProcessorCount * 3) {
                        // If decode is too small, use smaller tile sizes to smooth out quantization
                        run_true_fused_mha_fwd_<cutlass::half_t, kHeadDim, Is_causal, 11, DecodeSplit>(pp, pd, stream);
                    } else if (prefill_work < dprops->multiProcessorCount * 2 && pp.seqlen_k * 8 >= pd.seqlen_k * 5) {
                        // For high relative prefill sequence length the compute util is high, so we stick to 9.
                        // TODO: Construct a formula that encapsulates this succinctly
                        run_true_fused_mha_fwd_<cutlass::half_t, kHeadDim, Is_causal, 9, DecodeSplit>(pp, pd, stream);
                    } else if (prefill_work >= dprops->multiProcessorCount * 2 && decode_work < dprops->multiProcessorCount * 2 
                        && pp.seqlen_k * 8 >= pd.seqlen_k * 5) {
                        // If decode is too small, use smaller tile sizes to smooth out quantization
                        run_true_fused_mha_fwd_<cutlass::half_t, kHeadDim, Is_causal, 11, DecodeSplit>(pp, pd, stream);
                    } else if(decode_work >= prefill_work) {
                        run_true_fused_mha_fwd_<cutlass::half_t, kHeadDim, Is_causal, 11, DecodeSplit>(pp, pd, stream);
                    } else {
                        run_true_fused_mha_fwd_<cutlass::half_t, kHeadDim, Is_causal, 9, DecodeSplit>(pp, pd, stream);
                    }
                    return;
                }
                FUSED_SWITCH(params_prefill.fused_params, [&] {
                    run_true_fused_mha_fwd_<cutlass::half_t, kHeadDim, Is_causal, fusedOpt, DecodeSplit>(params_prefill, params_decode, stream);
                });
            });
        });
    });
}

std::vector<at::Tensor>
mha_fused_fwd_kvcache(at::Tensor &q_p,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache_p,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache_p,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                at::Tensor &q_d,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache_d,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache_d,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits,
                int ratio
                ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");
    // Do checks and stuff for prefill inputs
    // TODO: Repeat this for decode inputs
    auto &q = q_p;
    auto &kcache = kcache_p, &vcache = vcache_p;

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fused_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );
    // Ratio between prefill:decode in format 4:ratio
    params.fused_params = ratio;

    at::Tensor k, v, k_padded, v_padded;

    params.is_seqlens_k_cumulative = true;
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    params.rotary_dim = 0;

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }
    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_fused_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, /*dropout*/0.f, num_splits, dprops, opts);

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size = page_block_size;


    set_params_fused_alibi(params, alibi_slopes_, batch_size, num_heads);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    run_fused_mha_fwd(params, stream, /*force_split_kernel=*/cache_batch_idx_.has_value() || paged_KV);

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse};
}

std::vector<at::Tensor>
mha_true_fused_fwd_kvcache(at::Tensor &q_p,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache_p,            // batch_size_c x seqlen_k x num_heads_k x head_size.
                const at::Tensor &vcache_p,            // batch_size_c x seqlen_k x num_heads_k x head_size.
                c10::optional<const at::Tensor> &seqlens_k_p, // batch_size
                at::Tensor &q_d,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache_d,            // batch_size_c x seqlen_k x num_heads_k x head_size.
                const at::Tensor &vcache_d,            // batch_size_c x seqlen_k x num_heads_k x head_size.
                c10::optional<const at::Tensor> &seqlens_k_d, // batch_size
                c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache (decode only)
                c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                c10::optional<at::Tensor> &block_table_p_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &block_table_d_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits_p,
                int num_splits_d,
                uint64_t fused_params
                ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "PODAttention only supports Ampere GPUs or newer.");
    TORCH_CHECK(fused_params == 9 || fused_params == 11 || fused_params == 15 || fused_params == 64, 
        "Invalid fused_params value, need 9, 11, 15, or 64");

    at::Tensor block_table_p, block_table_d;
    const bool paged_KV_p = block_table_p_.has_value(), paged_KV_d = block_table_d_.has_value();
    if (paged_KV_p) {
        block_table_p = block_table_p_.value();
        CHECK_DEVICE(block_table_p);
        TORCH_CHECK(block_table_p.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table_p.stride(-1) == 1, "block_table must have contiguous last dimension");
    }
    if (paged_KV_d) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table_d = block_table_d_.value();
        CHECK_DEVICE(block_table_d);
        TORCH_CHECK(block_table_d.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table_d.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const int batch_size_p = q_p.sizes()[0];
    int seqlen_q_p = q_p.sizes()[1];
    const int num_heads_p = q_p.sizes()[2];
    //const int seqlen_k_p = kcache_p.size(1);
    const int num_heads_k_p = kcache_p.size(2);

    const int batch_size_d = q_d.sizes()[0];
    int seqlen_q_d = q_d.sizes()[1];
    const int seqlen_k_d = kcache_d.size(1);
    const int num_heads_k_d = kcache_d.size(2);
    const int num_heads_d = seqlen_q_d == 1 ? num_heads_k_d : q_d.sizes()[2];

    assert(q_p.sizes()[3] == q_d.sizes()[3]);
    const int head_size_og = q_p.sizes()[3];
    at::Tensor out_prefill, out_decode;
    Flash_fwd_params params_prefill, params_decode;

    const int num_m_blocks_p = (seqlen_q_p + 128 - 1) / 128;
    const int num_m_blocks_d = (seqlen_q_d + 64 - 1) / 64;
    // Generate at least 1 full wave of decode for better overlap; Used for 2 TBs per SM
    if(fused_params == 9 || 
        (fused_params == 15 && (batch_size_p * num_heads_p * num_m_blocks_p >= dprops->multiProcessorCount * 2 || 
                                batch_size_p * num_heads_p * num_m_blocks_p * 2 < 0.8f * dprops->multiProcessorCount * 2))) {
        // When prefill chunk size is at least 5/8th decode seqlen, we don't need to split. Enough compute intensity is available
        //if (num_splits_p == 0 && batch_size_p * num_heads_p * num_m_blocks_p * 2 >= 0.8f * dprops->multiProcessorCount * 2 &&
        //    batch_size_p * num_heads_p * num_m_blocks_p < dprops->multiProcessorCount * 2 &&
        //    seqlen_q_p * 8 < seqlen_k_d * 5)
        //    num_splits_p = dprops->multiProcessorCount * 2 / (batch_size_p * num_heads_p * num_m_blocks_p) + 1;
        //printf("num_splits_p: %d\n", num_splits_p);
        if (batch_size_d * num_heads_d * num_m_blocks_d >= 0.8f * dprops->multiProcessorCount * 2 &&
            batch_size_d * num_heads_d * num_m_blocks_d < dprops->multiProcessorCount * 2)
            num_splits_d = dprops->multiProcessorCount * 2 / (batch_size_d * num_heads_d * num_m_blocks_d) + 1;
    }
    // Minimize max splits to reduce memory accesses for prefills, where possible
    int max_splits_p = (!(fused_params & 8) || num_splits_p != 0) ? 128 : (2 * dprops->multiProcessorCount * 2 / (batch_size_p * num_heads_p * num_m_blocks_p * 2));// > 10 ? 128 : 10;
    
    // When we have enough decodes for 1 full wave, we do not need to split prefill
    if((fused_params & 8) && num_splits_p == 0 && batch_size_d * num_heads_d * num_m_blocks_d >= dprops->multiProcessorCount * 2)
        num_splits_p = 1;
    
    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_p, softmax_lse_accum_p, out_accum_p;
    at::Tensor softmax_lse_d, softmax_lse_accum_d, out_accum_d;
    
    at::Tensor q_padded_p, kcache_padded_p, vcache_padded_p;
    at::Tensor q_padded_d, kcache_padded_d, vcache_padded_d;
    bool seqlenq_ngroups_swapped_p = setup(q_p, kcache_p, vcache_p, q_padded_p, 
        kcache_padded_p, vcache_padded_p, out_prefill, window_size_left, 
        window_size_right, softmax_scale, num_splits_p, is_causal, seqlen_q_p, 
        softmax_lse_p, softmax_lse_accum_p, out_accum_p, block_table_p, paged_KV_p, params_prefill, max_splits_p);
    bool seqlenq_ngroups_swapped_d = setup(q_d, kcache_d, vcache_d, q_padded_d, 
        kcache_padded_d, vcache_padded_d, out_decode, window_size_left, 
        window_size_right, softmax_scale, num_splits_d, is_causal, seqlen_q_d, 
        softmax_lse_d, softmax_lse_accum_d, out_accum_d, block_table_d, paged_KV_d, params_decode);

    // These MUST be here because set_params_fprop resets params to NULL
    // Only for decode:
    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_d.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q_d <= seqlen_k_d, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        auto q_dtype = q_d.dtype();
        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size_d, seqlen_knew, num_heads_k_d, head_size_og);
        CHECK_SHAPE(v, batch_size_d, seqlen_knew, num_heads_k_d, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params_decode.seqlen_knew = seqlen_knew;
        params_decode.knew_ptr = k_padded.data_ptr();
        params_decode.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params_decode.knew_batch_stride = k_padded.stride(0);
        params_decode.vnew_batch_stride = v_padded.stride(0);
        params_decode.knew_row_stride = k_padded.stride(-3);
        params_decode.vnew_row_stride = v_padded.stride(-3);
        params_decode.knew_head_stride = k_padded.stride(-2);
        params_decode.vnew_head_stride = v_padded.stride(-2);
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params_decode.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    } else {
        params_decode.cache_batch_idx = nullptr;
    }

    if (seqlens_k_p.has_value()) {
        auto seqlens_k = seqlens_k_p.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size_p);
        params_prefill.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    params_prefill.is_seqlens_k_cumulative = !(seqlens_k_p.has_value());

    if (seqlens_k_d.has_value()) {
        auto seqlens_k = seqlens_k_d.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size_d);
        params_decode.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    params_decode.is_seqlens_k_cumulative = !(seqlens_k_d.has_value());

    params_prefill.fused_params = fused_params;
    params_decode.fused_params = fused_params;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    run_true_fused_mha_fwd(params_prefill, params_decode, stream);

    if (head_size_og % 8 != 0) {
        out_prefill = out_prefill.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        out_decode = out_decode.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache_d.copy_(kcache_padded_d.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache_d.copy_(vcache_padded_d.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }

    if (seqlenq_ngroups_swapped_p) {
        out_prefill = out_prefill.transpose(1, 2).reshape({batch_size_p, 1, num_heads_k_p * seqlen_q_p, head_size_og});
        //softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    
    if (seqlenq_ngroups_swapped_d) {
        out_decode = out_decode.transpose(1, 2).reshape({batch_size_d, 1, num_heads_k_d * seqlen_q_d, head_size_og});
        //softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out_prefill, out_decode};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FusedAttention";
    m.def("fused_fwd_kvcache", &mha_fused_fwd_kvcache, "Fused forward pass, with KV-cache");
    m.def("true_fused_fwd_kvcache", &mha_true_fused_fwd_kvcache, "Fused forward pass, with KV-cache");
}
