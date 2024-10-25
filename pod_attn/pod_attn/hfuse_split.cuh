/*
template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits119, bool Is_causal120, bool Is_local121, bool Has_alibi122, bool Is_even_MN123, bool Is_even_K124, bool Is_softcap125, bool Split126, bool Append_KV127>
 __global__ __launch_bounds__(128, 2) void flash_fwd_kernel_flash_fwd_splitkv_kernel_fused_kernel_vfuse_lb_idx_0(const Flash_fwd_params params9, const Flash_fwd_params params128)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    static_assert(!(Is_causal2 && Is_local3));
    const int m_block10 = blockIdx.x;
    const int bidb11 = blockIdx.y;
    const int bidh12 = blockIdx.z;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_13[] __attribute__((shared));
    const int tidx14 = threadIdx_x_0;
    constexpr int kBlockM15 = Kernel_traits0::kBlockM;
    constexpr int kBlockN16 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim17 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps18 = Kernel_traits0::kNWarps;
    auto seed_offset19 = at::cuda::philox::unpack(params9.philox_args);
    flash::Dropout dropout20(std::get<0>(seed_offset19), std::get<1>(seed_offset19), params9.p_dropout_in_uint8_t, bidb11, bidh12, tidx14, params9.h);
    if (Is_dropout1 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx14 == 0) {
        params9.rng_state[0] = std::get<0>(seed_offset19);
        params9.rng_state[1] = std::get<1>(seed_offset19);
    }
    const flash::BlockInfo<!Is_even_MN5> binfo21(params9, bidb11);
    if (m_block10 * kBlockM15 >= binfo21.actual_seqlen_q)
        return;
    const int n_block_min22 = !Is_local3 ? 0 : std::max(0, (m_block10 * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN16);
    int n_block_max23 = cute::ceil_div(binfo21.actual_seqlen_k, kBlockN16);
    if (Is_causal2 || Is_local3) {
        n_block_max23 = std::min(n_block_max23, cute::ceil_div((m_block10 + 1) * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN16));
    }
    if ((Is_causal2 || Is_local3 || !Is_even_MN5) && n_block_max23 <= n_block_min22) {
        Tensor mO93 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
        Tensor gO94 = local_tile(mO93(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
        Tensor gLSE95 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
        typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O96;
        auto gmem_thr_copy_O97 = gmem_tiled_copy_O96.get_thread_slice(tidx14);
        Tensor tOgO98 = gmem_thr_copy_O97.partition_D(gO94);
        Tensor tOrO99 = make_tensor<Element>(shape(tOgO98));
        clear(tOrO99);
        Tensor cO100 = make_identity_tensor(make_shape(size<0>(gO94), size<1>(gO94)));
        Tensor tOcO101 = gmem_thr_copy_O97.partition_D(cO100);
        Tensor tOpO102 = make_tensor<bool>(make_shape(size<2>(tOgO98)));
        if (!Is_even_K6) {
            for (int k = 0; k < size(tOpO102); ++k) {
                tOpO102(k) = get<1>(tOcO101(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O96, tOrO99, tOgO98, tOcO101, tOpO102, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
        for (int m = 0; m < size<1>(tOgO98); ++m) {
            const int row103 = get<0>(tOcO101(0, m, 0));
            if (row103 < binfo21.actual_seqlen_q - m_block10 * kBlockM15 && get<1>(tOcO101(0, m, 0)) == 0) {
                gLSE95(row103) = (__builtin_inff());
            }
        }
        return;
    }
    const index_t row_offset_p24 = ((bidb11 * params9.h + bidh12) * params9.seqlen_q_rounded + m_block10 * kBlockM15) * params9.seqlen_k_rounded + (n_block_max23 - 1) * kBlockN16;
    Tensor mQ25 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ26 = local_tile(mQ25(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor mK27 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.k_row_stride, params9.k_head_stride, _1{}));
    Tensor gK28 = local_tile(mK27(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor mV29 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.v_row_stride, params9.v_head_stride, _1{}));
    Tensor gV30 = local_tile(mV29(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor gP31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.p_ptr) + row_offset_p24), Shape<Int<kBlockM15>, Int<kBlockN16>>{}, make_stride(params9.seqlen_k_rounded, _1{}));
    Tensor sQ32 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_13)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK33 = make_tensor(sQ32.data() + (Kernel_traits0::Share_Q_K_smem ? 0 : size(sQ32)), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV34 = make_tensor(sK33.data() + size(sK33), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt35 = make_tensor(sV34.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle36 = make_tensor(sV34.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV37;
    auto gmem_thr_copy_QKV38 = gmem_tiled_copy_QKV37.get_thread_slice(tidx14);
    Tensor tQgQ39 = gmem_thr_copy_QKV38.partition_S(gQ26);
    Tensor tQsQ40 = gmem_thr_copy_QKV38.partition_D(sQ32);
    Tensor tKgK41 = gmem_thr_copy_QKV38.partition_S(gK28);
    Tensor tKsK42 = gmem_thr_copy_QKV38.partition_D(sK33);
    Tensor tVgV43 = gmem_thr_copy_QKV38.partition_S(gV30);
    Tensor tVsV44 = gmem_thr_copy_QKV38.partition_D(sV34);
    typename Kernel_traits0::TiledMma tiled_mma45;
    auto thr_mma46 = tiled_mma45.get_thread_slice(tidx14);
    Tensor tSrQ47 = thr_mma46.partition_fragment_A(sQ32);
    Tensor tSrK48 = thr_mma46.partition_fragment_B(sK33);
    Tensor tOrVt49 = thr_mma46.partition_fragment_B(sVtNoSwizzle36);
    Tensor tSgS50 = thr_mma46.partition_C(gP31);
    Tensor acc_o51 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    auto smem_tiled_copy_Q52 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_Q53 = smem_tiled_copy_Q52.get_thread_slice(tidx14);
    Tensor tSsQ54 = smem_thr_copy_Q53.partition_S(sQ32);
    auto smem_tiled_copy_K55 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_K56 = smem_tiled_copy_K55.get_thread_slice(tidx14);
    Tensor tSsK57 = smem_thr_copy_K56.partition_S(sK33);
    auto smem_tiled_copy_V58 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma45);
    auto smem_thr_copy_V59 = smem_tiled_copy_V58.get_thread_slice(tidx14);
    Tensor tOsVt60 = smem_thr_copy_V59.partition_S(sVt35);
    Tensor cQ61 = make_identity_tensor(make_shape(size<0>(sQ32), size<1>(sQ32)));
    Tensor cKV62 = make_identity_tensor(make_shape(size<0>(sK33), size<1>(sK33)));
    Tensor tQcQ63 = gmem_thr_copy_QKV38.partition_S(cQ61);
    Tensor tKVcKV64 = gmem_thr_copy_QKV38.partition_S(cKV62);
    Tensor tQpQ65 = make_tensor<bool>(make_shape(size<2>(tQsQ40)));
    Tensor tKVpKV66 = make_tensor<bool>(make_shape(size<2>(tKsK42)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tQpQ65); ++k) {
            tQpQ65(k) = get<1>(tQcQ63(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV66); ++k) {
            tKVpKV66(k) = get<1>(tKVcKV64(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tQgQ39, tQsQ40, tQcQ63, tQpQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
    if (Kernel_traits0::Is_Q_in_regs) {
        cute::cp_async_fence();
    }
    if (Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view104 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view104))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view104);
        asm ("bar.sync 1,128;");
        ;
    }
    int n_block67 = n_block_max23 - 1;
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67), tKsK42, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
    cute::cp_async_fence();
    if (Kernel_traits0::Is_Q_in_regs && ! Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view105 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view105))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view105);
    }
    clear(acc_o51);
    flash::Softmax<2 * size<1>(acc_o51)> softmax68;
    const float alibi_slope69 = !Has_alibi4 || params9.alibi_slopes_ptr == nullptr ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal2, Is_local3, Has_alibi4> mask70(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope69);
    constexpr int n_masking_steps71 = (!Is_causal2 && !Is_local3) ? 1 : ((Is_even_MN5 && Is_causal2) ? cute::ceil_div(kBlockM15, kBlockN16) : cute::ceil_div(kBlockM15, kBlockN16) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps71; ++masking_step , --n_block67) {
        Tensor acc_s106 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s106);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        } else {
            flash::copy<Is_even_MN5, Is_even_K6, true>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
        }
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s106, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s106, params9.softcap);
        }
        mask70.template apply_mask<Is_causal2, Is_even_MN5>(acc_s106, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax68.template softmax_rescale_o<true, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2) : softmax68.template softmax_rescale_o<false, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2);
        Tensor rP107 = flash::convert_type<Element>(acc_s106);
        int block_row_idx108 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx109 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop111 = make_fragment_like(rP107);
            cute::copy(rP107, rP_drop111);
            dropout20.template apply_dropout<true>(rP_drop111, block_row_idx108, block_col_idx109, kNWarps18);
            cute::copy(rP_drop111, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP107, block_row_idx108, block_col_idx109, kNWarps18);
        }
        Tensor tOrP110 = make_tensor(rP107.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP107.layout()));
        flash::gemm_rs(acc_o51, tOrP110, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
        if (n_masking_steps71 > 1 && n_block67 <= n_block_min22) {
            --n_block67;
            break;
        }
    }
    for (; n_block67 >= n_block_min22; --n_block67) {
        Tensor acc_s112 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s112);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s112, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s112, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        mask70.template apply_mask<false>(acc_s112, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        softmax68.template softmax_rescale_o<false, Is_local3>(acc_s112, acc_o51, params9.scale_softmax_log2);
        Tensor rP113 = flash::convert_type<Element>(acc_s112);
        int block_row_idx114 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx115 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop117 = make_fragment_like(rP113);
            cute::copy(rP113, rP_drop117);
            dropout20.template apply_dropout<true>(rP_drop117, block_row_idx114, block_col_idx115, kNWarps18);
            cute::copy(rP_drop117, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP113, block_row_idx114, block_col_idx115, kNWarps18);
        }
        Tensor tOrP116 = make_tensor(rP113.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP113.layout()));
        flash::gemm_rs(acc_o51, tOrP116, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
    }
    Tensor lse72 = softmax68.template normalize_softmax_lse<Is_dropout1>(acc_o51, params9.scale_softmax, params9.rp_dropout);
    Tensor rO73 = flash::convert_type<Element>(acc_o51);
    Tensor sO74 = make_tensor(sQ32.data(), typename Kernel_traits0::SmemLayoutO{});
    auto smem_tiled_copy_O75 = make_tiled_copy_C(typename Kernel_traits0::SmemCopyAtomO{}, tiled_mma45);
    auto smem_thr_copy_O76 = smem_tiled_copy_O75.get_thread_slice(tidx14);
    Tensor taccOrO77 = smem_thr_copy_O76.retile_S(rO73);
    Tensor taccOsO78 = smem_thr_copy_O76.partition_D(sO74);
    if (Kernel_traits0::Share_Q_K_smem) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_O75, taccOrO77, taccOsO78);
    Tensor mO79 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
    Tensor gO80 = local_tile(mO79(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor gLSE81 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
    typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O82;
    auto gmem_thr_copy_O83 = gmem_tiled_copy_O82.get_thread_slice(tidx14);
    Tensor tOsO84 = gmem_thr_copy_O83.partition_S(sO74);
    Tensor tOgO85 = gmem_thr_copy_O83.partition_D(gO80);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrO86 = make_tensor<Element>(shape(tOgO85));
    cute::copy(gmem_tiled_copy_O82, tOsO84, tOrO86);
    Tensor caccO87 = make_identity_tensor(Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    Tensor taccOcO88 = thr_mma46.partition_C(caccO87);
    static_assert(decltype(size<0>(taccOcO88))::value == 4);
    Tensor taccOcO_row89 = logical_divide(taccOcO88, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse72) == size(taccOcO_row89))::value);
    if (get<1>(taccOcO_row89(0)) == 0) {
        for (int mi = 0; mi < size(lse72); ++mi) {
            const int row118 = get<0>(taccOcO_row89(mi));
            if (row118 < binfo21.actual_seqlen_q - m_block10 * kBlockM15) {
                gLSE81(row118) = lse72(mi);
            }
        }
    }
    Tensor cO90 = make_identity_tensor(make_shape(size<0>(sO74), size<1>(sO74)));
    Tensor tOcO91 = gmem_thr_copy_O83.partition_D(cO90);
    Tensor tOpO92 = make_tensor<bool>(make_shape(size<2>(tOgO85)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tOpO92); ++k) {
            tOpO92(k) = get<1>(tOcO91(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O82, tOrO86, tOgO85, tOcO91, tOpO92, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int m_block129 = blockIdx.x;
    const int bidb130 = Split126 ? blockIdx.z / params128.h : blockIdx.y;
    const int bidh131 = Split126 ? blockIdx.z - bidb130 * params128.h : blockIdx.z;
    const int n_split_idx132 = Split126 ? blockIdx.y : 0;
    const int num_n_splits133 = Split126 ? gridDim.y : 1;
    using Element = typename Kernel_traits119::Element;
    using ElementAccum = typename Kernel_traits119::ElementAccum;
    using index_t = typename Kernel_traits119::index_t;
    extern char smem_134[] __attribute__((shared));
    const int tidx135 = threadIdx_x_1;
    constexpr int kBlockM136 = Kernel_traits119::kBlockM;
    constexpr int kBlockN137 = Kernel_traits119::kBlockN;
    constexpr int kHeadDim138 = Kernel_traits119::kHeadDim;
    constexpr int kNWarps139 = Kernel_traits119::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::GmemTiledCopyO, typename Kernel_traits119::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split126, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN123> binfo140(params128, bidb130);
    if (m_block129 * kBlockM136 >= binfo140.actual_seqlen_q)
        return;
    const int n_blocks_per_split141 = ((params128.seqlen_k + kBlockN137 - 1) / kBlockN137 + num_n_splits133 - 1) / num_n_splits133;
    const int n_block_min142 = !Is_local121 ? n_split_idx132 * n_blocks_per_split141 : std::max(n_split_idx132 * n_blocks_per_split141, (m_block129 * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q - params128.window_size_left) / kBlockN137);
    int n_block_max143 = std::min(cute::ceil_div(binfo140.actual_seqlen_k, kBlockN137), (n_split_idx132 + 1) * n_blocks_per_split141);
    if (Is_causal120 || Is_local121) {
        n_block_max143 = std::min(n_block_max143, cute::ceil_div((m_block129 + 1) * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q + params128.window_size_right, kBlockN137));
    }
    if (n_block_min142 >= n_block_max143) {
        const index_t row_offset_o220 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
        const index_t row_offset_oaccum221 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
        const index_t row_offset_lseaccum222 = ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136;
        Tensor gOaccum223 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum221 : row_offset_o220)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
        Tensor gLSEaccum224 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum222), Shape<Int<kBlockM136>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum225;
        auto gmem_thr_copy_Oaccum226 = gmem_tiled_copy_Oaccum225.get_thread_slice(tidx135);
        Tensor tOgOaccum227 = gmem_thr_copy_Oaccum226.partition_D(gOaccum223);
        Tensor tOrOaccum228 = make_tensor<ElementO>(shape(tOgOaccum227));
        clear(tOrOaccum228);
        Tensor cO229 = make_identity_tensor(make_shape(size<0>(gOaccum223), size<1>(gOaccum223)));
        Tensor tOcO230 = gmem_thr_copy_Oaccum226.partition_D(cO229);
        Tensor tOpO231 = make_tensor<bool>(make_shape(size<2>(tOgOaccum227)));
        if (!Is_even_K124) {
            for (int k = 0; k < size(tOpO231); ++k) {
                tOpO231(k) = get<1>(tOcO230(0, 0, k)) < params128.d;
            }
        }
        flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum225, tOrOaccum228, tOgOaccum227, tOcO230, tOpO231, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
        for (int m = 0; m < size<1>(tOgOaccum227); ++m) {
            const int row232 = get<0>(tOcO230(0, m, 0));
            if (row232 < binfo140.actual_seqlen_q - m_block129 * kBlockM136 && get<1>(tOcO230(0, m, 0)) == 0) {
                gLSEaccum224(row232) = Split126 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache144 = params128.cache_batch_idx == nullptr ? bidb130 : params128.cache_batch_idx[bidb130];
    const int *block_table145 = params128.block_table == nullptr ? nullptr : params128.block_table + bidb130 * params128.block_table_batch_stride;
    const int block_table_idx146 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 / params128.page_block_size;
    const int block_table_offset147 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 - block_table_idx146 * params128.page_block_size;
    const index_t row_offset_k148 = block_table145 == nullptr ? binfo140.k_offset(params128.k_batch_stride, params128.k_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride : block_table145[block_table_idx146] * params128.k_batch_stride + block_table_offset147 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride;
    const index_t row_offset_v149 = block_table145 == nullptr ? binfo140.k_offset(params128.v_batch_stride, params128.v_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride : block_table145[block_table_idx146] * params128.v_batch_stride + block_table_offset147 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride;
    Tensor mQ150 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.q_ptr) + binfo140.q_offset(params128.q_batch_stride, params128.q_row_stride, bidb130)), make_shape(binfo140.actual_seqlen_q, params128.h, params128.d), make_stride(params128.q_row_stride, params128.q_head_stride, _1{}));
    Tensor gQ151 = local_tile(mQ150(_, bidh131, _), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_coord(m_block129, 0));
    Tensor gK152 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.k_ptr) + row_offset_k148), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.k_row_stride, _1{}));
    Tensor gV153 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.v_ptr) + row_offset_v149), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.v_row_stride, _1{}));
    Tensor sQ154 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_134)), typename Kernel_traits119::SmemLayoutQ{});
    Tensor sK155 = make_tensor(sQ154.data() + size(sQ154), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sV156 = make_tensor(sK155.data() + size(sK155), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sVt157 = make_tensor(sV156.data(), typename Kernel_traits119::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle158 = make_tensor(sV156.data().get(), typename Kernel_traits119::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits119::GmemTiledCopyQKV gmem_tiled_copy_QKV159;
    auto gmem_thr_copy_QKV160 = gmem_tiled_copy_QKV159.get_thread_slice(tidx135);
    Tensor tQgQ161 = gmem_thr_copy_QKV160.partition_S(gQ151);
    Tensor tQsQ162 = gmem_thr_copy_QKV160.partition_D(sQ154);
    Tensor tKgK163 = gmem_thr_copy_QKV160.partition_S(gK152);
    Tensor tKsK164 = gmem_thr_copy_QKV160.partition_D(sK155);
    Tensor tVgV165 = gmem_thr_copy_QKV160.partition_S(gV153);
    Tensor tVsV166 = gmem_thr_copy_QKV160.partition_D(sV156);
    typename Kernel_traits119::TiledMma tiled_mma167;
    auto thr_mma168 = tiled_mma167.get_thread_slice(tidx135);
    Tensor tSrQ169 = thr_mma168.partition_fragment_A(sQ154);
    Tensor tSrK170 = thr_mma168.partition_fragment_B(sK155);
    Tensor tOrVt171 = thr_mma168.partition_fragment_B(sVtNoSwizzle158);
    Tensor acc_o172 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    auto smem_tiled_copy_Q173 = make_tiled_copy_A(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_Q174 = smem_tiled_copy_Q173.get_thread_slice(tidx135);
    Tensor tSsQ175 = smem_thr_copy_Q174.partition_S(sQ154);
    auto smem_tiled_copy_K176 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_K177 = smem_tiled_copy_K176.get_thread_slice(tidx135);
    Tensor tSsK178 = smem_thr_copy_K177.partition_S(sK155);
    auto smem_tiled_copy_V179 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtomTransposed{}, tiled_mma167);
    auto smem_thr_copy_V180 = smem_tiled_copy_V179.get_thread_slice(tidx135);
    Tensor tOsVt181 = smem_thr_copy_V180.partition_S(sVt157);
    Tensor cQ182 = make_identity_tensor(make_shape(size<0>(sQ154), size<1>(sQ154)));
    Tensor cKV183 = make_identity_tensor(make_shape(size<0>(sK155), size<1>(sK155)));
    Tensor tQcQ184 = gmem_thr_copy_QKV160.partition_S(cQ182);
    Tensor tKVcKV185 = gmem_thr_copy_QKV160.partition_S(cKV183);
    Tensor tQpQ186 = make_tensor<bool>(make_shape(size<2>(tQsQ162)));
    Tensor tKVpKV187 = make_tensor<bool>(make_shape(size<2>(tKsK164)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tQpQ186); ++k) {
            tQpQ186(k) = get<1>(tQcQ184(0, 0, k)) < params128.d;
        }
        for (int k = 0; k < size(tKVpKV187); ++k) {
            tKVpKV187(k) = get<1>(tKVcKV185(0, 0, k)) < params128.d;
        }
    }
    typename Kernel_traits119::GmemTiledCopyRotcossin gmem_tiled_copy_rotary188;
    auto gmem_thr_copy_rotary189 = gmem_tiled_copy_rotary188.get_thread_slice(tidx135);
    typename Kernel_traits119::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont190;
    auto gmem_thr_copy_rotary_cont191 = gmem_tiled_copy_rotary_cont190.get_thread_slice(tidx135);
    if (Append_KV127) {
        const index_t row_offset_cossin233 = ((n_block_max143 - 1) * kBlockN137 + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130])) * (params128.rotary_dim / 2);
        Tensor gCos234 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSin235 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gCosCont236 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSinCont237 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor tRgCos238 = gmem_thr_copy_rotary189.partition_S(gCos234);
        Tensor tRgSin239 = gmem_thr_copy_rotary189.partition_S(gSin235);
        Tensor tRgCosCont240 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont236);
        Tensor tRgSinCont241 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont237);
        const index_t row_offset_knew242 = bidb130 * params128.knew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.knew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.knew_head_stride;
        const index_t row_offset_vnew243 = bidb130 * params128.vnew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.vnew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.vnew_head_stride;
        Tensor gKnew244 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.knew_ptr) + row_offset_knew242 - binfo140.seqlen_k_cache * params128.knew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.knew_row_stride, _1{}));
        Tensor gVnew245 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.vnew_ptr) + row_offset_vnew243 - binfo140.seqlen_k_cache * params128.vnew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.vnew_row_stride, _1{}));
        Tensor tKgKnew246 = gmem_thr_copy_QKV160.partition_S(gKnew244);
        Tensor tVgVnew247 = gmem_thr_copy_QKV160.partition_S(gVnew245);
        const int n_block_copy_min248 = std::max(n_block_min142, binfo140.seqlen_k_cache / kBlockN137);
        auto tKgK_data249 = tKgK163.data();
        auto tVgV_data250 = tVgV165.data();
        for (int n_block = n_block_max143 - 1; n_block >= n_block_copy_min248; n_block--) {
            flash::copy_w_min_idx<Is_even_K124>(tVgVnew247, tVgV165, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            tVgVnew247.data() = tVgVnew247.data() + (-int(kBlockN137 * params128.vnew_row_stride));
            if (params128.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K124>(tKgKnew246, tKgK163, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            } else {
                if (params128.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCos238, tRgSin239, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCos238.data() = tRgCos238.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSin239.data() = tRgSin239.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCosCont240, tRgSinCont241, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCosCont240.data() = tRgCosCont240.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSinCont241.data() = tRgSinCont241.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                }
            }
            tKgKnew246.data() = tKgKnew246.data() + (-int(kBlockN137 * params128.knew_row_stride));
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                if (n_block > n_block_copy_min248) {
                    const int block_table_idx_cur251 = n_block * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_cur252 = n_block * kBlockN137 - block_table_idx_cur251 * params128.page_block_size;
                    const int block_table_idx_next253 = (n_block - 1) * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_next254 = (n_block - 1) * kBlockN137 - block_table_idx_next253 * params128.page_block_size;
                    const int table_diff255 = block_table145[block_table_idx_next253] - block_table145[block_table_idx_cur251];
                    const int offset_diff256 = block_table_offset_next254 - block_table_offset_cur252;
                    tVgV165.data() = tVgV165.data() + table_diff255 * params128.v_batch_stride + offset_diff256 * params128.v_row_stride;
                    tKgK163.data() = tKgK163.data() + table_diff255 * params128.k_batch_stride + offset_diff256 * params128.k_row_stride;
                }
            }
        }
        asm ("bar.sync 2,128;");
        ;
        tKgK163.data() = tKgK_data249;
        tVgV165.data() = tVgV_data250;
    }
    if (!Append_KV127 || params128.rotary_dim == 0) {
        flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tQgQ161, tQsQ162, tQcQ184, tQpQ186, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
    } else {
        const index_t row_offset_cossin257 = (binfo140.seqlen_k_cache + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130]) + (Is_causal120 || Is_local121 ? m_block129 * kBlockM136 : 0)) * (params128.rotary_dim / 2);
        Tensor gCos258 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSin259 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont260 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont261 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos262 = gmem_thr_copy_rotary189.partition_S(gCos258);
        Tensor tRgSin263 = gmem_thr_copy_rotary189.partition_S(gSin259);
        Tensor tRgCosCont264 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont260);
        Tensor tRgSinCont265 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont261);
        if (params128.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K124>(tQgQ161, tQsQ162, tRgCos262, tRgSin263, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K124>(tQgQ161, tQsQ162, tRgCosCont264, tRgSinCont265, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        }
    }
    int n_block192 = n_block_max143 - 1;
    flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
    cute::cp_async_fence();
    clear(acc_o172);
    flash::Softmax<2 * size<1>(acc_o172)> softmax193;
    const float alibi_slope194 = !Has_alibi122 ? 0.F : reinterpret_cast<float *>(params128.alibi_slopes_ptr)[bidb130 * params128.alibi_slopes_batch_stride + bidh131] / params128.scale_softmax;
    flash::Mask<Is_causal120, Is_local121, Has_alibi122> mask195(binfo140.actual_seqlen_k, binfo140.actual_seqlen_q, params128.window_size_left, params128.window_size_right, alibi_slope194);
    constexpr int n_masking_steps196 = (!Is_causal120 && !Is_local121) ? 1 : ((Is_even_MN123 && Is_causal120) ? cute::ceil_div(kBlockM136, kBlockN137) : cute::ceil_div(kBlockM136, kBlockN137) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps196; ++masking_step , --n_block192) {
        Tensor acc_s266 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s266);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (masking_step > 0) {
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
            } else {
                const int block_table_idx_cur269 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur270 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur269 * params128.page_block_size;
                const int block_table_idx_next271 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next272 = n_block192 * kBlockN137 - block_table_idx_next271 * params128.page_block_size;
                tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next271] - block_table145[block_table_idx_cur269]) * params128.v_batch_stride + (block_table_offset_next272 - block_table_offset_cur270) * params128.v_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        } else {
            flash::copy<Is_even_MN123, Is_even_K124, true>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s266, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s266, params128.softcap);
        }
        mask195.template apply_mask<Is_causal120, Is_even_MN123>(acc_s266, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur273 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur274 = n_block192 * kBlockN137 - block_table_idx_cur273 * params128.page_block_size;
                const int block_table_idx_next275 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next276 = (n_block192 - 1) * kBlockN137 - block_table_idx_next275 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next275] - block_table145[block_table_idx_cur273]) * params128.k_batch_stride + (block_table_offset_next276 - block_table_offset_cur274) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax193.template softmax_rescale_o<true, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2) : softmax193.template softmax_rescale_o<false, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2);
        Tensor rP267 = flash::convert_type<Element>(acc_s266);
        Tensor tOrP268 = make_tensor(rP267.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP267.layout()));
        flash::gemm_rs(acc_o172, tOrP268, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
        if (n_masking_steps196 > 1 && n_block192 <= n_block_min142) {
            --n_block192;
            break;
        }
    }
    for (; n_block192 >= n_block_min142; --n_block192) {
        Tensor acc_s277 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s277);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (block_table145 == nullptr) {
            tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
        } else {
            const int block_table_idx_cur280 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
            const int block_table_offset_cur281 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur280 * params128.page_block_size;
            const int block_table_idx_next282 = n_block192 * kBlockN137 / params128.page_block_size;
            const int block_table_offset_next283 = n_block192 * kBlockN137 - block_table_idx_next282 * params128.page_block_size;
            tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next282] - block_table145[block_table_idx_cur280]) * params128.v_batch_stride + (block_table_offset_next283 - block_table_offset_cur281) * params128.v_row_stride;
        }
        flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        cute::cp_async_fence();
        flash::gemm(acc_s277, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s277, params128.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur284 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur285 = n_block192 * kBlockN137 - block_table_idx_cur284 * params128.page_block_size;
                const int block_table_idx_next286 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next287 = (n_block192 - 1) * kBlockN137 - block_table_idx_next286 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next286] - block_table145[block_table_idx_cur284]) * params128.k_batch_stride + (block_table_offset_next287 - block_table_offset_cur285) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        mask195.template apply_mask<false>(acc_s277, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        softmax193.template softmax_rescale_o<false, Is_local121>(acc_s277, acc_o172, params128.scale_softmax_log2);
        Tensor rP278 = flash::convert_type<Element>(acc_s277);
        Tensor tOrP279 = make_tensor(rP278.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP278.layout()));
        flash::gemm_rs(acc_o172, tOrP279, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
    }
    Tensor lse197 = softmax193.template normalize_softmax_lse<false, Split126>(acc_o172, params128.scale_softmax);
    Tensor sOaccum198 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_134)), typename Kernel_traits119::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::SmemCopyAtomO, typename Kernel_traits119::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum199 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma167);
    auto smem_thr_copy_Oaccum200 = smem_tiled_copy_Oaccum199.get_thread_slice(tidx135);
    Tensor rO201 = flash::convert_type<ElementO>(acc_o172);
    Tensor taccOrOaccum202 = smem_thr_copy_Oaccum200.retile_S(rO201);
    Tensor taccOsOaccum203 = smem_thr_copy_Oaccum200.partition_D(sOaccum198);
    if (Split126) {
        asm ("bar.sync 2,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum199, taccOrOaccum202, taccOsOaccum203);
    const index_t row_offset_o204 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
    const index_t row_offset_oaccum205 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
    const index_t row_offset_lseaccum206 = (Split126 || !params128.unpadded_lse ? ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q : bidh131 * params128.total_q + binfo140.q_offset(params128.seqlen_q, 1, bidb130)) + m_block129 * kBlockM136;
    Tensor gOaccum207 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum205 : row_offset_o204)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
    Tensor gLSEaccum208 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum206), Shape<Int<kBlockM136>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum209;
    auto gmem_thr_copy_Oaccum210 = gmem_tiled_copy_Oaccum209.get_thread_slice(tidx135);
    Tensor tOsOaccum211 = gmem_thr_copy_Oaccum210.partition_S(sOaccum198);
    Tensor tOgOaccum212 = gmem_thr_copy_Oaccum210.partition_D(gOaccum207);
    asm ("bar.sync 2,128;");
    ;
    Tensor tOrOaccum213 = make_tensor<ElementO>(shape(tOgOaccum212));
    cute::copy(gmem_tiled_copy_Oaccum209, tOsOaccum211, tOrOaccum213);
    Tensor caccO214 = make_identity_tensor(Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    Tensor taccOcO215 = thr_mma168.partition_C(caccO214);
    static_assert(decltype(size<0>(taccOcO215))::value == 4);
    Tensor taccOcO_row216 = logical_divide(taccOcO215, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse197) == size(taccOcO_row216))::value);
    if (get<1>(taccOcO_row216(0)) == 0) {
        for (int mi = 0; mi < size(lse197); ++mi) {
            const int row288 = get<0>(taccOcO_row216(mi));
            if (row288 < binfo140.actual_seqlen_q - m_block129 * kBlockM136) {
                gLSEaccum208(row288) = lse197(mi);
            }
        }
    }
    Tensor cO217 = make_identity_tensor(make_shape(size<0>(sOaccum198), size<1>(sOaccum198)));
    Tensor tOcO218 = gmem_thr_copy_Oaccum210.partition_D(cO217);
    Tensor tOpO219 = make_tensor<bool>(make_shape(size<2>(tOgOaccum212)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tOpO219); ++k) {
            tOpO219(k) = get<1>(tOcO218(0, 0, k)) < params128.d;
        }
    }
    flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum209, tOrOaccum213, tOgOaccum212, tOcO218, tOpO219, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
}
}
template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits119, bool Is_causal120, bool Is_local121, bool Has_alibi122, bool Is_even_MN123, bool Is_even_K124, bool Is_softcap125, bool Split126, bool Append_KV127>
 __global__ __launch_bounds__(128, 0) void flash_fwd_kernel_flash_fwd_splitkv_kernel_fused_kernel_vfuse_idx_0(const Flash_fwd_params params9, const Flash_fwd_params params128)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    static_assert(!(Is_causal2 && Is_local3));
    const int m_block10 = blockIdx.x;
    const int bidb11 = blockIdx.y;
    const int bidh12 = blockIdx.z;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_13[] __attribute__((shared));
    const int tidx14 = threadIdx_x_0;
    constexpr int kBlockM15 = Kernel_traits0::kBlockM;
    constexpr int kBlockN16 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim17 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps18 = Kernel_traits0::kNWarps;
    auto seed_offset19 = at::cuda::philox::unpack(params9.philox_args);
    flash::Dropout dropout20(std::get<0>(seed_offset19), std::get<1>(seed_offset19), params9.p_dropout_in_uint8_t, bidb11, bidh12, tidx14, params9.h);
    if (Is_dropout1 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx14 == 0) {
        params9.rng_state[0] = std::get<0>(seed_offset19);
        params9.rng_state[1] = std::get<1>(seed_offset19);
    }
    const flash::BlockInfo<!Is_even_MN5> binfo21(params9, bidb11);
    if (m_block10 * kBlockM15 >= binfo21.actual_seqlen_q)
        return;
    const int n_block_min22 = !Is_local3 ? 0 : std::max(0, (m_block10 * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN16);
    int n_block_max23 = cute::ceil_div(binfo21.actual_seqlen_k, kBlockN16);
    if (Is_causal2 || Is_local3) {
        n_block_max23 = std::min(n_block_max23, cute::ceil_div((m_block10 + 1) * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN16));
    }
    if ((Is_causal2 || Is_local3 || !Is_even_MN5) && n_block_max23 <= n_block_min22) {
        Tensor mO93 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
        Tensor gO94 = local_tile(mO93(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
        Tensor gLSE95 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
        typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O96;
        auto gmem_thr_copy_O97 = gmem_tiled_copy_O96.get_thread_slice(tidx14);
        Tensor tOgO98 = gmem_thr_copy_O97.partition_D(gO94);
        Tensor tOrO99 = make_tensor<Element>(shape(tOgO98));
        clear(tOrO99);
        Tensor cO100 = make_identity_tensor(make_shape(size<0>(gO94), size<1>(gO94)));
        Tensor tOcO101 = gmem_thr_copy_O97.partition_D(cO100);
        Tensor tOpO102 = make_tensor<bool>(make_shape(size<2>(tOgO98)));
        if (!Is_even_K6) {
            for (int k = 0; k < size(tOpO102); ++k) {
                tOpO102(k) = get<1>(tOcO101(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O96, tOrO99, tOgO98, tOcO101, tOpO102, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
        for (int m = 0; m < size<1>(tOgO98); ++m) {
            const int row103 = get<0>(tOcO101(0, m, 0));
            if (row103 < binfo21.actual_seqlen_q - m_block10 * kBlockM15 && get<1>(tOcO101(0, m, 0)) == 0) {
                gLSE95(row103) = (__builtin_inff());
            }
        }
        return;
    }
    const index_t row_offset_p24 = ((bidb11 * params9.h + bidh12) * params9.seqlen_q_rounded + m_block10 * kBlockM15) * params9.seqlen_k_rounded + (n_block_max23 - 1) * kBlockN16;
    Tensor mQ25 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ26 = local_tile(mQ25(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor mK27 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.k_row_stride, params9.k_head_stride, _1{}));
    Tensor gK28 = local_tile(mK27(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor mV29 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.v_row_stride, params9.v_head_stride, _1{}));
    Tensor gV30 = local_tile(mV29(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor gP31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.p_ptr) + row_offset_p24), Shape<Int<kBlockM15>, Int<kBlockN16>>{}, make_stride(params9.seqlen_k_rounded, _1{}));
    Tensor sQ32 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_13)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK33 = make_tensor(sQ32.data() + (Kernel_traits0::Share_Q_K_smem ? 0 : size(sQ32)), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV34 = make_tensor(sK33.data() + size(sK33), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt35 = make_tensor(sV34.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle36 = make_tensor(sV34.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV37;
    auto gmem_thr_copy_QKV38 = gmem_tiled_copy_QKV37.get_thread_slice(tidx14);
    Tensor tQgQ39 = gmem_thr_copy_QKV38.partition_S(gQ26);
    Tensor tQsQ40 = gmem_thr_copy_QKV38.partition_D(sQ32);
    Tensor tKgK41 = gmem_thr_copy_QKV38.partition_S(gK28);
    Tensor tKsK42 = gmem_thr_copy_QKV38.partition_D(sK33);
    Tensor tVgV43 = gmem_thr_copy_QKV38.partition_S(gV30);
    Tensor tVsV44 = gmem_thr_copy_QKV38.partition_D(sV34);
    typename Kernel_traits0::TiledMma tiled_mma45;
    auto thr_mma46 = tiled_mma45.get_thread_slice(tidx14);
    Tensor tSrQ47 = thr_mma46.partition_fragment_A(sQ32);
    Tensor tSrK48 = thr_mma46.partition_fragment_B(sK33);
    Tensor tOrVt49 = thr_mma46.partition_fragment_B(sVtNoSwizzle36);
    Tensor tSgS50 = thr_mma46.partition_C(gP31);
    Tensor acc_o51 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    auto smem_tiled_copy_Q52 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_Q53 = smem_tiled_copy_Q52.get_thread_slice(tidx14);
    Tensor tSsQ54 = smem_thr_copy_Q53.partition_S(sQ32);
    auto smem_tiled_copy_K55 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_K56 = smem_tiled_copy_K55.get_thread_slice(tidx14);
    Tensor tSsK57 = smem_thr_copy_K56.partition_S(sK33);
    auto smem_tiled_copy_V58 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma45);
    auto smem_thr_copy_V59 = smem_tiled_copy_V58.get_thread_slice(tidx14);
    Tensor tOsVt60 = smem_thr_copy_V59.partition_S(sVt35);
    Tensor cQ61 = make_identity_tensor(make_shape(size<0>(sQ32), size<1>(sQ32)));
    Tensor cKV62 = make_identity_tensor(make_shape(size<0>(sK33), size<1>(sK33)));
    Tensor tQcQ63 = gmem_thr_copy_QKV38.partition_S(cQ61);
    Tensor tKVcKV64 = gmem_thr_copy_QKV38.partition_S(cKV62);
    Tensor tQpQ65 = make_tensor<bool>(make_shape(size<2>(tQsQ40)));
    Tensor tKVpKV66 = make_tensor<bool>(make_shape(size<2>(tKsK42)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tQpQ65); ++k) {
            tQpQ65(k) = get<1>(tQcQ63(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV66); ++k) {
            tKVpKV66(k) = get<1>(tKVcKV64(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tQgQ39, tQsQ40, tQcQ63, tQpQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
    if (Kernel_traits0::Is_Q_in_regs) {
        cute::cp_async_fence();
    }
    if (Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view104 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view104))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view104);
        asm ("bar.sync 1,128;");
        ;
    }
    int n_block67 = n_block_max23 - 1;
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67), tKsK42, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
    cute::cp_async_fence();
    if (Kernel_traits0::Is_Q_in_regs && ! Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view105 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view105))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view105);
    }
    clear(acc_o51);
    flash::Softmax<2 * size<1>(acc_o51)> softmax68;
    const float alibi_slope69 = !Has_alibi4 || params9.alibi_slopes_ptr == nullptr ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal2, Is_local3, Has_alibi4> mask70(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope69);
    constexpr int n_masking_steps71 = (!Is_causal2 && !Is_local3) ? 1 : ((Is_even_MN5 && Is_causal2) ? cute::ceil_div(kBlockM15, kBlockN16) : cute::ceil_div(kBlockM15, kBlockN16) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps71; ++masking_step , --n_block67) {
        Tensor acc_s106 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s106);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        } else {
            flash::copy<Is_even_MN5, Is_even_K6, true>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
        }
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s106, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s106, params9.softcap);
        }
        mask70.template apply_mask<Is_causal2, Is_even_MN5>(acc_s106, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax68.template softmax_rescale_o<true, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2) : softmax68.template softmax_rescale_o<false, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2);
        Tensor rP107 = flash::convert_type<Element>(acc_s106);
        int block_row_idx108 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx109 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop111 = make_fragment_like(rP107);
            cute::copy(rP107, rP_drop111);
            dropout20.template apply_dropout<true>(rP_drop111, block_row_idx108, block_col_idx109, kNWarps18);
            cute::copy(rP_drop111, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP107, block_row_idx108, block_col_idx109, kNWarps18);
        }
        Tensor tOrP110 = make_tensor(rP107.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP107.layout()));
        flash::gemm_rs(acc_o51, tOrP110, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
        if (n_masking_steps71 > 1 && n_block67 <= n_block_min22) {
            --n_block67;
            break;
        }
    }
    for (; n_block67 >= n_block_min22; --n_block67) {
        Tensor acc_s112 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s112);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s112, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s112, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        mask70.template apply_mask<false>(acc_s112, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        softmax68.template softmax_rescale_o<false, Is_local3>(acc_s112, acc_o51, params9.scale_softmax_log2);
        Tensor rP113 = flash::convert_type<Element>(acc_s112);
        int block_row_idx114 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx115 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop117 = make_fragment_like(rP113);
            cute::copy(rP113, rP_drop117);
            dropout20.template apply_dropout<true>(rP_drop117, block_row_idx114, block_col_idx115, kNWarps18);
            cute::copy(rP_drop117, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP113, block_row_idx114, block_col_idx115, kNWarps18);
        }
        Tensor tOrP116 = make_tensor(rP113.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP113.layout()));
        flash::gemm_rs(acc_o51, tOrP116, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
    }
    Tensor lse72 = softmax68.template normalize_softmax_lse<Is_dropout1>(acc_o51, params9.scale_softmax, params9.rp_dropout);
    Tensor rO73 = flash::convert_type<Element>(acc_o51);
    Tensor sO74 = make_tensor(sQ32.data(), typename Kernel_traits0::SmemLayoutO{});
    auto smem_tiled_copy_O75 = make_tiled_copy_C(typename Kernel_traits0::SmemCopyAtomO{}, tiled_mma45);
    auto smem_thr_copy_O76 = smem_tiled_copy_O75.get_thread_slice(tidx14);
    Tensor taccOrO77 = smem_thr_copy_O76.retile_S(rO73);
    Tensor taccOsO78 = smem_thr_copy_O76.partition_D(sO74);
    if (Kernel_traits0::Share_Q_K_smem) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_O75, taccOrO77, taccOsO78);
    Tensor mO79 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
    Tensor gO80 = local_tile(mO79(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor gLSE81 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
    typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O82;
    auto gmem_thr_copy_O83 = gmem_tiled_copy_O82.get_thread_slice(tidx14);
    Tensor tOsO84 = gmem_thr_copy_O83.partition_S(sO74);
    Tensor tOgO85 = gmem_thr_copy_O83.partition_D(gO80);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrO86 = make_tensor<Element>(shape(tOgO85));
    cute::copy(gmem_tiled_copy_O82, tOsO84, tOrO86);
    Tensor caccO87 = make_identity_tensor(Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    Tensor taccOcO88 = thr_mma46.partition_C(caccO87);
    static_assert(decltype(size<0>(taccOcO88))::value == 4);
    Tensor taccOcO_row89 = logical_divide(taccOcO88, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse72) == size(taccOcO_row89))::value);
    if (get<1>(taccOcO_row89(0)) == 0) {
        for (int mi = 0; mi < size(lse72); ++mi) {
            const int row118 = get<0>(taccOcO_row89(mi));
            if (row118 < binfo21.actual_seqlen_q - m_block10 * kBlockM15) {
                gLSE81(row118) = lse72(mi);
            }
        }
    }
    Tensor cO90 = make_identity_tensor(make_shape(size<0>(sO74), size<1>(sO74)));
    Tensor tOcO91 = gmem_thr_copy_O83.partition_D(cO90);
    Tensor tOpO92 = make_tensor<bool>(make_shape(size<2>(tOgO85)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tOpO92); ++k) {
            tOpO92(k) = get<1>(tOcO91(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O82, tOrO86, tOgO85, tOcO91, tOpO92, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int m_block129 = blockIdx.x;
    const int bidb130 = Split126 ? blockIdx.z / params128.h : blockIdx.y;
    const int bidh131 = Split126 ? blockIdx.z - bidb130 * params128.h : blockIdx.z;
    const int n_split_idx132 = Split126 ? blockIdx.y : 0;
    const int num_n_splits133 = Split126 ? gridDim.y : 1;
    using Element = typename Kernel_traits119::Element;
    using ElementAccum = typename Kernel_traits119::ElementAccum;
    using index_t = typename Kernel_traits119::index_t;
    extern char smem_134[] __attribute__((shared));
    const int tidx135 = threadIdx_x_1;
    constexpr int kBlockM136 = Kernel_traits119::kBlockM;
    constexpr int kBlockN137 = Kernel_traits119::kBlockN;
    constexpr int kHeadDim138 = Kernel_traits119::kHeadDim;
    constexpr int kNWarps139 = Kernel_traits119::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::GmemTiledCopyO, typename Kernel_traits119::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split126, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN123> binfo140(params128, bidb130);
    if (m_block129 * kBlockM136 >= binfo140.actual_seqlen_q)
        return;
    const int n_blocks_per_split141 = ((params128.seqlen_k + kBlockN137 - 1) / kBlockN137 + num_n_splits133 - 1) / num_n_splits133;
    const int n_block_min142 = !Is_local121 ? n_split_idx132 * n_blocks_per_split141 : std::max(n_split_idx132 * n_blocks_per_split141, (m_block129 * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q - params128.window_size_left) / kBlockN137);
    int n_block_max143 = std::min(cute::ceil_div(binfo140.actual_seqlen_k, kBlockN137), (n_split_idx132 + 1) * n_blocks_per_split141);
    if (Is_causal120 || Is_local121) {
        n_block_max143 = std::min(n_block_max143, cute::ceil_div((m_block129 + 1) * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q + params128.window_size_right, kBlockN137));
    }
    if (n_block_min142 >= n_block_max143) {
        const index_t row_offset_o220 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
        const index_t row_offset_oaccum221 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
        const index_t row_offset_lseaccum222 = ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136;
        Tensor gOaccum223 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum221 : row_offset_o220)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
        Tensor gLSEaccum224 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum222), Shape<Int<kBlockM136>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum225;
        auto gmem_thr_copy_Oaccum226 = gmem_tiled_copy_Oaccum225.get_thread_slice(tidx135);
        Tensor tOgOaccum227 = gmem_thr_copy_Oaccum226.partition_D(gOaccum223);
        Tensor tOrOaccum228 = make_tensor<ElementO>(shape(tOgOaccum227));
        clear(tOrOaccum228);
        Tensor cO229 = make_identity_tensor(make_shape(size<0>(gOaccum223), size<1>(gOaccum223)));
        Tensor tOcO230 = gmem_thr_copy_Oaccum226.partition_D(cO229);
        Tensor tOpO231 = make_tensor<bool>(make_shape(size<2>(tOgOaccum227)));
        if (!Is_even_K124) {
            for (int k = 0; k < size(tOpO231); ++k) {
                tOpO231(k) = get<1>(tOcO230(0, 0, k)) < params128.d;
            }
        }
        flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum225, tOrOaccum228, tOgOaccum227, tOcO230, tOpO231, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
        for (int m = 0; m < size<1>(tOgOaccum227); ++m) {
            const int row232 = get<0>(tOcO230(0, m, 0));
            if (row232 < binfo140.actual_seqlen_q - m_block129 * kBlockM136 && get<1>(tOcO230(0, m, 0)) == 0) {
                gLSEaccum224(row232) = Split126 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache144 = params128.cache_batch_idx == nullptr ? bidb130 : params128.cache_batch_idx[bidb130];
    const int *block_table145 = params128.block_table == nullptr ? nullptr : params128.block_table + bidb130 * params128.block_table_batch_stride;
    const int block_table_idx146 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 / params128.page_block_size;
    const int block_table_offset147 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 - block_table_idx146 * params128.page_block_size;
    const index_t row_offset_k148 = block_table145 == nullptr ? binfo140.k_offset(params128.k_batch_stride, params128.k_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride : block_table145[block_table_idx146] * params128.k_batch_stride + block_table_offset147 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride;
    const index_t row_offset_v149 = block_table145 == nullptr ? binfo140.k_offset(params128.v_batch_stride, params128.v_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride : block_table145[block_table_idx146] * params128.v_batch_stride + block_table_offset147 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride;
    Tensor mQ150 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.q_ptr) + binfo140.q_offset(params128.q_batch_stride, params128.q_row_stride, bidb130)), make_shape(binfo140.actual_seqlen_q, params128.h, params128.d), make_stride(params128.q_row_stride, params128.q_head_stride, _1{}));
    Tensor gQ151 = local_tile(mQ150(_, bidh131, _), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_coord(m_block129, 0));
    Tensor gK152 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.k_ptr) + row_offset_k148), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.k_row_stride, _1{}));
    Tensor gV153 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.v_ptr) + row_offset_v149), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.v_row_stride, _1{}));
    Tensor sQ154 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_134)), typename Kernel_traits119::SmemLayoutQ{});
    Tensor sK155 = make_tensor(sQ154.data() + size(sQ154), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sV156 = make_tensor(sK155.data() + size(sK155), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sVt157 = make_tensor(sV156.data(), typename Kernel_traits119::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle158 = make_tensor(sV156.data().get(), typename Kernel_traits119::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits119::GmemTiledCopyQKV gmem_tiled_copy_QKV159;
    auto gmem_thr_copy_QKV160 = gmem_tiled_copy_QKV159.get_thread_slice(tidx135);
    Tensor tQgQ161 = gmem_thr_copy_QKV160.partition_S(gQ151);
    Tensor tQsQ162 = gmem_thr_copy_QKV160.partition_D(sQ154);
    Tensor tKgK163 = gmem_thr_copy_QKV160.partition_S(gK152);
    Tensor tKsK164 = gmem_thr_copy_QKV160.partition_D(sK155);
    Tensor tVgV165 = gmem_thr_copy_QKV160.partition_S(gV153);
    Tensor tVsV166 = gmem_thr_copy_QKV160.partition_D(sV156);
    typename Kernel_traits119::TiledMma tiled_mma167;
    auto thr_mma168 = tiled_mma167.get_thread_slice(tidx135);
    Tensor tSrQ169 = thr_mma168.partition_fragment_A(sQ154);
    Tensor tSrK170 = thr_mma168.partition_fragment_B(sK155);
    Tensor tOrVt171 = thr_mma168.partition_fragment_B(sVtNoSwizzle158);
    Tensor acc_o172 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    auto smem_tiled_copy_Q173 = make_tiled_copy_A(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_Q174 = smem_tiled_copy_Q173.get_thread_slice(tidx135);
    Tensor tSsQ175 = smem_thr_copy_Q174.partition_S(sQ154);
    auto smem_tiled_copy_K176 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_K177 = smem_tiled_copy_K176.get_thread_slice(tidx135);
    Tensor tSsK178 = smem_thr_copy_K177.partition_S(sK155);
    auto smem_tiled_copy_V179 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtomTransposed{}, tiled_mma167);
    auto smem_thr_copy_V180 = smem_tiled_copy_V179.get_thread_slice(tidx135);
    Tensor tOsVt181 = smem_thr_copy_V180.partition_S(sVt157);
    Tensor cQ182 = make_identity_tensor(make_shape(size<0>(sQ154), size<1>(sQ154)));
    Tensor cKV183 = make_identity_tensor(make_shape(size<0>(sK155), size<1>(sK155)));
    Tensor tQcQ184 = gmem_thr_copy_QKV160.partition_S(cQ182);
    Tensor tKVcKV185 = gmem_thr_copy_QKV160.partition_S(cKV183);
    Tensor tQpQ186 = make_tensor<bool>(make_shape(size<2>(tQsQ162)));
    Tensor tKVpKV187 = make_tensor<bool>(make_shape(size<2>(tKsK164)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tQpQ186); ++k) {
            tQpQ186(k) = get<1>(tQcQ184(0, 0, k)) < params128.d;
        }
        for (int k = 0; k < size(tKVpKV187); ++k) {
            tKVpKV187(k) = get<1>(tKVcKV185(0, 0, k)) < params128.d;
        }
    }
    typename Kernel_traits119::GmemTiledCopyRotcossin gmem_tiled_copy_rotary188;
    auto gmem_thr_copy_rotary189 = gmem_tiled_copy_rotary188.get_thread_slice(tidx135);
    typename Kernel_traits119::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont190;
    auto gmem_thr_copy_rotary_cont191 = gmem_tiled_copy_rotary_cont190.get_thread_slice(tidx135);
    if (Append_KV127) {
        const index_t row_offset_cossin233 = ((n_block_max143 - 1) * kBlockN137 + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130])) * (params128.rotary_dim / 2);
        Tensor gCos234 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSin235 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gCosCont236 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSinCont237 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor tRgCos238 = gmem_thr_copy_rotary189.partition_S(gCos234);
        Tensor tRgSin239 = gmem_thr_copy_rotary189.partition_S(gSin235);
        Tensor tRgCosCont240 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont236);
        Tensor tRgSinCont241 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont237);
        const index_t row_offset_knew242 = bidb130 * params128.knew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.knew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.knew_head_stride;
        const index_t row_offset_vnew243 = bidb130 * params128.vnew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.vnew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.vnew_head_stride;
        Tensor gKnew244 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.knew_ptr) + row_offset_knew242 - binfo140.seqlen_k_cache * params128.knew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.knew_row_stride, _1{}));
        Tensor gVnew245 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.vnew_ptr) + row_offset_vnew243 - binfo140.seqlen_k_cache * params128.vnew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.vnew_row_stride, _1{}));
        Tensor tKgKnew246 = gmem_thr_copy_QKV160.partition_S(gKnew244);
        Tensor tVgVnew247 = gmem_thr_copy_QKV160.partition_S(gVnew245);
        const int n_block_copy_min248 = std::max(n_block_min142, binfo140.seqlen_k_cache / kBlockN137);
        auto tKgK_data249 = tKgK163.data();
        auto tVgV_data250 = tVgV165.data();
        for (int n_block = n_block_max143 - 1; n_block >= n_block_copy_min248; n_block--) {
            flash::copy_w_min_idx<Is_even_K124>(tVgVnew247, tVgV165, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            tVgVnew247.data() = tVgVnew247.data() + (-int(kBlockN137 * params128.vnew_row_stride));
            if (params128.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K124>(tKgKnew246, tKgK163, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            } else {
                if (params128.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCos238, tRgSin239, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCos238.data() = tRgCos238.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSin239.data() = tRgSin239.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCosCont240, tRgSinCont241, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCosCont240.data() = tRgCosCont240.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSinCont241.data() = tRgSinCont241.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                }
            }
            tKgKnew246.data() = tKgKnew246.data() + (-int(kBlockN137 * params128.knew_row_stride));
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                if (n_block > n_block_copy_min248) {
                    const int block_table_idx_cur251 = n_block * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_cur252 = n_block * kBlockN137 - block_table_idx_cur251 * params128.page_block_size;
                    const int block_table_idx_next253 = (n_block - 1) * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_next254 = (n_block - 1) * kBlockN137 - block_table_idx_next253 * params128.page_block_size;
                    const int table_diff255 = block_table145[block_table_idx_next253] - block_table145[block_table_idx_cur251];
                    const int offset_diff256 = block_table_offset_next254 - block_table_offset_cur252;
                    tVgV165.data() = tVgV165.data() + table_diff255 * params128.v_batch_stride + offset_diff256 * params128.v_row_stride;
                    tKgK163.data() = tKgK163.data() + table_diff255 * params128.k_batch_stride + offset_diff256 * params128.k_row_stride;
                }
            }
        }
        asm ("bar.sync 2,128;");
        ;
        tKgK163.data() = tKgK_data249;
        tVgV165.data() = tVgV_data250;
    }
    if (!Append_KV127 || params128.rotary_dim == 0) {
        flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tQgQ161, tQsQ162, tQcQ184, tQpQ186, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
    } else {
        const index_t row_offset_cossin257 = (binfo140.seqlen_k_cache + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130]) + (Is_causal120 || Is_local121 ? m_block129 * kBlockM136 : 0)) * (params128.rotary_dim / 2);
        Tensor gCos258 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSin259 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont260 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont261 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos262 = gmem_thr_copy_rotary189.partition_S(gCos258);
        Tensor tRgSin263 = gmem_thr_copy_rotary189.partition_S(gSin259);
        Tensor tRgCosCont264 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont260);
        Tensor tRgSinCont265 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont261);
        if (params128.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K124>(tQgQ161, tQsQ162, tRgCos262, tRgSin263, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K124>(tQgQ161, tQsQ162, tRgCosCont264, tRgSinCont265, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        }
    }
    int n_block192 = n_block_max143 - 1;
    flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
    cute::cp_async_fence();
    clear(acc_o172);
    flash::Softmax<2 * size<1>(acc_o172)> softmax193;
    const float alibi_slope194 = !Has_alibi122 ? 0.F : reinterpret_cast<float *>(params128.alibi_slopes_ptr)[bidb130 * params128.alibi_slopes_batch_stride + bidh131] / params128.scale_softmax;
    flash::Mask<Is_causal120, Is_local121, Has_alibi122> mask195(binfo140.actual_seqlen_k, binfo140.actual_seqlen_q, params128.window_size_left, params128.window_size_right, alibi_slope194);
    constexpr int n_masking_steps196 = (!Is_causal120 && !Is_local121) ? 1 : ((Is_even_MN123 && Is_causal120) ? cute::ceil_div(kBlockM136, kBlockN137) : cute::ceil_div(kBlockM136, kBlockN137) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps196; ++masking_step , --n_block192) {
        Tensor acc_s266 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s266);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (masking_step > 0) {
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
            } else {
                const int block_table_idx_cur269 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur270 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur269 * params128.page_block_size;
                const int block_table_idx_next271 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next272 = n_block192 * kBlockN137 - block_table_idx_next271 * params128.page_block_size;
                tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next271] - block_table145[block_table_idx_cur269]) * params128.v_batch_stride + (block_table_offset_next272 - block_table_offset_cur270) * params128.v_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        } else {
            flash::copy<Is_even_MN123, Is_even_K124, true>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s266, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s266, params128.softcap);
        }
        mask195.template apply_mask<Is_causal120, Is_even_MN123>(acc_s266, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur273 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur274 = n_block192 * kBlockN137 - block_table_idx_cur273 * params128.page_block_size;
                const int block_table_idx_next275 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next276 = (n_block192 - 1) * kBlockN137 - block_table_idx_next275 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next275] - block_table145[block_table_idx_cur273]) * params128.k_batch_stride + (block_table_offset_next276 - block_table_offset_cur274) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax193.template softmax_rescale_o<true, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2) : softmax193.template softmax_rescale_o<false, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2);
        Tensor rP267 = flash::convert_type<Element>(acc_s266);
        Tensor tOrP268 = make_tensor(rP267.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP267.layout()));
        flash::gemm_rs(acc_o172, tOrP268, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
        if (n_masking_steps196 > 1 && n_block192 <= n_block_min142) {
            --n_block192;
            break;
        }
    }
    for (; n_block192 >= n_block_min142; --n_block192) {
        Tensor acc_s277 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s277);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (block_table145 == nullptr) {
            tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
        } else {
            const int block_table_idx_cur280 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
            const int block_table_offset_cur281 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur280 * params128.page_block_size;
            const int block_table_idx_next282 = n_block192 * kBlockN137 / params128.page_block_size;
            const int block_table_offset_next283 = n_block192 * kBlockN137 - block_table_idx_next282 * params128.page_block_size;
            tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next282] - block_table145[block_table_idx_cur280]) * params128.v_batch_stride + (block_table_offset_next283 - block_table_offset_cur281) * params128.v_row_stride;
        }
        flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        cute::cp_async_fence();
        flash::gemm(acc_s277, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s277, params128.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur284 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur285 = n_block192 * kBlockN137 - block_table_idx_cur284 * params128.page_block_size;
                const int block_table_idx_next286 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next287 = (n_block192 - 1) * kBlockN137 - block_table_idx_next286 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next286] - block_table145[block_table_idx_cur284]) * params128.k_batch_stride + (block_table_offset_next287 - block_table_offset_cur285) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        mask195.template apply_mask<false>(acc_s277, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        softmax193.template softmax_rescale_o<false, Is_local121>(acc_s277, acc_o172, params128.scale_softmax_log2);
        Tensor rP278 = flash::convert_type<Element>(acc_s277);
        Tensor tOrP279 = make_tensor(rP278.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP278.layout()));
        flash::gemm_rs(acc_o172, tOrP279, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
    }
    Tensor lse197 = softmax193.template normalize_softmax_lse<false, Split126>(acc_o172, params128.scale_softmax);
    Tensor sOaccum198 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_134)), typename Kernel_traits119::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::SmemCopyAtomO, typename Kernel_traits119::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum199 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma167);
    auto smem_thr_copy_Oaccum200 = smem_tiled_copy_Oaccum199.get_thread_slice(tidx135);
    Tensor rO201 = flash::convert_type<ElementO>(acc_o172);
    Tensor taccOrOaccum202 = smem_thr_copy_Oaccum200.retile_S(rO201);
    Tensor taccOsOaccum203 = smem_thr_copy_Oaccum200.partition_D(sOaccum198);
    if (Split126) {
        asm ("bar.sync 2,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum199, taccOrOaccum202, taccOsOaccum203);
    const index_t row_offset_o204 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
    const index_t row_offset_oaccum205 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
    const index_t row_offset_lseaccum206 = (Split126 || !params128.unpadded_lse ? ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q : bidh131 * params128.total_q + binfo140.q_offset(params128.seqlen_q, 1, bidb130)) + m_block129 * kBlockM136;
    Tensor gOaccum207 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum205 : row_offset_o204)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
    Tensor gLSEaccum208 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum206), Shape<Int<kBlockM136>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum209;
    auto gmem_thr_copy_Oaccum210 = gmem_tiled_copy_Oaccum209.get_thread_slice(tidx135);
    Tensor tOsOaccum211 = gmem_thr_copy_Oaccum210.partition_S(sOaccum198);
    Tensor tOgOaccum212 = gmem_thr_copy_Oaccum210.partition_D(gOaccum207);
    asm ("bar.sync 2,128;");
    ;
    Tensor tOrOaccum213 = make_tensor<ElementO>(shape(tOgOaccum212));
    cute::copy(gmem_tiled_copy_Oaccum209, tOsOaccum211, tOrOaccum213);
    Tensor caccO214 = make_identity_tensor(Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    Tensor taccOcO215 = thr_mma168.partition_C(caccO214);
    static_assert(decltype(size<0>(taccOcO215))::value == 4);
    Tensor taccOcO_row216 = logical_divide(taccOcO215, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse197) == size(taccOcO_row216))::value);
    if (get<1>(taccOcO_row216(0)) == 0) {
        for (int mi = 0; mi < size(lse197); ++mi) {
            const int row288 = get<0>(taccOcO_row216(mi));
            if (row288 < binfo140.actual_seqlen_q - m_block129 * kBlockM136) {
                gLSEaccum208(row288) = lse197(mi);
            }
        }
    }
    Tensor cO217 = make_identity_tensor(make_shape(size<0>(sOaccum198), size<1>(sOaccum198)));
    Tensor tOcO218 = gmem_thr_copy_Oaccum210.partition_D(cO217);
    Tensor tOpO219 = make_tensor<bool>(make_shape(size<2>(tOgOaccum212)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tOpO219); ++k) {
            tOpO219(k) = get<1>(tOcO218(0, 0, k)) < params128.d;
        }
    }
    flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum209, tOrOaccum213, tOgOaccum212, tOcO218, tOpO219, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
}
}*/
template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits119, bool Is_causal120, bool Is_local121, bool Has_alibi122, bool Is_even_MN123, bool Is_even_K124, bool Is_softcap125, bool Split126, bool Append_KV127>
 __global__ __launch_bounds__(256) void flash_fwd_kernel_flash_fwd_splitkv_kernel_fused_kernel_hfuse_idx_0(const Flash_fwd_params params9, const Flash_fwd_params params128)
 {
    
    extern __shared__ char smem[];
if ((/*(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && */(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    //unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    //unsigned int blockDim_y_0 = 1;
    //unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    //unsigned int blockDim_z_0 = 1;
    //unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    static_assert(!(Is_causal2 && Is_local3));
    const int num_mblocks = (params9.seqlen_q + Kernel_traits0::kBlockM - 1) / Kernel_traits0::kBlockM;
    if(blockIdx.x >= num_mblocks * params9.b * params9.h){
        return;
    }
    const int m_block10 = (blockIdx.x % num_mblocks);
    // The block index for the batch.
    const int bidb11 = (blockIdx.x / num_mblocks) % params9.b;
    // The block index for the head.
    const int bidh12 = (blockIdx.x / num_mblocks / params9.b) % params9.h;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    char *smem_13 = smem;
    const int tidx14 = threadIdx_x_0;
    constexpr int kBlockM15 = Kernel_traits0::kBlockM;
    constexpr int kBlockN16 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim17 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps18 = Kernel_traits0::kNWarps;
    auto seed_offset19 = at::cuda::philox::unpack(params9.philox_args);
    flash::Dropout dropout20(std::get<0>(seed_offset19), std::get<1>(seed_offset19), params9.p_dropout_in_uint8_t, bidb11, bidh12, tidx14, params9.h);
    if (Is_dropout1 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx14 == 0) {
        params9.rng_state[0] = std::get<0>(seed_offset19);
        params9.rng_state[1] = std::get<1>(seed_offset19);
    }
    const flash::BlockInfo<!Is_even_MN5> binfo21(params9, bidb11);
    if (m_block10 * kBlockM15 >= binfo21.actual_seqlen_q)
        return;
    const int n_block_min22 = !Is_local3 ? 0 : std::max(0, (m_block10 * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN16);
    int n_block_max23 = cute::ceil_div(binfo21.actual_seqlen_k, kBlockN16);
    if (Is_causal2 || Is_local3) {
        n_block_max23 = std::min(n_block_max23, cute::ceil_div((m_block10 + 1) * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN16));
    }
    if ((Is_causal2 || Is_local3 || !Is_even_MN5) && n_block_max23 <= n_block_min22) {
        Tensor mO93 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
        Tensor gO94 = local_tile(mO93(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
        Tensor gLSE95 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
        typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O96;
        auto gmem_thr_copy_O97 = gmem_tiled_copy_O96.get_thread_slice(tidx14);
        Tensor tOgO98 = gmem_thr_copy_O97.partition_D(gO94);
        Tensor tOrO99 = make_tensor<Element>(shape(tOgO98));
        clear(tOrO99);
        Tensor cO100 = make_identity_tensor(make_shape(size<0>(gO94), size<1>(gO94)));
        Tensor tOcO101 = gmem_thr_copy_O97.partition_D(cO100);
        Tensor tOpO102 = make_tensor<bool>(make_shape(size<2>(tOgO98)));
        if (!Is_even_K6) {
            for (int k = 0; k < size(tOpO102); ++k) {
                tOpO102(k) = get<1>(tOcO101(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O96, tOrO99, tOgO98, tOcO101, tOpO102, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
        for (int m = 0; m < size<1>(tOgO98); ++m) {
            const int row103 = get<0>(tOcO101(0, m, 0));
            if (row103 < binfo21.actual_seqlen_q - m_block10 * kBlockM15 && get<1>(tOcO101(0, m, 0)) == 0) {
                gLSE95(row103) = (__builtin_inff());
            }
        }
        return;
    }
    const index_t row_offset_p24 = ((bidb11 * params9.h + bidh12) * params9.seqlen_q_rounded + m_block10 * kBlockM15) * params9.seqlen_k_rounded + (n_block_max23 - 1) * kBlockN16;
    Tensor mQ25 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ26 = local_tile(mQ25(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor mK27 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.k_row_stride, params9.k_head_stride, _1{}));
    Tensor gK28 = local_tile(mK27(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor mV29 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.v_row_stride, params9.v_head_stride, _1{}));
    Tensor gV30 = local_tile(mV29(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor gP31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.p_ptr) + row_offset_p24), Shape<Int<kBlockM15>, Int<kBlockN16>>{}, make_stride(params9.seqlen_k_rounded, _1{}));
    Tensor sQ32 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_13)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK33 = make_tensor(sQ32.data() + (Kernel_traits0::Share_Q_K_smem ? 0 : size(sQ32)), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV34 = make_tensor(sK33.data() + size(sK33), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt35 = make_tensor(sV34.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle36 = make_tensor(sV34.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV37;
    auto gmem_thr_copy_QKV38 = gmem_tiled_copy_QKV37.get_thread_slice(tidx14);
    Tensor tQgQ39 = gmem_thr_copy_QKV38.partition_S(gQ26);
    Tensor tQsQ40 = gmem_thr_copy_QKV38.partition_D(sQ32);
    Tensor tKgK41 = gmem_thr_copy_QKV38.partition_S(gK28);
    Tensor tKsK42 = gmem_thr_copy_QKV38.partition_D(sK33);
    Tensor tVgV43 = gmem_thr_copy_QKV38.partition_S(gV30);
    Tensor tVsV44 = gmem_thr_copy_QKV38.partition_D(sV34);
    typename Kernel_traits0::TiledMma tiled_mma45;
    auto thr_mma46 = tiled_mma45.get_thread_slice(tidx14);
    Tensor tSrQ47 = thr_mma46.partition_fragment_A(sQ32);
    Tensor tSrK48 = thr_mma46.partition_fragment_B(sK33);
    Tensor tOrVt49 = thr_mma46.partition_fragment_B(sVtNoSwizzle36);
    Tensor tSgS50 = thr_mma46.partition_C(gP31);
    Tensor acc_o51 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    auto smem_tiled_copy_Q52 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_Q53 = smem_tiled_copy_Q52.get_thread_slice(tidx14);
    Tensor tSsQ54 = smem_thr_copy_Q53.partition_S(sQ32);
    auto smem_tiled_copy_K55 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_K56 = smem_tiled_copy_K55.get_thread_slice(tidx14);
    Tensor tSsK57 = smem_thr_copy_K56.partition_S(sK33);
    auto smem_tiled_copy_V58 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma45);
    auto smem_thr_copy_V59 = smem_tiled_copy_V58.get_thread_slice(tidx14);
    Tensor tOsVt60 = smem_thr_copy_V59.partition_S(sVt35);
    Tensor cQ61 = make_identity_tensor(make_shape(size<0>(sQ32), size<1>(sQ32)));
    Tensor cKV62 = make_identity_tensor(make_shape(size<0>(sK33), size<1>(sK33)));
    Tensor tQcQ63 = gmem_thr_copy_QKV38.partition_S(cQ61);
    Tensor tKVcKV64 = gmem_thr_copy_QKV38.partition_S(cKV62);
    Tensor tQpQ65 = make_tensor<bool>(make_shape(size<2>(tQsQ40)));
    Tensor tKVpKV66 = make_tensor<bool>(make_shape(size<2>(tKsK42)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tQpQ65); ++k) {
            tQpQ65(k) = get<1>(tQcQ63(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV66); ++k) {
            tKVpKV66(k) = get<1>(tKVcKV64(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tQgQ39, tQsQ40, tQcQ63, tQpQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
    if (Kernel_traits0::Is_Q_in_regs) {
        cute::cp_async_fence();
    }
    if (Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view104 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view104))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view104);
        asm ("bar.sync 1,128;");
        ;
    }
    int n_block67 = n_block_max23 - 1;
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67), tKsK42, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
    cute::cp_async_fence();
    if (Kernel_traits0::Is_Q_in_regs && ! Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view105 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view105))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view105);
    }
    clear(acc_o51);
    flash::Softmax<2 * size<1>(acc_o51)> softmax68;
    const float alibi_slope69 = !Has_alibi4 || params9.alibi_slopes_ptr == nullptr ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal2, Is_local3, Has_alibi4> mask70(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope69);
    constexpr int n_masking_steps71 = (!Is_causal2 && !Is_local3) ? 1 : ((Is_even_MN5 && Is_causal2) ? cute::ceil_div(kBlockM15, kBlockN16) : cute::ceil_div(kBlockM15, kBlockN16) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps71; ++masking_step , --n_block67) {
        Tensor acc_s106 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s106);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        } else {
            flash::copy<Is_even_MN5, Is_even_K6, true>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
        }
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s106, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s106, params9.softcap);
        }
        mask70.template apply_mask<Is_causal2, Is_even_MN5>(acc_s106, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax68.template softmax_rescale_o<true, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2) : softmax68.template softmax_rescale_o<false, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2);
        Tensor rP107 = flash::convert_type<Element>(acc_s106);
        int block_row_idx108 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx109 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop111 = make_fragment_like(rP107);
            cute::copy(rP107, rP_drop111);
            dropout20.template apply_dropout<true>(rP_drop111, block_row_idx108, block_col_idx109, kNWarps18);
            cute::copy(rP_drop111, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP107, block_row_idx108, block_col_idx109, kNWarps18);
        }
        Tensor tOrP110 = make_tensor(rP107.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP107.layout()));
        flash::gemm_rs(acc_o51, tOrP110, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
        if (n_masking_steps71 > 1 && n_block67 <= n_block_min22) {
            --n_block67;
            break;
        }
    }
    for (; n_block67 >= n_block_min22; --n_block67) {
        Tensor acc_s112 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s112);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s112, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s112, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        mask70.template apply_mask<false>(acc_s112, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        softmax68.template softmax_rescale_o<false, Is_local3>(acc_s112, acc_o51, params9.scale_softmax_log2);
        Tensor rP113 = flash::convert_type<Element>(acc_s112);
        int block_row_idx114 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx115 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop117 = make_fragment_like(rP113);
            cute::copy(rP113, rP_drop117);
            dropout20.template apply_dropout<true>(rP_drop117, block_row_idx114, block_col_idx115, kNWarps18);
            cute::copy(rP_drop117, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP113, block_row_idx114, block_col_idx115, kNWarps18);
        }
        Tensor tOrP116 = make_tensor(rP113.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP113.layout()));
        flash::gemm_rs(acc_o51, tOrP116, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
    }
    Tensor lse72 = softmax68.template normalize_softmax_lse<Is_dropout1>(acc_o51, params9.scale_softmax, params9.rp_dropout);
    Tensor rO73 = flash::convert_type<Element>(acc_o51);
    Tensor sO74 = make_tensor(sQ32.data(), typename Kernel_traits0::SmemLayoutO{});
    auto smem_tiled_copy_O75 = make_tiled_copy_C(typename Kernel_traits0::SmemCopyAtomO{}, tiled_mma45);
    auto smem_thr_copy_O76 = smem_tiled_copy_O75.get_thread_slice(tidx14);
    Tensor taccOrO77 = smem_thr_copy_O76.retile_S(rO73);
    Tensor taccOsO78 = smem_thr_copy_O76.partition_D(sO74);
    if (Kernel_traits0::Share_Q_K_smem) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_O75, taccOrO77, taccOsO78);
    Tensor mO79 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
    Tensor gO80 = local_tile(mO79(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor gLSE81 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
    typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O82;
    auto gmem_thr_copy_O83 = gmem_tiled_copy_O82.get_thread_slice(tidx14);
    Tensor tOsO84 = gmem_thr_copy_O83.partition_S(sO74);
    Tensor tOgO85 = gmem_thr_copy_O83.partition_D(gO80);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrO86 = make_tensor<Element>(shape(tOgO85));
    cute::copy(gmem_tiled_copy_O82, tOsO84, tOrO86);
    Tensor caccO87 = make_identity_tensor(Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    Tensor taccOcO88 = thr_mma46.partition_C(caccO87);
    static_assert(decltype(size<0>(taccOcO88))::value == 4);
    Tensor taccOcO_row89 = logical_divide(taccOcO88, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse72) == size(taccOcO_row89))::value);
    if (get<1>(taccOcO_row89(0)) == 0) {
        for (int mi = 0; mi < size(lse72); ++mi) {
            const int row118 = get<0>(taccOcO_row89(mi));
            if (row118 < binfo21.actual_seqlen_q - m_block10 * kBlockM15) {
                gLSE81(row118) = lse72(mi);
            }
        }
    }
    Tensor cO90 = make_identity_tensor(make_shape(size<0>(sO74), size<1>(sO74)));
    Tensor tOcO91 = gmem_thr_copy_O83.partition_D(cO90);
    Tensor tOpO92 = make_tensor<bool>(make_shape(size<2>(tOgO85)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tOpO92); ++k) {
            tOpO92(k) = get<1>(tOcO91(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O82, tOrO86, tOgO85, tOcO91, tOpO92, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    //unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    //unsigned int blockDim_y_1 = 1;
    //unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    //unsigned int blockDim_z_1 = 1;
    //unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    
    const int num_mblocks = (params128.seqlen_q + Kernel_traits119::kBlockM - 1) / Kernel_traits119::kBlockM;
    const int linear_block_id = blockIdx.x;
    if(linear_block_id >= num_mblocks * params128.b * params128.h * params128.num_splits){
        return;
    }
    const int m_block129 = linear_block_id % num_mblocks;
    const int num_n_splits133 = params128.num_splits;
    const int n_split_idx132 = (linear_block_id / num_mblocks) % num_n_splits133;
    const int bidb130 = (linear_block_id / num_mblocks / num_n_splits133) % params128.b;
    const int bidh131 = (linear_block_id / num_mblocks / num_n_splits133 / params128.b) % params128.h;
    using Element = typename Kernel_traits119::Element;
    using ElementAccum = typename Kernel_traits119::ElementAccum;
    using index_t = typename Kernel_traits119::index_t;
    char *smem_134 = &smem[Kernel_traits0::kSmemSize];
    const int tidx135 = threadIdx_x_1;
    constexpr int kBlockM136 = Kernel_traits119::kBlockM;
    constexpr int kBlockN137 = Kernel_traits119::kBlockN;
    constexpr int kHeadDim138 = Kernel_traits119::kHeadDim;
    constexpr int kNWarps139 = Kernel_traits119::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::GmemTiledCopyO, typename Kernel_traits119::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split126, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN123> binfo140(params128, bidb130);
    if (m_block129 * kBlockM136 >= binfo140.actual_seqlen_q)
        return;
    const int n_blocks_per_split141 = ((params128.seqlen_k + kBlockN137 - 1) / kBlockN137 + num_n_splits133 - 1) / num_n_splits133;
    const int n_block_min142 = !Is_local121 ? n_split_idx132 * n_blocks_per_split141 : std::max(n_split_idx132 * n_blocks_per_split141, (m_block129 * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q - params128.window_size_left) / kBlockN137);
    int n_block_max143 = std::min(cute::ceil_div(binfo140.actual_seqlen_k, kBlockN137), (n_split_idx132 + 1) * n_blocks_per_split141);
    if (Is_causal120 || Is_local121) {
        n_block_max143 = std::min(n_block_max143, cute::ceil_div((m_block129 + 1) * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q + params128.window_size_right, kBlockN137));
    }
    if (n_block_min142 >= n_block_max143) {
        const index_t row_offset_o220 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
        const index_t row_offset_oaccum221 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
        const index_t row_offset_lseaccum222 = ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136;
        Tensor gOaccum223 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum221 : row_offset_o220)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
        Tensor gLSEaccum224 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum222), Shape<Int<kBlockM136>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum225;
        auto gmem_thr_copy_Oaccum226 = gmem_tiled_copy_Oaccum225.get_thread_slice(tidx135);
        Tensor tOgOaccum227 = gmem_thr_copy_Oaccum226.partition_D(gOaccum223);
        Tensor tOrOaccum228 = make_tensor<ElementO>(shape(tOgOaccum227));
        clear(tOrOaccum228);
        Tensor cO229 = make_identity_tensor(make_shape(size<0>(gOaccum223), size<1>(gOaccum223)));
        Tensor tOcO230 = gmem_thr_copy_Oaccum226.partition_D(cO229);
        Tensor tOpO231 = make_tensor<bool>(make_shape(size<2>(tOgOaccum227)));
        if (!Is_even_K124) {
            for (int k = 0; k < size(tOpO231); ++k) {
                tOpO231(k) = get<1>(tOcO230(0, 0, k)) < params128.d;
            }
        }
        flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum225, tOrOaccum228, tOgOaccum227, tOcO230, tOpO231, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
        for (int m = 0; m < size<1>(tOgOaccum227); ++m) {
            const int row232 = get<0>(tOcO230(0, m, 0));
            if (row232 < binfo140.actual_seqlen_q - m_block129 * kBlockM136 && get<1>(tOcO230(0, m, 0)) == 0) {
                gLSEaccum224(row232) = Split126 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache144 = params128.cache_batch_idx == nullptr ? bidb130 : params128.cache_batch_idx[bidb130];
    const int *block_table145 = params128.block_table == nullptr ? nullptr : params128.block_table + bidb130 * params128.block_table_batch_stride;
    const int block_table_idx146 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 / params128.page_block_size;
    const int block_table_offset147 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 - block_table_idx146 * params128.page_block_size;
    const index_t row_offset_k148 = block_table145 == nullptr ? binfo140.k_offset(params128.k_batch_stride, params128.k_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride : block_table145[block_table_idx146] * params128.k_batch_stride + block_table_offset147 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride;
    const index_t row_offset_v149 = block_table145 == nullptr ? binfo140.k_offset(params128.v_batch_stride, params128.v_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride : block_table145[block_table_idx146] * params128.v_batch_stride + block_table_offset147 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride;
    Tensor mQ150 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.q_ptr) + binfo140.q_offset(params128.q_batch_stride, params128.q_row_stride, bidb130)), make_shape(binfo140.actual_seqlen_q, params128.h, params128.d), make_stride(params128.q_row_stride, params128.q_head_stride, _1{}));
    Tensor gQ151 = local_tile(mQ150(_, bidh131, _), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_coord(m_block129, 0));
    Tensor gK152 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.k_ptr) + row_offset_k148), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.k_row_stride, _1{}));
    Tensor gV153 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.v_ptr) + row_offset_v149), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.v_row_stride, _1{}));
    Tensor sQ154 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_134)), typename Kernel_traits119::SmemLayoutQ{});
    Tensor sK155 = make_tensor(sQ154.data() + size(sQ154), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sV156 = make_tensor(sK155.data() + size(sK155), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sVt157 = make_tensor(sV156.data(), typename Kernel_traits119::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle158 = make_tensor(sV156.data().get(), typename Kernel_traits119::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits119::GmemTiledCopyQKV gmem_tiled_copy_QKV159;
    auto gmem_thr_copy_QKV160 = gmem_tiled_copy_QKV159.get_thread_slice(tidx135);
    Tensor tQgQ161 = gmem_thr_copy_QKV160.partition_S(gQ151);
    Tensor tQsQ162 = gmem_thr_copy_QKV160.partition_D(sQ154);
    Tensor tKgK163 = gmem_thr_copy_QKV160.partition_S(gK152);
    Tensor tKsK164 = gmem_thr_copy_QKV160.partition_D(sK155);
    Tensor tVgV165 = gmem_thr_copy_QKV160.partition_S(gV153);
    Tensor tVsV166 = gmem_thr_copy_QKV160.partition_D(sV156);
    typename Kernel_traits119::TiledMma tiled_mma167;
    auto thr_mma168 = tiled_mma167.get_thread_slice(tidx135);
    Tensor tSrQ169 = thr_mma168.partition_fragment_A(sQ154);
    Tensor tSrK170 = thr_mma168.partition_fragment_B(sK155);
    Tensor tOrVt171 = thr_mma168.partition_fragment_B(sVtNoSwizzle158);
    Tensor acc_o172 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    auto smem_tiled_copy_Q173 = make_tiled_copy_A(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_Q174 = smem_tiled_copy_Q173.get_thread_slice(tidx135);
    Tensor tSsQ175 = smem_thr_copy_Q174.partition_S(sQ154);
    auto smem_tiled_copy_K176 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_K177 = smem_tiled_copy_K176.get_thread_slice(tidx135);
    Tensor tSsK178 = smem_thr_copy_K177.partition_S(sK155);
    auto smem_tiled_copy_V179 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtomTransposed{}, tiled_mma167);
    auto smem_thr_copy_V180 = smem_tiled_copy_V179.get_thread_slice(tidx135);
    Tensor tOsVt181 = smem_thr_copy_V180.partition_S(sVt157);
    Tensor cQ182 = make_identity_tensor(make_shape(size<0>(sQ154), size<1>(sQ154)));
    Tensor cKV183 = make_identity_tensor(make_shape(size<0>(sK155), size<1>(sK155)));
    Tensor tQcQ184 = gmem_thr_copy_QKV160.partition_S(cQ182);
    Tensor tKVcKV185 = gmem_thr_copy_QKV160.partition_S(cKV183);
    Tensor tQpQ186 = make_tensor<bool>(make_shape(size<2>(tQsQ162)));
    Tensor tKVpKV187 = make_tensor<bool>(make_shape(size<2>(tKsK164)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tQpQ186); ++k) {
            tQpQ186(k) = get<1>(tQcQ184(0, 0, k)) < params128.d;
        }
        for (int k = 0; k < size(tKVpKV187); ++k) {
            tKVpKV187(k) = get<1>(tKVcKV185(0, 0, k)) < params128.d;
        }
    }
    typename Kernel_traits119::GmemTiledCopyRotcossin gmem_tiled_copy_rotary188;
    auto gmem_thr_copy_rotary189 = gmem_tiled_copy_rotary188.get_thread_slice(tidx135);
    typename Kernel_traits119::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont190;
    auto gmem_thr_copy_rotary_cont191 = gmem_tiled_copy_rotary_cont190.get_thread_slice(tidx135);
    if (Append_KV127) {
        const index_t row_offset_cossin233 = ((n_block_max143 - 1) * kBlockN137 + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130])) * (params128.rotary_dim / 2);
        Tensor gCos234 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSin235 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gCosCont236 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSinCont237 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor tRgCos238 = gmem_thr_copy_rotary189.partition_S(gCos234);
        Tensor tRgSin239 = gmem_thr_copy_rotary189.partition_S(gSin235);
        Tensor tRgCosCont240 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont236);
        Tensor tRgSinCont241 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont237);
        const index_t row_offset_knew242 = bidb130 * params128.knew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.knew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.knew_head_stride;
        const index_t row_offset_vnew243 = bidb130 * params128.vnew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.vnew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.vnew_head_stride;
        Tensor gKnew244 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.knew_ptr) + row_offset_knew242 - binfo140.seqlen_k_cache * params128.knew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.knew_row_stride, _1{}));
        Tensor gVnew245 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.vnew_ptr) + row_offset_vnew243 - binfo140.seqlen_k_cache * params128.vnew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.vnew_row_stride, _1{}));
        Tensor tKgKnew246 = gmem_thr_copy_QKV160.partition_S(gKnew244);
        Tensor tVgVnew247 = gmem_thr_copy_QKV160.partition_S(gVnew245);
        const int n_block_copy_min248 = std::max(n_block_min142, binfo140.seqlen_k_cache / kBlockN137);
        auto tKgK_data249 = tKgK163.data();
        auto tVgV_data250 = tVgV165.data();
        for (int n_block = n_block_max143 - 1; n_block >= n_block_copy_min248; n_block--) {
            flash::copy_w_min_idx<Is_even_K124>(tVgVnew247, tVgV165, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            tVgVnew247.data() = tVgVnew247.data() + (-int(kBlockN137 * params128.vnew_row_stride));
            if (params128.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K124>(tKgKnew246, tKgK163, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            } else {
                if (params128.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCos238, tRgSin239, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCos238.data() = tRgCos238.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSin239.data() = tRgSin239.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCosCont240, tRgSinCont241, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCosCont240.data() = tRgCosCont240.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSinCont241.data() = tRgSinCont241.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                }
            }
            tKgKnew246.data() = tKgKnew246.data() + (-int(kBlockN137 * params128.knew_row_stride));
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                if (n_block > n_block_copy_min248) {
                    const int block_table_idx_cur251 = n_block * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_cur252 = n_block * kBlockN137 - block_table_idx_cur251 * params128.page_block_size;
                    const int block_table_idx_next253 = (n_block - 1) * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_next254 = (n_block - 1) * kBlockN137 - block_table_idx_next253 * params128.page_block_size;
                    const int table_diff255 = block_table145[block_table_idx_next253] - block_table145[block_table_idx_cur251];
                    const int offset_diff256 = block_table_offset_next254 - block_table_offset_cur252;
                    tVgV165.data() = tVgV165.data() + table_diff255 * params128.v_batch_stride + offset_diff256 * params128.v_row_stride;
                    tKgK163.data() = tKgK163.data() + table_diff255 * params128.k_batch_stride + offset_diff256 * params128.k_row_stride;
                }
            }
        }
        asm ("bar.sync 2,128;");
        ;
        tKgK163.data() = tKgK_data249;
        tVgV165.data() = tVgV_data250;
    }
    if (!Append_KV127 || params128.rotary_dim == 0) {
        flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tQgQ161, tQsQ162, tQcQ184, tQpQ186, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
    } else {
        const index_t row_offset_cossin257 = (binfo140.seqlen_k_cache + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130]) + (Is_causal120 || Is_local121 ? m_block129 * kBlockM136 : 0)) * (params128.rotary_dim / 2);
        Tensor gCos258 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSin259 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont260 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont261 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos262 = gmem_thr_copy_rotary189.partition_S(gCos258);
        Tensor tRgSin263 = gmem_thr_copy_rotary189.partition_S(gSin259);
        Tensor tRgCosCont264 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont260);
        Tensor tRgSinCont265 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont261);
        if (params128.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K124>(tQgQ161, tQsQ162, tRgCos262, tRgSin263, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K124>(tQgQ161, tQsQ162, tRgCosCont264, tRgSinCont265, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        }
    }
    int n_block192 = n_block_max143 - 1;
    flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
    cute::cp_async_fence();
    clear(acc_o172);
    flash::Softmax<2 * size<1>(acc_o172)> softmax193;
    const float alibi_slope194 = !Has_alibi122 ? 0.F : reinterpret_cast<float *>(params128.alibi_slopes_ptr)[bidb130 * params128.alibi_slopes_batch_stride + bidh131] / params128.scale_softmax;
    flash::Mask<Is_causal120, Is_local121, Has_alibi122> mask195(binfo140.actual_seqlen_k, binfo140.actual_seqlen_q, params128.window_size_left, params128.window_size_right, alibi_slope194);
    constexpr int n_masking_steps196 = (!Is_causal120 && !Is_local121) ? 1 : ((Is_even_MN123 && Is_causal120) ? cute::ceil_div(kBlockM136, kBlockN137) : cute::ceil_div(kBlockM136, kBlockN137) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps196; ++masking_step , --n_block192) {
        Tensor acc_s266 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s266);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (masking_step > 0) {
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
            } else {
                const int block_table_idx_cur269 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur270 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur269 * params128.page_block_size;
                const int block_table_idx_next271 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next272 = n_block192 * kBlockN137 - block_table_idx_next271 * params128.page_block_size;
                tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next271] - block_table145[block_table_idx_cur269]) * params128.v_batch_stride + (block_table_offset_next272 - block_table_offset_cur270) * params128.v_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        } else {
            flash::copy<Is_even_MN123, Is_even_K124, true>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s266, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s266, params128.softcap);
        }
        mask195.template apply_mask<Is_causal120, Is_even_MN123>(acc_s266, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur273 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur274 = n_block192 * kBlockN137 - block_table_idx_cur273 * params128.page_block_size;
                const int block_table_idx_next275 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next276 = (n_block192 - 1) * kBlockN137 - block_table_idx_next275 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next275] - block_table145[block_table_idx_cur273]) * params128.k_batch_stride + (block_table_offset_next276 - block_table_offset_cur274) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax193.template softmax_rescale_o<true, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2) : softmax193.template softmax_rescale_o<false, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2);
        Tensor rP267 = flash::convert_type<Element>(acc_s266);
        Tensor tOrP268 = make_tensor(rP267.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP267.layout()));
        flash::gemm_rs(acc_o172, tOrP268, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
        if (n_masking_steps196 > 1 && n_block192 <= n_block_min142) {
            --n_block192;
            break;
        }
    }
    for (; n_block192 >= n_block_min142; --n_block192) {
        Tensor acc_s277 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s277);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (block_table145 == nullptr) {
            tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
        } else {
            const int block_table_idx_cur280 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
            const int block_table_offset_cur281 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur280 * params128.page_block_size;
            const int block_table_idx_next282 = n_block192 * kBlockN137 / params128.page_block_size;
            const int block_table_offset_next283 = n_block192 * kBlockN137 - block_table_idx_next282 * params128.page_block_size;
            tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next282] - block_table145[block_table_idx_cur280]) * params128.v_batch_stride + (block_table_offset_next283 - block_table_offset_cur281) * params128.v_row_stride;
        }
        flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        cute::cp_async_fence();
        flash::gemm(acc_s277, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s277, params128.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur284 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur285 = n_block192 * kBlockN137 - block_table_idx_cur284 * params128.page_block_size;
                const int block_table_idx_next286 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next287 = (n_block192 - 1) * kBlockN137 - block_table_idx_next286 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next286] - block_table145[block_table_idx_cur284]) * params128.k_batch_stride + (block_table_offset_next287 - block_table_offset_cur285) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        mask195.template apply_mask<false>(acc_s277, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        softmax193.template softmax_rescale_o<false, Is_local121>(acc_s277, acc_o172, params128.scale_softmax_log2);
        Tensor rP278 = flash::convert_type<Element>(acc_s277);
        Tensor tOrP279 = make_tensor(rP278.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP278.layout()));
        flash::gemm_rs(acc_o172, tOrP279, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
    }
    Tensor lse197 = softmax193.template normalize_softmax_lse<false, Split126>(acc_o172, params128.scale_softmax);
    Tensor sOaccum198 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_134)), typename Kernel_traits119::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::SmemCopyAtomO, typename Kernel_traits119::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum199 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma167);
    auto smem_thr_copy_Oaccum200 = smem_tiled_copy_Oaccum199.get_thread_slice(tidx135);
    Tensor rO201 = flash::convert_type<ElementO>(acc_o172);
    Tensor taccOrOaccum202 = smem_thr_copy_Oaccum200.retile_S(rO201);
    Tensor taccOsOaccum203 = smem_thr_copy_Oaccum200.partition_D(sOaccum198);
    if (Split126) {
        asm ("bar.sync 2,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum199, taccOrOaccum202, taccOsOaccum203);
    const index_t row_offset_o204 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
    const index_t row_offset_oaccum205 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
    const index_t row_offset_lseaccum206 = (Split126 || !params128.unpadded_lse ? ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q : bidh131 * params128.total_q + binfo140.q_offset(params128.seqlen_q, 1, bidb130)) + m_block129 * kBlockM136;
    Tensor gOaccum207 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum205 : row_offset_o204)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
    Tensor gLSEaccum208 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum206), Shape<Int<kBlockM136>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum209;
    auto gmem_thr_copy_Oaccum210 = gmem_tiled_copy_Oaccum209.get_thread_slice(tidx135);
    Tensor tOsOaccum211 = gmem_thr_copy_Oaccum210.partition_S(sOaccum198);
    Tensor tOgOaccum212 = gmem_thr_copy_Oaccum210.partition_D(gOaccum207);
    asm ("bar.sync 2,128;");
    ;
    Tensor tOrOaccum213 = make_tensor<ElementO>(shape(tOgOaccum212));
    cute::copy(gmem_tiled_copy_Oaccum209, tOsOaccum211, tOrOaccum213);
    Tensor caccO214 = make_identity_tensor(Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    Tensor taccOcO215 = thr_mma168.partition_C(caccO214);
    static_assert(decltype(size<0>(taccOcO215))::value == 4);
    Tensor taccOcO_row216 = logical_divide(taccOcO215, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse197) == size(taccOcO_row216))::value);
    if (get<1>(taccOcO_row216(0)) == 0) {
        for (int mi = 0; mi < size(lse197); ++mi) {
            const int row288 = get<0>(taccOcO_row216(mi));
            if (row288 < binfo140.actual_seqlen_q - m_block129 * kBlockM136) {
                gLSEaccum208(row288) = lse197(mi);
            }
        }
    }
    Tensor cO217 = make_identity_tensor(make_shape(size<0>(sOaccum198), size<1>(sOaccum198)));
    Tensor tOcO218 = gmem_thr_copy_Oaccum210.partition_D(cO217);
    Tensor tOpO219 = make_tensor<bool>(make_shape(size<2>(tOgOaccum212)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tOpO219); ++k) {
            tOpO219(k) = get<1>(tOcO218(0, 0, k)) < params128.d;
        }
    }
    flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum209, tOrOaccum213, tOgOaccum212, tOcO218, tOpO219, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
}
}
/*template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits119, bool Is_causal120, bool Is_local121, bool Has_alibi122, bool Is_even_MN123, bool Is_even_K124, bool Is_softcap125, bool Split126, bool Append_KV127>
 __global__ __launch_bounds__(256, 2) void flash_fwd_kernel_flash_fwd_splitkv_kernel_fused_kernel_hfuse_lb_idx_0(const Flash_fwd_params params9, const Flash_fwd_params params128)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    //unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    //unsigned int blockDim_y_0 = 1;
    //unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    //unsigned int blockDim_z_0 = 1;
    //unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    static_assert(!(Is_causal2 && Is_local3));
    const int m_block10 = blockIdx.x;
    const int bidb11 = blockIdx.y;
    const int bidh12 = blockIdx.z;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_13[] __attribute__((shared));
    const int tidx14 = threadIdx_x_0;
    constexpr int kBlockM15 = Kernel_traits0::kBlockM;
    constexpr int kBlockN16 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim17 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps18 = Kernel_traits0::kNWarps;
    auto seed_offset19 = at::cuda::philox::unpack(params9.philox_args);
    flash::Dropout dropout20(std::get<0>(seed_offset19), std::get<1>(seed_offset19), params9.p_dropout_in_uint8_t, bidb11, bidh12, tidx14, params9.h);
    if (Is_dropout1 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx14 == 0) {
        params9.rng_state[0] = std::get<0>(seed_offset19);
        params9.rng_state[1] = std::get<1>(seed_offset19);
    }
    const flash::BlockInfo<!Is_even_MN5> binfo21(params9, bidb11);
    if (m_block10 * kBlockM15 >= binfo21.actual_seqlen_q)
        return;
    const int n_block_min22 = !Is_local3 ? 0 : std::max(0, (m_block10 * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN16);
    int n_block_max23 = cute::ceil_div(binfo21.actual_seqlen_k, kBlockN16);
    if (Is_causal2 || Is_local3) {
        n_block_max23 = std::min(n_block_max23, cute::ceil_div((m_block10 + 1) * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN16));
    }
    if ((Is_causal2 || Is_local3 || !Is_even_MN5) && n_block_max23 <= n_block_min22) {
        Tensor mO93 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
        Tensor gO94 = local_tile(mO93(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
        Tensor gLSE95 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
        typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O96;
        auto gmem_thr_copy_O97 = gmem_tiled_copy_O96.get_thread_slice(tidx14);
        Tensor tOgO98 = gmem_thr_copy_O97.partition_D(gO94);
        Tensor tOrO99 = make_tensor<Element>(shape(tOgO98));
        clear(tOrO99);
        Tensor cO100 = make_identity_tensor(make_shape(size<0>(gO94), size<1>(gO94)));
        Tensor tOcO101 = gmem_thr_copy_O97.partition_D(cO100);
        Tensor tOpO102 = make_tensor<bool>(make_shape(size<2>(tOgO98)));
        if (!Is_even_K6) {
            for (int k = 0; k < size(tOpO102); ++k) {
                tOpO102(k) = get<1>(tOcO101(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O96, tOrO99, tOgO98, tOcO101, tOpO102, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
        for (int m = 0; m < size<1>(tOgO98); ++m) {
            const int row103 = get<0>(tOcO101(0, m, 0));
            if (row103 < binfo21.actual_seqlen_q - m_block10 * kBlockM15 && get<1>(tOcO101(0, m, 0)) == 0) {
                gLSE95(row103) = (__builtin_inff());
            }
        }
        return;
    }
    const index_t row_offset_p24 = ((bidb11 * params9.h + bidh12) * params9.seqlen_q_rounded + m_block10 * kBlockM15) * params9.seqlen_k_rounded + (n_block_max23 - 1) * kBlockN16;
    Tensor mQ25 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ26 = local_tile(mQ25(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor mK27 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.k_row_stride, params9.k_head_stride, _1{}));
    Tensor gK28 = local_tile(mK27(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor mV29 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params9.h_k, params9.d), make_stride(params9.v_row_stride, params9.v_head_stride, _1{}));
    Tensor gV30 = local_tile(mV29(_, bidh12 / params9.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor gP31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.p_ptr) + row_offset_p24), Shape<Int<kBlockM15>, Int<kBlockN16>>{}, make_stride(params9.seqlen_k_rounded, _1{}));
    Tensor sQ32 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_13)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK33 = make_tensor(sQ32.data() + (Kernel_traits0::Share_Q_K_smem ? 0 : size(sQ32)), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV34 = make_tensor(sK33.data() + size(sK33), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt35 = make_tensor(sV34.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle36 = make_tensor(sV34.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV37;
    auto gmem_thr_copy_QKV38 = gmem_tiled_copy_QKV37.get_thread_slice(tidx14);
    Tensor tQgQ39 = gmem_thr_copy_QKV38.partition_S(gQ26);
    Tensor tQsQ40 = gmem_thr_copy_QKV38.partition_D(sQ32);
    Tensor tKgK41 = gmem_thr_copy_QKV38.partition_S(gK28);
    Tensor tKsK42 = gmem_thr_copy_QKV38.partition_D(sK33);
    Tensor tVgV43 = gmem_thr_copy_QKV38.partition_S(gV30);
    Tensor tVsV44 = gmem_thr_copy_QKV38.partition_D(sV34);
    typename Kernel_traits0::TiledMma tiled_mma45;
    auto thr_mma46 = tiled_mma45.get_thread_slice(tidx14);
    Tensor tSrQ47 = thr_mma46.partition_fragment_A(sQ32);
    Tensor tSrK48 = thr_mma46.partition_fragment_B(sK33);
    Tensor tOrVt49 = thr_mma46.partition_fragment_B(sVtNoSwizzle36);
    Tensor tSgS50 = thr_mma46.partition_C(gP31);
    Tensor acc_o51 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    auto smem_tiled_copy_Q52 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_Q53 = smem_tiled_copy_Q52.get_thread_slice(tidx14);
    Tensor tSsQ54 = smem_thr_copy_Q53.partition_S(sQ32);
    auto smem_tiled_copy_K55 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_K56 = smem_tiled_copy_K55.get_thread_slice(tidx14);
    Tensor tSsK57 = smem_thr_copy_K56.partition_S(sK33);
    auto smem_tiled_copy_V58 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma45);
    auto smem_thr_copy_V59 = smem_tiled_copy_V58.get_thread_slice(tidx14);
    Tensor tOsVt60 = smem_thr_copy_V59.partition_S(sVt35);
    Tensor cQ61 = make_identity_tensor(make_shape(size<0>(sQ32), size<1>(sQ32)));
    Tensor cKV62 = make_identity_tensor(make_shape(size<0>(sK33), size<1>(sK33)));
    Tensor tQcQ63 = gmem_thr_copy_QKV38.partition_S(cQ61);
    Tensor tKVcKV64 = gmem_thr_copy_QKV38.partition_S(cKV62);
    Tensor tQpQ65 = make_tensor<bool>(make_shape(size<2>(tQsQ40)));
    Tensor tKVpKV66 = make_tensor<bool>(make_shape(size<2>(tKsK42)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tQpQ65); ++k) {
            tQpQ65(k) = get<1>(tQcQ63(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV66); ++k) {
            tKVpKV66(k) = get<1>(tKVcKV64(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tQgQ39, tQsQ40, tQcQ63, tQpQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
    if (Kernel_traits0::Is_Q_in_regs) {
        cute::cp_async_fence();
    }
    if (Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view104 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view104))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view104);
        asm ("bar.sync 1,128;");
        ;
    }
    int n_block67 = n_block_max23 - 1;
    flash::copy<Is_even_MN5, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67), tKsK42, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
    cute::cp_async_fence();
    if (Kernel_traits0::Is_Q_in_regs && ! Kernel_traits0::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        asm ("bar.sync 1,128;");
        ;
        Tensor tSrQ_copy_view105 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view105))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view105);
    }
    clear(acc_o51);
    flash::Softmax<2 * size<1>(acc_o51)> softmax68;
    const float alibi_slope69 = !Has_alibi4 || params9.alibi_slopes_ptr == nullptr ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal2, Is_local3, Has_alibi4> mask70(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope69);
    constexpr int n_masking_steps71 = (!Is_causal2 && !Is_local3) ? 1 : ((Is_even_MN5 && Is_causal2) ? cute::ceil_div(kBlockM15, kBlockN16) : cute::ceil_div(kBlockM15, kBlockN16) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps71; ++masking_step , --n_block67) {
        Tensor acc_s106 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s106);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        } else {
            flash::copy<Is_even_MN5, Is_even_K6, true>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
        }
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s106, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s106, params9.softcap);
        }
        mask70.template apply_mask<Is_causal2, Is_even_MN5>(acc_s106, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax68.template softmax_rescale_o<true, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2) : softmax68.template softmax_rescale_o<false, Is_causal2 || Is_local3>(acc_s106, acc_o51, params9.scale_softmax_log2);
        Tensor rP107 = flash::convert_type<Element>(acc_s106);
        int block_row_idx108 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx109 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop111 = make_fragment_like(rP107);
            cute::copy(rP107, rP_drop111);
            dropout20.template apply_dropout<true>(rP_drop111, block_row_idx108, block_col_idx109, kNWarps18);
            cute::copy(rP_drop111, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP107, block_row_idx108, block_col_idx109, kNWarps18);
        }
        Tensor tOrP110 = make_tensor(rP107.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP107.layout()));
        flash::gemm_rs(acc_o51, tOrP110, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
        if (n_masking_steps71 > 1 && n_block67 <= n_block_min22) {
            --n_block67;
            break;
        }
    }
    for (; n_block67 >= n_block_min22; --n_block67) {
        Tensor acc_s112 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s112);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        cute::cp_async_fence();
        flash::gemm<Kernel_traits0::Is_Q_in_regs>(acc_s112, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s112, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        mask70.template apply_mask<false>(acc_s112, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        softmax68.template softmax_rescale_o<false, Is_local3>(acc_s112, acc_o51, params9.scale_softmax_log2);
        Tensor rP113 = flash::convert_type<Element>(acc_s112);
        int block_row_idx114 = m_block10 * (kBlockM15 / 16) + tidx14 / 32;
        int block_col_idx115 = n_block67 * (kBlockN16 / 32);
        if (Return_softmax8) {
            Tensor rP_drop117 = make_fragment_like(rP113);
            cute::copy(rP113, rP_drop117);
            dropout20.template apply_dropout<true>(rP_drop117, block_row_idx114, block_col_idx115, kNWarps18);
            cute::copy(rP_drop117, tSgS50);
            tSgS50.data() = tSgS50.data() + (-kBlockN16);
        }
        if (Is_dropout1) {
            dropout20.apply_dropout(rP113, block_row_idx114, block_col_idx115, kNWarps18);
        }
        Tensor tOrP116 = make_tensor(rP113.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP113.layout()));
        flash::gemm_rs(acc_o51, tOrP116, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
    }
    Tensor lse72 = softmax68.template normalize_softmax_lse<Is_dropout1>(acc_o51, params9.scale_softmax, params9.rp_dropout);
    Tensor rO73 = flash::convert_type<Element>(acc_o51);
    Tensor sO74 = make_tensor(sQ32.data(), typename Kernel_traits0::SmemLayoutO{});
    auto smem_tiled_copy_O75 = make_tiled_copy_C(typename Kernel_traits0::SmemCopyAtomO{}, tiled_mma45);
    auto smem_thr_copy_O76 = smem_tiled_copy_O75.get_thread_slice(tidx14);
    Tensor taccOrO77 = smem_thr_copy_O76.retile_S(rO73);
    Tensor taccOsO78 = smem_thr_copy_O76.partition_D(sO74);
    if (Kernel_traits0::Share_Q_K_smem) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_O75, taccOrO77, taccOsO78);
    Tensor mO79 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.o_ptr) + binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.o_row_stride, params9.o_head_stride, _1{}));
    Tensor gO80 = local_tile(mO79(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor gLSE81 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN5>(params9, bidb11, bidh12, m_block10, binfo21);
    typename Kernel_traits0::GmemTiledCopyO gmem_tiled_copy_O82;
    auto gmem_thr_copy_O83 = gmem_tiled_copy_O82.get_thread_slice(tidx14);
    Tensor tOsO84 = gmem_thr_copy_O83.partition_S(sO74);
    Tensor tOgO85 = gmem_thr_copy_O83.partition_D(gO80);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrO86 = make_tensor<Element>(shape(tOgO85));
    cute::copy(gmem_tiled_copy_O82, tOsO84, tOrO86);
    Tensor caccO87 = make_identity_tensor(Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    Tensor taccOcO88 = thr_mma46.partition_C(caccO87);
    static_assert(decltype(size<0>(taccOcO88))::value == 4);
    Tensor taccOcO_row89 = logical_divide(taccOcO88, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse72) == size(taccOcO_row89))::value);
    if (get<1>(taccOcO_row89(0)) == 0) {
        for (int mi = 0; mi < size(lse72); ++mi) {
            const int row118 = get<0>(taccOcO_row89(mi));
            if (row118 < binfo21.actual_seqlen_q - m_block10 * kBlockM15) {
                gLSE81(row118) = lse72(mi);
            }
        }
    }
    Tensor cO90 = make_identity_tensor(make_shape(size<0>(sO74), size<1>(sO74)));
    Tensor tOcO91 = gmem_thr_copy_O83.partition_D(cO90);
    Tensor tOpO92 = make_tensor<bool>(make_shape(size<2>(tOgO85)));
    if (!Is_even_K6) {
        for (int k = 0; k < size(tOpO92); ++k) {
            tOpO92(k) = get<1>(tOcO91(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN5, Is_even_K6, false, false>(gmem_tiled_copy_O82, tOrO86, tOgO85, tOcO91, tOpO92, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    const int m_block129 = blockIdx.x;
    const int bidb130 = Split126 ? blockIdx.z / params128.h : blockIdx.y;
    const int bidh131 = Split126 ? blockIdx.z - bidb130 * params128.h : blockIdx.z;
    const int n_split_idx132 = Split126 ? blockIdx.y : 0;
    const int num_n_splits133 = Split126 ? gridDim.y : 1;
    using Element = typename Kernel_traits119::Element;
    using ElementAccum = typename Kernel_traits119::ElementAccum;
    using index_t = typename Kernel_traits119::index_t;
    extern char smem_134[] __attribute__((shared));
    const int tidx135 = threadIdx_x_1;
    constexpr int kBlockM136 = Kernel_traits119::kBlockM;
    constexpr int kBlockN137 = Kernel_traits119::kBlockN;
    constexpr int kHeadDim138 = Kernel_traits119::kHeadDim;
    constexpr int kNWarps139 = Kernel_traits119::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::GmemTiledCopyO, typename Kernel_traits119::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split126, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN123> binfo140(params128, bidb130);
    if (m_block129 * kBlockM136 >= binfo140.actual_seqlen_q)
        return;
    const int n_blocks_per_split141 = ((params128.seqlen_k + kBlockN137 - 1) / kBlockN137 + num_n_splits133 - 1) / num_n_splits133;
    const int n_block_min142 = !Is_local121 ? n_split_idx132 * n_blocks_per_split141 : std::max(n_split_idx132 * n_blocks_per_split141, (m_block129 * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q - params128.window_size_left) / kBlockN137);
    int n_block_max143 = std::min(cute::ceil_div(binfo140.actual_seqlen_k, kBlockN137), (n_split_idx132 + 1) * n_blocks_per_split141);
    if (Is_causal120 || Is_local121) {
        n_block_max143 = std::min(n_block_max143, cute::ceil_div((m_block129 + 1) * kBlockM136 + binfo140.actual_seqlen_k - binfo140.actual_seqlen_q + params128.window_size_right, kBlockN137));
    }
    if (n_block_min142 >= n_block_max143) {
        const index_t row_offset_o220 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
        const index_t row_offset_oaccum221 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
        const index_t row_offset_lseaccum222 = ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136;
        Tensor gOaccum223 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum221 : row_offset_o220)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
        Tensor gLSEaccum224 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum222), Shape<Int<kBlockM136>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum225;
        auto gmem_thr_copy_Oaccum226 = gmem_tiled_copy_Oaccum225.get_thread_slice(tidx135);
        Tensor tOgOaccum227 = gmem_thr_copy_Oaccum226.partition_D(gOaccum223);
        Tensor tOrOaccum228 = make_tensor<ElementO>(shape(tOgOaccum227));
        clear(tOrOaccum228);
        Tensor cO229 = make_identity_tensor(make_shape(size<0>(gOaccum223), size<1>(gOaccum223)));
        Tensor tOcO230 = gmem_thr_copy_Oaccum226.partition_D(cO229);
        Tensor tOpO231 = make_tensor<bool>(make_shape(size<2>(tOgOaccum227)));
        if (!Is_even_K124) {
            for (int k = 0; k < size(tOpO231); ++k) {
                tOpO231(k) = get<1>(tOcO230(0, 0, k)) < params128.d;
            }
        }
        flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum225, tOrOaccum228, tOgOaccum227, tOcO230, tOpO231, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
        for (int m = 0; m < size<1>(tOgOaccum227); ++m) {
            const int row232 = get<0>(tOcO230(0, m, 0));
            if (row232 < binfo140.actual_seqlen_q - m_block129 * kBlockM136 && get<1>(tOcO230(0, m, 0)) == 0) {
                gLSEaccum224(row232) = Split126 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache144 = params128.cache_batch_idx == nullptr ? bidb130 : params128.cache_batch_idx[bidb130];
    const int *block_table145 = params128.block_table == nullptr ? nullptr : params128.block_table + bidb130 * params128.block_table_batch_stride;
    const int block_table_idx146 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 / params128.page_block_size;
    const int block_table_offset147 = block_table145 == nullptr ? 0 : (n_block_max143 - 1) * kBlockN137 - block_table_idx146 * params128.page_block_size;
    const index_t row_offset_k148 = block_table145 == nullptr ? binfo140.k_offset(params128.k_batch_stride, params128.k_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride : block_table145[block_table_idx146] * params128.k_batch_stride + block_table_offset147 * params128.k_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.k_head_stride;
    const index_t row_offset_v149 = block_table145 == nullptr ? binfo140.k_offset(params128.v_batch_stride, params128.v_row_stride, bidb_cache144) + (n_block_max143 - 1) * kBlockN137 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride : block_table145[block_table_idx146] * params128.v_batch_stride + block_table_offset147 * params128.v_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.v_head_stride;
    Tensor mQ150 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.q_ptr) + binfo140.q_offset(params128.q_batch_stride, params128.q_row_stride, bidb130)), make_shape(binfo140.actual_seqlen_q, params128.h, params128.d), make_stride(params128.q_row_stride, params128.q_head_stride, _1{}));
    Tensor gQ151 = local_tile(mQ150(_, bidh131, _), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_coord(m_block129, 0));
    Tensor gK152 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.k_ptr) + row_offset_k148), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.k_row_stride, _1{}));
    Tensor gV153 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.v_ptr) + row_offset_v149), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.v_row_stride, _1{}));
    Tensor sQ154 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_134)), typename Kernel_traits119::SmemLayoutQ{});
    Tensor sK155 = make_tensor(sQ154.data() + size(sQ154), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sV156 = make_tensor(sK155.data() + size(sK155), typename Kernel_traits119::SmemLayoutKV{});
    Tensor sVt157 = make_tensor(sV156.data(), typename Kernel_traits119::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle158 = make_tensor(sV156.data().get(), typename Kernel_traits119::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits119::GmemTiledCopyQKV gmem_tiled_copy_QKV159;
    auto gmem_thr_copy_QKV160 = gmem_tiled_copy_QKV159.get_thread_slice(tidx135);
    Tensor tQgQ161 = gmem_thr_copy_QKV160.partition_S(gQ151);
    Tensor tQsQ162 = gmem_thr_copy_QKV160.partition_D(sQ154);
    Tensor tKgK163 = gmem_thr_copy_QKV160.partition_S(gK152);
    Tensor tKsK164 = gmem_thr_copy_QKV160.partition_D(sK155);
    Tensor tVgV165 = gmem_thr_copy_QKV160.partition_S(gV153);
    Tensor tVsV166 = gmem_thr_copy_QKV160.partition_D(sV156);
    typename Kernel_traits119::TiledMma tiled_mma167;
    auto thr_mma168 = tiled_mma167.get_thread_slice(tidx135);
    Tensor tSrQ169 = thr_mma168.partition_fragment_A(sQ154);
    Tensor tSrK170 = thr_mma168.partition_fragment_B(sK155);
    Tensor tOrVt171 = thr_mma168.partition_fragment_B(sVtNoSwizzle158);
    Tensor acc_o172 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    auto smem_tiled_copy_Q173 = make_tiled_copy_A(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_Q174 = smem_tiled_copy_Q173.get_thread_slice(tidx135);
    Tensor tSsQ175 = smem_thr_copy_Q174.partition_S(sQ154);
    auto smem_tiled_copy_K176 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtom{}, tiled_mma167);
    auto smem_thr_copy_K177 = smem_tiled_copy_K176.get_thread_slice(tidx135);
    Tensor tSsK178 = smem_thr_copy_K177.partition_S(sK155);
    auto smem_tiled_copy_V179 = make_tiled_copy_B(typename Kernel_traits119::SmemCopyAtomTransposed{}, tiled_mma167);
    auto smem_thr_copy_V180 = smem_tiled_copy_V179.get_thread_slice(tidx135);
    Tensor tOsVt181 = smem_thr_copy_V180.partition_S(sVt157);
    Tensor cQ182 = make_identity_tensor(make_shape(size<0>(sQ154), size<1>(sQ154)));
    Tensor cKV183 = make_identity_tensor(make_shape(size<0>(sK155), size<1>(sK155)));
    Tensor tQcQ184 = gmem_thr_copy_QKV160.partition_S(cQ182);
    Tensor tKVcKV185 = gmem_thr_copy_QKV160.partition_S(cKV183);
    Tensor tQpQ186 = make_tensor<bool>(make_shape(size<2>(tQsQ162)));
    Tensor tKVpKV187 = make_tensor<bool>(make_shape(size<2>(tKsK164)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tQpQ186); ++k) {
            tQpQ186(k) = get<1>(tQcQ184(0, 0, k)) < params128.d;
        }
        for (int k = 0; k < size(tKVpKV187); ++k) {
            tKVpKV187(k) = get<1>(tKVcKV185(0, 0, k)) < params128.d;
        }
    }
    typename Kernel_traits119::GmemTiledCopyRotcossin gmem_tiled_copy_rotary188;
    auto gmem_thr_copy_rotary189 = gmem_tiled_copy_rotary188.get_thread_slice(tidx135);
    typename Kernel_traits119::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont190;
    auto gmem_thr_copy_rotary_cont191 = gmem_tiled_copy_rotary_cont190.get_thread_slice(tidx135);
    if (Append_KV127) {
        const index_t row_offset_cossin233 = ((n_block_max143 - 1) * kBlockN137 + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130])) * (params128.rotary_dim / 2);
        Tensor gCos234 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSin235 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138 / 2>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gCosCont236 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor gSinCont237 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin233), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.rotary_dim / 2, _1{}));
        Tensor tRgCos238 = gmem_thr_copy_rotary189.partition_S(gCos234);
        Tensor tRgSin239 = gmem_thr_copy_rotary189.partition_S(gSin235);
        Tensor tRgCosCont240 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont236);
        Tensor tRgSinCont241 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont237);
        const index_t row_offset_knew242 = bidb130 * params128.knew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.knew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.knew_head_stride;
        const index_t row_offset_vnew243 = bidb130 * params128.vnew_batch_stride + ((n_block_max143 - 1) * kBlockN137) * params128.vnew_row_stride + (bidh131 / params128.h_h_k_ratio) * params128.vnew_head_stride;
        Tensor gKnew244 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.knew_ptr) + row_offset_knew242 - binfo140.seqlen_k_cache * params128.knew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.knew_row_stride, _1{}));
        Tensor gVnew245 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.vnew_ptr) + row_offset_vnew243 - binfo140.seqlen_k_cache * params128.vnew_row_stride), Shape<Int<kBlockN137>, Int<kHeadDim138>>{}, make_stride(params128.vnew_row_stride, _1{}));
        Tensor tKgKnew246 = gmem_thr_copy_QKV160.partition_S(gKnew244);
        Tensor tVgVnew247 = gmem_thr_copy_QKV160.partition_S(gVnew245);
        const int n_block_copy_min248 = std::max(n_block_min142, binfo140.seqlen_k_cache / kBlockN137);
        auto tKgK_data249 = tKgK163.data();
        auto tVgV_data250 = tVgV165.data();
        for (int n_block = n_block_max143 - 1; n_block >= n_block_copy_min248; n_block--) {
            flash::copy_w_min_idx<Is_even_K124>(tVgVnew247, tVgV165, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            tVgVnew247.data() = tVgVnew247.data() + (-int(kBlockN137 * params128.vnew_row_stride));
            if (params128.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K124>(tKgKnew246, tKgK163, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137);
            } else {
                if (params128.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCos238, tRgSin239, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCos238.data() = tRgCos238.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSin239.data() = tRgSin239.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K124, false>(tKgKnew246, tKgK163, tRgCosCont240, tRgSinCont241, tKVcKV185, binfo140.actual_seqlen_k - n_block * kBlockN137, binfo140.seqlen_k_cache - n_block * kBlockN137, params128.d, params128.rotary_dim);
                    tRgCosCont240.data() = tRgCosCont240.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                    tRgSinCont241.data() = tRgSinCont241.data() + (-int(kBlockN137 * params128.rotary_dim / 2));
                }
            }
            tKgKnew246.data() = tKgKnew246.data() + (-int(kBlockN137 * params128.knew_row_stride));
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                if (n_block > n_block_copy_min248) {
                    const int block_table_idx_cur251 = n_block * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_cur252 = n_block * kBlockN137 - block_table_idx_cur251 * params128.page_block_size;
                    const int block_table_idx_next253 = (n_block - 1) * kBlockN137 / params128.page_block_size;
                    const int block_table_offset_next254 = (n_block - 1) * kBlockN137 - block_table_idx_next253 * params128.page_block_size;
                    const int table_diff255 = block_table145[block_table_idx_next253] - block_table145[block_table_idx_cur251];
                    const int offset_diff256 = block_table_offset_next254 - block_table_offset_cur252;
                    tVgV165.data() = tVgV165.data() + table_diff255 * params128.v_batch_stride + offset_diff256 * params128.v_row_stride;
                    tKgK163.data() = tKgK163.data() + table_diff255 * params128.k_batch_stride + offset_diff256 * params128.k_row_stride;
                }
            }
        }
        asm ("bar.sync 2,128;");
        ;
        tKgK163.data() = tKgK_data249;
        tVgV165.data() = tVgV_data250;
    }
    if (!Append_KV127 || params128.rotary_dim == 0) {
        flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tQgQ161, tQsQ162, tQcQ184, tQpQ186, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
    } else {
        const index_t row_offset_cossin257 = (binfo140.seqlen_k_cache + (params128.leftpad_k == nullptr ? 0 : params128.leftpad_k[bidb130]) + (Is_causal120 || Is_local121 ? m_block129 * kBlockM136 : 0)) * (params128.rotary_dim / 2);
        Tensor gCos258 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSin259 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138 / 2>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont260 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_cos_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont261 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params128.rotary_sin_ptr) + row_offset_cossin257), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Is_causal120 || Is_local121 ? params128.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos262 = gmem_thr_copy_rotary189.partition_S(gCos258);
        Tensor tRgSin263 = gmem_thr_copy_rotary189.partition_S(gSin259);
        Tensor tRgCosCont264 = gmem_thr_copy_rotary_cont191.partition_S(gCosCont260);
        Tensor tRgSinCont265 = gmem_thr_copy_rotary_cont191.partition_S(gSinCont261);
        if (params128.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K124>(tQgQ161, tQsQ162, tRgCos262, tRgSin263, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K124>(tQgQ161, tQsQ162, tRgCosCont264, tRgSinCont265, tQcQ184, binfo140.actual_seqlen_q - m_block129 * kBlockM136, 0, params128.d, params128.rotary_dim);
        }
    }
    int n_block192 = n_block_max143 - 1;
    flash::copy<Is_even_MN123, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
    cute::cp_async_fence();
    clear(acc_o172);
    flash::Softmax<2 * size<1>(acc_o172)> softmax193;
    const float alibi_slope194 = !Has_alibi122 ? 0.F : reinterpret_cast<float *>(params128.alibi_slopes_ptr)[bidb130 * params128.alibi_slopes_batch_stride + bidh131] / params128.scale_softmax;
    flash::Mask<Is_causal120, Is_local121, Has_alibi122> mask195(binfo140.actual_seqlen_k, binfo140.actual_seqlen_q, params128.window_size_left, params128.window_size_right, alibi_slope194);
    constexpr int n_masking_steps196 = (!Is_causal120 && !Is_local121) ? 1 : ((Is_even_MN123 && Is_causal120) ? cute::ceil_div(kBlockM136, kBlockN137) : cute::ceil_div(kBlockM136, kBlockN137) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps196; ++masking_step , --n_block192) {
        Tensor acc_s266 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s266);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (masking_step > 0) {
            if (block_table145 == nullptr) {
                tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
            } else {
                const int block_table_idx_cur269 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur270 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur269 * params128.page_block_size;
                const int block_table_idx_next271 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next272 = n_block192 * kBlockN137 - block_table_idx_next271 * params128.page_block_size;
                tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next271] - block_table145[block_table_idx_cur269]) * params128.v_batch_stride + (block_table_offset_next272 - block_table_offset_cur270) * params128.v_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        } else {
            flash::copy<Is_even_MN123, Is_even_K124, true>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187, binfo140.actual_seqlen_k - n_block192 * kBlockN137);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s266, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s266, params128.softcap);
        }
        mask195.template apply_mask<Is_causal120, Is_even_MN123>(acc_s266, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur273 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur274 = n_block192 * kBlockN137 - block_table_idx_cur273 * params128.page_block_size;
                const int block_table_idx_next275 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next276 = (n_block192 - 1) * kBlockN137 - block_table_idx_next275 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next275] - block_table145[block_table_idx_cur273]) * params128.k_batch_stride + (block_table_offset_next276 - block_table_offset_cur274) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax193.template softmax_rescale_o<true, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2) : softmax193.template softmax_rescale_o<false, Is_causal120 || Is_local121 || !Is_even_MN123>(acc_s266, acc_o172, params128.scale_softmax_log2);
        Tensor rP267 = flash::convert_type<Element>(acc_s266);
        Tensor tOrP268 = make_tensor(rP267.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP267.layout()));
        flash::gemm_rs(acc_o172, tOrP268, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
        if (n_masking_steps196 > 1 && n_block192 <= n_block_min142) {
            --n_block192;
            break;
        }
    }
    for (; n_block192 >= n_block_min142; --n_block192) {
        Tensor acc_s277 = partition_fragment_C(tiled_mma167, Shape<Int<kBlockM136>, Int<kBlockN137>>{});
        clear(acc_s277);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (block_table145 == nullptr) {
            tVgV165.data() = tVgV165.data() + (-int(kBlockN137 * params128.v_row_stride));
        } else {
            const int block_table_idx_cur280 = (n_block192 + 1) * kBlockN137 / params128.page_block_size;
            const int block_table_offset_cur281 = (n_block192 + 1) * kBlockN137 - block_table_idx_cur280 * params128.page_block_size;
            const int block_table_idx_next282 = n_block192 * kBlockN137 / params128.page_block_size;
            const int block_table_offset_next283 = n_block192 * kBlockN137 - block_table_idx_next282 * params128.page_block_size;
            tVgV165.data() = tVgV165.data() + (block_table145[block_table_idx_next282] - block_table145[block_table_idx_cur280]) * params128.v_batch_stride + (block_table_offset_next283 - block_table_offset_cur281) * params128.v_row_stride;
        }
        flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tVgV165, tVsV166, tKVcKV185, tKVpKV187);
        cute::cp_async_fence();
        flash::gemm(acc_s277, tSrQ169, tSrK170, tSsQ175, tSsK178, tiled_mma167, smem_tiled_copy_Q173, smem_tiled_copy_K176, smem_thr_copy_Q174, smem_thr_copy_K177);
        if (Is_softcap125) {
            fused::apply_softcap(acc_s277, params128.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block192 > n_block_min142) {
            if (block_table145 == nullptr) {
                tKgK163.data() = tKgK163.data() + (-int(kBlockN137 * params128.k_row_stride));
            } else {
                const int block_table_idx_cur284 = n_block192 * kBlockN137 / params128.page_block_size;
                const int block_table_offset_cur285 = n_block192 * kBlockN137 - block_table_idx_cur284 * params128.page_block_size;
                const int block_table_idx_next286 = (n_block192 - 1) * kBlockN137 / params128.page_block_size;
                const int block_table_offset_next287 = (n_block192 - 1) * kBlockN137 - block_table_idx_next286 * params128.page_block_size;
                tKgK163.data() = tKgK163.data() + (block_table145[block_table_idx_next286] - block_table145[block_table_idx_cur284]) * params128.k_batch_stride + (block_table_offset_next287 - block_table_offset_cur285) * params128.k_row_stride;
            }
            flash::copy<true, Is_even_K124>(gmem_tiled_copy_QKV159, tKgK163, tKsK164, tKVcKV185, tKVpKV187);
            cute::cp_async_fence();
        }
        mask195.template apply_mask<false>(acc_s277, n_block192 * kBlockN137, m_block129 * kBlockM136 + (tidx135 / 32) * 16 + (tidx135 % 32) / 4, kNWarps139 * 16);
        softmax193.template softmax_rescale_o<false, Is_local121>(acc_s277, acc_o172, params128.scale_softmax_log2);
        Tensor rP278 = flash::convert_type<Element>(acc_s277);
        Tensor tOrP279 = make_tensor(rP278.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits119::TiledMma>(rP278.layout()));
        flash::gemm_rs(acc_o172, tOrP279, tOrVt171, tOsVt181, tiled_mma167, smem_tiled_copy_V179, smem_thr_copy_V180);
    }
    Tensor lse197 = softmax193.template normalize_softmax_lse<false, Split126>(acc_o172, params128.scale_softmax);
    Tensor sOaccum198 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_134)), typename Kernel_traits119::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split126, typename Kernel_traits119::SmemCopyAtomO, typename Kernel_traits119::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum199 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma167);
    auto smem_thr_copy_Oaccum200 = smem_tiled_copy_Oaccum199.get_thread_slice(tidx135);
    Tensor rO201 = flash::convert_type<ElementO>(acc_o172);
    Tensor taccOrOaccum202 = smem_thr_copy_Oaccum200.retile_S(rO201);
    Tensor taccOsOaccum203 = smem_thr_copy_Oaccum200.partition_D(sOaccum198);
    if (Split126) {
        asm ("bar.sync 2,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum199, taccOrOaccum202, taccOsOaccum203);
    const index_t row_offset_o204 = binfo140.q_offset(params128.o_batch_stride, params128.o_row_stride, bidb130) + m_block129 * kBlockM136 * params128.o_row_stride + bidh131 * params128.o_head_stride;
    const index_t row_offset_oaccum205 = (((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q + m_block129 * kBlockM136) * params128.d_rounded;
    const index_t row_offset_lseaccum206 = (Split126 || !params128.unpadded_lse ? ((n_split_idx132 * params128.b + bidb130) * params128.h + bidh131) * params128.seqlen_q : bidh131 * params128.total_q + binfo140.q_offset(params128.seqlen_q, 1, bidb130)) + m_block129 * kBlockM136;
    Tensor gOaccum207 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split126 ? params128.oaccum_ptr : params128.o_ptr) + (Split126 ? row_offset_oaccum205 : row_offset_o204)), Shape<Int<kBlockM136>, Int<kHeadDim138>>{}, make_stride(Split126 ? kHeadDim138 : params128.o_row_stride, _1{}));
    Tensor gLSEaccum208 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split126 ? params128.softmax_lseaccum_ptr : params128.softmax_lse_ptr) + row_offset_lseaccum206), Shape<Int<kBlockM136>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum209;
    auto gmem_thr_copy_Oaccum210 = gmem_tiled_copy_Oaccum209.get_thread_slice(tidx135);
    Tensor tOsOaccum211 = gmem_thr_copy_Oaccum210.partition_S(sOaccum198);
    Tensor tOgOaccum212 = gmem_thr_copy_Oaccum210.partition_D(gOaccum207);
    asm ("bar.sync 2,128;");
    ;
    Tensor tOrOaccum213 = make_tensor<ElementO>(shape(tOgOaccum212));
    cute::copy(gmem_tiled_copy_Oaccum209, tOsOaccum211, tOrOaccum213);
    Tensor caccO214 = make_identity_tensor(Shape<Int<kBlockM136>, Int<kHeadDim138>>{});
    Tensor taccOcO215 = thr_mma168.partition_C(caccO214);
    static_assert(decltype(size<0>(taccOcO215))::value == 4);
    Tensor taccOcO_row216 = logical_divide(taccOcO215, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse197) == size(taccOcO_row216))::value);
    if (get<1>(taccOcO_row216(0)) == 0) {
        for (int mi = 0; mi < size(lse197); ++mi) {
            const int row288 = get<0>(taccOcO_row216(mi));
            if (row288 < binfo140.actual_seqlen_q - m_block129 * kBlockM136) {
                gLSEaccum208(row288) = lse197(mi);
            }
        }
    }
    Tensor cO217 = make_identity_tensor(make_shape(size<0>(sOaccum198), size<1>(sOaccum198)));
    Tensor tOcO218 = gmem_thr_copy_Oaccum210.partition_D(cO217);
    Tensor tOpO219 = make_tensor<bool>(make_shape(size<2>(tOgOaccum212)));
    if (!Is_even_K124) {
        for (int k = 0; k < size(tOpO219); ++k) {
            tOpO219(k) = get<1>(tOcO218(0, 0, k)) < params128.d;
        }
    }
    flash::copy<Is_even_MN123, Is_even_K124, false, false>(gmem_tiled_copy_Oaccum209, tOrOaccum213, tOgOaccum212, tOcO218, tOpO219, binfo140.actual_seqlen_q - m_block129 * kBlockM136);
}
}
*/