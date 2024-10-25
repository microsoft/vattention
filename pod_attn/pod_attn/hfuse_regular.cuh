/*template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8>
 __global__ __launch_bounds__(128, 2) void flash_fwd_kernel_flash_fwd_kernel_fused_kernel_vfuse_lb_idx_0(const Flash_fwd_params params9const Flash_fwd_params params9)
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
}
template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8>
 __global__ __launch_bounds__(128, 0) void flash_fwd_kernel_flash_fwd_kernel_fused_kernel_vfuse_idx_0(const Flash_fwd_params params9const Flash_fwd_params params9)
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
}*/
template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits01, bool Is_dropout11, bool Is_causal21, bool Is_local31, bool Has_alibi41, bool Is_even_MN51, bool Is_even_K61, bool Is_softcap71, bool Return_softmax81>
 __global__ __launch_bounds__(256) void flash_fwd_kernel_flash_fwd_kernel_fused_kernel_hfuse_idx_0(const Flash_fwd_params params9, const Flash_fwd_params params91)
 {
    extern __shared__ char smem[];
if ((/*(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && */(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    //unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    //unsigned int blockDim_y_0 = 1;
    //unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    //unsigned int blockDim_z_0 = 1;
    //unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
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
    //unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    //unsigned int blockDim_y_0 = 1;
    //unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    //unsigned int blockDim_z_0 = 1;
    //unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    static_assert(!(Is_causal21 && Is_local31));
    const int num_mblocks = (params91.seqlen_q + Kernel_traits01::kBlockM - 1) / Kernel_traits01::kBlockM;
    if(blockIdx.x >= num_mblocks * params91.b * params91.h){
        return;
    }
    const int m_block10 = (blockIdx.x % num_mblocks);
    // The block index for the batch.
    const int bidb11 = (blockIdx.x / num_mblocks) % params91.b;
    // The block index for the head.
    const int bidh12 = (blockIdx.x / num_mblocks / params91.b) % params91.h;
    using Element = typename Kernel_traits01::Element;
    using ElementAccum = typename Kernel_traits01::ElementAccum;
    using index_t = typename Kernel_traits01::index_t;
    char *smem_13 = &smem[Kernel_traits0::kSmemSize];
    const int tidx14 = threadIdx_x_0;
    constexpr int kBlockM15 = Kernel_traits01::kBlockM;
    constexpr int kBlockN16 = Kernel_traits01::kBlockN;
    constexpr int kHeadDim17 = Kernel_traits01::kHeadDim;
    constexpr int kNWarps18 = Kernel_traits01::kNWarps;
    auto seed_offset19 = at::cuda::philox::unpack(params91.philox_args);
    flash::Dropout dropout20(std::get<0>(seed_offset19), std::get<1>(seed_offset19), params91.p_dropout_in_uint8_t, bidb11, bidh12, tidx14, params91.h);
    if (Is_dropout11 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx14 == 0) {
        params91.rng_state[0] = std::get<0>(seed_offset19);
        params91.rng_state[1] = std::get<1>(seed_offset19);
    }
    const flash::BlockInfo<!Is_even_MN51> binfo21(params91, bidb11);
    if (m_block10 * kBlockM15 >= binfo21.actual_seqlen_q)
        return;
    const int n_block_min22 = !Is_local31 ? 0 : std::max(0, (m_block10 * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params91.window_size_left) / kBlockN16);
    int n_block_max23 = cute::ceil_div(binfo21.actual_seqlen_k, kBlockN16);
    if (Is_causal21 || Is_local31) {
        n_block_max23 = std::min(n_block_max23, cute::ceil_div((m_block10 + 1) * kBlockM15 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params91.window_size_right, kBlockN16));
    }
    if ((Is_causal21 || Is_local31 || !Is_even_MN51) && n_block_max23 <= n_block_min22) {
        Tensor mO93 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params91.o_ptr) + binfo21.q_offset(params91.o_batch_stride, params91.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params91.h, params91.d), make_stride(params91.o_row_stride, params91.o_head_stride, _1{}));
        Tensor gO94 = local_tile(mO93(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
        Tensor gLSE95 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN51>(params91, bidb11, bidh12, m_block10, binfo21);
        typename Kernel_traits01::GmemTiledCopyO gmem_tiled_copy_O96;
        auto gmem_thr_copy_O97 = gmem_tiled_copy_O96.get_thread_slice(tidx14);
        Tensor tOgO98 = gmem_thr_copy_O97.partition_D(gO94);
        Tensor tOrO99 = make_tensor<Element>(shape(tOgO98));
        clear(tOrO99);
        Tensor cO100 = make_identity_tensor(make_shape(size<0>(gO94), size<1>(gO94)));
        Tensor tOcO101 = gmem_thr_copy_O97.partition_D(cO100);
        Tensor tOpO102 = make_tensor<bool>(make_shape(size<2>(tOgO98)));
        if (!Is_even_K6) {
            for (int k = 0; k < size(tOpO102); ++k) {
                tOpO102(k) = get<1>(tOcO101(0, 0, k)) < params91.d;
            }
        }
        flash::copy<Is_even_MN51, Is_even_K6, false, false>(gmem_tiled_copy_O96, tOrO99, tOgO98, tOcO101, tOpO102, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
        for (int m = 0; m < size<1>(tOgO98); ++m) {
            const int row103 = get<0>(tOcO101(0, m, 0));
            if (row103 < binfo21.actual_seqlen_q - m_block10 * kBlockM15 && get<1>(tOcO101(0, m, 0)) == 0) {
                gLSE95(row103) = (__builtin_inff());
            }
        }
        return;
    }
    const index_t row_offset_p24 = ((bidb11 * params91.h + bidh12) * params91.seqlen_q_rounded + m_block10 * kBlockM15) * params91.seqlen_k_rounded + (n_block_max23 - 1) * kBlockN16;
    Tensor mQ25 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params91.q_ptr) + binfo21.q_offset(params91.q_batch_stride, params91.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params91.h, params91.d), make_stride(params91.q_row_stride, params91.q_head_stride, _1{}));
    Tensor gQ26 = local_tile(mQ25(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor mK27 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params91.k_ptr) + binfo21.k_offset(params91.k_batch_stride, params91.k_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params91.h_k, params91.d), make_stride(params91.k_row_stride, params91.k_head_stride, _1{}));
    Tensor gK28 = local_tile(mK27(_, bidh12 / params91.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor mV29 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params91.v_ptr) + binfo21.k_offset(params91.v_batch_stride, params91.v_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_k, params91.h_k, params91.d), make_stride(params91.v_row_stride, params91.v_head_stride, _1{}));
    Tensor gV30 = local_tile(mV29(_, bidh12 / params91.h_h_k_ratio, _), Shape<Int<kBlockN16>, Int<kHeadDim17>>{}, make_coord(_, 0));
    Tensor gP31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params91.p_ptr) + row_offset_p24), Shape<Int<kBlockM15>, Int<kBlockN16>>{}, make_stride(params91.seqlen_k_rounded, _1{}));
    Tensor sQ32 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_13)), typename Kernel_traits01::SmemLayoutQ{});
    Tensor sK33 = make_tensor(sQ32.data() + (Kernel_traits01::Share_Q_K_smem ? 0 : size(sQ32)), typename Kernel_traits01::SmemLayoutKV{});
    Tensor sV34 = make_tensor(sK33.data() + size(sK33), typename Kernel_traits01::SmemLayoutKV{});
    Tensor sVt35 = make_tensor(sV34.data(), typename Kernel_traits01::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle36 = make_tensor(sV34.data().get(), typename Kernel_traits01::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits01::GmemTiledCopyQKV gmem_tiled_copy_QKV37;
    auto gmem_thr_copy_QKV38 = gmem_tiled_copy_QKV37.get_thread_slice(tidx14);
    Tensor tQgQ39 = gmem_thr_copy_QKV38.partition_S(gQ26);
    Tensor tQsQ40 = gmem_thr_copy_QKV38.partition_D(sQ32);
    Tensor tKgK41 = gmem_thr_copy_QKV38.partition_S(gK28);
    Tensor tKsK42 = gmem_thr_copy_QKV38.partition_D(sK33);
    Tensor tVgV43 = gmem_thr_copy_QKV38.partition_S(gV30);
    Tensor tVsV44 = gmem_thr_copy_QKV38.partition_D(sV34);
    typename Kernel_traits01::TiledMma tiled_mma45;
    auto thr_mma46 = tiled_mma45.get_thread_slice(tidx14);
    Tensor tSrQ47 = thr_mma46.partition_fragment_A(sQ32);
    Tensor tSrK48 = thr_mma46.partition_fragment_B(sK33);
    Tensor tOrVt49 = thr_mma46.partition_fragment_B(sVtNoSwizzle36);
    Tensor tSgS50 = thr_mma46.partition_C(gP31);
    Tensor acc_o51 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kHeadDim17>>{});
    auto smem_tiled_copy_Q52 = make_tiled_copy_A(typename Kernel_traits01::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_Q53 = smem_tiled_copy_Q52.get_thread_slice(tidx14);
    Tensor tSsQ54 = smem_thr_copy_Q53.partition_S(sQ32);
    auto smem_tiled_copy_K55 = make_tiled_copy_B(typename Kernel_traits01::SmemCopyAtom{}, tiled_mma45);
    auto smem_thr_copy_K56 = smem_tiled_copy_K55.get_thread_slice(tidx14);
    Tensor tSsK57 = smem_thr_copy_K56.partition_S(sK33);
    auto smem_tiled_copy_V58 = make_tiled_copy_B(typename Kernel_traits01::SmemCopyAtomTransposed{}, tiled_mma45);
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
            tQpQ65(k) = get<1>(tQcQ63(0, 0, k)) < params91.d;
        }
        for (int k = 0; k < size(tKVpKV66); ++k) {
            tKVpKV66(k) = get<1>(tKVcKV64(0, 0, k)) < params91.d;
        }
    }
    flash::copy<Is_even_MN51, Is_even_K6>(gmem_tiled_copy_QKV37, tQgQ39, tQsQ40, tQcQ63, tQpQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
    if (Kernel_traits01::Is_Q_in_regs) {
        cute::cp_async_fence();
    }
    if (Kernel_traits01::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        Tensor tSrQ_copy_view104 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view104))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view104);
        asm ("bar.sync 2,128;");
        ;
    }
    int n_block67 = n_block_max23 - 1;
    flash::copy<Is_even_MN51, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67), tKsK42, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
    cute::cp_async_fence();
    if (Kernel_traits01::Is_Q_in_regs && ! Kernel_traits01::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        asm ("bar.sync 2,128;");
        ;
        Tensor tSrQ_copy_view105 = smem_thr_copy_Q53.retile_D(tSrQ47);
        static_assert(decltype(size<1>(tSsQ54) == size<1>(tSrQ_copy_view105))::value);
        cute::copy(smem_tiled_copy_Q52, tSsQ54, tSrQ_copy_view105);
    }
    clear(acc_o51);
    flash::Softmax<2 * size<1>(acc_o51)> softmax68;
    const float alibi_slope69 = !Has_alibi4 || params91.alibi_slopes_ptr == nullptr ? 0.F : reinterpret_cast<float *>(params91.alibi_slopes_ptr)[bidb11 * params91.alibi_slopes_batch_stride + bidh12] / params91.scale_softmax;
    flash::Mask<Is_causal21, Is_local31, Has_alibi4> mask70(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params91.window_size_left, params91.window_size_right, alibi_slope69);
    constexpr int n_masking_steps71 = (!Is_causal21 && !Is_local31) ? 1 : ((Is_even_MN51 && Is_causal21) ? cute::ceil_div(kBlockM15, kBlockN16) : cute::ceil_div(kBlockM15, kBlockN16) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps71; ++masking_step , --n_block67) {
        Tensor acc_s106 = partition_fragment_C(tiled_mma45, Shape<Int<kBlockM15>, Int<kBlockN16>>{});
        clear(acc_s106);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (masking_step > 0) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        } else {
            flash::copy<Is_even_MN51, Is_even_K6, true>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66, binfo21.actual_seqlen_k - n_block67 * kBlockN16);
        }
        cute::cp_async_fence();
        flash::gemm<Kernel_traits01::Is_Q_in_regs>(acc_s106, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s106, params91.softcap);
        }
        mask70.template apply_mask<Is_causal21, Is_even_MN51>(acc_s106, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax68.template softmax_rescale_o<true, Is_causal21 || Is_local31>(acc_s106, acc_o51, params91.scale_softmax_log2) : softmax68.template softmax_rescale_o<false, Is_causal21 || Is_local31>(acc_s106, acc_o51, params91.scale_softmax_log2);
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
        if (Is_dropout11) {
            dropout20.apply_dropout(rP107, block_row_idx108, block_col_idx109, kNWarps18);
        }
        Tensor tOrP110 = make_tensor(rP107.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits01::TiledMma>(rP107.layout()));
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
        asm ("bar.sync 2,128;");
        ;
        flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tVgV43(_, _, _, n_block67), tVsV44, tKVcKV64, tKVpKV66);
        cute::cp_async_fence();
        flash::gemm<Kernel_traits01::Is_Q_in_regs>(acc_s112, tSrQ47, tSrK48, tSsQ54, tSsK57, tiled_mma45, smem_tiled_copy_Q52, smem_tiled_copy_K55, smem_thr_copy_Q53, smem_thr_copy_K56);
        if (Is_softcap7) {
            fused::apply_softcap(acc_s112, params91.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block67 > n_block_min22) {
            flash::copy<true, Is_even_K6>(gmem_tiled_copy_QKV37, tKgK41(_, _, _, n_block67 - 1), tKsK42, tKVcKV64, tKVpKV66);
            cute::cp_async_fence();
        }
        mask70.template apply_mask<false>(acc_s112, n_block67 * kBlockN16, m_block10 * kBlockM15 + (tidx14 / 32) * 16 + (tidx14 % 32) / 4, kNWarps18 * 16);
        softmax68.template softmax_rescale_o<false, Is_local31>(acc_s112, acc_o51, params91.scale_softmax_log2);
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
        if (Is_dropout11) {
            dropout20.apply_dropout(rP113, block_row_idx114, block_col_idx115, kNWarps18);
        }
        Tensor tOrP116 = make_tensor(rP113.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits01::TiledMma>(rP113.layout()));
        flash::gemm_rs(acc_o51, tOrP116, tOrVt49, tOsVt60, tiled_mma45, smem_tiled_copy_V58, smem_thr_copy_V59);
    }
    Tensor lse72 = softmax68.template normalize_softmax_lse<Is_dropout11>(acc_o51, params91.scale_softmax, params91.rp_dropout);
    Tensor rO73 = flash::convert_type<Element>(acc_o51);
    Tensor sO74 = make_tensor(sQ32.data(), typename Kernel_traits01::SmemLayoutO{});
    auto smem_tiled_copy_O75 = make_tiled_copy_C(typename Kernel_traits01::SmemCopyAtomO{}, tiled_mma45);
    auto smem_thr_copy_O76 = smem_tiled_copy_O75.get_thread_slice(tidx14);
    Tensor taccOrO77 = smem_thr_copy_O76.retile_S(rO73);
    Tensor taccOsO78 = smem_thr_copy_O76.partition_D(sO74);
    if (Kernel_traits01::Share_Q_K_smem) {
        asm ("bar.sync 2,128;");
        ;
    }
    cute::copy(smem_tiled_copy_O75, taccOrO77, taccOsO78);
    Tensor mO79 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params91.o_ptr) + binfo21.q_offset(params91.o_batch_stride, params91.o_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params91.h, params91.d), make_stride(params91.o_row_stride, params91.o_head_stride, _1{}));
    Tensor gO80 = local_tile(mO79(_, bidh12, _), Shape<Int<kBlockM15>, Int<kHeadDim17>>{}, make_coord(m_block10, 0));
    Tensor gLSE81 = fused::get_lse_tile<ElementAccum, Flash_fwd_params, kBlockM15, Is_even_MN51>(params91, bidb11, bidh12, m_block10, binfo21);
    typename Kernel_traits01::GmemTiledCopyO gmem_tiled_copy_O82;
    auto gmem_thr_copy_O83 = gmem_tiled_copy_O82.get_thread_slice(tidx14);
    Tensor tOsO84 = gmem_thr_copy_O83.partition_S(sO74);
    Tensor tOgO85 = gmem_thr_copy_O83.partition_D(gO80);
    asm ("bar.sync 2,128;");
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
            tOpO92(k) = get<1>(tOcO91(0, 0, k)) < params91.d;
        }
    }
    flash::copy<Is_even_MN51, Is_even_K6, false, false>(gmem_tiled_copy_O82, tOrO86, tOgO85, tOcO91, tOpO92, binfo21.actual_seqlen_q - m_block10 * kBlockM15);
}
}
/*
template <typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8, typename Kernel_traits0, bool Is_dropout1, bool Is_causal2, bool Is_local3, bool Has_alibi4, bool Is_even_MN5, bool Is_even_K6, bool Is_softcap7, bool Return_softmax8>
 __global__ __launch_bounds__(256, 2) void flash_fwd_kernel_flash_fwd_kernel_fused_kernel_hfuse_lb_idx_0(const Flash_fwd_params params9, const Flash_fwd_params params91)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
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
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
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
}
*/