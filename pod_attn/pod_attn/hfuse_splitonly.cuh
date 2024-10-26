/*
template <typename Kernel_traits0, bool Is_causal1, bool Is_local2, bool Has_alibi3, bool Is_even_MN4, bool Is_even_K5, bool Is_softcap6, bool Split7, bool Append_KV8, typename Kernel_traits0, bool Is_causal1, bool Is_local2, bool Has_alibi3, bool Is_even_MN4, bool Is_even_K5, bool Is_softcap6, bool Split7, bool Append_KV8>
 __global__ __launch_bounds__(128, 2) void flash_fwd_splitkv_kernel_flash_fwd_splitkv_kernel_fused_kernel_vfuse_lb_idx_0(const Flash_fwd_params params9const Flash_fwd_params params9)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int m_block10 = blockIdx.x;
    const int bidb11 = Split7 ? blockIdx.z / params9.h : blockIdx.y;
    const int bidh12 = Split7 ? blockIdx.z - bidb11 * params9.h : blockIdx.z;
    const int n_split_idx13 = Split7 ? blockIdx.y : 0;
    const int num_n_splits14 = Split7 ? gridDim.y : 1;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_15[] __attribute__((shared));
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits0::kBlockM;
    constexpr int kBlockN18 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits0::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::GmemTiledCopyO, typename Kernel_traits0::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split7, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN4> binfo21(params9, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params9.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local2 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal1 || Is_local2) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split7 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params9.cache_batch_idx == nullptr ? bidb11 : params9.cache_batch_idx[bidb11];
    const int *block_table26 = params9.block_table == nullptr ? nullptr : params9.block_table + bidb11 * params9.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params9.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params9.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride : block_table26[block_table_idx27] * params9.k_batch_stride + block_table_offset28 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride : block_table26[block_table_idx27] * params9.v_batch_stride + block_table_offset28 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits0::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params9.d;
        }
    }
    typename Kernel_traits0::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits0::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV8) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11])) * (params9.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params9.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.knew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params9.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.vnew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params9.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params9.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params9.vnew_row_stride));
            if (params9.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params9.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params9.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params9.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params9.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params9.v_batch_stride + offset_diff137 * params9.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params9.k_batch_stride + offset_diff137 * params9.k_row_stride;
                }
            }
        }
        asm ("bar.sync 1,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV8 || params9.rotary_dim == 0) {
        flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11]) + (Is_causal1 || Is_local2 ? m_block10 * kBlockM17 : 0)) * (params9.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params9.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal1, Is_local2, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal1 && !Is_local2) ? 1 : ((Is_even_MN4 && Is_causal1) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params9.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params9.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params9.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params9.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN4, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params9.softcap);
        }
        mask76.template apply_mask<Is_causal1, Is_even_MN4>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params9.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params9.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params9.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params9.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params9.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params9.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params9.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params9.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params9.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params9.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split7>(acc_o53, params9.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits0::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::SmemCopyAtomO, typename Kernel_traits0::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split7) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
    const index_t row_offset_lseaccum87 = (Split7 || !params9.unpadded_lse ? ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q : bidh12 * params9.total_q + binfo21.q_offset(params9.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int m_block10 = blockIdx.x;
    const int bidb11 = Split7 ? blockIdx.z / params9.h : blockIdx.y;
    const int bidh12 = Split7 ? blockIdx.z - bidb11 * params9.h : blockIdx.z;
    const int n_split_idx13 = Split7 ? blockIdx.y : 0;
    const int num_n_splits14 = Split7 ? gridDim.y : 1;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_15[] __attribute__((shared));
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits0::kBlockM;
    constexpr int kBlockN18 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits0::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::GmemTiledCopyO, typename Kernel_traits0::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split7, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN4> binfo21(params9, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params9.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local2 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal1 || Is_local2) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split7 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params9.cache_batch_idx == nullptr ? bidb11 : params9.cache_batch_idx[bidb11];
    const int *block_table26 = params9.block_table == nullptr ? nullptr : params9.block_table + bidb11 * params9.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params9.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params9.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride : block_table26[block_table_idx27] * params9.k_batch_stride + block_table_offset28 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride : block_table26[block_table_idx27] * params9.v_batch_stride + block_table_offset28 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits0::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params9.d;
        }
    }
    typename Kernel_traits0::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits0::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV8) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11])) * (params9.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params9.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.knew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params9.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.vnew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params9.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params9.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params9.vnew_row_stride));
            if (params9.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params9.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params9.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params9.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params9.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params9.v_batch_stride + offset_diff137 * params9.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params9.k_batch_stride + offset_diff137 * params9.k_row_stride;
                }
            }
        }
        asm ("bar.sync 1,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV8 || params9.rotary_dim == 0) {
        flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11]) + (Is_causal1 || Is_local2 ? m_block10 * kBlockM17 : 0)) * (params9.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params9.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal1, Is_local2, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal1 && !Is_local2) ? 1 : ((Is_even_MN4 && Is_causal1) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params9.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params9.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params9.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params9.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN4, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params9.softcap);
        }
        mask76.template apply_mask<Is_causal1, Is_even_MN4>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params9.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params9.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params9.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params9.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params9.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params9.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params9.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params9.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params9.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params9.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split7>(acc_o53, params9.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits0::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::SmemCopyAtomO, typename Kernel_traits0::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split7) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
    const index_t row_offset_lseaccum87 = (Split7 || !params9.unpadded_lse ? ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q : bidh12 * params9.total_q + binfo21.q_offset(params9.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
}
template <typename Kernel_traits0, bool Is_causal1, bool Is_local2, bool Has_alibi3, bool Is_even_MN4, bool Is_even_K5, bool Is_softcap6, bool Split7, bool Append_KV8, typename Kernel_traits0, bool Is_causal1, bool Is_local2, bool Has_alibi3, bool Is_even_MN4, bool Is_even_K5, bool Is_softcap6, bool Split7, bool Append_KV8>
 __global__ __launch_bounds__(128, 0) void flash_fwd_splitkv_kernel_flash_fwd_splitkv_kernel_fused_kernel_vfuse_idx_0(const Flash_fwd_params params9const Flash_fwd_params params9)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int m_block10 = blockIdx.x;
    const int bidb11 = Split7 ? blockIdx.z / params9.h : blockIdx.y;
    const int bidh12 = Split7 ? blockIdx.z - bidb11 * params9.h : blockIdx.z;
    const int n_split_idx13 = Split7 ? blockIdx.y : 0;
    const int num_n_splits14 = Split7 ? gridDim.y : 1;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_15[] __attribute__((shared));
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits0::kBlockM;
    constexpr int kBlockN18 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits0::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::GmemTiledCopyO, typename Kernel_traits0::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split7, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN4> binfo21(params9, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params9.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local2 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal1 || Is_local2) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split7 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params9.cache_batch_idx == nullptr ? bidb11 : params9.cache_batch_idx[bidb11];
    const int *block_table26 = params9.block_table == nullptr ? nullptr : params9.block_table + bidb11 * params9.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params9.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params9.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride : block_table26[block_table_idx27] * params9.k_batch_stride + block_table_offset28 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride : block_table26[block_table_idx27] * params9.v_batch_stride + block_table_offset28 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits0::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params9.d;
        }
    }
    typename Kernel_traits0::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits0::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV8) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11])) * (params9.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params9.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.knew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params9.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.vnew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params9.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params9.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params9.vnew_row_stride));
            if (params9.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params9.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params9.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params9.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params9.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params9.v_batch_stride + offset_diff137 * params9.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params9.k_batch_stride + offset_diff137 * params9.k_row_stride;
                }
            }
        }
        asm ("bar.sync 1,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV8 || params9.rotary_dim == 0) {
        flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11]) + (Is_causal1 || Is_local2 ? m_block10 * kBlockM17 : 0)) * (params9.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params9.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal1, Is_local2, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal1 && !Is_local2) ? 1 : ((Is_even_MN4 && Is_causal1) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params9.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params9.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params9.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params9.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN4, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params9.softcap);
        }
        mask76.template apply_mask<Is_causal1, Is_even_MN4>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params9.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params9.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params9.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params9.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params9.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params9.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params9.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params9.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params9.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params9.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split7>(acc_o53, params9.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits0::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::SmemCopyAtomO, typename Kernel_traits0::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split7) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
    const index_t row_offset_lseaccum87 = (Split7 || !params9.unpadded_lse ? ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q : bidh12 * params9.total_q + binfo21.q_offset(params9.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int m_block10 = blockIdx.x;
    const int bidb11 = Split7 ? blockIdx.z / params9.h : blockIdx.y;
    const int bidh12 = Split7 ? blockIdx.z - bidb11 * params9.h : blockIdx.z;
    const int n_split_idx13 = Split7 ? blockIdx.y : 0;
    const int num_n_splits14 = Split7 ? gridDim.y : 1;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_15[] __attribute__((shared));
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits0::kBlockM;
    constexpr int kBlockN18 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits0::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::GmemTiledCopyO, typename Kernel_traits0::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split7, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN4> binfo21(params9, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params9.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local2 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal1 || Is_local2) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split7 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params9.cache_batch_idx == nullptr ? bidb11 : params9.cache_batch_idx[bidb11];
    const int *block_table26 = params9.block_table == nullptr ? nullptr : params9.block_table + bidb11 * params9.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params9.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params9.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride : block_table26[block_table_idx27] * params9.k_batch_stride + block_table_offset28 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride : block_table26[block_table_idx27] * params9.v_batch_stride + block_table_offset28 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits0::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params9.d;
        }
    }
    typename Kernel_traits0::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits0::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV8) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11])) * (params9.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params9.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.knew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params9.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.vnew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params9.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params9.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params9.vnew_row_stride));
            if (params9.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params9.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params9.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params9.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params9.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params9.v_batch_stride + offset_diff137 * params9.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params9.k_batch_stride + offset_diff137 * params9.k_row_stride;
                }
            }
        }
        asm ("bar.sync 1,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV8 || params9.rotary_dim == 0) {
        flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11]) + (Is_causal1 || Is_local2 ? m_block10 * kBlockM17 : 0)) * (params9.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params9.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal1, Is_local2, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal1 && !Is_local2) ? 1 : ((Is_even_MN4 && Is_causal1) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params9.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params9.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params9.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params9.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN4, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params9.softcap);
        }
        mask76.template apply_mask<Is_causal1, Is_even_MN4>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params9.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params9.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params9.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params9.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params9.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params9.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params9.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params9.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params9.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params9.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split7>(acc_o53, params9.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits0::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::SmemCopyAtomO, typename Kernel_traits0::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split7) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
    const index_t row_offset_lseaccum87 = (Split7 || !params9.unpadded_lse ? ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q : bidh12 * params9.total_q + binfo21.q_offset(params9.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
}*/
template <typename Kernel_traits0, bool Is_causal1, bool Is_local2, bool Has_alibi3, bool Is_even_MN4, bool Is_even_K5, bool Is_softcap6, bool Split7, bool Append_KV8, typename Kernel_traits10, bool Is_causal11, bool Is_local12, bool Has_alibi13, bool Is_even_MN14, bool Is_even_K15, bool Is_softcap16, bool Split17, bool Append_KV18>
 __global__ __launch_bounds__(256) void flash_fwd_splitkv_kernel_flash_fwd_splitkv_kernel_fused_kernel_hfuse_idx_0(const Flash_fwd_params params9, const Flash_fwd_params params19)
 {
    extern __shared__ char smem[];
if ((/*(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && */(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    //unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    //unsigned int blockDim_y_0 = 1;
    //unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    //unsigned int blockDim_z_0 = 1;
    //unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    const int num_mblocks = (params9.seqlen_q + Kernel_traits0::kBlockM - 1) / Kernel_traits0::kBlockM;
    const int linear_block_id = blockIdx.x;
    if(linear_block_id >= num_mblocks * params9.b * params9.h * params9.num_splits){
        return;
    }
    const int m_block10 = linear_block_id % num_mblocks;
    const int num_n_splits14 = params9.num_splits;
    const int n_split_idx13 = (linear_block_id / num_mblocks) % num_n_splits14;
    const int bidb11 = (linear_block_id / num_mblocks / num_n_splits14) % params9.b;
    const int bidh12 = (linear_block_id / num_mblocks / num_n_splits14 / params9.b) % params9.h;
    
    char *smem_15 = smem;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits0::kBlockM;
    constexpr int kBlockN18 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits0::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::GmemTiledCopyO, typename Kernel_traits0::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split7, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN4> binfo21(params9, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params9.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local2 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal1 || Is_local2) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split7 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params9.cache_batch_idx == nullptr ? bidb11 : params9.cache_batch_idx[bidb11];
    const int *block_table26 = params9.block_table == nullptr ? nullptr : params9.block_table + bidb11 * params9.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params9.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params9.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride : block_table26[block_table_idx27] * params9.k_batch_stride + block_table_offset28 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride : block_table26[block_table_idx27] * params9.v_batch_stride + block_table_offset28 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits0::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params9.d;
        }
    }
    typename Kernel_traits0::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits0::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV8) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11])) * (params9.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params9.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.knew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params9.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.vnew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params9.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params9.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params9.vnew_row_stride));
            if (params9.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params9.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params9.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params9.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params9.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params9.v_batch_stride + offset_diff137 * params9.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params9.k_batch_stride + offset_diff137 * params9.k_row_stride;
                }
            }
        }
        asm ("bar.sync 1,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV8 || params9.rotary_dim == 0) {
        flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11]) + (Is_causal1 || Is_local2 ? m_block10 * kBlockM17 : 0)) * (params9.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params9.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal1, Is_local2, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal1 && !Is_local2) ? 1 : ((Is_even_MN4 && Is_causal1) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params9.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params9.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params9.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params9.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN4, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params9.softcap);
        }
        mask76.template apply_mask<Is_causal1, Is_even_MN4>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params9.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params9.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params9.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params9.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params9.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params9.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params9.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params9.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params9.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params9.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split7>(acc_o53, params9.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits0::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::SmemCopyAtomO, typename Kernel_traits0::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split7) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
    const index_t row_offset_lseaccum87 = (Split7 || !params9.unpadded_lse ? ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q : bidh12 * params9.total_q + binfo21.q_offset(params9.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    //unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    //unsigned int blockDim_y_0 = 1;
    //unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    //unsigned int blockDim_z_0 = 1;
    //unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;

    const int num_mblocks = (params19.seqlen_q + Kernel_traits10::kBlockM - 1) / Kernel_traits10::kBlockM;
    const int linear_block_id = blockIdx.x;
    if(linear_block_id >= num_mblocks * params19.b * params19.h * params19.num_splits){
        return;
    }
    const int m_block10 = linear_block_id % num_mblocks;
    const int num_n_splits14 = params19.num_splits;
    const int n_split_idx13 = (linear_block_id / num_mblocks) % num_n_splits14;
    const int bidb11 = (linear_block_id / num_mblocks / num_n_splits14) % params19.b;
    const int bidh12 = (linear_block_id / num_mblocks / num_n_splits14 / params19.b) % params19.h;
    
    char *smem_15 = &smem[Kernel_traits0::kSmemSize];

    using Element = typename Kernel_traits10::Element;
    using ElementAccum = typename Kernel_traits10::ElementAccum;
    using index_t = typename Kernel_traits10::index_t;
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits10::kBlockM;
    constexpr int kBlockN18 = Kernel_traits10::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits10::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits10::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split17, typename Kernel_traits10::GmemTiledCopyO, typename Kernel_traits10::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split17, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN14> binfo21(params19, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params19.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local12 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params19.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal11 || Is_local12) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params19.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params19.o_batch_stride, params19.o_row_stride, bidb11) + m_block10 * kBlockM17 * params19.o_row_stride + bidh12 * params19.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params19.b + bidb11) * params19.h + bidh12) * params19.seqlen_q + m_block10 * kBlockM17) * params19.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params19.b + bidb11) * params19.h + bidh12) * params19.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split17 ? params19.oaccum_ptr : params19.o_ptr) + (Split17 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split17 ? kHeadDim19 : params19.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split17 ? params19.softmax_lseaccum_ptr : params19.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params19.d;
            }
        }
        flash::copy<Is_even_MN14, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split17 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params19.cache_batch_idx == nullptr ? bidb11 : params19.cache_batch_idx[bidb11];
    const int *block_table26 = params19.block_table == nullptr ? nullptr : params19.block_table + bidb11 * params19.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params19.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params19.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params19.k_batch_stride, params19.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params19.k_row_stride + (bidh12 / params19.h_h_k_ratio) * params19.k_head_stride : block_table26[block_table_idx27] * params19.k_batch_stride + block_table_offset28 * params19.k_row_stride + (bidh12 / params19.h_h_k_ratio) * params19.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params19.v_batch_stride, params19.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params19.v_row_stride + (bidh12 / params19.h_h_k_ratio) * params19.v_head_stride : block_table26[block_table_idx27] * params19.v_batch_stride + block_table_offset28 * params19.v_row_stride + (bidh12 / params19.h_h_k_ratio) * params19.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.q_ptr) + binfo21.q_offset(params19.q_batch_stride, params19.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params19.h, params19.d), make_stride(params19.q_row_stride, params19.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params19.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params19.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits10::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits10::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits10::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits10::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits10::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits10::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits10::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits10::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits10::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits10::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params19.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params19.d;
        }
    }
    typename Kernel_traits10::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits10::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV18) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params19.leftpad_k == nullptr ? 0 : params19.leftpad_k[bidb11])) * (params19.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params19.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params19.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params19.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params19.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params19.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params19.knew_row_stride + (bidh12 / params19.h_h_k_ratio) * params19.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params19.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params19.vnew_row_stride + (bidh12 / params19.h_h_k_ratio) * params19.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params19.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params19.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params19.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params19.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params19.vnew_row_stride));
            if (params19.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params19.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params19.d, params19.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params19.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params19.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params19.d, params19.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params19.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params19.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params19.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params19.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params19.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params19.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params19.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params19.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params19.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params19.v_batch_stride + offset_diff137 * params19.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params19.k_batch_stride + offset_diff137 * params19.k_row_stride;
                }
            }
        }
        asm ("bar.sync 2,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV18 || params19.rotary_dim == 0) {
        flash::copy<Is_even_MN14, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params19.leftpad_k == nullptr ? 0 : params19.leftpad_k[bidb11]) + (Is_causal11 || Is_local12 ? m_block10 * kBlockM17 : 0)) * (params19.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal11 || Is_local12 ? params19.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal11 || Is_local12 ? params19.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal11 || Is_local12 ? params19.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params19.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal11 || Is_local12 ? params19.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params19.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params19.d, params19.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params19.d, params19.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN14, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params19.alibi_slopes_ptr)[bidb11 * params19.alibi_slopes_batch_stride + bidh12] / params19.scale_softmax;
    flash::Mask<Is_causal11, Is_local12, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params19.window_size_left, params19.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal11 && !Is_local12) ? 1 : ((Is_even_MN14 && Is_causal11) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params19.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params19.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params19.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params19.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params19.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params19.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params19.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN14, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params19.softcap);
        }
        mask76.template apply_mask<Is_causal11, Is_even_MN14>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params19.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params19.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params19.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params19.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params19.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params19.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params19.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal11 || Is_local12 || !Is_even_MN14>(acc_s147, acc_o53, params19.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal11 || Is_local12 || !Is_even_MN14>(acc_s147, acc_o53, params19.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits10::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params19.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params19.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params19.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params19.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params19.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params19.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params19.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params19.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 2,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params19.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params19.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params19.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params19.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params19.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params19.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params19.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params19.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits10::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split17>(acc_o53, params19.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits10::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split17, typename Kernel_traits10::SmemCopyAtomO, typename Kernel_traits10::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split17) {
        asm ("bar.sync 2,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params19.o_batch_stride, params19.o_row_stride, bidb11) + m_block10 * kBlockM17 * params19.o_row_stride + bidh12 * params19.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params19.b + bidb11) * params19.h + bidh12) * params19.seqlen_q + m_block10 * kBlockM17) * params19.d_rounded;
    const index_t row_offset_lseaccum87 = (Split17 || !params19.unpadded_lse ? ((n_split_idx13 * params19.b + bidb11) * params19.h + bidh12) * params19.seqlen_q : bidh12 * params19.total_q + binfo21.q_offset(params19.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split17 ? params19.oaccum_ptr : params19.o_ptr) + (Split17 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split17 ? kHeadDim19 : params19.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split17 ? params19.softmax_lseaccum_ptr : params19.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 2,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params19.d;
        }
    }
    flash::copy<Is_even_MN14, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
}
/*template <typename Kernel_traits0, bool Is_causal1, bool Is_local2, bool Has_alibi3, bool Is_even_MN4, bool Is_even_K5, bool Is_softcap6, bool Split7, bool Append_KV8, typename Kernel_traits0, bool Is_causal1, bool Is_local2, bool Has_alibi3, bool Is_even_MN4, bool Is_even_K5, bool Is_softcap6, bool Split7, bool Append_KV8>
 __global__ __launch_bounds__(256, 2) void flash_fwd_splitkv_kernel_flash_fwd_splitkv_kernel_fused_kernel_hfuse_lb_idx_0(const Flash_fwd_params params9const Flash_fwd_params params9)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    const int m_block10 = blockIdx.x;
    const int bidb11 = Split7 ? blockIdx.z / params9.h : blockIdx.y;
    const int bidh12 = Split7 ? blockIdx.z - bidb11 * params9.h : blockIdx.z;
    const int n_split_idx13 = Split7 ? blockIdx.y : 0;
    const int num_n_splits14 = Split7 ? gridDim.y : 1;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_15[] __attribute__((shared));
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits0::kBlockM;
    constexpr int kBlockN18 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits0::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::GmemTiledCopyO, typename Kernel_traits0::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split7, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN4> binfo21(params9, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params9.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local2 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal1 || Is_local2) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split7 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params9.cache_batch_idx == nullptr ? bidb11 : params9.cache_batch_idx[bidb11];
    const int *block_table26 = params9.block_table == nullptr ? nullptr : params9.block_table + bidb11 * params9.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params9.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params9.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride : block_table26[block_table_idx27] * params9.k_batch_stride + block_table_offset28 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride : block_table26[block_table_idx27] * params9.v_batch_stride + block_table_offset28 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits0::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params9.d;
        }
    }
    typename Kernel_traits0::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits0::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV8) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11])) * (params9.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params9.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.knew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params9.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.vnew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params9.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params9.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params9.vnew_row_stride));
            if (params9.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params9.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params9.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params9.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params9.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params9.v_batch_stride + offset_diff137 * params9.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params9.k_batch_stride + offset_diff137 * params9.k_row_stride;
                }
            }
        }
        asm ("bar.sync 1,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV8 || params9.rotary_dim == 0) {
        flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11]) + (Is_causal1 || Is_local2 ? m_block10 * kBlockM17 : 0)) * (params9.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params9.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal1, Is_local2, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal1 && !Is_local2) ? 1 : ((Is_even_MN4 && Is_causal1) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params9.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params9.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params9.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params9.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN4, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params9.softcap);
        }
        mask76.template apply_mask<Is_causal1, Is_even_MN4>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params9.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params9.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params9.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params9.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params9.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params9.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params9.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params9.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params9.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params9.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split7>(acc_o53, params9.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits0::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::SmemCopyAtomO, typename Kernel_traits0::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split7) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
    const index_t row_offset_lseaccum87 = (Split7 || !params9.unpadded_lse ? ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q : bidh12 * params9.total_q + binfo21.q_offset(params9.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    const int m_block10 = blockIdx.x;
    const int bidb11 = Split7 ? blockIdx.z / params9.h : blockIdx.y;
    const int bidh12 = Split7 ? blockIdx.z - bidb11 * params9.h : blockIdx.z;
    const int n_split_idx13 = Split7 ? blockIdx.y : 0;
    const int num_n_splits14 = Split7 ? gridDim.y : 1;
    using Element = typename Kernel_traits0::Element;
    using ElementAccum = typename Kernel_traits0::ElementAccum;
    using index_t = typename Kernel_traits0::index_t;
    extern char smem_15[] __attribute__((shared));
    const int tidx16 = threadIdx_x_0;
    constexpr int kBlockM17 = Kernel_traits0::kBlockM;
    constexpr int kBlockN18 = Kernel_traits0::kBlockN;
    constexpr int kHeadDim19 = Kernel_traits0::kHeadDim;
    constexpr int kNWarps20 = Kernel_traits0::kNWarps;
    using GmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::GmemTiledCopyO, typename Kernel_traits0::GmemTiledCopyOaccum>;
    using ElementO = std::conditional_t<!Split7, Element, ElementAccum>;
    const flash::BlockInfo<!Is_even_MN4> binfo21(params9, bidb11);
    if (m_block10 * kBlockM17 >= binfo21.actual_seqlen_q)
        return;
    const int n_blocks_per_split22 = ((params9.seqlen_k + kBlockN18 - 1) / kBlockN18 + num_n_splits14 - 1) / num_n_splits14;
    const int n_block_min23 = !Is_local2 ? n_split_idx13 * n_blocks_per_split22 : std::max(n_split_idx13 * n_blocks_per_split22, (m_block10 * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q - params9.window_size_left) / kBlockN18);
    int n_block_max24 = std::min(cute::ceil_div(binfo21.actual_seqlen_k, kBlockN18), (n_split_idx13 + 1) * n_blocks_per_split22);
    if (Is_causal1 || Is_local2) {
        n_block_max24 = std::min(n_block_max24, cute::ceil_div((m_block10 + 1) * kBlockM17 + binfo21.actual_seqlen_k - binfo21.actual_seqlen_q + params9.window_size_right, kBlockN18));
    }
    if (n_block_min23 >= n_block_max24) {
        const index_t row_offset_o101 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
        const index_t row_offset_oaccum102 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
        const index_t row_offset_lseaccum103 = ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17;
        Tensor gOaccum104 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum102 : row_offset_o101)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
        Tensor gLSEaccum105 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum103), Shape<Int<kBlockM17>>{}, Stride<_1>{});
        GmemTiledCopyO gmem_tiled_copy_Oaccum106;
        auto gmem_thr_copy_Oaccum107 = gmem_tiled_copy_Oaccum106.get_thread_slice(tidx16);
        Tensor tOgOaccum108 = gmem_thr_copy_Oaccum107.partition_D(gOaccum104);
        Tensor tOrOaccum109 = make_tensor<ElementO>(shape(tOgOaccum108));
        clear(tOrOaccum109);
        Tensor cO110 = make_identity_tensor(make_shape(size<0>(gOaccum104), size<1>(gOaccum104)));
        Tensor tOcO111 = gmem_thr_copy_Oaccum107.partition_D(cO110);
        Tensor tOpO112 = make_tensor<bool>(make_shape(size<2>(tOgOaccum108)));
        if (!Is_even_K5) {
            for (int k = 0; k < size(tOpO112); ++k) {
                tOpO112(k) = get<1>(tOcO111(0, 0, k)) < params9.d;
            }
        }
        flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum106, tOrOaccum109, tOgOaccum108, tOcO111, tOpO112, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
        for (int m = 0; m < size<1>(tOgOaccum108); ++m) {
            const int row113 = get<0>(tOcO111(0, m, 0));
            if (row113 < binfo21.actual_seqlen_q - m_block10 * kBlockM17 && get<1>(tOcO111(0, m, 0)) == 0) {
                gLSEaccum105(row113) = Split7 ? -(__builtin_inff()) : (__builtin_inff());
            }
        }
        return;
    }
    const int bidb_cache25 = params9.cache_batch_idx == nullptr ? bidb11 : params9.cache_batch_idx[bidb11];
    const int *block_table26 = params9.block_table == nullptr ? nullptr : params9.block_table + bidb11 * params9.block_table_batch_stride;
    const int block_table_idx27 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 / params9.page_block_size;
    const int block_table_offset28 = block_table26 == nullptr ? 0 : (n_block_max24 - 1) * kBlockN18 - block_table_idx27 * params9.page_block_size;
    const index_t row_offset_k29 = block_table26 == nullptr ? binfo21.k_offset(params9.k_batch_stride, params9.k_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride : block_table26[block_table_idx27] * params9.k_batch_stride + block_table_offset28 * params9.k_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.k_head_stride;
    const index_t row_offset_v30 = block_table26 == nullptr ? binfo21.k_offset(params9.v_batch_stride, params9.v_row_stride, bidb_cache25) + (n_block_max24 - 1) * kBlockN18 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride : block_table26[block_table_idx27] * params9.v_batch_stride + block_table_offset28 * params9.v_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.v_head_stride;
    Tensor mQ31 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.q_ptr) + binfo21.q_offset(params9.q_batch_stride, params9.q_row_stride, bidb11)), make_shape(binfo21.actual_seqlen_q, params9.h, params9.d), make_stride(params9.q_row_stride, params9.q_head_stride, _1{}));
    Tensor gQ32 = local_tile(mQ31(_, bidh12, _), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_coord(m_block10, 0));
    Tensor gK33 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.k_ptr) + row_offset_k29), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.k_row_stride, _1{}));
    Tensor gV34 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.v_ptr) + row_offset_v30), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.v_row_stride, _1{}));
    Tensor sQ35 = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_15)), typename Kernel_traits0::SmemLayoutQ{});
    Tensor sK36 = make_tensor(sQ35.data() + size(sQ35), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sV37 = make_tensor(sK36.data() + size(sK36), typename Kernel_traits0::SmemLayoutKV{});
    Tensor sVt38 = make_tensor(sV37.data(), typename Kernel_traits0::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle39 = make_tensor(sV37.data().get(), typename Kernel_traits0::SmemLayoutVtransposedNoSwizzle{});
    typename Kernel_traits0::GmemTiledCopyQKV gmem_tiled_copy_QKV40;
    auto gmem_thr_copy_QKV41 = gmem_tiled_copy_QKV40.get_thread_slice(tidx16);
    Tensor tQgQ42 = gmem_thr_copy_QKV41.partition_S(gQ32);
    Tensor tQsQ43 = gmem_thr_copy_QKV41.partition_D(sQ35);
    Tensor tKgK44 = gmem_thr_copy_QKV41.partition_S(gK33);
    Tensor tKsK45 = gmem_thr_copy_QKV41.partition_D(sK36);
    Tensor tVgV46 = gmem_thr_copy_QKV41.partition_S(gV34);
    Tensor tVsV47 = gmem_thr_copy_QKV41.partition_D(sV37);
    typename Kernel_traits0::TiledMma tiled_mma48;
    auto thr_mma49 = tiled_mma48.get_thread_slice(tidx16);
    Tensor tSrQ50 = thr_mma49.partition_fragment_A(sQ35);
    Tensor tSrK51 = thr_mma49.partition_fragment_B(sK36);
    Tensor tOrVt52 = thr_mma49.partition_fragment_B(sVtNoSwizzle39);
    Tensor acc_o53 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    auto smem_tiled_copy_Q54 = make_tiled_copy_A(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_Q55 = smem_tiled_copy_Q54.get_thread_slice(tidx16);
    Tensor tSsQ56 = smem_thr_copy_Q55.partition_S(sQ35);
    auto smem_tiled_copy_K57 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtom{}, tiled_mma48);
    auto smem_thr_copy_K58 = smem_tiled_copy_K57.get_thread_slice(tidx16);
    Tensor tSsK59 = smem_thr_copy_K58.partition_S(sK36);
    auto smem_tiled_copy_V60 = make_tiled_copy_B(typename Kernel_traits0::SmemCopyAtomTransposed{}, tiled_mma48);
    auto smem_thr_copy_V61 = smem_tiled_copy_V60.get_thread_slice(tidx16);
    Tensor tOsVt62 = smem_thr_copy_V61.partition_S(sVt38);
    Tensor cQ63 = make_identity_tensor(make_shape(size<0>(sQ35), size<1>(sQ35)));
    Tensor cKV64 = make_identity_tensor(make_shape(size<0>(sK36), size<1>(sK36)));
    Tensor tQcQ65 = gmem_thr_copy_QKV41.partition_S(cQ63);
    Tensor tKVcKV66 = gmem_thr_copy_QKV41.partition_S(cKV64);
    Tensor tQpQ67 = make_tensor<bool>(make_shape(size<2>(tQsQ43)));
    Tensor tKVpKV68 = make_tensor<bool>(make_shape(size<2>(tKsK45)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tQpQ67); ++k) {
            tQpQ67(k) = get<1>(tQcQ65(0, 0, k)) < params9.d;
        }
        for (int k = 0; k < size(tKVpKV68); ++k) {
            tKVpKV68(k) = get<1>(tKVcKV66(0, 0, k)) < params9.d;
        }
    }
    typename Kernel_traits0::GmemTiledCopyRotcossin gmem_tiled_copy_rotary69;
    auto gmem_thr_copy_rotary70 = gmem_tiled_copy_rotary69.get_thread_slice(tidx16);
    typename Kernel_traits0::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont71;
    auto gmem_thr_copy_rotary_cont72 = gmem_tiled_copy_rotary_cont71.get_thread_slice(tidx16);
    if (Append_KV8) {
        const index_t row_offset_cossin114 = ((n_block_max24 - 1) * kBlockN18 + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11])) * (params9.rotary_dim / 2);
        Tensor gCos115 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSin116 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19 / 2>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gCosCont117 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor gSinCont118 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin114), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.rotary_dim / 2, _1{}));
        Tensor tRgCos119 = gmem_thr_copy_rotary70.partition_S(gCos115);
        Tensor tRgSin120 = gmem_thr_copy_rotary70.partition_S(gSin116);
        Tensor tRgCosCont121 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont117);
        Tensor tRgSinCont122 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont118);
        const index_t row_offset_knew123 = bidb11 * params9.knew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.knew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.knew_head_stride;
        const index_t row_offset_vnew124 = bidb11 * params9.vnew_batch_stride + ((n_block_max24 - 1) * kBlockN18) * params9.vnew_row_stride + (bidh12 / params9.h_h_k_ratio) * params9.vnew_head_stride;
        Tensor gKnew125 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.knew_ptr) + row_offset_knew123 - binfo21.seqlen_k_cache * params9.knew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.knew_row_stride, _1{}));
        Tensor gVnew126 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.vnew_ptr) + row_offset_vnew124 - binfo21.seqlen_k_cache * params9.vnew_row_stride), Shape<Int<kBlockN18>, Int<kHeadDim19>>{}, make_stride(params9.vnew_row_stride, _1{}));
        Tensor tKgKnew127 = gmem_thr_copy_QKV41.partition_S(gKnew125);
        Tensor tVgVnew128 = gmem_thr_copy_QKV41.partition_S(gVnew126);
        const int n_block_copy_min129 = std::max(n_block_min23, binfo21.seqlen_k_cache / kBlockN18);
        auto tKgK_data130 = tKgK44.data();
        auto tVgV_data131 = tVgV46.data();
        for (int n_block = n_block_max24 - 1; n_block >= n_block_copy_min129; n_block--) {
            flash::copy_w_min_idx<Is_even_K5>(tVgVnew128, tVgV46, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            tVgVnew128.data() = tVgVnew128.data() + (-int(kBlockN18 * params9.vnew_row_stride));
            if (params9.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K5>(tKgKnew127, tKgK44, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18);
            } else {
                if (params9.is_rotary_interleaved) {
                    flash::copy_rotary_interleaved<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCos119, tRgSin120, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCos119.data() = tRgCos119.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSin120.data() = tRgSin120.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                } else {
                    flash::copy_rotary_contiguous<Is_even_K5, false>(tKgKnew127, tKgK44, tRgCosCont121, tRgSinCont122, tKVcKV66, binfo21.actual_seqlen_k - n_block * kBlockN18, binfo21.seqlen_k_cache - n_block * kBlockN18, params9.d, params9.rotary_dim);
                    tRgCosCont121.data() = tRgCosCont121.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                    tRgSinCont122.data() = tRgSinCont122.data() + (-int(kBlockN18 * params9.rotary_dim / 2));
                }
            }
            tKgKnew127.data() = tKgKnew127.data() + (-int(kBlockN18 * params9.knew_row_stride));
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                if (n_block > n_block_copy_min129) {
                    const int block_table_idx_cur132 = n_block * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_cur133 = n_block * kBlockN18 - block_table_idx_cur132 * params9.page_block_size;
                    const int block_table_idx_next134 = (n_block - 1) * kBlockN18 / params9.page_block_size;
                    const int block_table_offset_next135 = (n_block - 1) * kBlockN18 - block_table_idx_next134 * params9.page_block_size;
                    const int table_diff136 = block_table26[block_table_idx_next134] - block_table26[block_table_idx_cur132];
                    const int offset_diff137 = block_table_offset_next135 - block_table_offset_cur133;
                    tVgV46.data() = tVgV46.data() + table_diff136 * params9.v_batch_stride + offset_diff137 * params9.v_row_stride;
                    tKgK44.data() = tKgK44.data() + table_diff136 * params9.k_batch_stride + offset_diff137 * params9.k_row_stride;
                }
            }
        }
        asm ("bar.sync 1,128;");
        ;
        tKgK44.data() = tKgK_data130;
        tVgV46.data() = tVgV_data131;
    }
    if (!Append_KV8 || params9.rotary_dim == 0) {
        flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tQgQ42, tQsQ43, tQcQ65, tQpQ67, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
    } else {
        const index_t row_offset_cossin138 = (binfo21.seqlen_k_cache + (params9.leftpad_k == nullptr ? 0 : params9.leftpad_k[bidb11]) + (Is_causal1 || Is_local2 ? m_block10 * kBlockM17 : 0)) * (params9.rotary_dim / 2);
        Tensor gCos139 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSin140 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19 / 2>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont141 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_cos_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont142 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params9.rotary_sin_ptr) + row_offset_cossin138), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Is_causal1 || Is_local2 ? params9.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos143 = gmem_thr_copy_rotary70.partition_S(gCos139);
        Tensor tRgSin144 = gmem_thr_copy_rotary70.partition_S(gSin140);
        Tensor tRgCosCont145 = gmem_thr_copy_rotary_cont72.partition_S(gCosCont141);
        Tensor tRgSinCont146 = gmem_thr_copy_rotary_cont72.partition_S(gSinCont142);
        if (params9.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K5>(tQgQ42, tQsQ43, tRgCos143, tRgSin144, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        } else {
            flash::copy_rotary_contiguous<Is_even_K5>(tQgQ42, tQsQ43, tRgCosCont145, tRgSinCont146, tQcQ65, binfo21.actual_seqlen_q - m_block10 * kBlockM17, 0, params9.d, params9.rotary_dim);
        }
    }
    int n_block73 = n_block_max24 - 1;
    flash::copy<Is_even_MN4, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
    cute::cp_async_fence();
    clear(acc_o53);
    flash::Softmax<2 * size<1>(acc_o53)> softmax74;
    const float alibi_slope75 = !Has_alibi3 ? 0.F : reinterpret_cast<float *>(params9.alibi_slopes_ptr)[bidb11 * params9.alibi_slopes_batch_stride + bidh12] / params9.scale_softmax;
    flash::Mask<Is_causal1, Is_local2, Has_alibi3> mask76(binfo21.actual_seqlen_k, binfo21.actual_seqlen_q, params9.window_size_left, params9.window_size_right, alibi_slope75);
    constexpr int n_masking_steps77 = (!Is_causal1 && !Is_local2) ? 1 : ((Is_even_MN4 && Is_causal1) ? cute::ceil_div(kBlockM17, kBlockN18) : cute::ceil_div(kBlockM17, kBlockN18) + 1);
    for (int masking_step = 0; masking_step < n_masking_steps77; ++masking_step , --n_block73) {
        Tensor acc_s147 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s147);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (masking_step > 0) {
            if (block_table26 == nullptr) {
                tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
            } else {
                const int block_table_idx_cur150 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur151 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur150 * params9.page_block_size;
                const int block_table_idx_next152 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next153 = n_block73 * kBlockN18 - block_table_idx_next152 * params9.page_block_size;
                tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next152] - block_table26[block_table_idx_cur150]) * params9.v_batch_stride + (block_table_offset_next153 - block_table_offset_cur151) * params9.v_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        } else {
            flash::copy<Is_even_MN4, Is_even_K5, true>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68, binfo21.actual_seqlen_k - n_block73 * kBlockN18);
        }
        cute::cp_async_fence();
        flash::gemm(acc_s147, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s147, params9.softcap);
        }
        mask76.template apply_mask<Is_causal1, Is_even_MN4>(acc_s147, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur154 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur155 = n_block73 * kBlockN18 - block_table_idx_cur154 * params9.page_block_size;
                const int block_table_idx_next156 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next157 = (n_block73 - 1) * kBlockN18 - block_table_idx_next156 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next156] - block_table26[block_table_idx_cur154]) * params9.k_batch_stride + (block_table_offset_next157 - block_table_offset_cur155) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        masking_step == 0 ? softmax74.template softmax_rescale_o<true, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2) : softmax74.template softmax_rescale_o<false, Is_causal1 || Is_local2 || !Is_even_MN4>(acc_s147, acc_o53, params9.scale_softmax_log2);
        Tensor rP148 = flash::convert_type<Element>(acc_s147);
        Tensor tOrP149 = make_tensor(rP148.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP148.layout()));
        flash::gemm_rs(acc_o53, tOrP149, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
        if (n_masking_steps77 > 1 && n_block73 <= n_block_min23) {
            --n_block73;
            break;
        }
    }
    for (; n_block73 >= n_block_min23; --n_block73) {
        Tensor acc_s158 = partition_fragment_C(tiled_mma48, Shape<Int<kBlockM17>, Int<kBlockN18>>{});
        clear(acc_s158);
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (block_table26 == nullptr) {
            tVgV46.data() = tVgV46.data() + (-int(kBlockN18 * params9.v_row_stride));
        } else {
            const int block_table_idx_cur161 = (n_block73 + 1) * kBlockN18 / params9.page_block_size;
            const int block_table_offset_cur162 = (n_block73 + 1) * kBlockN18 - block_table_idx_cur161 * params9.page_block_size;
            const int block_table_idx_next163 = n_block73 * kBlockN18 / params9.page_block_size;
            const int block_table_offset_next164 = n_block73 * kBlockN18 - block_table_idx_next163 * params9.page_block_size;
            tVgV46.data() = tVgV46.data() + (block_table26[block_table_idx_next163] - block_table26[block_table_idx_cur161]) * params9.v_batch_stride + (block_table_offset_next164 - block_table_offset_cur162) * params9.v_row_stride;
        }
        flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tVgV46, tVsV47, tKVcKV66, tKVpKV68);
        cute::cp_async_fence();
        flash::gemm(acc_s158, tSrQ50, tSrK51, tSsQ56, tSsK59, tiled_mma48, smem_tiled_copy_Q54, smem_tiled_copy_K57, smem_thr_copy_Q55, smem_thr_copy_K58);
        if (Is_softcap6) {
            fused::apply_softcap(acc_s158, params9.softcap);
        }
        flash::cp_async_wait<0>();
        asm ("bar.sync 1,128;");
        ;
        if (n_block73 > n_block_min23) {
            if (block_table26 == nullptr) {
                tKgK44.data() = tKgK44.data() + (-int(kBlockN18 * params9.k_row_stride));
            } else {
                const int block_table_idx_cur165 = n_block73 * kBlockN18 / params9.page_block_size;
                const int block_table_offset_cur166 = n_block73 * kBlockN18 - block_table_idx_cur165 * params9.page_block_size;
                const int block_table_idx_next167 = (n_block73 - 1) * kBlockN18 / params9.page_block_size;
                const int block_table_offset_next168 = (n_block73 - 1) * kBlockN18 - block_table_idx_next167 * params9.page_block_size;
                tKgK44.data() = tKgK44.data() + (block_table26[block_table_idx_next167] - block_table26[block_table_idx_cur165]) * params9.k_batch_stride + (block_table_offset_next168 - block_table_offset_cur166) * params9.k_row_stride;
            }
            flash::copy<true, Is_even_K5>(gmem_tiled_copy_QKV40, tKgK44, tKsK45, tKVcKV66, tKVpKV68);
            cute::cp_async_fence();
        }
        mask76.template apply_mask<false>(acc_s158, n_block73 * kBlockN18, m_block10 * kBlockM17 + (tidx16 / 32) * 16 + (tidx16 % 32) / 4, kNWarps20 * 16);
        softmax74.template softmax_rescale_o<false, Is_local2>(acc_s158, acc_o53, params9.scale_softmax_log2);
        Tensor rP159 = flash::convert_type<Element>(acc_s158);
        Tensor tOrP160 = make_tensor(rP159.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits0::TiledMma>(rP159.layout()));
        flash::gemm_rs(acc_o53, tOrP160, tOrVt52, tOsVt62, tiled_mma48, smem_tiled_copy_V60, smem_thr_copy_V61);
    }
    Tensor lse78 = softmax74.template normalize_softmax_lse<false, Split7>(acc_o53, params9.scale_softmax);
    Tensor sOaccum79 = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_15)), typename Kernel_traits0::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<!Split7, typename Kernel_traits0::SmemCopyAtomO, typename Kernel_traits0::SmemCopyAtomOaccum>;
    auto smem_tiled_copy_Oaccum80 = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma48);
    auto smem_thr_copy_Oaccum81 = smem_tiled_copy_Oaccum80.get_thread_slice(tidx16);
    Tensor rO82 = flash::convert_type<ElementO>(acc_o53);
    Tensor taccOrOaccum83 = smem_thr_copy_Oaccum81.retile_S(rO82);
    Tensor taccOsOaccum84 = smem_thr_copy_Oaccum81.partition_D(sOaccum79);
    if (Split7) {
        asm ("bar.sync 1,128;");
        ;
    }
    cute::copy(smem_tiled_copy_Oaccum80, taccOrOaccum83, taccOsOaccum84);
    const index_t row_offset_o85 = binfo21.q_offset(params9.o_batch_stride, params9.o_row_stride, bidb11) + m_block10 * kBlockM17 * params9.o_row_stride + bidh12 * params9.o_head_stride;
    const index_t row_offset_oaccum86 = (((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q + m_block10 * kBlockM17) * params9.d_rounded;
    const index_t row_offset_lseaccum87 = (Split7 || !params9.unpadded_lse ? ((n_split_idx13 * params9.b + bidb11) * params9.h + bidh12) * params9.seqlen_q : bidh12 * params9.total_q + binfo21.q_offset(params9.seqlen_q, 1, bidb11)) + m_block10 * kBlockM17;
    Tensor gOaccum88 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split7 ? params9.oaccum_ptr : params9.o_ptr) + (Split7 ? row_offset_oaccum86 : row_offset_o85)), Shape<Int<kBlockM17>, Int<kHeadDim19>>{}, make_stride(Split7 ? kHeadDim19 : params9.o_row_stride, _1{}));
    Tensor gLSEaccum89 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split7 ? params9.softmax_lseaccum_ptr : params9.softmax_lse_ptr) + row_offset_lseaccum87), Shape<Int<kBlockM17>>{}, Stride<_1>{});
    GmemTiledCopyO gmem_tiled_copy_Oaccum90;
    auto gmem_thr_copy_Oaccum91 = gmem_tiled_copy_Oaccum90.get_thread_slice(tidx16);
    Tensor tOsOaccum92 = gmem_thr_copy_Oaccum91.partition_S(sOaccum79);
    Tensor tOgOaccum93 = gmem_thr_copy_Oaccum91.partition_D(gOaccum88);
    asm ("bar.sync 1,128;");
    ;
    Tensor tOrOaccum94 = make_tensor<ElementO>(shape(tOgOaccum93));
    cute::copy(gmem_tiled_copy_Oaccum90, tOsOaccum92, tOrOaccum94);
    Tensor caccO95 = make_identity_tensor(Shape<Int<kBlockM17>, Int<kHeadDim19>>{});
    Tensor taccOcO96 = thr_mma49.partition_C(caccO95);
    static_assert(decltype(size<0>(taccOcO96))::value == 4);
    Tensor taccOcO_row97 = logical_divide(taccOcO96, Shape<_2>{})(make_coord(0, _), _, 0);
    static_assert(decltype(size(lse78) == size(taccOcO_row97))::value);
    if (get<1>(taccOcO_row97(0)) == 0) {
        for (int mi = 0; mi < size(lse78); ++mi) {
            const int row169 = get<0>(taccOcO_row97(mi));
            if (row169 < binfo21.actual_seqlen_q - m_block10 * kBlockM17) {
                gLSEaccum89(row169) = lse78(mi);
            }
        }
    }
    Tensor cO98 = make_identity_tensor(make_shape(size<0>(sOaccum79), size<1>(sOaccum79)));
    Tensor tOcO99 = gmem_thr_copy_Oaccum91.partition_D(cO98);
    Tensor tOpO100 = make_tensor<bool>(make_shape(size<2>(tOgOaccum93)));
    if (!Is_even_K5) {
        for (int k = 0; k < size(tOpO100); ++k) {
            tOpO100(k) = get<1>(tOcO99(0, 0, k)) < params9.d;
        }
    }
    flash::copy<Is_even_MN4, Is_even_K5, false, false>(gmem_tiled_copy_Oaccum90, tOrOaccum94, tOgOaccum93, tOcO99, tOpO100, binfo21.actual_seqlen_q - m_block10 * kBlockM17);
}
}
*/