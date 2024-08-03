inline CUPage pop_cuda_page() {
    if (cuda_pages.empty())
        throw std::runtime_error("***** page pool is empty *****");

    CUPage page = cuda_pages.back();
    cuda_pages.pop_back();
    return page;
}

inline NvU64 pop_uvm_page() {
    if (uvm_pages.empty())
        throw std::runtime_error("***** uvm_page pool is empty *****");
    NvU64 page = uvm_pages.back();
    uvm_pages.pop_back();
    return page;
}

static inline u64 get_num_free_pages(u64 page_size) {
    if (is_uvm_backend(page_size))
        return uvm_pages.size();
    return cuda_pages.size();
}

#define DO_KVCACHE_CLEANUP(page_size) \
    do { \
        for (int reqId = 0; reqId < max_batch_size; reqId++) \
            release_kvcache_pages_all(reqId); \
        if (is_uvm_backend(page_size)) { \
            do_uvm_kvcache_cleanup(); \
            uvm_pages.clear(); \
        } else { \
            do_cuda_kvcache_cleanup(); \
            cuda_pages.clear(); \
        } \
    } while (0)

#define MAP_PAGES(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, page_size) \
    do { \
        if (!is_uvm_backend(page_size)) { \
            CUPage k_page = pop_cuda_page(); \
            CUPage v_page = pop_cuda_page(); \
            map_cuda_pages(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, k_page, v_page); \
        } else { \
            NvU64 k_page = pop_uvm_page(); \
            NvU64 v_page = pop_uvm_page(); \
            map_uvm_pages(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, k_page, v_page); \
        } \
    } while (0)


#define UNMAP_PAGES(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, page_size) \
    do { \
        if (!is_uvm_backend(page_size)) { \
            CHECK_CUDA(cuMemUnmap(kcache_ptr + req_offset, page_size)); \
            CHECK_CUDA(cuMemUnmap(vcache_ptr + req_offset, page_size)); \
            std::pair pages = cuda_pagemap[std::make_tuple(reqId, req_offset, layer_idx)]; \
            cuda_pages.push_back(pages.first); \
            cuda_pages.push_back(pages.second); \
            cuda_pagemap.erase(std::make_tuple(reqId, req_offset, layer_idx)); \
        } else { \
            std::pair pages = uvm_pagemap[std::make_tuple(reqId, req_offset, layer_idx)]; \
            uvm_pages.push_back(pages.first); \
            uvm_pages.push_back(pages.second); \
            uvm_pagemap.erase(std::make_tuple(reqId, req_offset, layer_idx)); \
        } \
    } while (0)

#define MAP_COMMON_PAGES(layer_idx, kcache_ptr, vcache_ptr, page_size) \
    do { \
        if (!is_uvm_backend(page_size)) { \
            CUPage k_page = pop_cuda_page(); \
            CUPage v_page = pop_cuda_page(); \
            for (int reqId = 0; reqId < max_batch_size; reqId++) { \
                req_offset = get_req_current_map_offset(reqId); \
                map_cuda_pages(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, k_page, v_page); \
            } \
        } else { \
            NvU64 k_page = pop_uvm_page(); \
            NvU64 v_page = pop_uvm_page(); \
            for (int reqId = 0; reqId < max_batch_size; reqId++) { \
                req_offset = get_req_current_map_offset(reqId); \
                map_uvm_pages(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, k_page, v_page); \
            } \
        } \
    } while (0)
