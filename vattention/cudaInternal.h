#define CHECK_CUDA(x)                                                                 \
    do                                                                                \
    {                                                                                 \
        CUresult res = x;                                                             \
        if (res != CUDA_SUCCESS)                                                      \
        {                                                                             \
            const char *errStr = NULL;                                                \
            (void)cuGetErrorString(res, &errStr);                                     \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x                     \
                      << "failed (" << (unsigned)res << "): " << errStr << std::endl; \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

u64 do_cuda_default_init(int device, u64 page_size)
{
    u64 phys_granularity;
    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuCtxGetCurrent(&ctx));
    if (ctx == NULL)
    {
        std::cerr << "[vAttention] No CUDA context found.";
        std::cerr << " Please initialize PyTorch before configuring vAttention." << std::endl;
        exit(1);
    }
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = device;
    CHECK_CUDA(cuMemGetAllocationGranularity(&phys_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    assert (phys_granularity == page_size);
    return phys_granularity;
}

u64 do_cuda_init(int device, u64 page_size)
{
    if (is_uvm_backend(page_size))
        return do_cuda_uvm_init(device, page_size);

    return do_cuda_default_init(device, page_size);
}

u64 reserve_cuda_pages(u64 num_layers, u64 free_memory, u64 page_size)
{
    Log log;
    u64 num_phys_blocks = get_num_phys_blocks(num_layers, free_memory, page_size);
    log.log("Reserving " + std::to_string(num_phys_blocks) + " pages of size " + std::to_string(page_size) + " ...");

    while (cuda_pages.size() < num_phys_blocks)
    {
        CUmemGenericAllocationHandle cuda_page;
        CHECK_CUDA(cuMemCreate(&cuda_page, page_size, &prop, 0));
        cuda_pages.push_back(cuda_page);
    }

    return cuda_pages.size();
}

/* This function must be called only after do_cuda_init */
u64 reserve_gpu_pages(u64 num_layers, u64 free_memory, u64 page_size)
{
    if (is_uvm_backend(page_size))
        return reserve_uvm_pages(num_layers, free_memory, page_size);

    return reserve_cuda_pages(num_layers, free_memory, page_size);
}

inline void map_cuda_pages(int reqId,
                        int layer_idx,
                        u64 req_offset,
                        CUdeviceptr kcache_ptr,
                        CUdeviceptr vcache_ptr,
                        CUPage k_page,
                        CUPage v_page) {
    CHECK_CUDA(cuMemMap(kcache_ptr + req_offset, page_size, 0, k_page, 0));
    CHECK_CUDA(cuMemMap(vcache_ptr + req_offset, page_size, 0, v_page, 0));
    CHECK_CUDA(cuMemSetAccess(kcache_ptr + req_offset, page_size, &accessDesc, 1));
    CHECK_CUDA(cuMemSetAccess(vcache_ptr + req_offset, page_size, &accessDesc, 1));
    cuda_pagemap[std::make_tuple(reqId, req_offset, layer_idx)] = std::make_pair(k_page, v_page);
}

void do_cuda_kvcache_cleanup() {
    for (int i = 0; i < k_tensors.size(); i++) {
        CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(k_tensors[i].data_ptr()), virt_buff_size));
        CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(v_tensors[i].data_ptr()), virt_buff_size));
        CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(k_tensors[i].data_ptr()), virt_buff_size));
        CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(v_tensors[i].data_ptr()), virt_buff_size));
    }

    for(int i = 0; i < cuda_pages.size(); i++)
        CHECK_CUDA(cuMemRelease(cuda_pages[i]));
}
