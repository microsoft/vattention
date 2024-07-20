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

#define CHECK_VATTN(x)                                                      \
    do                                                                      \
    {                                                                       \
        NvU64 res = x;                                                      \
        if (res != VATTN_OK)                                                \
        {                                                                   \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x           \
                      << "failed (" << (unsigned)res << "): " << std::endl; \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

unsigned long get_num_phys_blocks(size_t num_layers, size_t free_memory)
{
    // size_t free, total;
    unsigned long num_phys_blocks;
    // CHECK_CUDA(cuMemGetInfo(&free, &total));
    /* reserving 80% for kv cache */
    // free = (free * 80) / 100;
    /* do not allocate if we can't use a handle. we need multiples of 2 * num_layers */
    num_phys_blocks = free_memory / page_size_bytes;
    num_phys_blocks -= num_phys_blocks % (2 * num_layers);
    return num_phys_blocks;
}

size_t do_cuda_default_init(int device)
{
    size_t phys_granularity;
    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuCtxGetCurrent(&ctx));
    if (ctx == NULL)
    {
        std::cerr << "[vAttention ]No CUDA context found.";
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
    return phys_granularity;
}

size_t do_cuda_uvm_init(int device)
{
    CHECK_VATTN(vattn_init(device, VATTN_DEF_HANDLE_SIZE));
    return VATTN_DEF_HANDLE_SIZE;
}

size_t do_cuda_init(int device, bool use_uvm_backend = false)
{
    if (!use_uvm_backend)
        return do_cuda_default_init(device);
    return do_cuda_uvm_init(device);
}

int reserve_cuda_pages(size_t num_layers, size_t free_memory)
{
    Log log;
    unsigned long num_phys_blocks = get_num_phys_blocks(num_layers, free_memory);
    log.log("Reserving " + std::to_string(num_phys_blocks) + " pages of size " + std::to_string(page_size_bytes) + " ...");
    while (page_handles.size() < num_phys_blocks)
    {
        CUmemGenericAllocationHandle handle;
        CHECK_CUDA(cuMemCreate(&handle, page_size_bytes, &prop, 0));
        page_handles.push_back(handle);
    }
    return page_handles.size();
}

int reserve_uvm_pages(size_t num_layers, size_t free_memory)
{
    /* This method must be called only after do_cuda_init */
    Log log;
    unsigned long num_phys_blocks = get_num_phys_blocks(num_layers, free_memory);
    log.log("Reserving " + std::to_string(num_phys_blocks) + " pages " + std::to_string(page_size_bytes) + " ...");
    while (uvm_page_handles.size() < num_phys_blocks)
    {
        NvU64 handle;
        CHECK_VATTN(vattn_get_mem_handle(&handle));
        uvm_page_handles.push_back(handle);
    }
    return uvm_page_handles.size();
}

int reserve_gpu_pages(size_t num_layers, size_t free_memory, bool use_uvm_backend = false)
{
    if (!use_uvm_backend)
        return reserve_cuda_pages(num_layers, free_memory);
    return reserve_uvm_pages(num_layers, free_memory);
}

void test_func()
{
    do_cuda_init(0, false);
    // TODO: change this test to also consider UVM backend
    for (int i = 0; i < 100; i++)
    {
        std::cout << "Iteration: " << i << std::endl;
        CUdeviceptr ptr;
        std::cout << "Reserving address..." << std::endl;
        CHECK_CUDA(cuMemAddressReserve(&ptr, 2 * MB, 0, 0, 0));
        std::cout << "ptr: " << ptr << std::endl;
        CUmemGenericAllocationHandle handle;
        std::cout << "Creating memory..." << std::endl;
        CHECK_CUDA(cuMemCreate(&handle, 2 * MB, &prop, 0));
        std::cout << "handle: " << handle << std::endl;
        std::cout << "Mapping memory..." << std::endl;
        CHECK_CUDA(cuMemMap(ptr, 2 * MB, 0, handle, 0));
        std::cout << "Setting access..." << std::endl;
        CHECK_CUDA(cuMemSetAccess(ptr, 2 * MB, &accessDesc, 1));
        std::cout << "Unmapping memory..." << std::endl;
        CHECK_CUDA(cuMemUnmap(ptr, 2 * MB));
        std::cout << "Releasing memory..." << std::endl;
        CHECK_CUDA(cuMemRelease(handle));
        std::cout << "Freeing address..." << std::endl;
        CHECK_CUDA(cuMemAddressFree(ptr, 2 * MB));
        std::cout << "test_func done..." << std::endl;
    }
}
