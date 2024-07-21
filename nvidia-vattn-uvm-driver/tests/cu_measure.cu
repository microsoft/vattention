#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define ALLOC_SIZE                (2ULL * 1024 * 1024)
#define MAPPING_SIZE              (256ULL * 1024)
#define REQUESTS                  100                      // num of requests
#define LAYERS                    32                       // num of layers
#define TYPE                      2                        // KV

using namespace std;

CUcontext ctx;
CUmemAllocationProp prop = {};
CUmemAccessDesc accessDesc = {};

CUmemGenericAllocationHandle *handles;
CUdeviceptr *buffers;
size_t num_handles;
size_t block_size;

#define CHECK_CUDA(x) \
    do { \
        CUresult res = x; \
        if (res != CUDA_SUCCESS) { \
            const char *errStr = NULL; \
            (void)cuGetErrorString(res, &errStr); \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x \
                      << "failed (" << (unsigned)res << "): " << errStr << std::endl; \
        } \
    } while (0)

#define PROFILE_CALL(x, show) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        x; \
        auto end = std::chrono::high_resolution_clock::now(); \
        if (show) \
            std::cout << #x << " took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl; \
    } while (0)

static void do_cuda_init() {
    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuCtxCreate(&ctx, 0, 0));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = 0;
    CHECK_CUDA(cuMemGetAllocationGranularity(&block_size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
}

void reserve_memory() {
    buffers = (CUdeviceptr*)malloc(num_handles * sizeof(CUdeviceptr));
    for (int i = 0; i < num_handles; i++) {
        cuMemAddressReserve(&buffers[i], block_size, 0, 0, 0);
    }
}

void get_handles() {
    handles = (CUmemGenericAllocationHandle*)malloc(num_handles * sizeof(CUmemGenericAllocationHandle));
    for (int i = 0; i < num_handles; i++) {
        cuMemCreate(&handles[i], block_size, &prop, 0);
    }
}

void map_handles() {
    for (int i = 0; i < num_handles; i++) {
        cuMemMap(buffers[i], block_size, 0, handles[i], 0);
        cuMemSetAccess(buffers[i], block_size, &accessDesc, 1);
    }
}

void free_buffers() {
    for (int i = 0; i < num_handles; i++) {
        cuMemUnmap(buffers[i], block_size);
        cuMemAddressFree(buffers[i], block_size);
    }
    free(buffers);
}

void release_handles() {
    for (int i = 0; i < num_handles; i++) {
        cuMemRelease(handles[i]);
    }
    free(handles);
}

void print_config() {
    printf("Number of handles: %lu\n", num_handles);
    printf("Number of requests: %d\n", REQUESTS);
    printf("Number of layers: %d\n", LAYERS);
    printf("Number of types: %d\n", TYPE);
    printf("Mapping size: %lu B\n", block_size);
    printf("Block size: %lu\n", block_size);
    printf("Total memory mapped: %lu MB\n", (num_handles * block_size) / (1024 * 1024));
}

int main(int argc, char **argv) {
    /* setup environment */
    do_cuda_init();

    // calculate the number of handles needed per token, per layer, per request for each KV
    num_handles = MAPPING_SIZE / block_size;
    if (num_handles <= 0)
        // block size > MAPPING_SIZE
        num_handles = 1;
    // calculate number of handles needed in total
    num_handles = (num_handles * TYPE * REQUESTS * LAYERS);

    print_config();
    PROFILE_CALL(reserve_memory(), false);
    PROFILE_CALL(get_handles(), false);
    PROFILE_CALL(map_handles(), true);

    PROFILE_CALL(free_buffers(), false);
    PROFILE_CALL(release_handles(), false);
    return 0;
}
