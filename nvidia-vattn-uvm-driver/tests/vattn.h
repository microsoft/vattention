#ifndef _VATTN_MEM_MGMT_H_
#define _VATTN_MEM_MGMT_H_

#include <chrono>
#include "cuda.h"
#include <dirent.h>
#include <iostream>
#include <sys/ioctl.h>
#include <sys/types.h>

#define VATTN_GET_MEM_HANDLE_IOCTL_NUM   200
#define VATTN_MEM_MAP_IOCTL_NUM          201
#define VATTN_FREE_MEM_HANDLE_IOCTL_NUM  202
#define VATTN_CLEAR_ADDRESS_IOCTL_NUM    203

#define VATTN_HANDLE_SIZE_64K          (64ULL * 1024)
#define VATTN_HANDLE_SIZE_128K         (128ULL * 1024)
#define VATTN_HANDLE_SIZE_256K         (256ULL * 1024)
#define VATTN_DEF_HANDLE_SIZE          (VATTN_HANDLE_SIZE_256K)
#define VATTN_MEM_RESERVE_LOW_LIMIT    (2ULL * 1024 * 1024)

#define UUID_SIZE 16
#define NV_OK     0

#define PSF_DIR         "/proc/self/fd"
#define NVIDIA_UVM_PATH "/dev/nvidia-uvm"

typedef long long unsigned int NvU64;

uint8_t cached_uuid[UUID_SIZE];
static int nvidia_uvm_fd = -1;

static NvU64 handle_size = VATTN_DEF_HANDLE_SIZE;

typedef struct {
    NvU64 size;
    uint8_t uuid[UUID_SIZE];
    NvU64 handle;
    unsigned status;
} vattn_get_mem_handle_params;

typedef struct {
    NvU64 base_address;
    NvU64 handle;
    uint8_t uuid[UUID_SIZE];
    NvU64 size;
    unsigned status;
} vattn_mem_map_params;

typedef struct {
    NvU64 base_address;
    NvU64 size;
    uint8_t uuid[UUID_SIZE];
    unsigned status;
} vattn_clear_address_params;

typedef struct {
    NvU64 handle;
    NvU64 size;
    uint8_t uuid[UUID_SIZE];
    unsigned status;
} vattn_free_mem_handle_params;

typedef enum {
    VATTN_IOCTL_MEM_MAP_ERR = (NvU64)-6,
    VATTN_IOCTL_CLEAR_ADDRESS_ERR = (NvU64)-5,
    VATTN_IOCTL_MEM_HANDLE_ERR = (NvU64)-4,
    VATTN_IOCTL_FREE_MEM_HANDLE_ERR = (NvU64)-3,
    VATTN_IOCTL_ERR = (NvU64)-2,
    VATTN_ERR_PATH = (NvU64)-1,
    VATTN_OK = (NvU64)0,
} vattn_err_t;

#define PROFILE_CALL(x, show) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        x; \
        auto end = std::chrono::high_resolution_clock::now(); \
        if (show) \
            std::cout << #x << " took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl; \
    } while (0)

#define CHECK_VATTN(x) \
    do { \
        NvU64 res = x; \
        if (res != VATTN_OK) { \
            std::cerr << __FILE__ << ':' << __LINE__ << ' ' << #x \
                      << "failed (" << (unsigned)res << ") " << std::endl; \
            exit(1); \
        } \
    } while (0)

/* Step 1. initialize for running vattn.
 *         Specifically, create arguments needed for later
 *         calls to vattn IOCTLs */
vattn_err_t vattn_init(int in_device, NvU64 in_handle_size) {
    int *dummy_alloc;
    cudaDeviceProp prop;
    DIR *d;
    struct dirent *dir;
    char psf_path[512];
    char *psf_realpath;

    cudaSetDevice(in_device);
    /* create a small allocation for creating /dev/nvidua-uvm file */
    cudaMallocManaged(&dummy_alloc, sizeof(int));
    cudaFree(dummy_alloc);

    /* get device information */
    cudaGetDeviceProperties(&prop, in_device);
    /* set uuid and cache it for later use */
    for(int i = 0; i < UUID_SIZE; i++) {
        cached_uuid[i] = prop.uuid.bytes[i];
    }

    d = opendir(PSF_DIR);
    if (d) {
       while ((dir = readdir(d)) != NULL) {
           if (dir->d_type == DT_LNK) {
               sprintf(psf_path, "%s/%s", PSF_DIR, dir->d_name);
               psf_realpath = realpath(psf_path, NULL);
	       if (!psf_realpath)
                   continue;
               if (strcmp(psf_realpath, NVIDIA_UVM_PATH) == 0)
                   nvidia_uvm_fd = atoi(dir->d_name);
               free(psf_realpath);
               if (nvidia_uvm_fd >= 0)
                   break;
           }
       }
       closedir(d);
    }
    if (nvidia_uvm_fd < 0) {
        fprintf(stderr, "Cannot find %s in %s\n", NVIDIA_UVM_PATH, PSF_DIR);
        return VATTN_ERR_PATH;
    }

    /* setting handle size during init --- it cannot be changed later in the run */
    handle_size = in_handle_size;

    return VATTN_OK;
}

/* STEP 2. Reserve memory allocations. Current implementation
 *         expects creation of VA spaces at 2M alignments. This
 *         is a generally observed situation. */
void* vattn_reserve_memory(size_t size_bytes) {
    void *allocation;
    cudaMallocManaged((void**)&allocation, VATTN_MEM_RESERVE_LOW_LIMIT);
    /* TODO: allocate at 2M alignments for now --- call UP_ALIGN */
    return allocation;
}

/* STEP 3. Get a mem handle of the requested size.
 *         Currently, the size argument is ignored */
NvU64 vattn_get_mem_handle(NvU64 *handle) {
    vattn_get_mem_handle_params request;

    /* set uuid to get a physical chunk from that specific device */
    for(int i = 0; i < UUID_SIZE; i++) {
        request.uuid[i] = cached_uuid[i];
    }
    /* fixing it to 64K for now */
    request.size = handle_size;
    /* get the chunk */
    if (ioctl(nvidia_uvm_fd, VATTN_GET_MEM_HANDLE_IOCTL_NUM, &request) != NV_OK) {
        return VATTN_IOCTL_MEM_HANDLE_ERR;
    }
    /* set handle */
    *handle = request.handle;
    return VATTN_OK;
}

/* STEP 4. Map the passed virtual address to the handle. */
NvU64 vattn_mem_map(void *address, NvU64 handle) {
    vattn_mem_map_params request;

    /* set uuid to get a physical chunk from that specific device */
    for(int i = 0; i < UUID_SIZE; i++) {
        request.uuid[i] = cached_uuid[i];
    }
    request.base_address = (NvU64)address;
    request.handle = handle;
    request.size = handle_size;
    /* assert that nvidia_uvm_fd is initialized */
    if (ioctl(nvidia_uvm_fd, VATTN_MEM_MAP_IOCTL_NUM, &request) != NV_OK) {
        return VATTN_IOCTL_MEM_MAP_ERR;
    }
    return VATTN_OK;
}

/* STEP 5. Need to clear the data structures associated with the virtual
 *         address range on which mapping happened */
NvU64 vattn_free_reserved_address(void* address, size_t size) {
    vattn_clear_address_params request;

    for (int i = 0; i < UUID_SIZE; i++) {
        request.uuid[i] = cached_uuid[i];
    }
    request.base_address = (NvU64)address;
    request.size = size;
    /* assert that nvidia_uvm_fd is initialized */
    if (ioctl(nvidia_uvm_fd, VATTN_CLEAR_ADDRESS_IOCTL_NUM, &request) != NV_OK) {
        return VATTN_IOCTL_CLEAR_ADDRESS_ERR;
    }
    /* free the virtual memory as well */
    // TODO: maybe for torch integration --- this should not be done
    cudaFree(address);
    return VATTN_OK;
}

/* STEP 6. Need to free the created handles else there will be trouble to
 *         rerun experiments as the created handles are in PINNED state.
 *         This should deal with the issue */
NvU64 vattn_release_mem_handle(NvU64 handle) {
    vattn_free_mem_handle_params request;

    for (int i = 0; i < UUID_SIZE; i++) {
        request.uuid[i] = cached_uuid[i];
    }
    request.handle = handle;
    request.size = handle_size;
    /* assert that nvidia_uvm_fd is initialized */
    if (ioctl(nvidia_uvm_fd, VATTN_FREE_MEM_HANDLE_IOCTL_NUM, &request) != NV_OK) {
        return VATTN_IOCTL_FREE_MEM_HANDLE_ERR;
    }
    return VATTN_OK;
}

#endif
