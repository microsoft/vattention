#ifndef _VATTN_MEM_MGMT_H_
#define _VATTN_MEM_MGMT_H_

#include <chrono>
#include "cuda.h"
#include <dirent.h>
#include <iostream>
#include <sys/ioctl.h>
#include <sys/types.h>

#define VATTN_GET_MEM_PAGE_IOCTL_NUM   200
#define VATTN_MEM_MAP_IOCTL_NUM          201
#define VATTN_FREE_MEM_PAGE_IOCTL_NUM  202
#define VATTN_CLEAR_ADDRESS_IOCTL_NUM    203

#define VATTN_PAGE_SIZE_64K          (64ULL * 1024)
#define VATTN_PAGE_SIZE_128K         (128ULL * 1024)
#define VATTN_PAGE_SIZE_256K         (256ULL * 1024)
#define VATTN_DEF_PAGE_SIZE          (VATTN_PAGE_SIZE_256K)
#define VATTN_MEM_RESERVE_LOW_LIMIT    (2ULL * 1024 * 1024)

#define UUID_SIZE 16
#define NV_OK     0

#define PSF_DIR         "/proc/self/fd"
#define NVIDIA_UVM_PATH "/dev/nvidia-uvm"

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

static uint8_t cached_uuid[UUID_SIZE];
static int nvidia_uvm_fd = -1;
static NvU64 uvm_page_size;

typedef struct {
    NvU64 size;
    uint8_t uuid[UUID_SIZE];
    NvU64 page;
    unsigned status;
} vattn_get_mem_page_params;

typedef struct {
    NvU64 base_address;
    NvU64 page;
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
    NvU64 page;
    NvU64 size;
    uint8_t uuid[UUID_SIZE];
    unsigned status;
} vattn_free_mem_page_params;

typedef enum {
    VATTN_IOCTL_MEM_MAP_ERR = (NvU64)-6,
    VATTN_IOCTL_CLEAR_ADDRESS_ERR = (NvU64)-5,
    VATTN_IOCTL_MEM_PAGE_ERR = (NvU64)-4,
    VATTN_IOCTL_FREE_MEM_PAGE_ERR = (NvU64)-3,
    VATTN_IOCTL_ERR = (NvU64)-2,
    VATTN_ERR_PATH = (NvU64)-1,
    VATTN_OK = (NvU64)0,
} vattn_err_t;

/* Step 1. initialize for running vattn.
 *         Specifically, create arguments needed for later
 *         calls to vattn IOCTLs */
vattn_err_t vattn_init(int in_device, NvU64 in_page_size) {
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

    /* setting page size during init --- it cannot be changed later in the run */
    uvm_page_size = in_page_size;

    return VATTN_OK;
}

/* STEP 2. Reserve memory allocations. Current implementation
 *         expects creation of VA spaces at 2M alignments. This
 *         is a generally observed situation. */
void* vattn_reserve_memory(u64 size_bytes) {
    void *allocation;
    cudaMallocManaged((void**)&allocation, VATTN_MEM_RESERVE_LOW_LIMIT);
    /* TODO: allocate at 2M alignments for now --- call UP_ALIGN */
    return allocation;
}

/* STEP 3. Get a mem page of the requested size.
 *         Currently, the size argument is ignored */
NvU64 vattn_get_mem_page(NvU64 *page) {
    vattn_get_mem_page_params request;

    /* set uuid to get a physical chunk from that specific device */
    for(int i = 0; i < UUID_SIZE; i++) {
        request.uuid[i] = cached_uuid[i];
    }
    /* fixing it to 64K for now */
    request.size = uvm_page_size;
    /* get the chunk */
    if (ioctl(nvidia_uvm_fd, VATTN_GET_MEM_PAGE_IOCTL_NUM, &request) != NV_OK) {
        return VATTN_IOCTL_MEM_PAGE_ERR;
    }
    /* set page */
    *page = request.page;
    return VATTN_OK;
}

/* STEP 4. Map the passed virtual address to the page. */
NvU64 vattn_mem_map(void *address, NvU64 page) {
    vattn_mem_map_params request;

    /* set uuid to get a physical chunk from that specific device */
    for(int i = 0; i < UUID_SIZE; i++) {
        request.uuid[i] = cached_uuid[i];
    }
    request.base_address = (NvU64)address;
    request.page = page;
    request.size = uvm_page_size;
    /* assert that nvidia_uvm_fd is initialized */
    if (ioctl(nvidia_uvm_fd, VATTN_MEM_MAP_IOCTL_NUM, &request) != NV_OK) {
        return VATTN_IOCTL_MEM_MAP_ERR;
    }
    return VATTN_OK;
}

/* STEP 5. Need to clear the data structures associated with the virtual
 *         address range on which mapping happened */
NvU64 vattn_free_reserved_address(void* address, u64 size) {
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
    cudaFree(address);
    return VATTN_OK;
}

/* STEP 6. Need to free the created pages else there will be trouble to
 *         rerun experiments as the created pages are in PINNED state.
 *         This should deal with the issue */
NvU64 vattn_release_mem_page(NvU64 page) {
    vattn_free_mem_page_params request;

    for (int i = 0; i < UUID_SIZE; i++) {
        request.uuid[i] = cached_uuid[i];
    }
    request.page = page;
    request.size = uvm_page_size;
    /* assert that nvidia_uvm_fd is initialized */
    if (ioctl(nvidia_uvm_fd, VATTN_FREE_MEM_PAGE_IOCTL_NUM, &request) != NV_OK) {
        return VATTN_IOCTL_FREE_MEM_PAGE_ERR;
    }
    return VATTN_OK;
}

u64 do_cuda_uvm_init(int device, u64 page_size)
{
    /* our uvm driver support only 64KB, 128KB and 256KB page sizes */
    assert (page_size == 64*KB || page_size == 128*KB ||
                page_size == 256*KB);
    CHECK_VATTN(vattn_init(device, (NvU64)page_size));
    return page_size;
}

u64 reserve_uvm_pages(u64 num_layers, u64 free_memory, u64 page_size)
{
    Log log;
    u64 num_phys_blocks = get_num_phys_blocks(num_layers, free_memory, page_size);
    log.log("Reserving " + std::to_string(num_phys_blocks) + " pages " + std::to_string(page_size) + " ...");

    while (uvm_pages.size() < num_phys_blocks)
    {
        NvU64 uvm_page;
        CHECK_VATTN(vattn_get_mem_page(&uvm_page));
        uvm_pages.push_back(uvm_page);
    }

    return uvm_pages.size();
}

inline void map_uvm_pages(int reqId,
                        int layer_idx,
                        u64 req_offset,
                        CUdeviceptr kcache_ptr,
                        CUdeviceptr vcache_ptr,
                        NvU64 k_page,
                        NvU64 v_page) {
    CHECK_VATTN(vattn_mem_map((void*)(kcache_ptr + req_offset), k_page));
    CHECK_VATTN(vattn_mem_map((void*)(vcache_ptr + req_offset), v_page));
    uvm_pagemap[std::make_tuple(reqId, req_offset, layer_idx)] = std::make_pair(k_page, v_page);
}

/* NOTE: This function must be called after wait_kvcache_manager_sync */
void do_uvm_kvcache_cleanup() {
    u64 nelements = (max_batch_size * max_context_length * num_kv_heads * head_size);
    for(int j = 0; j < k_tensors.size(); j++) {
        CHECK_VATTN(vattn_free_reserved_address((void*)(k_tensors[j].data_ptr()), k_tensors[j].element_size() * nelements));
        CHECK_VATTN(vattn_free_reserved_address((void*)(v_tensors[j].data_ptr()), v_tensors[j].element_size() * nelements));
    }

    for(int i = 0; i < uvm_pages.size(); i++)
        CHECK_VATTN(vattn_release_mem_page(uvm_pages[i]));
}

#endif
