#include <stdio.h>
#include "vattn.h"

#define REQUESTS      100                      // num of requests
#define LAYERS        32                       // num of layers
#define TYPE          2                        // KV
#define MAPPING_SIZE  (256ULL * 1024)          // memory needed for 1 token per layer for k

int **buffers;
NvU64 *handles;
NvU64 num_handles;
NvU64 block_size;

void reserve_memory() {
    buffers = (int**)malloc(num_handles * sizeof(int*));
    for (int i = 0; i < num_handles; i++) {
        int *d = (int*)vattn_reserve_memory(VATTN_MEM_RESERVE_LOW_LIMIT);
        buffers[i] = d;
    }
}

void get_handles() {
    handles = (NvU64*)malloc(num_handles * sizeof(NvU64));
    for (int i = 0; i < num_handles; i++) {
        CHECK_VATTN(vattn_get_mem_handle(&handles[i]));
    }
}

void map_handles() {
    for (int i = 0; i < num_handles; i++) {
        CHECK_VATTN(vattn_mem_map((void*)buffers[i], handles[i]));
    }
}

void free_buffers() {
    for (int i = 0; i < num_handles; i++) {
        CHECK_VATTN(vattn_free_reserved_address((void*)buffers[i], VATTN_MEM_RESERVE_LOW_LIMIT));
    }
    free(buffers);
}

void release_handles() {
    for (int i = 0; i < num_handles; i++) {
        CHECK_VATTN(vattn_release_mem_handle(handles[i]));
    }
    free(handles);
}

void print_config() {
    printf("Number of handles: %llu\n", num_handles);
    printf("Number of requests: %d\n", REQUESTS);
    printf("Number of layers: %d\n", LAYERS);
    printf("Number of types: %d\n", TYPE);
    printf("Mapping size: %llu B\n", MAPPING_SIZE);
    printf("Block size: %llu B\n", block_size);
    printf("Total memory mapped: %llu MB\n", (num_handles * block_size) / (1024 * 1024));
}

int main(int argc, char **argv) {
    block_size = (VATTN_DEF_HANDLE_SIZE / 1024);
    if (argc > 1) {
        block_size = atoi(argv[1]);
    }
    block_size *= 1024;
    if (block_size != VATTN_HANDLE_SIZE_64K && block_size != VATTN_HANDLE_SIZE_128K && block_size != VATTN_HANDLE_SIZE_256K) {
        printf("Invalid block size: %llu. Only supports 64, 128 and 256 (default)\n", block_size);
        exit(-1);
    }
    // calculate the number of handles needed per token, per layer, per request for each KV
    num_handles = MAPPING_SIZE / block_size;
    if (num_handles <= 0)
        num_handles = 1;

    // calculate number of handles needed in total
    num_handles = (num_handles * TYPE * REQUESTS * LAYERS);

    print_config();
    /* setup environment */
    CHECK_VATTN(vattn_init(0, block_size));

    PROFILE_CALL(reserve_memory(), true);
    PROFILE_CALL(get_handles(), true);
    PROFILE_CALL(map_handles(), true);

    PROFILE_CALL(free_buffers(), true);
    PROFILE_CALL(release_handles(), true);
    return 0;
}
