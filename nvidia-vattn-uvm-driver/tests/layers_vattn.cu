#include <stdio.h>
#include "vattn.h"

#define REQUESTS    1
#define LAYERS      32
#define HANDLES     (REQUESTS * LAYERS)
#define TEST_VAL    42
#define INDEX_2     (VATTN_DEF_HANDLE_SIZE / sizeof(int))
#define ARRAY_SIZE  (VATTN_MEM_RESERVE_LOW_LIMIT / sizeof(int))

int *buffers[HANDLES];
NvU64 handles[HANDLES];
int host;

__global__ void test(int *d) {
    d[0] = TEST_VAL;
}

void reserve_memory() {
    for (int i = 0; i < HANDLES; i++) {
        int *d = (int*)vattn_reserve_memory(VATTN_MEM_RESERVE_LOW_LIMIT);
        buffers[i] = d;
    }
}

void get_handles() {
    for (int i = 0; i < HANDLES; i++) {
        CHECK_VATTN(vattn_get_mem_handle(&handles[i]));
    }
}

void map_handles() {
    for (int i = 0; i < HANDLES; i++) {
        CHECK_VATTN(vattn_mem_map((void*)buffers[i], handles[i]));
    }
}

void free_buffers() {
    for (int i = 0; i < HANDLES; i++) {
        CHECK_VATTN(vattn_free_reserved_address((void*)buffers[i], VATTN_MEM_RESERVE_LOW_LIMIT));
    }
}

void release_handles() {
    for (int i = 0; i < HANDLES; i++) {
        CHECK_VATTN(vattn_release_mem_handle(handles[i]));
    }
}

int main() {
    /* setup environment */
    CHECK_VATTN(vattn_init(0, VATTN_DEF_HANDLE_SIZE));

    PROFILE_CALL(reserve_memory(), true);
    PROFILE_CALL(get_handles(), true);
    PROFILE_CALL(map_handles(), true);

    for(int i = 0; i < HANDLES; i++) {
        test<<<1,1>>>(buffers[i]);
        cudaMemcpy(&host, buffers[i], sizeof(int), cudaMemcpyDeviceToHost);
        if (host != TEST_VAL) {
            printf("NOT YAY\n");
            goto out;
        }
    }
    printf("YAY\n");
out:
    PROFILE_CALL(free_buffers(), true);
    PROFILE_CALL(release_handles(), true);
    return 0;
}
