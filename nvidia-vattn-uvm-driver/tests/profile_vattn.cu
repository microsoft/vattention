#include <stdio.h>
#include "vattn.h"

#define THREADS 1
#define BLOCKS 1

#define TEST_VAL    10
#define HANDLE_SIZE (VATTN_HANDLE_SIZE_256K)
#define INDEX_1     (HANDLE_SIZE / sizeof(int))

int host[2];

void __global__ test(int *a) {
    /* access multiple indices within an array */
    a[0] = TEST_VAL;
    a[INDEX_1] = TEST_VAL;
}

int main() {
    /* setup environment */
    if (vattn_init(0, HANDLE_SIZE) != VATTN_OK) {
        printf("ERR!\n");
        exit(-1);
    }

    NvU64 handle;
    /* reserve memory */
    int *d;
    PROFILE_CALL(d = (int*)vattn_reserve_memory(VATTN_MEM_RESERVE_LOW_LIMIT), true);

    /* get_mem_handle */
    PROFILE_CALL(vattn_get_mem_handle(&handle), true);
    printf("HANDLE 1: %llx\n", handle);
    /* map first region with handle size (VATTN_DEF_HANDLE_SIZE) */
    PROFILE_CALL(vattn_mem_map((void*)d, handle), true);

    NvU64 handle_2;
    /* get_mem_handle */
    PROFILE_CALL(vattn_get_mem_handle(&handle_2), true);
    printf("HANDLE 2: %llx\n", handle_2);
    /* map second region with handle size (VATTN_DEF_HANDLE_SIZE) */
    PROFILE_CALL(vattn_mem_map((void*)(d + INDEX_1), handle_2), true);

    // Start the test
    test<<<1,1>>>(d);
    cudaMemcpy(host, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host + 1, d + INDEX_1, sizeof(int), cudaMemcpyDeviceToHost);

    if (host[0] == TEST_VAL && host[1] == TEST_VAL) {
        printf("YAY\n");
    }

    PROFILE_CALL(vattn_free_reserved_address((void*)d, VATTN_MEM_RESERVE_LOW_LIMIT), true);
    PROFILE_CALL(vattn_release_mem_handle(handle), true);
    PROFILE_CALL(vattn_release_mem_handle(handle_2), true);
}
