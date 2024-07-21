#include <stdio.h>
#include "vattn.h"

#define THREADS 1
#define BLOCKS 1

#define TEST_VAL    10
#define INDEX_2     (VATTN_DEF_HANDLE_SIZE / sizeof(int))
#define ARRAY_SIZE  (VATTN_MEM_RESERVE_LOW_LIMIT / sizeof(int))

int host[ARRAY_SIZE];

void __global__ test(int *a) {
    /* access multiple indices within an array */
    a[0] = TEST_VAL;
    a[INDEX_2] = TEST_VAL + 10;
}

int main() {
    /* setup environment */
    if (vattn_init(0, VATTN_DEF_HANDLE_SIZE) != VATTN_OK) {
        printf("ERR!\n");
        exit(-1);
    }

    /* reserve memory */
    int *d = (int*)vattn_reserve_memory(VATTN_MEM_RESERVE_LOW_LIMIT);

    /* get_mem_handle */
    NvU64 handle_1;
    NvU64 handle_2;
    vattn_get_mem_handle(&handle_1);
    printf("HANDLE 1: %llx\n", handle_1);
    /* map handle to first 64k */
    vattn_get_mem_handle(&handle_2);
    printf("HANDLE 2: %llx\n", handle_2);
    /* map new handle to second 64k */
    vattn_mem_map((void*)d, handle_2);
    vattn_mem_map((void*)(d + INDEX_2), handle_1);

    // Start the test
    test<<<1,1>>>(d);
    cudaMemcpy(host, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host + INDEX_2, d + INDEX_2, sizeof(int), cudaMemcpyDeviceToHost);

    if (host[0] == TEST_VAL && host[INDEX_2] == (TEST_VAL + 10)) {
        printf("YAY\n");
    }
    vattn_free_reserved_address((void*)d, VATTN_MEM_RESERVE_LOW_LIMIT);
    // release handles
    vattn_release_mem_handle(handle_1);
    vattn_release_mem_handle(handle_2);
}
