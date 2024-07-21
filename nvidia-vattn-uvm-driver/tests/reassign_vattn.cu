#include <stdio.h>
#include "vattn.h"

#define THREADS 1
#define BLOCKS 1

#define TEST_VAL_1    10
#define TEST_VAL_2    11
#define INDEX_2     (VATTN_DEF_HANDLE_SIZE / sizeof(int))
#define ARRAY_SIZE  (VATTN_MEM_RESERVE_LOW_LIMIT / sizeof(int))

int host[ARRAY_SIZE];

void __global__ test_1(int *a) {
    a[0] = TEST_VAL_1;
}

void __global__ test_2(int *a) {
    a[INDEX_2] = TEST_VAL_2;
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
    NvU64 handle;
    vattn_get_mem_handle(&handle);
    printf("HANDLE 1: %llx\n", handle);
    /* map handle to first 64k */
    vattn_mem_map((void*)d, handle);
    // Start the test
    test_1<<<1,1>>>(d);
    cudaMemcpy(host, d, sizeof(int), cudaMemcpyDeviceToHost);

    // Reassigning the same handle to a different offset
    printf("Reusing HANDLE 1: %llx\n", handle);
    /* map new handle to second 64k */
    vattn_mem_map((void*)(d + INDEX_2), handle);
    test_2<<<1,1>>>(d);
    cudaMemcpy(host + INDEX_2, d + INDEX_2, sizeof(int), cudaMemcpyDeviceToHost);

    if (host[0] == TEST_VAL_1 && host[INDEX_2] == TEST_VAL_2) {
        printf("YAY\n");
    }

    vattn_free_reserved_address((void*)d, VATTN_MEM_RESERVE_LOW_LIMIT);
    vattn_release_mem_handle(handle);
}
