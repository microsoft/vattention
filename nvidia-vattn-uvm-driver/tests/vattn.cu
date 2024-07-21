#include <stdio.h>
#include "vattn.h"

#define THREADS      1
#define BLOCKS       1
#define TEST_VAL     10
#define HANDLE_SIZE  (VATTN_HANDLE_SIZE_64K)
#define LAST_INDEX   (HANDLE_SIZE / sizeof(int))
int host[2];

void __global__ test(int *a) {
    /* 0 index */
    a[0] = TEST_VAL;
    /* last index in the mapped region */
    a[LAST_INDEX - 1] = TEST_VAL;
}

int main() {
    /* setup environment */
    CHECK_VATTN(vattn_init(0, HANDLE_SIZE));

    /* reserve memory */
    int *d = (int*)vattn_reserve_memory(VATTN_MEM_RESERVE_LOW_LIMIT);
    /* get_mem_handle */
    NvU64 handle;
    CHECK_VATTN(vattn_get_mem_handle(&handle));
    printf("HANDLE: %llx\n", handle);
    /* map handle */
    CHECK_VATTN(vattn_mem_map((void*)d, handle));

    // Start the test
    test<<<1,1>>>(d);
    cudaMemcpy(&host[0], d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host[1], d + LAST_INDEX - 1, sizeof(int), cudaMemcpyDeviceToHost);

    if (host[0] == TEST_VAL && host[1] == TEST_VAL)
        printf("SUCCESS!\n");

    CHECK_VATTN(vattn_free_reserved_address((void*)d, VATTN_MEM_RESERVE_LOW_LIMIT));
    CHECK_VATTN(vattn_release_mem_handle(handle));
}
