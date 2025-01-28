#include <stdio.h>
#include <chrono>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#define THREADS 256
//#define ITERS 150

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(a, b) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count()
#define cudaError(a) { \
    cudaError_t error = a; \
    if (error != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
}
#define cudaErrChk() { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
}

/***********************************************
                Helper functions
 ***********************************************/
__global__ void reset_data(float *a, float *b, float *c, float *d, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = i; j < n; j += blockDim.x * gridDim.x) {
        a[j] = 1.0f;
        b[j] = 2.0f;
        c[j] = 1.0f;
        d[j] = 1.0f;
    }
}

__global__ void verify_data(float *a, float *b, float *c, float *d, size_t n1, size_t n2, size_t iters) {
    float c_d = 1.0f;
    for(int i = 0; i < iters; ++i)
        c_d = 2 * c_d;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = i; j < n1; j += blockDim.x * gridDim.x) {
        if(a[j] + b[j] + 1.0f != c[j]) {
            printf("Error at %ld: %f %f %f; wanted c = %f\n", j, a[j], b[j], c[j], a[j] + b[j] + 1.0f);
            return;
        }
    }
    for(size_t j = i; j < n2; j += blockDim.x * gridDim.x) {
        if(d[j] != c_d) {
            printf("Error %ld: Got %f wanted %f\n", j, d[j], c_d);
            return;
        }
    }
}

/***********************************************
                Evaluation functions
 ***********************************************/
__global__ void membound_kernel(float *a, float *b, float *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = i; j < n; j += blockDim.x * gridDim.x) {
        c[j] += a[j] + b[j];
        __syncthreads();
    }
}

__global__ void computebound_kernel(float *d, size_t n, const int ITERS) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = i; j < n; j += blockDim.x * gridDim.x) {
        for(int iters = 0; iters < ITERS; ++iters)
            d[j] = 2 * d[j];
        __syncthreads();
    }
}

__global__ void fused_kernel(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = i;
    size_t n = min(n1, n2);
    for (; j < n; j += blockDim.x * gridDim.x) {
        c[j] += a[j] + b[j];
        for(int iters = 0; iters < ITERS; ++iters)
            d[j] = 2 * d[j];
    }
    for(; j < n1; j += blockDim.x * gridDim.x)
        c[j] += a[j] + b[j];
    for(; j < n2; j += blockDim.x * gridDim.x)
        for(int iters = 0; iters < ITERS; ++iters)
            d[j] = 2 * d[j];
}

__global__ void fused_kernel_barrier(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = i;
    //size_t n = min(n1, n2);
    for (; j < n1; j += blockDim.x * gridDim.x) {
        c[j] += a[j] + b[j];
        __syncthreads();
    }
    for(j = i; j < n2; j += blockDim.x * gridDim.x){
        for(int iters = 0; iters < ITERS; ++iters)
            d[j] = 2 * d[j];
        __syncthreads();
    }
}

__global__ void fused_kernel2(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    extern __shared__ int mem[];
    size_t i = (blockIdx.x * blockDim.x / 2 + threadIdx.x % (blockDim.x / 2));
    if(threadIdx.x < blockDim.x / 2)
        for (size_t j = i; j < n1; j += blockDim.x * gridDim.x / 2) {
            c[j] += a[j] + b[j];
        }
    else
        for (size_t j = i; j < n2; j += blockDim.x * gridDim.x / 2) {
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
        }
}

#define MOD_DIV(a, b) (((a / b) + 1) * b)

__global__ void fused_kernel2_barrier(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    extern __shared__ int mem[];
    size_t i = (blockIdx.x * blockDim.x / 2 + threadIdx.x % (blockDim.x / 2));
    size_t n1_new = MOD_DIV(n1, (blockDim.x / 2));
    size_t n2_new = MOD_DIV(n2, (blockDim.x / 2));
    if(threadIdx.x < blockDim.x / 2)
        for (size_t j = i; j < n1_new; j += blockDim.x * gridDim.x / 2) {
            if(j < n1)
                c[j] += a[j] + b[j];
            asm volatile("bar.sync 0, %0;"::"n"(THREADS / 2));
        }
    else
        for (size_t j = i; j < n2_new; j += blockDim.x * gridDim.x / 2) {
            if(j < n2)
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
            asm volatile("bar.sync 1, %0;"::"n"(THREADS / 2));
        }
}

__global__ void fused_kernel3(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    size_t i = ((blockIdx.x % (gridDim.x / 2)) * blockDim.x + threadIdx.x);
    if(blockIdx.x < gridDim.x / 2)
        for (size_t j = i; j < n1; j += blockDim.x * gridDim.x / 2) {
            c[j] += a[j] + b[j];
        }
    else
        for (size_t j = i; j < n2; j += blockDim.x * gridDim.x / 2) {
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
        }
}

__global__ void fused_kernel3_barrier(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    size_t i = ((blockIdx.x % (gridDim.x / 2)) * blockDim.x + threadIdx.x);
    if(blockIdx.x < gridDim.x / 2)
        for (size_t j = i; j < n1; j += blockDim.x * gridDim.x / 2) {
            c[j] += a[j] + b[j];
            __syncthreads();
        }
    else
        for (size_t j = i; j < n2; j += blockDim.x * gridDim.x / 2) {
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
            __syncthreads();
        }
}

__global__ void fused_kernel_tbinterleave(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    size_t i = (blockIdx.x / 2) * blockDim.x + threadIdx.x;
    if(blockIdx.x % 2 == 0)
        for (size_t j = i; j < n1; j += blockDim.x * gridDim.x / 2) {
            c[j] += a[j] + b[j];
        }
    else
        for (size_t j = i; j < n2; j += blockDim.x * gridDim.x / 2) {
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
        }
}

__global__ void fused_kernel_tb_hwaware(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS, const int SMs, int *test) {
    __shared__ int bid, tag;
    if(threadIdx.x == 0) {
        int num_SMs, sm_id;
        asm volatile("mov.u32 %0, %nsmid;" : "=r"(num_SMs));
        asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id));
        tag = (atomicAdd(&test[sm_id], 1) % 2);
        bid = atomicAdd(&test[num_SMs + tag], 1);
        if(bid >= gridDim.x / 2) {
            tag = (tag + 1) % 2;
            bid = atomicAdd(&test[num_SMs + tag], 1);
        }
    }
    __syncthreads();
    size_t i = bid * blockDim.x + threadIdx.x;
    if(tag % 2 == 0)
        for (size_t j = i; j < n1; j += blockDim.x * gridDim.x / 2) {
            c[j] += a[j] + b[j];
        }
    else
        for (size_t j = i; j < n2; j += blockDim.x * gridDim.x / 2) {
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
        }
}

__global__ void fused_kernel_tb_hwaware_barrier(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS, const int SMs, int *test) {
    __shared__ int bid, tag;
    if(threadIdx.x == 0) {
        int num_SMs, sm_id;
        asm volatile("mov.u32 %0, %nsmid;" : "=r"(num_SMs));
        asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id));
        tag = (atomicAdd(&test[sm_id], 1) % 2);
        bid = atomicAdd(&test[num_SMs + tag], 1);
        if(bid >= gridDim.x / 2) {
            tag = (tag + 1) % 2;
            bid = atomicAdd(&test[num_SMs + tag], 1);
        }
    }
    __syncthreads();
    size_t i = bid * blockDim.x + threadIdx.x;
    if(tag % 2 == 0)
        for (size_t j = i; j < n1; j += blockDim.x * gridDim.x / 2) {
            c[j] += a[j] + b[j];
            __syncthreads();
        }
    else
        for (size_t j = i; j < n2; j += blockDim.x * gridDim.x / 2) {
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
            __syncthreads();
        }
}

__global__ void fused_kernel_tb_smbind_barrier(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS, const int SMs, int *test) {
    __shared__ int bid, tag;
    if(threadIdx.x == 0) {
        int num_SMs, sm_id;
        asm volatile("mov.u32 %0, %nsmid;" : "=r"(num_SMs));
        asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id));
        tag = (sm_id % 2);
        bid = atomicAdd(&test[num_SMs + tag], 1);
        if(bid >= gridDim.x / 2) {
            tag = (tag + 1) % 2;
            bid = atomicAdd(&test[num_SMs + tag], 1);
        }
    }
    __syncthreads();
    size_t i = bid * blockDim.x + threadIdx.x;
    if(tag % 2 == 0)
        for (size_t j = i; j < n1; j += blockDim.x * gridDim.x / 2) {
            c[j] += a[j] + b[j];
            __syncthreads();
        }
    else
        for (size_t j = i; j < n2; j += blockDim.x * gridDim.x / 2) {
            for(int iters = 0; iters < ITERS; ++iters)
                d[j] = 2 * d[j];
            __syncthreads();
        }
}

__global__ void tma_kernel(float *a, float *b, float *c, float *d, size_t n1, size_t n2, const int ITERS) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    /*__shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier1;
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier2;
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier3;
    if (block.thread_rank() == 0) {
        init(&barrier1, block.size()); // Friend function initializes barrier
        init(&barrier2, block.size()); // Friend function initializes barrier
        init(&barrier3, block.size()); // Friend function initializes barrier
    }
    block.sync();*/

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    __shared__ float s_a[THREADS], s_b[THREADS], s_c[THREADS];
    size_t gridSize = gridDim.x * blockDim.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = min(n1, n2);
    size_t j = i;
    for (; j < n; j += gridSize) {
        //size_t batchOffset = j / gridSize * gridSize + blockIdx.x * blockDim.x;
        pipe.producer_acquire();
        /*cuda::memcpy_async(block, s_a, a + batchOffset, sizeof(float) * block.size(), barrier1);
        cuda::memcpy_async(block, s_b, b + batchOffset, sizeof(float) * block.size(), barrier2);
        cuda::memcpy_async(block, s_c, c + batchOffset, sizeof(float) * block.size(), barrier3);*/
        cuda::memcpy_async(&s_a[threadIdx.x], &a[j], sizeof(float), pipe);
        cuda::memcpy_async(&s_b[threadIdx.x], &b[j], sizeof(float), pipe);
        cuda::memcpy_async(&s_c[threadIdx.x], &c[j], sizeof(float), pipe);
        
        pipe.producer_commit();
        for(int iters = 0; iters < ITERS; ++iters)
            d[j] = 2 * d[j];
        //barrier1.arrive_and_wait();
        //barrier2.arrive_and_wait();
        //barrier3.arrive_and_wait();
        pipe.consumer_wait();
        c[j] = s_a[threadIdx.x] + s_b[threadIdx.x] + s_c[threadIdx.x];
        pipe.consumer_release();
    }
    for(; j < n1; j += gridSize) {
        pipe.producer_acquire();
        cuda::memcpy_async(&s_a[threadIdx.x], &a[j], sizeof(float), pipe);
        cuda::memcpy_async(&s_b[threadIdx.x], &b[j], sizeof(float), pipe);
        cuda::memcpy_async(&s_c[threadIdx.x], &c[j], sizeof(float), pipe);
        pipe.producer_commit();
        pipe.consumer_wait();
        c[j] = s_a[threadIdx.x] + s_b[threadIdx.x] + s_c[threadIdx.x];
        pipe.consumer_release();
    }
    for(; j < n2; j += gridSize) {
        for(int iters = 0; iters < ITERS; ++iters)
            d[j] = 2 * d[j];
    }
}

#define START \
    reset_data<<<108, 512>>>(a, b, c, d, N); \
    cudaDeviceSynchronize(); \
    cudaErrChk(); \
    start = TIME_NOW;

#define END(name, n1, n2) \
    cudaErrChk(); \
    cudaDeviceSynchronize(); \
    cudaErrChk(); \
    end = TIME_NOW; \
    verify_data<<<108, 512>>>(a, b, c, d, n1, n2, iters); \
    printf("%s: %ld us; throughput %ld GigaOps/s\n", name, \
        TIME_DIFF(start, end), N / TIME_DIFF(start, end) / 1000L);

int main() {
    // 16 GB
    size_t N = (1L << 34L) / sizeof(float);
    float *a, *b, *c, *d;
    auto start = TIME_NOW, end = TIME_NOW;
    cudaStream_t stream1, stream2 ;
    cudaStreamCreate ( &stream1) ;
    cudaStreamCreate ( &stream2) ;

    printf("N = %zu GB\n", N * sizeof(float) / (1L << 30L));
    cudaMalloc((void **)&a, N * sizeof(float));
    cudaMalloc((void **)&b, N * sizeof(float));
    cudaMalloc((void **)&c, N * sizeof(float));
    cudaMalloc((void **)&d, N * sizeof(float));
    cudaErrChk();
    int iters = 200;
    const int MAX_SIZE = 200;
    //int size = MAX_SIZE;
    for(int size = 20; size <= MAX_SIZE; size += 20)
        for(int blks = 2048; blks <= 2048; blks *= 2) {
            float frac = (float)size / (float)MAX_SIZE;
            printf("Blks: %d, Iters: %d\n", blks, size);	
            START
            membound_kernel<<<blks, THREADS>>>(a, b, c, N);
            END("Mb", N, 0)

            START
            computebound_kernel<<<blks, THREADS>>>(d, N * frac, iters);
            END("Cb", 0, N * frac)
    
            START
            membound_kernel<<<blks, THREADS>>>(a, b, c, N);
            computebound_kernel<<<blks, THREADS>>>(d, N * frac, iters);
            END("serial", N, N * frac)
    
            START
            membound_kernel<<<blks, THREADS, 0, stream1>>>(a, b, c, N);
            computebound_kernel<<<blks, THREADS, 0, stream2>>>(d, N * frac, iters);
            END("kernel(streams)", N, N * frac)
    
            START
            fused_kernel<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters);
            END("intra-thread", N, N * frac)
    
            START
            fused_kernel_barrier<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters);
            END("intra-thread-barrier", N, N * frac)
            
            START
            fused_kernel2<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters);
            END("warp", N, N * frac)
            
            START
            fused_kernel2_barrier<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters);
            END("warp-barrier", N, N * frac)
            
            /*const int smem_size = 40 * 1024;
            auto kernel = &fused_kernel2;
            if (smem_size >= 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            }
            START
            kernel<<<blks / 2, 2 * THREADS, smem_size>>>(a, b, c, d, N, N * frac, iters);
            END("Warp-parallel-shmem", N, N * frac)
            */

            START
            fused_kernel3<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters);
            END("cta(sequential)", N, N * frac)

            START
            fused_kernel3_barrier<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters);
            END("cta(sequential)-barrier", N, N * frac)
            
            /*START
            fused_kernel_tbinterleave<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters);
            END("TB-interleaved", N, N * frac)*/

            int *test;
            cudaMalloc((void **)&test, sizeof(int) * 110);
            cudaMemset(test, 0, sizeof(int) * 110);
            START
            fused_kernel_tb_hwaware<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters, 108, test);
            END("cta(sm-aware)", N, N * frac)

            cudaMemset(test, 0, sizeof(int) * 110);
            START
            fused_kernel_tb_hwaware_barrier<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters, 108, test);
            END("cta(sm-aware)-barrier", N, N * frac)

            cudaMemset(test, 0, sizeof(int) * 110);
            START
            fused_kernel_tb_smbind_barrier<<<blks, THREADS>>>(a, b, c, d, N, N * frac, iters, 108, test);
            END("cta(smbind)-barrier", N, N * frac)

            printf("\n");
        }
    return 0;
}