# CUDA Micro-Benchmarks

> Small, focused programs that isolate and measure **ONE specific GPU behavior**.
> Build intuition by seeing real numbers — not just reading about them.

**How to use this guide:**
1. Read the **Question** — understand what we're isolating
2. Compile and run the **Code** — `nvcc benchmark.cu -o benchmark && ./benchmark`
3. Compare your numbers to the **Expected Results**
4. Internalize the **Insight** — this is what changes how you write code

**Hardware reference:** Expected results target A100 (80 GB) and H100 SXM. Your numbers will differ — the *ratios* between measurements are what matter.

---

## Benchmark 1: Global Memory Bandwidth

**Question:** How close can a simple kernel get to the GPU's theoretical memory bandwidth?

**Code:**
```cuda
// bench_global_bw.cu — Measure global memory read, write, and copy bandwidth
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void read_kernel(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride)
        sum += src[i];
    if (sum == -999.0f) dst[0] = sum;  // prevent optimization
}

__global__ void write_kernel(float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        dst[i] = 1.0f;
}

__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        dst[i] = src[i];
}

int main() {
    const int N = 256 * 1024 * 1024;  // 1 GB of floats
    const int bytes = N * sizeof(float);
    const int ITERS = 20;

    float *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, bytes));
    CHECK(cudaMalloc(&d_dst, bytes));
    CHECK(cudaMemset(d_src, 1, bytes));
    CHECK(cudaMemset(d_dst, 0, bytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = 256;

    auto benchmark = [&](const char* name, auto kernel, int rw_bytes) {
        // Warmup
        kernel<<<blocks, threads>>>();
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            kernel<<<blocks, threads>>>();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        double sec = ms / 1000.0 / ITERS;
        double gbps = (double)rw_bytes / sec / 1e9;
        printf("%-12s  %7.1f GB/s  (%.3f ms)\n", name, gbps, ms / ITERS);
    };

    printf("Global Memory Bandwidth (N = %d floats, %.0f MB)\n", N, bytes / 1e6);
    printf("%-12s  %9s  %s\n", "Operation", "BW", "Time");
    printf("--------------------------------------------\n");

    benchmark("Read",  [&]() { read_kernel<<<blocks, threads>>>(d_src, d_dst, N); }, bytes);
    benchmark("Write", [&]() { write_kernel<<<blocks, threads>>>(d_dst, N); }, bytes);
    benchmark("Copy",  [&]() { copy_kernel<<<blocks, threads>>>(d_src, d_dst, N); }, 2 * bytes);

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return 0;
}
```

**Expected Results:**

| Operation | A100 (GB/s) | H100 (GB/s) | % of Peak |
|-----------|-------------|-------------|-----------|
| Read      | ~1,500      | ~2,600      | ~75-80%   |
| Write     | ~1,400      | ~2,500      | ~72-78%   |
| Copy      | ~1,550      | ~2,700      | ~78-83%   |
| **Theoretical Peak** | **2,039** | **3,350** | **100%** |

**Insight:** You'll never hit 100% of theoretical peak. 75-85% is **excellent** for a simple kernel. The gap comes from: (1) memory controller overhead, (2) address computation, (3) ECC if enabled. If you're below 60%, your access pattern has a problem.

---

## Benchmark 2: Coalesced vs Strided Access

**Question:** How much bandwidth do we lose when threads access non-contiguous memory addresses?

**Code:**
```cuda
// bench_coalesced.cu — Measure bandwidth at different access strides
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

__global__ void strided_read(const float* data, float* out, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (int i = tid * stride; i < n; i += total_threads * stride)
        sum += data[i];
    if (sum == -999.0f) out[0] = sum;
}

int main() {
    const int N = 256 * 1024 * 1024;
    const int bytes = N * sizeof(float);
    const int ITERS = 10;

    float *d_data, *d_out;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemset(d_data, 1, bytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = 512;

    printf("Coalesced vs Strided Memory Access\n");
    printf("%-8s  %10s  %12s\n", "Stride", "BW (GB/s)", "Slowdown");
    printf("--------------------------------------\n");

    float baseline_gbps = 0;

    for (int stride : {1, 2, 4, 8, 16, 32}) {
        int elements_accessed = N / stride;
        int access_bytes = elements_accessed * sizeof(float);

        strided_read<<<blocks, threads>>>(d_data, d_out, N, stride);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            strided_read<<<blocks, threads>>>(d_data, d_out, N, stride);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        double sec = ms / 1000.0 / ITERS;
        double gbps = access_bytes / sec / 1e9;

        if (stride == 1) baseline_gbps = gbps;
        printf("%-8d  %10.1f  %11.1fx\n", stride, gbps, baseline_gbps / gbps);
    }

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_out));
    return 0;
}
```

**Expected Results:**

| Stride | BW (GB/s) A100 | Slowdown | Why |
|--------|---------------|----------|-----|
| 1      | ~1,500        | 1.0x     | Perfect coalescing: 1 cache line per warp |
| 2      | ~750          | 2.0x     | 50% of each cache line wasted |
| 4      | ~380          | 4.0x     | 25% utilization per cache line |
| 8      | ~190          | 8.0x     | 12.5% utilization |
| 16     | ~95           | 16.0x   | Each thread triggers its own cache line |
| 32     | ~47           | 32.0x   | Worst case: 32 cache lines per warp access |

**Insight:** The bandwidth drops **linearly** with stride — there is no partial recovery. At stride=32, each thread in a warp loads a separate 128-byte cache line but uses only 4 bytes (3.1% utilization). This is the single most important optimization in CUDA: **make adjacent threads access adjacent memory**.

---

## Benchmark 3: Shared Memory Bandwidth and Bank Conflicts

**Question:** How fast is shared memory, and what happens when multiple threads hit the same bank?

**Code:**
```cuda
// bench_smem_banks.cu — Shared memory bandwidth with varying bank conflicts
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

// Stride=1: no bank conflicts (each thread hits a different bank)
__global__ void smem_no_conflict(float* out, int iters) {
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iters; i++)
        sum += smem[tid];               // stride=1, 0 conflicts

    if (sum == -1.0f) out[0] = sum;
}

// Stride=2: 2-way bank conflict
__global__ void smem_2way_conflict(float* out, int iters) {
    __shared__ float smem[2048];
    int tid = threadIdx.x;
    smem[tid * 2] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iters; i++)
        sum += smem[tid * 2];            // stride=2, 2-way conflict

    if (sum == -1.0f) out[0] = sum;
}

// Stride=32: 32-way bank conflict (all threads hit same bank)
__global__ void smem_32way_conflict(float* out, int iters) {
    __shared__ float smem[32 * 1024 / 4];
    int tid = threadIdx.x;
    int idx = (tid * 32) % (32 * 1024 / 4);
    smem[idx] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < iters; i++)
        sum += smem[idx];                // stride=32, 32-way conflict

    if (sum == -1.0f) out[0] = sum;
}

int main() {
    const int ITERS = 100000;
    float *d_out;
    CHECK(cudaMalloc(&d_out, sizeof(float)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = 1;  // Single block to isolate shared memory behavior

    auto bench = [&](const char* name, auto kernel) {
        kernel<<<blocks, threads>>>(d_out, ITERS);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int rep = 0; rep < 10; rep++)
            kernel<<<blocks, threads>>>(d_out, ITERS);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        long long total_ops = (long long)threads * ITERS * 10;
        double gops = total_ops / (ms / 1000.0) / 1e9;
        printf("%-22s  %8.1f Gops/s  (%.3f ms)\n", name, gops, ms / 10);
    };

    printf("Shared Memory Bank Conflict Benchmark\n");
    printf("%-22s  %10s  %s\n", "Pattern", "Throughput", "Time");
    printf("----------------------------------------------\n");

    bench("No conflict (stride=1)", smem_no_conflict);
    bench("2-way (stride=2)",       smem_2way_conflict);
    bench("32-way (stride=32)",     smem_32way_conflict);

    CHECK(cudaFree(d_out));
    return 0;
}
```

**Expected Results:**

| Pattern | Throughput (A100) | Relative |
|---------|-------------------|----------|
| No conflict (stride=1) | ~950 Gops/s | 1.0x |
| 2-way (stride=2) | ~475 Gops/s | 0.5x |
| 32-way (stride=32) | ~30 Gops/s | 0.03x |

**Insight:** Shared memory has **32 banks**, each 4 bytes wide. When two threads in a warp access the same bank, the access is serialized. 32-way conflict means all 32 threads queue up — throughput drops **~32x**. Fix: pad shared memory arrays (`float smem[32][33]` instead of `[32][32]`) to shift columns across banks.

---

## Benchmark 4: Shared Memory vs Global Memory Latency

**Question:** What is the raw latency of a single load from shared memory vs global memory?

**Code:**
```cuda
// bench_latency.cu — Pointer-chase latency measurement
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

// Pointer chase through global memory — measures true latency
__global__ void global_latency(int* chain, int* out, int steps) {
    int idx = 0;
    for (int i = 0; i < steps; i++)
        idx = chain[idx];  // dependent load — cannot be pipelined
    out[0] = idx;
}

// Pointer chase through shared memory
__global__ void shared_latency(int* init_data, int* out, int steps, int chain_len) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;

    // Load chain into shared memory
    for (int i = tid; i < chain_len; i += blockDim.x)
        smem[i] = init_data[i];
    __syncthreads();

    if (tid == 0) {
        int idx = 0;
        for (int i = 0; i < steps; i++)
            idx = smem[idx];  // dependent load from shared memory
        out[0] = idx;
    }
}

int main() {
    const int CHAIN_LEN = 1024;
    const int STEPS = 100000;

    // Build a random pointer-chase chain (permutation)
    int h_chain[CHAIN_LEN];
    // Simple sequential chain for predictable behavior
    for (int i = 0; i < CHAIN_LEN; i++)
        h_chain[i] = (i + 1) % CHAIN_LEN;

    // Shuffle to defeat prefetcher
    for (int i = CHAIN_LEN - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = h_chain[i]; h_chain[i] = h_chain[j]; h_chain[j] = tmp;
    }

    int *d_chain, *d_out;
    CHECK(cudaMalloc(&d_chain, CHAIN_LEN * sizeof(int)));
    CHECK(cudaMalloc(&d_out, sizeof(int)));
    CHECK(cudaMemcpy(d_chain, h_chain, CHAIN_LEN * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Global memory latency
    global_latency<<<1, 1>>>(d_chain, d_out, STEPS);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    global_latency<<<1, 1>>>(d_chain, d_out, STEPS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gms;
    CHECK(cudaEventElapsedTime(&gms, start, stop));
    double g_ns = (gms * 1e6) / STEPS;

    // Shared memory latency
    shared_latency<<<1, 256, CHAIN_LEN * sizeof(int)>>>(d_chain, d_out, STEPS, CHAIN_LEN);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    shared_latency<<<1, 256, CHAIN_LEN * sizeof(int)>>>(d_chain, d_out, STEPS, CHAIN_LEN);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float sms;
    CHECK(cudaEventElapsedTime(&sms, start, stop));
    double s_ns = (sms * 1e6) / STEPS;

    printf("Memory Latency Comparison\n");
    printf("%-18s  %8s\n", "Memory Type", "Latency");
    printf("-----------------------------------\n");
    printf("%-18s  %6.1f ns  (~%d cycles)\n", "Global (L2 miss)", g_ns, (int)(g_ns * 1.5));
    printf("%-18s  %6.1f ns  (~%d cycles)\n", "Shared memory",   s_ns, (int)(s_ns * 1.5));
    printf("%-18s  %6.1fx\n", "Ratio (G/S)", g_ns / s_ns);

    CHECK(cudaFree(d_chain));
    CHECK(cudaFree(d_out));
    return 0;
}
```

**Expected Results:**

| Memory Type | Latency (A100) | Cycles (~1.4 GHz) |
|-------------|----------------|-------------------|
| Global (DRAM, L2 miss) | ~400-600 ns | ~500-800 cycles |
| Global (L2 hit) | ~150-200 ns | ~200-280 cycles |
| Global (L1 hit) | ~30-35 ns | ~40-50 cycles |
| **Shared memory** | **~5-7 ns** | **~7-10 cycles** |
| Ratio (DRAM / Shared) | **~80-100x** | — |

**Insight:** Shared memory is essentially a software-managed L1 cache. A single global memory load takes 500+ clock cycles — during which the SM must have other warps ready to execute. This is why **occupancy matters**: you need enough warps to hide this latency. Shared memory lets you avoid it entirely for reused data.

---

## Benchmark 5: Pinned vs Pageable Host Memory Transfer

**Question:** How much faster is `cudaMemcpy` when using pinned (page-locked) host memory?

**Code:**
```cuda
// bench_pinned.cu — Pinned vs pageable host-device transfer
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

void benchmark_transfer(const char* label, int bytes, bool pinned) {
    float *h_data, *d_data;

    if (pinned)
        CHECK(cudaMallocHost(&h_data, bytes));
    else
        h_data = (float*)malloc(bytes);

    CHECK(cudaMalloc(&d_data, bytes));
    memset(h_data, 0, bytes);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    const int ITERS = 20;

    // Host to Device
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float h2d_ms;
    CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
    double h2d_gbps = (double)bytes * ITERS / (h2d_ms / 1000.0) / 1e9;

    // Device to Host
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float d2h_ms;
    CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
    double d2h_gbps = (double)bytes * ITERS / (d2h_ms / 1000.0) / 1e9;

    printf("%-12s  H2D: %6.1f GB/s   D2H: %6.1f GB/s\n", label, h2d_gbps, d2h_gbps);

    CHECK(cudaFree(d_data));
    if (pinned) CHECK(cudaFreeHost(h_data));
    else free(h_data);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("Pinned vs Pageable Memory Transfer\n");
    printf("%-12s  %-15s  %-15s\n", "Type", "H2D BW", "D2H BW");
    printf("-----------------------------------------------\n");

    int sizes[] = {1 << 20, 16 << 20, 256 << 20};
    const char* labels[] = {"1 MB", "16 MB", "256 MB"};

    for (int s = 0; s < 3; s++) {
        printf("\n--- Transfer size: %s ---\n", labels[s]);
        benchmark_transfer("Pageable", sizes[s], false);
        benchmark_transfer("Pinned",   sizes[s], true);
    }

    return 0;
}
```

**Expected Results (256 MB transfer, PCIe Gen4 x16):**

| Memory Type | H2D (GB/s) | D2H (GB/s) | PCIe Utilization |
|-------------|-----------|-----------|------------------|
| Pageable    | ~8-12     | ~8-10     | ~35-50%          |
| Pinned      | ~24-26    | ~24-26    | ~95-100%         |
| **Speedup** | **2-3x**  | **2-3x**  | —                |

**Insight:** Pageable memory requires an extra copy: `user buffer → pinned staging buffer → GPU`. The driver does this transparently but it halves throughput. For any transfer >1 MB that happens repeatedly, **always use pinned memory**. But don't over-allocate — pinned memory is locked in physical RAM and reduces memory available to the OS.

---

## Benchmark 6: PCIe vs NVLink Bandwidth

**Question:** What is the actual bandwidth hierarchy for different transfer paths?

**Code:**
```cuda
// bench_interconnect.cu — Measure transfer bandwidth across different paths
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

double measure_bandwidth(int src_dev, int dst_dev, int bytes, bool is_h2d, bool is_d2h) {
    float *src_ptr, *dst_ptr;
    const int ITERS = 20;

    if (is_h2d) {
        CHECK(cudaMallocHost(&src_ptr, bytes));
        CHECK(cudaSetDevice(dst_dev));
        CHECK(cudaMalloc(&dst_ptr, bytes));
    } else if (is_d2h) {
        CHECK(cudaSetDevice(src_dev));
        CHECK(cudaMalloc(&src_ptr, bytes));
        CHECK(cudaMallocHost(&dst_ptr, bytes));
    } else {
        CHECK(cudaSetDevice(src_dev));
        CHECK(cudaMalloc(&src_ptr, bytes));
        CHECK(cudaSetDevice(dst_dev));
        CHECK(cudaMalloc(&dst_ptr, bytes));
        // Enable peer access if possible
        int can_access;
        CHECK(cudaDeviceCanAccessPeer(&can_access, dst_dev, src_dev));
        if (can_access) {
            CHECK(cudaSetDevice(dst_dev));
            cudaDeviceEnablePeerAccess(src_dev, 0);
            CHECK(cudaSetDevice(src_dev));
            cudaDeviceEnablePeerAccess(dst_dev, 0);
        }
    }

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        CHECK(cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDefault));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gbps = (double)bytes * ITERS / (ms / 1000.0) / 1e9;

    if (is_h2d || is_d2h) {
        if (is_h2d) { CHECK(cudaFreeHost(src_ptr)); CHECK(cudaFree(dst_ptr)); }
        else        { CHECK(cudaFree(src_ptr)); CHECK(cudaFreeHost(dst_ptr)); }
    } else {
        CHECK(cudaSetDevice(src_dev)); CHECK(cudaFree(src_ptr));
        CHECK(cudaSetDevice(dst_dev)); CHECK(cudaFree(dst_ptr));
    }
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return gbps;
}

int main() {
    int device_count;
    CHECK(cudaGetDeviceCount(&device_count));

    const int SIZE = 256 << 20;  // 256 MB

    printf("Interconnect Bandwidth Benchmark\n");
    printf("Transfer size: %d MB, Devices: %d\n\n", SIZE >> 20, device_count);
    printf("%-25s  %10s\n", "Path", "BW (GB/s)");
    printf("--------------------------------------\n");

    // Host <-> Device 0
    double h2d = measure_bandwidth(0, 0, SIZE, true, false);
    printf("%-25s  %10.1f\n", "Host -> GPU 0 (pinned)", h2d);

    double d2h = measure_bandwidth(0, 0, SIZE, false, true);
    printf("%-25s  %10.1f\n", "GPU 0 -> Host (pinned)", d2h);

    // Device to Device (same GPU)
    float *d_src, *d_dst;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d_src, SIZE));
    CHECK(cudaMalloc(&d_dst, SIZE));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 20; i++)
        CHECK(cudaMemcpy(d_dst, d_src, SIZE, cudaMemcpyDeviceToDevice));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    double d2d = (double)SIZE * 20 / (ms / 1000.0) / 1e9;
    printf("%-25s  %10.1f\n", "GPU 0 -> GPU 0 (D2D)", d2d);
    CHECK(cudaFree(d_src)); CHECK(cudaFree(d_dst));

    // GPU to GPU (if multi-GPU)
    if (device_count >= 2) {
        double p2p = measure_bandwidth(0, 1, SIZE, false, false);
        printf("%-25s  %10.1f\n", "GPU 0 -> GPU 1 (P2P)", p2p);
    }

    printf("\n--- Bandwidth Hierarchy ---\n");
    printf("HBM (on-chip):  ~2000-3350 GB/s\n");
    printf("NVLink 4.0:     ~450-900 GB/s (bidi)\n");
    printf("PCIe Gen5 x16:  ~64 GB/s (bidi)\n");
    printf("PCIe Gen4 x16:  ~32 GB/s (bidi)\n");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return 0;
}
```

**Expected Results:**

| Path | Bandwidth | Technology |
|------|-----------|------------|
| GPU DRAM (on-chip) | 2,039 / 3,350 GB/s | HBM2e / HBM3 |
| GPU 0 ↔ GPU 1 (NVLink) | 300-450 GB/s (uni) | NVLink 4.0 |
| Host → GPU (pinned) | 25-26 GB/s | PCIe Gen4 x16 |
| GPU → Host (pinned) | 25-26 GB/s | PCIe Gen4 x16 |
| Host → GPU (pageable) | 10-12 GB/s | PCIe + staging |

**Insight:** The bandwidth hierarchy spans **100x**: HBM → NVLink → PCIe. This is why data placement is everything. Moving 1 GB over PCIe takes 40 ms; reading it from HBM takes 0.5 ms. **Keep data on the GPU**. When multi-GPU, prefer NVLink-connected pairs (check `nvidia-smi topo -m`).

---

## Benchmark 7: Warp Divergence Cost

**Question:** How much throughput do we lose when threads in a warp take different branches?

**Code:**
```cuda
// bench_divergence.cu — Measure cost of warp divergence
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

// No divergence: all threads take the same path
__global__ void no_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // All threads do the same work
        for (int i = 0; i < 100; i++)
            val = val * 1.01f + 0.5f;
        data[idx] = val;
    }
}

// 50% divergence: even/odd threads take different paths
__global__ void half_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        if (threadIdx.x % 2 == 0) {
            for (int i = 0; i < 100; i++)
                val = val * 1.01f + 0.5f;
        } else {
            for (int i = 0; i < 100; i++)
                val = val * 0.99f - 0.5f;
        }
        data[idx] = val;
    }
}

// 100% divergence: every thread takes a unique path length
__global__ void full_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        int iters = 50 + (threadIdx.x % 32) * (100 / 32);  // 50-150 range
        for (int i = 0; i < iters; i++)
            val = val * 1.01f + 0.5f;
        data[idx] = val;
    }
}

int main() {
    const int N = 16 * 1024 * 1024;
    const int bytes = N * sizeof(float);
    const int ITERS = 50;

    float *d_data;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMemset(d_data, 0, bytes));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto bench = [&](const char* name, auto kernel) {
        kernel<<<blocks, threads>>>(d_data, N);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            kernel<<<blocks, threads>>>(d_data, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("%-22s  %.3f ms/iter\n", name, ms / ITERS);
        return ms / ITERS;
    };

    printf("Warp Divergence Benchmark (%d M elements)\n", N / (1024 * 1024));
    printf("%-22s  %s\n", "Pattern", "Time");
    printf("--------------------------------------\n");

    float t0 = bench("0% divergence",   no_divergence);
    float t50 = bench("50% divergence",  half_divergence);
    float t100 = bench("100% divergence", full_divergence);

    printf("\n--- Analysis ---\n");
    printf("50%% divergence overhead:  %.1fx slower\n", t50 / t0);
    printf("100%% divergence overhead: %.1fx slower\n", t100 / t0);

    CHECK(cudaFree(d_data));
    return 0;
}
```

**Expected Results:**

| Pattern | Time (ms) | Relative | Explanation |
|---------|-----------|----------|-------------|
| 0% divergence | ~0.35 | 1.0x | All 32 threads execute together |
| 50% divergence (even/odd) | ~0.65 | ~1.85x | Warp executes both paths serially |
| 100% divergence (variable) | ~0.70 | ~2.0x | Worst case: all paths serialized |

**Insight:** With 50% divergence, the warp executes BOTH branches — the if-path AND the else-path — masking threads that shouldn't participate. It takes roughly **2x** as long, not 1.5x. The cost is proportional to the number of **distinct execution paths** within a warp, not the fraction of threads diverging. Fix: reorganize data so threads in the same warp follow the same path.

---

## Benchmark 8: Arithmetic Throughput (FP32, FP16, INT32, FP64)

**Question:** What is the actual compute throughput for different data types?

**Code:**
```cuda
// bench_arithmetic.cu — Measure peak FLOPS for different types
// Compile with: nvcc -arch=sm_80 bench_arithmetic.cu -o bench_arithmetic
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

template<typename T>
__global__ void fma_kernel(T* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T a = (T)1.0001, b = (T)0.9999, c = (T)0.5;

    // 128 FMA operations — all dependent to prevent optimization
    #pragma unroll
    for (int rep = 0; rep < 1024; rep++) {
        a = a * b + c;  b = b * a + c;  c = c * b + a;  a = a * c + b;
        b = b * c + a;  c = c * a + b;  a = a * b + c;  b = b * a + c;
    }

    if (idx < n) out[idx] = a + b + c;
}

// Special FP16 version using half2 for 2x throughput
__global__ void fma_fp16_kernel(half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    half2 a = __float2half2_rn(1.0001f);
    half2 b = __float2half2_rn(0.9999f);
    half2 c = __float2half2_rn(0.5f);

    #pragma unroll
    for (int rep = 0; rep < 1024; rep++) {
        a = __hfma2(a, b, c); b = __hfma2(b, a, c);
        c = __hfma2(c, b, a); a = __hfma2(a, c, b);
        b = __hfma2(b, c, a); c = __hfma2(c, a, b);
        a = __hfma2(a, b, c); b = __hfma2(b, a, c);
    }

    if (idx < n) out[idx] = __low2half(a);
}

int main() {
    const int N = 1024 * 1024;
    const int THREADS = 256;
    const int BLOCKS = 256;
    const long long OPS_PER_THREAD = 1024LL * 8 * 2;  // 1024 reps * 8 FMAs * 2 ops each

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto bench = [&](const char* name, auto kernel, long long ops_per_thread, int extra_factor = 1) {
        // Allocate minimal output buffer
        void* d_out;
        CHECK(cudaMalloc(&d_out, N * sizeof(double)));

        kernel<<<BLOCKS, THREADS>>>((decltype(+kernel)::argument_type)d_out, N);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 10; i++)
            kernel<<<BLOCKS, THREADS>>>((decltype(+kernel)::argument_type)d_out, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        long long total_ops = (long long)BLOCKS * THREADS * ops_per_thread * 10 * extra_factor;
        double tflops = total_ops / (ms / 1000.0) / 1e12;
        printf("%-10s  %8.2f TFLOPS  (%.3f ms)\n", name, tflops, ms / 10);

        CHECK(cudaFree(d_out));
    };

    printf("Arithmetic Throughput Benchmark\n");
    printf("%-10s  %10s  %s\n", "Type", "Throughput", "Time");
    printf("---------------------------------------\n");

    // Need to handle each type separately due to template issues
    float* f32_out;   CHECK(cudaMalloc(&f32_out, N * sizeof(float)));
    double* f64_out;  CHECK(cudaMalloc(&f64_out, N * sizeof(double)));
    half* f16_out;    CHECK(cudaMalloc(&f16_out, N * sizeof(half)));

    auto bench_type = [&](const char* name, auto kernel, void* out, long long ops, int factor = 1) {
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 10; i++)
            kernel<<<BLOCKS, THREADS>>>();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        long long total_ops = (long long)BLOCKS * THREADS * ops * 10 * factor;
        double tflops = total_ops / (ms / 1000.0) / 1e12;
        printf("%-10s  %8.2f TFLOPS  (%.3f ms)\n", name, tflops, ms / 10);
    };

    bench_type("FP32", [&](){ fma_kernel<<<BLOCKS, THREADS>>>(f32_out, N); },
               f32_out, OPS_PER_THREAD);
    bench_type("FP64", [&](){ fma_kernel<<<BLOCKS, THREADS>>>(f64_out, N); },
               f64_out, OPS_PER_THREAD);
    bench_type("FP16", [&](){ fma_fp16_kernel<<<BLOCKS, THREADS>>>(f16_out, N); },
               f16_out, OPS_PER_THREAD, 2);  // half2 does 2 ops per instruction

    printf("\n--- Theoretical Peaks ---\n");
    printf("A100: FP32=19.5 TFLOPS, FP64=9.7 TFLOPS, FP16=78 TFLOPS (Tensor)\n");
    printf("H100: FP32=67 TFLOPS, FP64=34 TFLOPS, FP16=990 TFLOPS (Tensor)\n");

    CHECK(cudaFree(f32_out));
    CHECK(cudaFree(f64_out));
    CHECK(cudaFree(f16_out));
    return 0;
}
```

**Expected Results:**

| Type | A100 Achieved | A100 Peak | Utilization |
|------|---------------|-----------|-------------|
| FP32 | ~16 TFLOPS | 19.5 TFLOPS | ~82% |
| FP64 | ~8 TFLOPS | 9.7 TFLOPS | ~82% |
| FP16 (CUDA cores) | ~35 TFLOPS | 78 TFLOPS | ~45% |
| FP16 (Tensor cores) | ~250+ TFLOPS | 312 TFLOPS | ~80% |

**Insight:** FP64 is exactly **half** of FP32 on A100 (1:2 ratio). FP16 on CUDA cores is 2x FP32 (two operations per instruction via `half2`). But Tensor Cores blow everything away — they're **4-16x** faster for FP16 matrix ops. This is why all modern ML uses Tensor Cores via cuBLAS/cuDNN, not hand-written CUDA kernels.

---

## Benchmark 9: Atomic Operation Contention

**Question:** How much does atomic contention cost, and does reducing contention help?

**Code:**
```cuda
// bench_atomics.cu — Atomic contention: single location vs distributed
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

// All threads contend on a single memory location
__global__ void atomic_single(int* counter) {
    for (int i = 0; i < 1000; i++)
        atomicAdd(counter, 1);
}

// Each warp has its own counter (32x less contention)
__global__ void atomic_per_warp(int* counters) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    for (int i = 0; i < 1000; i++)
        atomicAdd(&counters[warp_id], 1);
}

// Each thread has its own counter (no contention, but uses more memory)
__global__ void atomic_per_thread(int* counters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 1000; i++)
        atomicAdd(&counters[tid], 1);
}

// Shared memory atomics with final reduction
__global__ void atomic_smem_reduce(int* global_counter) {
    __shared__ int smem_counter;
    if (threadIdx.x == 0) smem_counter = 0;
    __syncthreads();

    for (int i = 0; i < 1000; i++)
        atomicAdd(&smem_counter, 1);
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(global_counter, smem_counter);
}

int main() {
    const int THREADS = 256;
    const int BLOCKS = 256;
    const int TOTAL_THREADS = THREADS * BLOCKS;
    const int WARPS = TOTAL_THREADS / 32;

    int *d_single, *d_warp, *d_thread, *d_smem;
    CHECK(cudaMalloc(&d_single, sizeof(int)));
    CHECK(cudaMalloc(&d_warp, WARPS * sizeof(int)));
    CHECK(cudaMalloc(&d_thread, TOTAL_THREADS * sizeof(int)));
    CHECK(cudaMalloc(&d_smem, sizeof(int)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto bench = [&](const char* name, auto kernel) {
        CHECK(cudaMemset(d_single, 0, sizeof(int)));
        CHECK(cudaMemset(d_warp, 0, WARPS * sizeof(int)));
        CHECK(cudaMemset(d_thread, 0, TOTAL_THREADS * sizeof(int)));
        CHECK(cudaMemset(d_smem, 0, sizeof(int)));

        kernel<<<BLOCKS, THREADS>>>();
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            kernel<<<BLOCKS, THREADS>>>();
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        long long ops = (long long)TOTAL_THREADS * 1000 * 10;
        double gops = ops / (ms / 1000.0) / 1e9;
        printf("%-24s  %8.3f ms  %8.1f Gops/s\n", name, ms / 10, gops);
        return ms / 10;
    };

    printf("Atomic Contention Benchmark (%d threads, 1000 atomics each)\n", TOTAL_THREADS);
    printf("%-24s  %10s  %10s\n", "Strategy", "Time", "Throughput");
    printf("---------------------------------------------------\n");

    float t1 = bench("1 location (max contend)", [&](){ atomic_single<<<BLOCKS, THREADS>>>(d_single); });
    float t2 = bench("Per-warp (/32 contend)",   [&](){ atomic_per_warp<<<BLOCKS, THREADS>>>(d_warp); });
    float t3 = bench("Per-thread (0 contend)",   [&](){ atomic_per_thread<<<BLOCKS, THREADS>>>(d_thread); });
    float t4 = bench("Shared mem + reduce",      [&](){ atomic_smem_reduce<<<BLOCKS, THREADS>>>(d_smem); });

    printf("\n--- Speedup vs single-location ---\n");
    printf("Per-warp:         %.0fx faster\n", t1 / t2);
    printf("Per-thread:       %.0fx faster\n", t1 / t3);
    printf("Shared + reduce:  %.0fx faster\n", t1 / t4);

    CHECK(cudaFree(d_single));
    CHECK(cudaFree(d_warp));
    CHECK(cudaFree(d_thread));
    CHECK(cudaFree(d_smem));
    return 0;
}
```

**Expected Results:**

| Strategy | Time (ms) | Throughput | Speedup |
|----------|-----------|------------|---------|
| 1 location (all contend) | ~25 ms | ~2.6 Gops/s | 1x |
| Per-warp (1/32 contention) | ~1.5 ms | ~43 Gops/s | ~17x |
| Per-thread (no contention) | ~0.25 ms | ~260 Gops/s | ~100x |
| Shared mem + reduce | ~0.8 ms | ~80 Gops/s | ~30x |

**Insight:** A single contended `atomicAdd` serializes the **entire GPU**. 65,536 threads fighting for one int is a disaster. The fix pattern: **hierarchical reduction** — accumulate in registers → reduce in shared memory → single atomic to global. This is why `cub::DeviceReduce` exists.

---

## Benchmark 10: Kernel Launch Overhead

**Question:** How long does it take just to *launch* a kernel, ignoring the work it does?

**Code:**
```cuda
// bench_launch.cu — Measure kernel launch overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

__global__ void empty_kernel() {}

__global__ void tiny_kernel(float* data) {
    data[threadIdx.x] = 1.0f;
}

__global__ void real_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++)
            val = val * 1.001f + 0.001f;
        data[idx] = val;
    }
}

int main() {
    const int N = 1 << 20;
    float* d_data;
    CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK(cudaMemset(d_data, 0, N * sizeof(float)));
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    const int LAUNCHES = 10000;

    // Measure host-side launch latency (async)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < LAUNCHES; i++)
        empty_kernel<<<1, 1>>>();
    auto t1 = std::chrono::high_resolution_clock::now();
    CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    double async_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / LAUNCHES;
    double total_us = std::chrono::duration<double, std::micro>(t2 - t0).count() / LAUNCHES;

    // Measure GPU-side execution (including launch)
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < LAUNCHES; i++)
        empty_kernel<<<1, 1>>>();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gpu_ms;
    CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    double gpu_us = (gpu_ms * 1000.0) / LAUNCHES;

    // Measure with sync after each launch
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        empty_kernel<<<1, 1>>>();
        CHECK(cudaDeviceSynchronize());
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double sync_us = std::chrono::duration<double, std::micro>(t4 - t3).count() / 1000;

    printf("Kernel Launch Overhead\n");
    printf("%-35s  %8s\n", "Measurement", "Latency");
    printf("----------------------------------------------\n");
    printf("%-35s  %6.1f µs\n", "Host-side launch (async)", async_us);
    printf("%-35s  %6.1f µs\n", "GPU-side empty kernel", gpu_us);
    printf("%-35s  %6.1f µs\n", "Launch + sync (round-trip)", sync_us);

    // Show why tiny kernels are bad
    printf("\n--- Why Tiny Kernels Are Bad ---\n");
    printf("%-35s  %8s  %8s\n", "Scenario", "Time", "Overhead%");
    printf("----------------------------------------------\n");

    int blocks = (N + 255) / 256;

    CHECK(cudaEventRecord(start));
    real_kernel<<<blocks, 256>>>(d_data, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float real_ms;
    CHECK(cudaEventElapsedTime(&real_ms, start, stop));

    printf("%-35s  %6.3f ms  %7.1f%%\n", "1 kernel (1M elements)", real_ms, 0.0);
    printf("%-35s  %6.3f ms  %7.1f%%\n", "Equiv. launch overhead",
           (float)gpu_us / 1000, gpu_us / 1000 / real_ms * 100);

    // Simulate launching 1000 tiny kernels instead of 1 big one
    int tiny_n = N / 1000;
    int tiny_blocks = (tiny_n + 255) / 256;
    if (tiny_blocks < 1) tiny_blocks = 1;

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++)
        real_kernel<<<tiny_blocks, 256>>>(d_data + i * tiny_n, tiny_n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float tiny_ms;
    CHECK(cudaEventElapsedTime(&tiny_ms, start, stop));

    printf("%-35s  %6.3f ms  %7.1f%%\n", "1000 tiny kernels (same work)",
           tiny_ms, (tiny_ms - real_ms) / tiny_ms * 100);

    CHECK(cudaFree(d_data));
    return 0;
}
```

**Expected Results:**

| Measurement | Latency |
|-------------|---------|
| Host-side launch (async, no sync) | ~3-5 µs |
| GPU-side empty kernel overhead | ~2-3 µs |
| Launch + cudaDeviceSynchronize | ~8-15 µs |
| 1 real kernel (1M elements) | ~0.15 ms |
| 1000 tiny kernels (same total work) | ~3-5 ms |

**Insight:** Each kernel launch costs **~5 µs** of pure overhead. Launch 1000 tiny kernels and you pay **5 ms** in overhead alone — this can be more than the computation itself. **Fuse kernels** whenever possible. CUDA Graphs can batch multiple launches into one submission, reducing per-launch overhead to ~0.5 µs.

---

## Benchmark 11: Occupancy vs Performance

**Question:** Does higher occupancy always mean higher performance?

**Code:**
```cuda
// bench_occupancy.cu — Artificially control occupancy and measure performance
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

// Use shared memory to artificially limit occupancy
// More shared memory per block → fewer blocks per SM → lower occupancy

template<int SMEM_KB>
__global__ void compute_kernel(float* data, int n) {
    // Declare shared memory to limit occupancy
    __shared__ float smem[SMEM_KB * 256];  // SMEM_KB * 1024 bytes

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];
    smem[threadIdx.x] = val;
    __syncthreads();

    // Compute-heavy loop (not memory bound)
    #pragma unroll 4
    for (int i = 0; i < 200; i++) {
        val = val * 1.0001f + 0.0001f;
        val = sqrtf(fabsf(val)) + 0.1f;
    }

    data[idx] = val + smem[threadIdx.x % blockDim.x];
}

int main() {
    const int N = 16 * 1024 * 1024;
    const int bytes = N * sizeof(float);
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    float *d_data;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMemset(d_data, 0, bytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    printf("Occupancy vs Performance\n");
    printf("%-12s  %-12s  %-12s  %-10s\n",
           "Shared/Blk", "Occupancy", "Time (ms)", "Relative");
    printf("---------------------------------------------------\n");

    // A100 has 164 KB shared memory per SM, max 2048 threads per SM
    // 256 threads/block = 8 warps. Max blocks per SM depends on shared memory.

    auto bench = [&](const char* label, auto kernel, int smem_bytes) {
        int min_grid, block_size;
        CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, kernel, smem_bytes));

        int max_blocks_per_sm;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, kernel, THREADS, smem_bytes));

        float occupancy = (float)(max_blocks_per_sm * THREADS) / 2048.0f;

        kernel<<<BLOCKS, THREADS>>>(d_data, N);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 20; i++)
            kernel<<<BLOCKS, THREADS>>>(d_data, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 20;

        printf("%-12s  %10.0f%%  %10.3f ms\n", label, occupancy * 100, ms);
        return ms;
    };

    float t1 = bench("1 KB",  compute_kernel<1>,  1 * 1024);
    float t2 = bench("4 KB",  compute_kernel<4>,  4 * 1024);
    float t3 = bench("16 KB", compute_kernel<16>, 16 * 1024);
    float t4 = bench("32 KB", compute_kernel<32>, 32 * 1024);
    float t5 = bench("48 KB", compute_kernel<48>, 48 * 1024);

    printf("\n--- Key Takeaway ---\n");
    printf("Performance often plateaus around 50-75%% occupancy.\n");
    printf("Sometimes LOWER occupancy is FASTER if the kernel benefits\n");
    printf("from having more registers/shared memory per thread.\n");

    CHECK(cudaFree(d_data));
    return 0;
}
```

**Expected Results:**

| Shared/Block | Occupancy | Time (ms) | Notes |
|-------------|-----------|-----------|-------|
| 1 KB        | 100%      | ~2.1      | Max occupancy but thrashing caches |
| 4 KB        | 100%      | ~2.0      | Still max — no benefit |
| 16 KB       | 75%       | ~1.95     | Slightly faster with more resources |
| 32 KB       | 50%       | ~2.0      | **Often the sweet spot** |
| 48 KB       | 25%       | ~2.5      | Too few warps to hide latency |

**Insight:** Occupancy is **not** "the higher the better." For compute-bound kernels, 50% occupancy often matches or beats 100% because each thread gets more registers (fewer spills) and more cache. The rule: **start at max occupancy, then try reducing it by using more shared memory or registers. Measure both.**

---

## Benchmark 12: Block Size Sweep

**Question:** What block size gives the best performance for a typical kernel?

**Code:**
```cuda
// bench_blocksize.cu — Sweep block sizes and measure performance
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

// A "realistic" kernel: memory access + moderate compute
__global__ void workload(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        float val = src[i];
        // Moderate compute: enough to not be purely memory-bound
        #pragma unroll
        for (int j = 0; j < 20; j++)
            val = val * 1.001f + 0.001f;
        dst[i] = val;
    }
}

int main() {
    const int N = 64 * 1024 * 1024;  // 64M elements
    const int bytes = N * sizeof(float);
    const int ITERS = 20;

    float *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, bytes));
    CHECK(cudaMalloc(&d_dst, bytes));
    CHECK(cudaMemset(d_src, 1, bytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Get device properties for analysis
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Block Size Sweep on %s\n", prop.name);
    printf("SMs: %d, Max threads/SM: %d, Warp size: %d\n\n",
           prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.warpSize);

    printf("%-8s  %-8s  %-10s  %-12s  %-10s  %-8s\n",
           "Block", "Blocks", "Occupancy", "Time (ms)", "GFlops", "Relative");
    printf("--------------------------------------------------------------\n");

    float best_ms = 1e9;
    int best_block = 0;

    for (int block_size : {32, 64, 128, 256, 512, 1024}) {
        int blocks = (N + block_size - 1) / block_size;

        // Compute occupancy
        int max_blocks_per_sm;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, workload, block_size, 0));
        float occupancy = (float)(max_blocks_per_sm * block_size) /
                          prop.maxThreadsPerMultiProcessor;

        // Warmup
        workload<<<blocks, block_size>>>(d_dst, d_src, N);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            workload<<<blocks, block_size>>>(d_dst, d_src, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= ITERS;

        double flops = (double)N * 20 * 2;  // 20 FMAs = 40 flops per element
        double gflops = flops / (ms / 1000.0) / 1e9;

        if (ms < best_ms) { best_ms = ms; best_block = block_size; }

        printf("%-8d  %-8d  %8.0f%%  %10.3f  %10.1f  %8.2fx\n",
               block_size, blocks, occupancy * 100, ms, gflops, ms / best_ms);
    }

    printf("\n--- Best block size: %d ---\n", best_block);
    printf("Use cudaOccupancyMaxPotentialBlockSize() to auto-select:\n");

    int min_grid, auto_block;
    CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &auto_block, workload, 0, N));
    printf("  Recommended: blockSize=%d, minGrid=%d\n", auto_block, min_grid);

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    return 0;
}
```

**Expected Results:**

| Block Size | Occupancy | Time (ms) | Relative | Notes |
|-----------|-----------|-----------|----------|-------|
| 32  | 25-50%  | ~3.5  | 1.75x | Only 1 warp/block — poor SM utilization |
| 64  | 50-75%  | ~2.2  | 1.10x | Better, but still underutilizing |
| 128 | 75-100% | ~2.05 | 1.02x | Good balance |
| **256** | **100%** | **~2.0** | **1.00x** | **Sweet spot for most kernels** |
| 512 | 100%  | ~2.0  | 1.00x | Same occupancy, no benefit |
| 1024 | 50-100% | ~2.1 | 1.05x | May limit occupancy due to register pressure |

**Insight:** Block size 256 is the **default choice** for a reason: it gives full occupancy on most GPUs while keeping register pressure manageable. Block size 128 is sometimes better for register-heavy kernels. Block size 32 (one warp) is almost always bad — you lose the ability to hide latency within a block. Use `cudaOccupancyMaxPotentialBlockSize()` for automatic selection.

---

## GPU Performance Numbers Cheat Sheet

All measurements from the benchmarks above, collected in one reference table.

### Memory Bandwidth

| What | A100 80GB | H100 SXM | Unit |
|------|-----------|----------|------|
| HBM peak (theoretical) | 2,039 | 3,350 | GB/s |
| HBM achieved (simple kernel) | ~1,500 | ~2,600 | GB/s |
| HBM utilization (typical) | 75-85% | 75-85% | % |
| Shared memory (no conflict) | ~19,000 | ~33,000 | GB/s (aggregate) |
| Shared memory (32-way conflict) | ~600 | ~1,000 | GB/s |
| L2 cache bandwidth | ~5,000 | ~12,000 | GB/s |

### Memory Latency

| What | Cycles | Nanoseconds |
|------|--------|-------------|
| Shared memory load | 7-10 | ~5 ns |
| L1 cache hit | 30-50 | ~30 ns |
| L2 cache hit | 200-300 | ~150 ns |
| HBM (DRAM) | 400-800 | ~400 ns |
| Global → Shared ratio | — | **~80-100x** |

### Host-Device Transfer

| What | Bandwidth | Notes |
|------|-----------|-------|
| PCIe Gen4 x16 (pinned) | ~25 GB/s | Per direction |
| PCIe Gen4 x16 (pageable) | ~10 GB/s | Extra CPU-side copy |
| PCIe Gen5 x16 (pinned) | ~50 GB/s | Per direction |
| NVLink 4.0 (H100) | ~450 GB/s | Per direction, GPU-GPU |
| Pinned vs pageable speedup | **2-3x** | Always use pinned for bulk |

### Compute Throughput

| What | A100 | H100 | Unit |
|------|------|------|------|
| FP32 (CUDA cores) | 19.5 | 67 | TFLOPS |
| FP64 (CUDA cores) | 9.7 | 34 | TFLOPS |
| FP16 (Tensor cores) | 312 | 990 | TFLOPS |
| INT8 (Tensor cores) | 624 | 1,979 | TOPS |
| FP32:FP64 ratio | 2:1 | 2:1 | — |
| Tensor:CUDA ratio (FP16) | ~16:1 | ~15:1 | — |

### Overhead & Penalties

| What | Cost | Impact |
|------|------|--------|
| Kernel launch (async) | ~3-5 µs | 1000 launches = 5 ms wasted |
| Kernel launch (with sync) | ~8-15 µs | Avoid sync in hot loops |
| Stride-32 vs stride-1 bandwidth | **32x** penalty | Coalescing is critical |
| 32-way bank conflict | **~32x** slower | Pad shared memory arrays |
| Single-location atomicAdd (65K threads) | **~100x** slower | Use hierarchical reduction |
| Warp divergence (50/50) | **~2x** slower | Sort data by branch path |

### Rules of Thumb

| Rule | Number |
|------|--------|
| Default block size | 256 threads |
| Minimum occupancy target | 50% |
| Bytes per FLOP (A100 HBM/FP32) | ~105 B/FLOP |
| Arithmetic intensity breakeven | ~10 FLOP/byte |
| Max useful shared memory/block | 48 KB (portable) |
| Warps needed to hide DRAM latency | ~30-40 per SM |
| PCIe transfer amortization | >100 KB to be worthwhile |
