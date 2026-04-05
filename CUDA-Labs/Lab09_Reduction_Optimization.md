# Lab 09: Reduction Optimization 🔴

| Detail | Value |
|---|---|
| **Difficulty** | 🔴 Advanced |
| **Estimated Time** | 90–120 minutes |
| **Prerequisites** | Labs 01-08; understanding of shared memory and warps |
| **GPU Required** | NVIDIA GPU with Compute Capability 3.0+ (warp shuffle needs CC 3.0+) |

---

## Objective

By the end of this lab you will:
- Implement parallel reduction (sum) in 5 progressively optimized versions
- Understand the performance impact of each optimization
- Master warp-level primitives (`__shfl_down_sync`)
- Compare your hand-written reduction against NVIDIA's CUB library
- See the full optimization journey: from naive to near-peak bandwidth

---

## Setup

Create a working directory for this lab's parallel reduction experiments.

```bash
mkdir -p ~/cuda-labs/lab09 && cd ~/cuda-labs/lab09
```

### Background: The Reduction Problem

Reducing N elements to a single value (sum, max, min) is embarrassingly simple on a CPU but fundamentally challenging on a GPU. The problem is inherently serial — each step depends on the previous. GPU reductions use a **tree-based** approach: N threads produce N/2 partial results, then N/4, then N/8... until one value remains.

---

## Step 1: Five Versions of Reduction

Create `reduction.cu`:

```cuda
// reduction.cu — Parallel reduction from naive to optimized
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 24)  // 16M elements
#define BLOCK_SIZE 256

// ============================================================
// VERSION 1: Naive interleaved addressing (DIVERGENT)
// Problem: threadIdx.x % (2*stride) causes massive warp divergence
// ============================================================
__global__ void reduce_v1_naive(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Interleaved addressing with divergent branching
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // ← DIVERGENT! Threads 1,3,5,7... are idle
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// VERSION 2: Interleaved addressing without divergence
// Fix: use sequential thread IDs, compute index from stride
// Still has bank conflicts from strided shared memory access
// ============================================================
__global__ void reduce_v2_nodivergence(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Sequential addressing — no divergence within active warps
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// VERSION 3: Sequential addressing (no bank conflicts)
// Reverse the reduction direction: stride from blockDim/2 down to 1
// Adjacent threads access adjacent memory — no bank conflicts!
// ============================================================
__global__ void reduce_v3_sequential(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reversed loop: stride from large to small
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// VERSION 4: First add during global load (2× elements per thread)
// Halve the number of blocks by having each thread load 2 elements
// This doubles useful work per thread and reduces block overhead
// ============================================================
__global__ void reduce_v4_firstAdd(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load 2 elements and add during load
    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// VERSION 5: Warp shuffle (no shared memory for final 32 elements)
// The last 32 elements fit in a single warp — use __shfl_down_sync
// instead of shared memory + __syncthreads for the final reduction
// ============================================================
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce_v5_warpShuffle(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Reduce in shared memory until we're down to 32 elements
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Final warp: use shuffle instead of shared memory
    if (tid < 32) {
        val = sdata[tid];
        val = warpReduceSum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}

// ============================================================
// HOST REDUCTION (for verification)
// ============================================================
double hostReduce(const float *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += data[i];
    return sum;
}

// ============================================================
// BENCHMARKING
// ============================================================
typedef void (*ReduceKernel)(const float*, float*, int);

float benchmarkReduction(ReduceKernel kernel, const float *d_input,
                         float *d_output, float *d_output2,
                         int n, int blockSize, bool halvedGrid,
                         float *result) {
    int gridSize;
    if (halvedGrid)
        gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    else
        gridSize = (n + blockSize - 1) / blockSize;

    int smemSize = blockSize * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Single pass to get partial sums
    kernel<<<gridSize, blockSize, smemSize>>>(d_input, d_output, n);

    // Second pass to reduce partial sums (simple V3)
    if (gridSize > 1) {
        int grid2 = (gridSize + blockSize - 1) / blockSize;
        reduce_v3_sequential<<<grid2, blockSize, smemSize>>>(d_output, d_output2, gridSize);
        // If still >1 block, do a third pass (unlikely for our sizes)
        if (grid2 > 1) {
            reduce_v3_sequential<<<1, blockSize, smemSize>>>(d_output2, d_output, grid2);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get result
    float h_result;
    float *resultPtr = (gridSize == 1) ? d_output :
                       (((gridSize + blockSize - 1) / blockSize) == 1 ? d_output2 : d_output);
    CUDA_CHECK(cudaMemcpy(&h_result, resultPtr, sizeof(float), cudaMemcpyDeviceToHost));
    *result = h_result;

    // Benchmark (first pass only — that's where the optimization matters)
    int runs = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        kernel<<<gridSize, blockSize, smemSize>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / runs;
}

int main() {
    size_t bytes = N * sizeof(float);
    printf("=== Parallel Reduction Optimization Journey ===\n");
    printf("N = %d elements (%.0f MB)\n\n", N, bytes / (1024.0 * 1024.0));

    // Initialize
    float *h_data = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++)
        h_data[i] = (float)(rand() % 100) * 0.01f;

    double hostSum = hostReduce(h_data, N);
    printf("Host (double precision) sum: %.6f\n\n", hostSum);

    float *d_input, *d_output, *d_output2;
    int maxBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, maxBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output2, maxBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice));

    printf("%-35s  %10s  %12s  %10s  %s\n",
           "Version", "Time (ms)", "Bandwidth", "Speedup", "Sum");
    printf("%-35s  %10s  %12s  %10s  %s\n",
           "-------", "---------", "---------", "-------", "---");

    struct {
        const char *name;
        ReduceKernel kernel;
        bool halvedGrid;
    } versions[] = {
        {"V1: Naive (divergent)",          reduce_v1_naive,          false},
        {"V2: No divergence",              reduce_v2_nodivergence,   false},
        {"V3: Sequential addressing",      reduce_v3_sequential,     false},
        {"V4: First add during load",      reduce_v4_firstAdd,       true},
        {"V5: Warp shuffle",               reduce_v5_warpShuffle,    true},
    };
    int numVersions = sizeof(versions) / sizeof(versions[0]);

    float baselineMs = 0;
    for (int v = 0; v < numVersions; v++) {
        float result;
        float ms = benchmarkReduction(versions[v].kernel, d_input, d_output, d_output2,
                                       N, BLOCK_SIZE, versions[v].halvedGrid, &result);
        if (v == 0) baselineMs = ms;

        float bw = bytes / (ms / 1000.0f) / 1e9;
        float error = fabsf(result - (float)hostSum) / (float)hostSum * 100.0f;

        printf("%-35s  %10.3f  %9.1f GB/s  %9.2fx  %.4f %s\n",
               versions[v].name, ms, bw, baselineMs / ms, result,
               error < 1.0f ? "✓" : "✗");
    }

    // Compare with theoretical peak bandwidth
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU memory bandwidth (theoretical): ~%.0f GB/s\n",
           prop.memoryClockRate * 2.0 * prop.memoryBusWidth / 8.0 / 1e6);
    printf("Reduction is memory-bound: bandwidth achieved vs peak is your efficiency.\n");

    free(h_data);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_output2));
    return 0;
}
```

### Compile and run

Compile and run to see the performance journey from naive reduction through five optimization levels.

```bash
nvcc -O2 -o reduction reduction.cu
./reduction
```

### Expected Output

```
=== Parallel Reduction Optimization Journey ===
N = 16777216 elements (64 MB)

Host (double precision) sum: 826197.812500

Version                              Time (ms)     Bandwidth     Speedup  Sum
-------                              ---------     ---------     -------  ---
V1: Naive (divergent)                    0.812      79.0 GB/s      1.00x  826197.8125 ✓
V2: No divergence                        0.534     120.1 GB/s      1.52x  826197.8125 ✓
V3: Sequential addressing               0.398     161.2 GB/s      2.04x  826197.8125 ✓
V4: First add during load               0.245     261.9 GB/s      3.31x  826197.8125 ✓
V5: Warp shuffle                         0.212     302.7 GB/s      3.83x  826197.8125 ✓

GPU memory bandwidth (theoretical): ~760 GB/s
Reduction is memory-bound: bandwidth achieved vs peak is your efficiency.
```

---

## Step 2: Understanding Each Optimization

### V1 → V2: Eliminating Divergence
```
V1: if (tid % (2*stride) == 0)  → threads 1,3,5,7... idle → divergent warps
V2: index = 2*stride*tid        → threads 0,1,2,3... active → no divergence
```

### V2 → V3: Eliminating Bank Conflicts
```
V2: sdata[2*stride*tid] += sdata[2*stride*tid + stride]  → strided access → bank conflicts
V3: sdata[tid] += sdata[tid + stride]                     → sequential access → no conflicts
```

### V3 → V4: Double Throughput on Load
```
V3: Each thread loads 1 element → half the block is idle after first iteration
V4: Each thread loads 2 elements → grid is halved, all threads do useful work
```

### V4 → V5: Warp-Level Optimization
```
V4: Last 5 iterations use shared memory + __syncthreads (expensive)
V5: Last 5 iterations use __shfl_down_sync (register-level, no sync needed)
```

---

## Step 3: CUB Comparison (Optional)

Create `reduction_cub.cu` (requires CUB, included with CUDA Toolkit 11+):

```cuda
// reduction_cub.cu — Compare hand-written reduction with NVIDIA CUB
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 24)

int main() {
    size_t bytes = N * sizeof(float);
    printf("=== CUB Device Reduce ===\n");
    printf("N = %d elements\n\n", N);

    float *h_data = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)(rand() % 100) * 0.01f;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice));

    // Determine temporary storage size
    void *d_temp = NULL;
    size_t tempBytes = 0;
    cub::DeviceReduce::Sum(d_temp, tempBytes, d_input, d_output, N);
    CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));
    printf("CUB temporary storage: %zu bytes\n", tempBytes);

    // Warm up
    cub::DeviceReduce::Sum(d_temp, tempBytes, d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int runs = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        cub::DeviceReduce::Sum(d_temp, tempBytes, d_input, d_output, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= runs;

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    float bw = bytes / (ms / 1000.0f) / 1e9;
    printf("CUB time:       %.3f ms\n", ms);
    printf("CUB bandwidth:  %.1f GB/s\n", bw);
    printf("CUB result:     %.4f\n", result);
    printf("\n→ CUB is typically within 5-10%% of hand-optimized code.\n");
    printf("→ In production, use CUB/Thrust unless you need custom reductions.\n");

    free(h_data);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
```

### Compile and run

Compile and run to compare your hand-optimized reduction against NVIDIA's CUB library implementation.

```bash
nvcc -O2 -o reduction_cub reduction_cub.cu
./reduction_cub
```

---

## Experiments

### Experiment 1: Block size impact on reduction
Run V5 with block sizes 64, 128, 256, 512, 1024. Which gives the best bandwidth? Why?

### Experiment 2: Integer reduction
Change from `float` to `int` sum. Does integer reduction achieve higher bandwidth than float? (Integer addition is cheaper.)

### Experiment 3: Max reduction
Replace `+=` with `max()`. How does max-reduction performance compare to sum-reduction?

### Experiment 4: Multiple elements per thread
In V4, each thread loads 2 elements. Try 4, 8, or 16 elements per thread (grid-stride loop). What's the optimal elements-per-thread ratio?

### Experiment 5: Atomic-based reduction
Write a kernel where each thread does `atomicAdd(&output[0], input[i])`. How much slower is this compared to V5? This shows why tree-based reduction matters.

---

## What Just Happened?

1. **V1 → V2 (1.5×): Divergence elimination.** The naive modulo check caused half the threads in each warp to be idle. Sequential thread IDs ensure all active threads are in the first warps.

2. **V2 → V3 (1.3×): Bank conflict removal.** Strided shared memory access caused 2-way to 32-way bank conflicts. Reversed iteration order gives sequential access patterns.

3. **V3 → V4 (1.6×): Doubling useful work.** Loading 2 elements per thread means half as many blocks, reducing scheduling overhead and using every thread productively from the start.

4. **V4 → V5 (1.15×): Warp-level primitives.** The last 5 iterations (32 elements to 1) use register-level shuffles instead of shared memory. No `__syncthreads` needed within a warp.

5. **Total journey: ~3.8× speedup** from V1 to V5, achieving 40-50% of theoretical peak bandwidth. The remaining gap is due to launch overhead and the inherent serialization of reduction.

---

## Key Insight

> **Reduction is the "Hello World" of GPU optimization.** It teaches every major concept: warp divergence, bank conflicts, memory coalescing, work efficiency, and warp-level primitives. If you understand why each version is faster, you understand GPU optimization.

---

## Checkpoint Quiz

**Q1:** Why can't we use `__shfl_down_sync` for the entire reduction (not just the last 32 elements)?
<details><summary>Answer</summary>
`__shfl_down_sync` only works within a single warp (32 threads). For a block of 256 threads (8 warps), the first reduction steps must combine values across warps, which requires shared memory. Only when we're down to 32 or fewer active threads (one warp) can we switch to shuffle instructions.
</details>

**Q2:** V4 loads 2 elements per thread. Why not load ALL elements per thread (single block)?
<details><summary>Answer</summary>
A single block is limited to 1024 threads. With 16M elements and 1024 threads, each thread would load ~16K elements. While this works, it severely underutilizes the GPU — only one SM would be active. The GPU has dozens of SMs that need work. The optimal balance is enough blocks to fill all SMs while keeping per-thread work high enough to amortize overhead.
</details>

**Q3:** Reduction achieves lower bandwidth than a simple copy kernel. Why?
<details><summary>Answer</summary>
Copy is a 1:1 read-write operation — every byte read is a byte written. Reduction reads N elements but writes only ~N/blockSize partial sums (and ultimately 1 value). The GPU memory system's load/store units are partially idle during the later reduction stages when fewer threads are active. Additionally, `__syncthreads` barriers cause stalls, and the reduction pattern has inherent serialization (each step depends on the previous).
</details>

---

*Next Lab: [Lab 10 — Build a Mini GEMM](Lab10_Build_A_Mini_GEMM.md)*
