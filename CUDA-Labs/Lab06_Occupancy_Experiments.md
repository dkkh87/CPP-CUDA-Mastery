# Lab 06: Occupancy Experiments 🟡

| Detail | Value |
|---|---|
| **Difficulty** | 🟡 Intermediate |
| **Estimated Time** | 60–80 minutes |
| **Prerequisites** | Labs 01-05; awareness of GPU architecture (SMs, warps) |
| **GPU Required** | Any NVIDIA GPU (Compute Capability 3.0+) |

---

## Objective

By the end of this lab you will:
- Understand what occupancy means and how to calculate it
- Discover the three limiters: registers, shared memory, and block size
- Measure achieved occupancy and correlate it with performance
- Learn that **max occupancy ≠ max performance**
- Use `cudaOccupancyMaxPotentialBlockSize` to find optimal launch configs

---

## Setup

Create a working directory for this lab's occupancy experiments.

```bash
mkdir -p ~/cuda-labs/lab06 && cd ~/cuda-labs/lab06
```

### Background: What Is Occupancy?

**Occupancy** = (active warps per SM) / (maximum warps per SM). An SM can schedule a hardware-limited number of warps (e.g., 48 or 64). If your kernel uses too many registers or too much shared memory per block, fewer blocks fit on each SM, reducing occupancy.

**But higher occupancy doesn't always mean better performance.** Sometimes lower occupancy with better per-thread resource usage wins.

---

## Step 1: Query Device Limits

Create `device_limits.cu`:

```cuda
// device_limits.cu — See your GPU's occupancy-related limits
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== GPU: %s ===\n\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("\n--- Occupancy Limiters ---\n");
    printf("Max threads per SM:       %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block:    %d\n", prop.maxThreadsPerBlock);
    printf("Max warps per SM:         %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("Max blocks per SM:        %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Registers per SM:         %d\n", prop.regsPerMultiprocessor);
    printf("Registers per block:      %d\n", prop.regsPerBlock);
    printf("Shared memory per SM:     %zu bytes (%.1f KB)\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Shared memory per block:  %zu bytes (%.1f KB)\n",
           prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
    printf("Warp size:                %d\n", prop.warpSize);

    printf("\n--- Derived ---\n");
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
    printf("Max warps per SM: %d\n", maxWarpsPerSM);
    printf("At 256 threads/block: %d warps/block → max %d blocks/SM → %d warps (%.0f%% occ)\n",
           256/32, maxWarpsPerSM/(256/32), maxWarpsPerSM, 100.0);

    return 0;
}
```

### Compile and run

Compile and run to see your GPU's occupancy-related hardware limits.

```bash
nvcc -o device_limits device_limits.cu
./device_limits
```

---

## Step 2: Block Size vs Performance

Create `blocksize_sweep.cu`:

```cuda
// blocksize_sweep.cu — How block size affects occupancy and performance
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 24)  // 16M elements

__global__ void compute(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        #pragma unroll 4
        for (int k = 0; k < 50; k++) {
            val = val * 1.0001f + 0.0001f;
            val = sqrtf(fabsf(val) + 1.0f);
        }
        data[i] = val;
    }
}

int main() {
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("=== Block Size vs Performance & Occupancy ===\n");
    printf("N = %d elements\n\n", N);

    printf("%-8s  %10s  %10s  %12s  %12s\n",
           "BlkSize", "Time (ms)", "Throughput", "Theo. Occ.", "Perf Index");
    printf("%-8s  %10s  %10s  %12s  %12s\n",
           "-------", "---------", "----------", "----------", "----------");

    int blockSizes[] = {32, 64, 128, 192, 256, 384, 512, 768, 1024};
    int numSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

    float bestMs = 1e9;

    for (int s = 0; s < numSizes; s++) {
        int bs = blockSizes[s];
        int grid = (N + bs - 1) / bs;

        // Query theoretical occupancy
        int maxActiveBlocks;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, compute, bs, 0));

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int maxWarps = prop.maxThreadsPerMultiProcessor / 32;
        float occupancy = (float)(maxActiveBlocks * (bs / 32)) / maxWarps * 100.0f;

        // Benchmark
        compute<<<grid, bs>>>(d_data, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        int runs = 20;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            compute<<<grid, bs>>>(d_data, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        if (ms < bestMs) bestMs = ms;
        float throughput = N / (ms / 1000.0f) / 1e9;  // G elements/s

        printf("%-8d  %10.3f  %7.2f GE/s  %10.1f%%  %12.2f\n",
               bs, ms, throughput, occupancy, bestMs / ms);
    }

    // Use CUDA occupancy API to find recommended block size
    int minGridSize, bestBlockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &bestBlockSize, compute, 0, N));
    printf("\ncudaOccupancyMaxPotentialBlockSize recommends: %d threads/block\n",
           bestBlockSize);
    printf("Minimum grid size for full occupancy: %d blocks\n", minGridSize);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Compile and run

Compile and run to sweep block sizes from 32 to 1024 and see how each affects occupancy and performance.

```bash
nvcc -O2 -o blocksize_sweep blocksize_sweep.cu
./blocksize_sweep
```

### Expected Output

```
=== Block Size vs Performance & Occupancy ===
N = 16777216 elements

BlkSize    Time (ms)  Throughput   Theo. Occ.    Perf Index
-------    ---------  ----------   ----------    ----------
32             7.234    2.32 GE/s       25.0%          0.89
64             6.512    2.58 GE/s       50.0%          0.99
128            6.423    2.61 GE/s      100.0%          1.00
192            6.445    2.60 GE/s       75.0%          1.00
256            6.431    2.61 GE/s      100.0%          1.00
384            6.489    2.59 GE/s       75.0%          0.99
512            6.441    2.60 GE/s      100.0%          1.00
768            6.501    2.58 GE/s       75.0%          0.99
1024           6.455    2.60 GE/s      100.0%          1.00

cudaOccupancyMaxPotentialBlockSize recommends: 256 threads/block
Minimum grid size for full occupancy: 160 blocks
```

> Notice: 128, 256, 512, 1024 all achieve similar performance despite different occupancy levels.

---

## Step 3: Register Pressure and Occupancy

Create `register_pressure.cu`:

```cuda
// register_pressure.cu — More registers per thread = lower occupancy
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 22)

// Few registers: compiler will use ~20 registers
__global__ void lowRegKernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = data[i];
        a = a * 1.001f + 0.001f;
        data[i] = a;
    }
}

// Many registers: force the compiler to use many registers
__global__ void highRegKernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = data[i], b = a*1.1f, c = b*1.2f, d = c*1.3f;
        float e = d*1.4f, f = e*1.5f, g = f*1.6f, h = g*1.7f;
        float i2 = h*1.8f, j = i2*1.9f, k = j*2.0f, l = k*2.1f;
        float m = l*2.2f, nn = m*2.3f, o = nn*2.4f, p = o*2.5f;
        float q = sinf(a) + cosf(b) + sinf(c) + cosf(d);
        float r = sinf(e) + cosf(f) + sinf(g) + cosf(h);
        float s = sinf(i2) + cosf(j) + sinf(k) + cosf(l);
        float t = sinf(m) + cosf(nn) + sinf(o) + cosf(p);
        data[i] = q + r + s + t;
    }
}

// Force register limit with __launch_bounds__
__global__ void __launch_bounds__(256, 2)
limitedRegKernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = data[i], b = a*1.1f, c = b*1.2f, d = c*1.3f;
        float e = d*1.4f, f = e*1.5f, g = f*1.6f, h = g*1.7f;
        float q = sinf(a) + cosf(b) + sinf(c) + cosf(d);
        float r = sinf(e) + cosf(f) + sinf(g) + cosf(h);
        data[i] = q + r;
    }
}

void analyzeKernel(const char *name, const void *kernel, int blockSize) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));

    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel, blockSize, 0));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxWarps = prop.maxThreadsPerMultiProcessor / 32;
    float occupancy = (float)(maxActiveBlocks * (blockSize / 32)) / maxWarps * 100.0f;

    printf("%-20s  Regs: %3d  Smem: %5zu B  MaxBlocks/SM: %d  Occupancy: %.0f%%\n",
           name, attr.numRegs, attr.sharedSizeBytes,
           maxActiveBlocks, occupancy);
}

int main() {
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

    int blockSize = 256;
    printf("=== Register Pressure vs Occupancy (block size = %d) ===\n\n", blockSize);

    analyzeKernel("Low registers", (const void *)lowRegKernel, blockSize);
    analyzeKernel("High registers", (const void *)highRegKernel, blockSize);
    analyzeKernel("Limited (__launch_bounds__)", (const void *)limitedRegKernel, blockSize);

    printf("\n→ More registers per thread → fewer threads per SM → lower occupancy.\n");
    printf("→ __launch_bounds__(maxThreads, minBlocks) hints the compiler to\n");
    printf("  limit register usage so minBlocks can be resident per SM.\n");

    // Benchmark
    printf("\n=== Performance Comparison ===\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int grid = (N + blockSize - 1) / blockSize;
    int runs = 50;

    struct { const char *name; void (*fn)(float*, int); } kernels[] = {
        {"Low registers", lowRegKernel},
        {"High registers", highRegKernel},
        {"Limited regs", limitedRegKernel},
    };

    for (int k = 0; k < 3; k++) {
        kernels[k].fn<<<grid, blockSize>>>(d_data, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            kernels[k].fn<<<grid, blockSize>>>(d_data, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;
        printf("%-20s  %8.3f ms\n", kernels[k].name, ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Compile and run

Compile and run to see how register usage per thread affects occupancy and whether `__launch_bounds__` can help.

```bash
nvcc -O2 -o register_pressure register_pressure.cu
./register_pressure
```

---

## Step 4: Shared Memory and Occupancy

Create `smem_occupancy.cu`:

```cuda
// smem_occupancy.cu — Shared memory usage limits occupancy
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 22)

// Template to vary shared memory usage
template<int SMEM_FLOATS>
__global__ void smemKernel(float *data, int n) {
    __shared__ float smem[SMEM_FLOATS];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Use shared memory (prevent compiler from optimizing it away)
    if (tid < SMEM_FLOATS) smem[tid] = (float)tid;
    __syncthreads();

    if (i < n) {
        float val = data[i];
        val += smem[tid % SMEM_FLOATS];
        for (int k = 0; k < 50; k++)
            val = val * 1.001f + 0.001f;
        data[i] = val;
    }
}

template<int SMEM_FLOATS>
void benchSmem(float *d_data, int blockSize, const char *label) {
    int grid = (N + blockSize - 1) / blockSize;
    int smemBytes = SMEM_FLOATS * sizeof(float);

    int maxBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, smemKernel<SMEM_FLOATS>, blockSize, 0));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxWarps = prop.maxThreadsPerMultiProcessor / 32;
    float occ = (float)(maxBlocks * (blockSize / 32)) / maxWarps * 100.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    smemKernel<SMEM_FLOATS><<<grid, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int runs = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        smemKernel<SMEM_FLOATS><<<grid, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= runs;

    printf("%-14s  %6d B smem  %2d blk/SM  %5.1f%% occ  %8.3f ms\n",
           label, smemBytes, maxBlocks, occ, ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    int bs = 256;
    printf("=== Shared Memory vs Occupancy (block size = %d) ===\n\n", bs);
    printf("%-14s  %8s  %9s  %9s  %10s\n",
           "Config", "Smem", "Blk/SM", "Occupancy", "Time");
    printf("%-14s  %8s  %9s  %9s  %10s\n",
           "------", "----", "------", "---------", "----");

    benchSmem<256>(d_data, bs, "1 KB smem");
    benchSmem<1024>(d_data, bs, "4 KB smem");
    benchSmem<4096>(d_data, bs, "16 KB smem");
    benchSmem<8192>(d_data, bs, "32 KB smem");
    benchSmem<12288>(d_data, bs, "48 KB smem");

    printf("\n→ As shared memory per block increases, fewer blocks fit per SM.\n");
    printf("→ This directly reduces occupancy.\n");

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Compile and run

Compile and run to see how increasing shared memory usage per block reduces occupancy.

```bash
nvcc -O2 -o smem_occupancy smem_occupancy.cu
./smem_occupancy
```

---

## Experiments

### Experiment 1: Compile with `-maxrregcount`
Recompile `register_pressure.cu` with `nvcc -O2 -maxrregcount=32` and then `-maxrregcount=64`. How does forced register limiting change occupancy and performance?

### Experiment 2: The "50% occupancy" sweet spot
For many kernels, 50% occupancy performs as well as 100%. Test this with `blocksize_sweep.cu` by using different kernels (compute-heavy vs memory-heavy).

### Experiment 3: Dynamic shared memory
Replace the template-based shared memory with dynamic allocation: `extern __shared__ float smem[];` and launch with `kernel<<<grid, block, smemBytes>>>`. Verify occupancy matches the static version.

### Experiment 4: Occupancy vs ILP
Write a kernel that uses many registers to achieve instruction-level parallelism (ILP) — independent operations that the compiler can overlap. Does higher ILP with lower occupancy beat high occupancy with low ILP?

### Experiment 5: Profile with nvcc
Compile with `nvcc --ptxas-options=-v` to see register and shared memory usage per kernel. Compare with `cudaFuncGetAttributes` results.

---

## What Just Happened?

1. **Occupancy is determined by three resources:** registers per thread, shared memory per block, and block size. The GPU fills each SM with as many blocks as these resources allow. The most constraining resource determines the actual occupancy.

2. **More occupancy helps hide latency.** The GPU switches between warps to hide memory latency. More active warps = more switching opportunities. But if your kernel has enough instruction-level parallelism, fewer warps can suffice.

3. **Max occupancy ≠ max performance.** A kernel at 50% occupancy with good data reuse (via shared memory or registers) often beats a kernel at 100% occupancy with poor data reuse. The goal is to use resources effectively, not maximize a percentage.

4. **`cudaOccupancyMaxPotentialBlockSize` is your starting point.** It returns the block size that maximizes occupancy for your kernel. Start there, then experiment with smaller sizes that may give better per-thread performance.

---

## Key Insight

> **Occupancy is a means, not a goal.** High occupancy helps the GPU hide memory latency by switching between warps. But the real goal is throughput: elements processed per second. Sometimes using more registers or shared memory per thread (lower occupancy) gives each thread a much faster execution path, yielding better overall throughput.

---

## Checkpoint Quiz

**Q1:** Your kernel uses 48 registers per thread, block size is 256. SM has 65536 registers. How many blocks can fit per SM based on register usage alone?
<details><summary>Answer</summary>
Each block needs 256 × 48 = 12,288 registers. The SM has 65,536 registers. 65,536 / 12,288 = 5.33 → **5 blocks** per SM (truncated, not rounded). That's 5 × 256 = 1,280 threads = 40 warps. If the SM supports 64 warps max, occupancy = 40/64 = 62.5%.
</details>

**Q2:** You want to use 48 KB of shared memory per block. Your SM has 96 KB of shared memory. What's the maximum blocks per SM from shared memory alone?
<details><summary>Answer</summary>
96 KB / 48 KB = 2 blocks per SM. If each block is 256 threads = 8 warps, that's only 16 warps. On a GPU with 64 max warps/SM, occupancy = 16/64 = 25%. Using large shared memory dramatically limits occupancy.
</details>

**Q3:** Your kernel achieves 25% occupancy but 95% of peak memory bandwidth. Should you increase occupancy?
<details><summary>Answer</summary>
Probably not. At 95% of peak bandwidth, the kernel is memory-bound and nearly saturating the memory system. Increasing occupancy would add more warps competing for the same memory bandwidth, unlikely to help. In fact, it might hurt if the increased occupancy requires reducing register usage and introducing register spilling (which adds more memory traffic).
</details>

---

*Next Lab: [Lab 07 — Coalescing Patterns](Lab07_Coalescing_Patterns.md)*
