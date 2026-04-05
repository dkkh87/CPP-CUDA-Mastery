# Lab 03: Memory Transfer Costs 🟢

| Detail | Value |
|---|---|
| **Difficulty** | 🟢 Beginner |
| **Estimated Time** | 50–70 minutes |
| **Prerequisites** | Lab 01, Lab 02 complete |
| **GPU Required** | Any NVIDIA GPU |

---

## Objective

By the end of this lab you will:
- Measure PCIe transfer times accurately using CUDA events
- Understand the compute-to-transfer ratio and when GPU becomes worthwhile
- Compare pageable vs pinned (page-locked) memory transfer speeds
- Know why memory transfers are the #1 bottleneck in most CUDA programs
- Calculate effective PCIe bandwidth and compare to theoretical limits

---

## Setup

Create a working directory for this lab's memory transfer experiments.

```bash
mkdir -p ~/cuda-labs/lab03 && cd ~/cuda-labs/lab03
```

---

## Step 1: Measure Raw Transfer Speeds

Create `transfer_bench.cu`:

```cuda
// transfer_bench.cu — Measure cudaMemcpy bandwidth for different sizes
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void benchmarkTransfer(size_t bytes, int iterations) {
    float *h_data = (float *)malloc(bytes);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Initialize host data
    for (size_t i = 0; i < bytes / sizeof(float); i++)
        h_data[i] = 1.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // Measure Host → Device
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
    h2d_ms /= iterations;

    // Measure Device → Host
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
    d2h_ms /= iterations;

    double h2d_gbps = (bytes / 1e9) / (h2d_ms / 1e3);
    double d2h_gbps = (bytes / 1e9) / (d2h_ms / 1e3);
    double mb = bytes / (1024.0 * 1024.0);

    printf("%10.1f MB  |  H→D: %8.3f ms (%6.2f GB/s)  |  D→H: %8.3f ms (%6.2f GB/s)\n",
           mb, h2d_ms, h2d_gbps, d2h_ms, d2h_gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    printf("%-12s  |  %-30s  |  %-30s\n", "Size", "Host → Device", "Device → Host");
    printf("%-12s  |  %-30s  |  %-30s\n", "----", "-------------", "-------------");

    size_t sizes[] = {
        1024,                   // 1 KB
        64 * 1024,              // 64 KB
        256 * 1024,             // 256 KB
        1024 * 1024,            // 1 MB
        4 * 1024 * 1024,        // 4 MB
        16 * 1024 * 1024,       // 16 MB
        64 * 1024 * 1024,       // 64 MB
        256 * 1024 * 1024,      // 256 MB
    };
    int iters[] = {1000, 500, 200, 100, 50, 20, 10, 5};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < numSizes; i++) {
        benchmarkTransfer(sizes[i], iters[i]);
    }

    return 0;
}
```

### Compile and run

Compile and run to measure PCIe transfer bandwidth for different data sizes.

```bash
nvcc -O2 -o transfer_bench transfer_bench.cu
./transfer_bench
```

### Expected Output (PCIe Gen3 x16)

```
GPU: NVIDIA GeForce RTX 3080

Size          |  Host → Device                  |  Device → Host
----          |  -------------                  |  -------------
       0.0 MB  |  H→D:    0.006 ms (  0.17 GB/s)  |  D→H:    0.006 ms (  0.16 GB/s)
       0.1 MB  |  H→D:    0.012 ms (  5.12 GB/s)  |  D→H:    0.013 ms (  4.73 GB/s)
       0.2 MB  |  H→D:    0.029 ms (  8.43 GB/s)  |  D→H:    0.030 ms (  8.13 GB/s)
       1.0 MB  |  H→D:    0.098 ms ( 10.24 GB/s)  |  D→H:    0.102 ms (  9.84 GB/s)
       4.0 MB  |  H→D:    0.357 ms ( 11.21 GB/s)  |  D→H:    0.370 ms ( 10.81 GB/s)
      16.0 MB  |  H→D:    1.372 ms ( 11.66 GB/s)  |  D→H:    1.412 ms ( 11.33 GB/s)
      64.0 MB  |  H→D:    5.421 ms ( 11.81 GB/s)  |  D→H:    5.602 ms ( 11.42 GB/s)
     256.0 MB  |  H→D:   21.633 ms ( 11.83 GB/s)  |  D→H:   22.411 ms ( 11.42 GB/s)
```

> **Key observation:** Small transfers have terrible bandwidth due to fixed overhead. Large transfers approach the PCIe theoretical max (~12-13 GB/s for Gen3 x16).

---

## Step 2: Compute-to-Transfer Ratio

When is the GPU worth using? Create `ratio_sweep.cu`:

```cuda
// ratio_sweep.cu — Find the compute-to-transfer ratio where GPU wins
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

// CPU: do FLOPS_PER_ELEMENT operations per element
void computeCPU(float *a, float *b, float *c, int n, int flops) {
    for (int i = 0; i < n; i++) {
        float val = a[i];
        for (int f = 0; f < flops; f++) {
            val = sinf(val) * cosf(b[i]) + val * 0.999f;
        }
        c[i] = val;
    }
}

__global__ void computeGPU(float *a, float *b, float *c, int n, int flops) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = a[i];
        for (int f = 0; f < flops; f++) {
            val = sinf(val) * cosf(b[i]) + val * 0.999f;
        }
        c[i] = val;
    }
}

double wallClock() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 1000) * 0.001f;
        h_b[i] = (float)(i % 500) * 0.002f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("N = %d elements (%zu MB)\n\n", N, bytes / (1024*1024));
    printf("%-10s  %10s  %10s  %10s  %10s  %s\n",
           "FLOPs/Elem", "CPU (ms)", "GPU Total", "GPU Kernel", "Speedup", "Winner");
    printf("%-10s  %10s  %10s  %10s  %10s  %s\n",
           "----------", "--------", "---------", "----------", "-------", "------");

    int flopsValues[] = {1, 2, 5, 10, 20, 50, 100, 200};
    int numFlops = sizeof(flopsValues) / sizeof(flopsValues[0]);

    for (int fi = 0; fi < numFlops; fi++) {
        int flops = flopsValues[fi];

        // CPU timing
        double t0 = wallClock();
        computeCPU(h_a, h_b, h_c, N, flops);
        double cpuMs = (wallClock() - t0) * 1000.0;

        // GPU timing (total: transfer + compute + transfer back)
        float gpuTotalMs, gpuKernelMs;

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        computeGPU<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N, flops);
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuTotalMs, start, stop));

        // GPU timing (kernel only)
        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        computeGPU<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N, flops);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&gpuKernelMs, start, stop));

        float speedup = cpuMs / gpuTotalMs;

        printf("%-10d  %10.3f  %10.3f  %10.3f  %9.2fx  %s\n",
               flops, cpuMs, gpuTotalMs, gpuKernelMs, speedup,
               speedup > 1.0 ? "GPU ✓" : "CPU ✓");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

### Compile and run

Compile and run to find the compute-to-transfer ratio where the GPU becomes worthwhile.

```bash
nvcc -O2 -o ratio_sweep ratio_sweep.cu
./ratio_sweep
```

### Expected Output (approximate)

```
N = 1048576 elements (4 MB)

FLOPs/Elem    CPU (ms)  GPU Total  GPU Kernel     Speedup  Winner
----------    --------  ---------  ----------     -------  ------
1                1.762      0.834       0.052       2.11x  GPU ✓
2                3.490      0.869       0.063       4.02x  GPU ✓
5                8.597      0.946       0.127       9.09x  GPU ✓
10              17.102      1.068       0.241      16.01x  GPU ✓
20              34.125      1.362       0.478      25.05x  GPU ✓
50              85.113      2.248       1.176      37.86x  GPU ✓
100            170.345      3.712       2.341      45.89x  GPU ✓
200            341.012      6.621       4.669      51.51x  GPU ✓
```

> As compute per element increases, the transfer time becomes a smaller fraction, and GPU speedup grows dramatically.

---

## Step 3: Pageable vs Pinned Memory

Create `pinned_vs_pageable.cu`:

```cuda
// pinned_vs_pageable.cu — Pinned memory can 2x your transfer speed
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

void benchTransfer(float *h_data, float *d_data, size_t bytes, int iters,
                   const char *label) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // H→D
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));
    h2d_ms /= iters;

    // D→H
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));
    d2h_ms /= iters;

    double h2d_gbps = (bytes / 1e9) / (h2d_ms / 1e3);
    double d2h_gbps = (bytes / 1e9) / (d2h_ms / 1e3);

    printf("%-12s  H→D: %7.3f ms (%6.2f GB/s)  D→H: %7.3f ms (%6.2f GB/s)\n",
           label, h2d_ms, h2d_gbps, d2h_ms, d2h_gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("=== Pageable vs Pinned Memory Transfer Speed ===\n\n");

    size_t sizes[] = {
        1 * 1024 * 1024,    // 1 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024,   // 64 MB
        256 * 1024 * 1024,  // 256 MB
    };
    int iters[] = {100, 50, 20, 5};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < numSizes; s++) {
        size_t bytes = sizes[s];
        printf("--- Size: %zu MB ---\n", bytes / (1024 * 1024));

        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));

        // Pageable memory (normal malloc)
        float *h_pageable = (float *)malloc(bytes);
        for (size_t i = 0; i < bytes / sizeof(float); i++) h_pageable[i] = 1.0f;
        benchTransfer(h_pageable, d_data, bytes, iters[s], "Pageable");

        // Pinned memory (cudaMallocHost)
        float *h_pinned;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
        for (size_t i = 0; i < bytes / sizeof(float); i++) h_pinned[i] = 1.0f;
        benchTransfer(h_pinned, d_data, bytes, iters[s], "Pinned");

        printf("\n");

        free(h_pageable);
        CUDA_CHECK(cudaFreeHost(h_pinned));
        CUDA_CHECK(cudaFree(d_data));
    }

    printf("=== Why the difference? ===\n");
    printf("Pageable: OS can swap pages out. CUDA must first copy to a pinned\n");
    printf("          staging buffer, then DMA to GPU → double copy!\n");
    printf("Pinned:   Memory is locked in physical RAM. CUDA DMAs directly\n");
    printf("          from your buffer to the GPU → single copy.\n");
    printf("\nWarning: Pinned memory is scarce. Allocating too much starves the OS.\n");

    return 0;
}
```

### Compile and run

Compile and run to compare pageable (normal `malloc`) versus pinned (`cudaMallocHost`) memory transfer speeds.

```bash
nvcc -O2 -o pinned_vs_pageable pinned_vs_pageable.cu
./pinned_vs_pageable
```

### Expected Output

```
=== Pageable vs Pinned Memory Transfer Speed ===

--- Size: 1 MB ---
Pageable      H→D:   0.125 ms (  8.00 GB/s)  D→H:   0.130 ms (  7.69 GB/s)
Pinned        H→D:   0.082 ms ( 12.20 GB/s)  D→H:   0.084 ms ( 11.90 GB/s)

--- Size: 16 MB ---
Pageable      H→D:   1.654 ms (  9.67 GB/s)  D→H:   1.721 ms (  9.30 GB/s)
Pinned        H→D:   1.297 ms ( 12.34 GB/s)  D→H:   1.310 ms ( 12.21 GB/s)

--- Size: 64 MB ---
Pageable      H→D:   6.492 ms (  9.86 GB/s)  D→H:   6.780 ms (  9.44 GB/s)
Pinned        H→D:   5.192 ms ( 12.33 GB/s)  D→H:   5.298 ms ( 12.08 GB/s)

--- Size: 256 MB ---
Pageable      H→D:  25.931 ms (  9.87 GB/s)  D→H:  27.200 ms (  9.41 GB/s)
Pinned        H→D:  20.752 ms ( 12.34 GB/s)  D→H:  21.184 ms ( 12.09 GB/s)
```

> Pinned memory consistently achieves ~25-50% higher bandwidth by avoiding the extra copy through a staging buffer.

---

## Step 4: Visualize Where Time Goes

Create `time_breakdown.cu`:

```cuda
// time_breakdown.cu — See exactly how time is split between transfer and compute
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

__global__ void heavyCompute(float *data, int n, int iterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        for (int iter = 0; iter < iterations; iter++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val) + 1.0f);
        }
        data[i] = val;
    }
}

int main() {
    int N = 1 << 22;  // 4M elements
    size_t bytes = N * sizeof(float);

    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));  // Use pinned for best transfer
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;

    cudaEvent_t e0, e1, e2, e3, e4;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));
    CUDA_CHECK(cudaEventCreate(&e4));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int computeIterations[] = {1, 10, 50, 100, 500};
    int numTests = sizeof(computeIterations) / sizeof(computeIterations[0]);

    printf("N = %d elements (%.0f MB)\n\n", N, bytes / (1024.0 * 1024.0));
    printf("%-8s  %8s  %8s  %8s  %8s  %10s\n",
           "Iters", "H→D", "Compute", "D→H", "Total", "Compute%");
    printf("%-8s  %8s  %8s  %8s  %8s  %10s\n",
           "-----", "---", "-------", "---", "-----", "--------");

    for (int t = 0; t < numTests; t++) {
        int iters = computeIterations[t];

        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        heavyCompute<<<blocks, threadsPerBlock>>>(d_data, N, iters);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));

        float h2d_ms, compute_ms, d2h_ms;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&compute_ms, e1, e2));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, e2, e3));
        float total = h2d_ms + compute_ms + d2h_ms;
        float computePct = (compute_ms / total) * 100.0f;

        printf("%-8d  %7.3f  %8.3f  %7.3f  %8.3f  %9.1f%%\n",
               iters, h2d_ms, compute_ms, d2h_ms, total, computePct);
    }

    printf("\n→ The GPU only 'earns its keep' when compute dominates transfer.\n");
    printf("→ Target: compute should be >80%% of total time.\n");

    // Draw a simple ASCII bar chart
    printf("\n=== Visual: Time Breakdown (iters=500) ===\n");
    {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        heavyCompute<<<blocks, threadsPerBlock>>>(d_data, N, 500);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));

        float h2d, comp, d2h;
        CUDA_CHECK(cudaEventElapsedTime(&h2d, e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&comp, e1, e2));
        CUDA_CHECK(cudaEventElapsedTime(&d2h, e2, e3));
        float total = h2d + comp + d2h;

        int barW = 60;
        int h2d_chars = (int)(h2d / total * barW);
        int comp_chars = (int)(comp / total * barW);
        int d2h_chars = barW - h2d_chars - comp_chars;

        printf("[");
        for (int i = 0; i < h2d_chars; i++) printf("<");
        for (int i = 0; i < comp_chars; i++) printf("=");
        for (int i = 0; i < d2h_chars; i++) printf(">");
        printf("]\n");
        printf(" < = H→D    = = Compute    > = D→H\n");
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaEventDestroy(e3));
    CUDA_CHECK(cudaEventDestroy(e4));
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Compile and run

Compile and run to see an ASCII visualization of how time is split between data transfer and GPU computation.

```bash
nvcc -O2 -o time_breakdown time_breakdown.cu
./time_breakdown
```

### Expected Output

```
N = 4194304 elements (16 MB)

Iters       H→D   Compute      D→H     Total   Compute%
-----       ---   -------      ---     -----   --------
1          1.312     0.098    1.358     2.768       3.5%
10         1.315     0.482    1.360     3.157      15.3%
50         1.318     2.210    1.352     4.880      45.3%
100        1.320     4.389    1.358     7.067      62.1%
500        1.325    21.876    1.362    24.563      89.1%

→ The GPU only 'earns its keep' when compute dominates transfer.
→ Target: compute should be >80% of total time.

=== Visual: Time Breakdown (iters=500) ===
[<<<=============================================>>>>>>>]
 < = H→D    = = Compute    > = D→H
```

---

## Experiments

### Experiment 1: PCIe Generation
Look up your system's PCIe generation with `nvidia-smi -q | grep -i pcie` or `lspci -vv`. Compare your measured bandwidth to the theoretical max (Gen3 x16 ≈ 15.75 GB/s, Gen4 x16 ≈ 31.5 GB/s).

### Experiment 2: Bidirectional transfers
With pinned memory, try overlapping H→D and D→H using two streams. Does total bandwidth increase? (This previews Lab 08.)

### Experiment 3: Pinned memory allocation overhead
Time `cudaMallocHost` vs `malloc` for different sizes. Pinned allocation is much slower — it's a one-time cost you pay upfront.

### Experiment 4: Unified Memory
Replace `cudaMalloc`/`cudaMemcpy` with `cudaMallocManaged`. Measure performance. When does the automatic migration help vs hurt?

### Experiment 5: Transfer size sweet spot
Run `transfer_bench` with sizes from 1 byte to 1 GB. Plot bandwidth vs size. Find the minimum transfer size needed to saturate PCIe bandwidth.

---

## What Just Happened?

1. **PCIe is the bottleneck.** GPU memory bandwidth is 500-2000 GB/s, but the PCIe link connecting host and device is only 12-32 GB/s. That's a 50-100× gap. Every byte you transfer pays this toll.

2. **Small transfers are catastrophically inefficient.** A 1 KB transfer still pays ~5-10μs of setup overhead, achieving < 1 GB/s effective bandwidth. Batch your transfers!

3. **Pinned memory eliminates a hidden copy.** With pageable memory, CUDA silently copies your data to a pinned staging buffer before DMAing to the GPU. `cudaMallocHost` removes this intermediate step.

4. **The compute-to-transfer ratio determines whether the GPU is worthwhile.** If your kernel does 1 FLOP per byte transferred, the GPU is wasted. If it does 100 FLOPs per byte, the GPU dominates. This is called *arithmetic intensity*.

---

## Key Insight

> **Every CUDA optimization story starts and ends with memory transfers.** Before optimizing your kernel, ask: "Can I transfer less data? Can I overlap transfers with compute? Can I keep data on the GPU longer?" Reducing transfers often matters more than making the kernel itself faster.

---

## Checkpoint Quiz

**Q1:** Your kernel processes 100 MB of data with 5 ms of compute time. PCIe bandwidth is 12 GB/s. What fraction of total time is spent on transfers?
<details><summary>Answer</summary>
Transfer time = 100 MB ÷ 12 GB/s = 100/12000 s ≈ 8.33 ms (for one direction). Round trip = ~16.7 ms. Total time = 16.7 + 5 = 21.7 ms. Transfer fraction = 16.7/21.7 ≈ 77%. The kernel is **transfer-bound** — optimizing compute won't help much.
</details>

**Q2:** Why is pinned memory faster for transfers but you shouldn't use it for all allocations?
<details><summary>Answer</summary>
Pinned memory is locked in physical RAM and cannot be paged out to disk. This allows direct DMA transfer to the GPU. However, allocating too much pinned memory starves the OS of pageable memory, potentially causing system-wide performance degradation or out-of-memory errors. Use it strategically for GPU transfer buffers only.
</details>

**Q3:** You have a pipeline that processes images: transfer to GPU → process → transfer back. Each image is 8 MB. What's the minimum compute time per image needed for the GPU pipeline to be faster than a CPU taking 10 ms per image?
<details><summary>Answer</summary>
At 12 GB/s PCIe: H→D = 8/12000 s ≈ 0.67 ms, D→H ≈ 0.67 ms. Total transfer ≈ 1.34 ms. For GPU to be faster: transfer + compute < 10 ms → compute < 8.66 ms. But the question is: can the GPU kernel finish in < 8.66 ms AND be correct? Yes, but the real win comes when you overlap transfers with compute using streams (Lab 08), where you pay the transfer cost only once for the pipeline.
</details>

---

*Next Lab: [Lab 04 — Shared Memory Speedup](Lab04_Shared_Memory_Speedup.md)*
