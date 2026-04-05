# Lab 08: Stream Overlap 🔴

| Detail | Value |
|---|---|
| **Difficulty** | 🔴 Advanced |
| **Estimated Time** | 70–90 minutes |
| **Prerequisites** | Labs 01-07; understanding of PCIe transfers and kernel execution |
| **GPU Required** | Any NVIDIA GPU with concurrent copy and execute capability |

---

## Objective

By the end of this lab you will:
- Understand CUDA streams and their role in concurrent execution
- Build a multi-stream pipeline that overlaps H2D copy, compute, and D2H copy
- Measure speedup from 1 stream to 2, 4, and more streams
- Use CUDA events for precise per-stream timing
- See the point of diminishing returns for stream count

---

## Setup

```bash
mkdir -p ~/cuda-labs/lab08 && cd ~/cuda-labs/lab08
```

### Background: Why Streams?

By default, all CUDA operations go into the **default stream** (stream 0) and execute sequentially. CUDA streams allow you to issue operations that can execute **concurrently**:
- H→D copy on copy engine 1
- Kernel on compute engine
- D→H copy on copy engine 2

This overlap can hide transfer latency entirely for pipeline-style workloads.

---

## Step 1: Check Concurrent Copy + Execute Capability

Create `check_overlap.cu`:

```cuda
// check_overlap.cu — Verify your GPU supports overlapped operations
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU: %s\n\n", prop.name);
    printf("Concurrent copy and execution:  %s\n",
           prop.deviceOverlap ? "YES ✓" : "NO ✗");
    printf("Async engine count:             %d\n", prop.asyncEngineCount);
    printf("  (1 = can overlap 1 copy + compute)\n");
    printf("  (2 = can overlap H2D + compute + D2H simultaneously)\n");
    printf("Concurrent kernels:             %s\n",
           prop.concurrentKernels ? "YES" : "NO");
    printf("Max streams (practical):        ~16-32 (hardware dependent)\n");

    if (!prop.deviceOverlap) {
        printf("\n⚠ Your GPU does NOT support overlap. Stream experiments will\n");
        printf("  still work but won't show speedup from concurrency.\n");
    }

    return 0;
}
```

### Compile and run

```bash
nvcc -o check_overlap check_overlap.cu
./check_overlap
```

---

## Step 2: Single Stream vs Multi-Stream

Create `stream_overlap.cu`:

```cuda
// stream_overlap.cu — Overlap copy and compute with multiple streams
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define TOTAL_N (1 << 24)  // 16M elements total

// Compute-heavy kernel to give copy engines time to overlap
__global__ void heavyCompute(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        for (int k = 0; k < 200; k++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val) + 1.0f);
        }
        data[i] = val;
    }
}

// ==================== SEQUENTIAL (1 STREAM) ====================
float runSequential(float *h_data, float *d_data, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    size_t bytes = n * sizeof(float);
    int block = 256;
    int grid = (n + block - 1) / block;

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    heavyCompute<<<grid, block>>>(d_data, n);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// ==================== MULTI-STREAM ====================
float runMultiStream(float *h_data, float *d_data, int n, int numStreams) {
    size_t bytes = n * sizeof(float);
    int chunkSize = n / numStreams;
    size_t chunkBytes = chunkSize * sizeof(float);

    // Create streams
    cudaStream_t *streams = new cudaStream_t[numStreams];
    for (int s = 0; s < numStreams; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;

    CUDA_CHECK(cudaEventRecord(start));

    for (int s = 0; s < numStreams; s++) {
        int offset = s * chunkSize;
        int thisChunk = (s == numStreams - 1) ? (n - offset) : chunkSize;
        size_t thisBytes = thisChunk * sizeof(float);
        int thisGrid = (thisChunk + block - 1) / block;

        // Each stream: H2D → compute → D2H
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                    thisBytes, cudaMemcpyHostToDevice, streams[s]));
        heavyCompute<<<thisGrid, block, 0, streams[s]>>>(d_data + offset, thisChunk);
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                    thisBytes, cudaMemcpyDeviceToHost, streams[s]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    for (int s = 0; s < numStreams; s++)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    delete[] streams;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    int N = TOTAL_N;
    size_t bytes = N * sizeof(float);

    printf("=== CUDA Stream Overlap ===\n");
    printf("N = %d elements (%.0f MB)\n\n", N, bytes / (1024.0 * 1024.0));

    // MUST use pinned memory for async transfers!
    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Warm up
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    heavyCompute<<<(N + 255) / 256, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark sequential
    float seqMs = runSequential(h_data, d_data, N);

    printf("%-20s  %10s  %10s\n", "Configuration", "Time (ms)", "Speedup");
    printf("%-20s  %10s  %10s\n", "-------------", "---------", "-------");
    printf("%-20s  %10.3f  %10.2fx\n", "Sequential (1 stream)", seqMs, 1.0f);

    // Benchmark multi-stream
    int streamCounts[] = {2, 4, 8, 16, 32};
    for (int i = 0; i < 5; i++) {
        int ns = streamCounts[i];
        float ms = runMultiStream(h_data, d_data, N, ns);
        char label[64];
        snprintf(label, sizeof(label), "%d streams", ns);
        printf("%-20s  %10.3f  %10.2fx\n", label, ms, seqMs / ms);
    }

    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Compile and run

```bash
nvcc -O2 -o stream_overlap stream_overlap.cu
./stream_overlap
```

### Expected Output

```
=== CUDA Stream Overlap ===
N = 16777216 elements (64 MB)

Configuration         Time (ms)     Speedup
-------------         ---------     -------
Sequential (1 stream)    28.456       1.00x
2 streams                19.234       1.48x
4 streams                15.678       1.82x
8 streams                14.123       2.01x
16 streams               13.891       2.05x
32 streams               13.950       2.04x
```

---

## Step 3: Detailed Timeline with Events

Create `stream_timeline.cu`:

```cuda
// stream_timeline.cu — Visualize the overlap pattern with per-phase timing
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 22)  // 4M elements

__global__ void compute(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        for (int k = 0; k < 500; k++) {
            val = sinf(val) + cosf(val);
        }
        data[i] = val;
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    int numStreams = 4;
    int chunkSize = N / numStreams;
    size_t chunkBytes = chunkSize * sizeof(float);

    printf("=== Stream Timeline (N=%d, %d streams, chunk=%.1f MB) ===\n\n",
           N, numStreams, chunkBytes / (1024.0 * 1024.0));

    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));
    for (int i = 0; i < N; i++) h_data[i] = 0.5f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Create streams and events
    cudaStream_t streams[4];
    cudaEvent_t h2d_start[4], h2d_end[4];
    cudaEvent_t comp_start[4], comp_end[4];
    cudaEvent_t d2h_start[4], d2h_end[4];
    cudaEvent_t globalStart, globalEnd;

    CUDA_CHECK(cudaEventCreate(&globalStart));
    CUDA_CHECK(cudaEventCreate(&globalEnd));

    for (int s = 0; s < numStreams; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaEventCreate(&h2d_start[s]));
        CUDA_CHECK(cudaEventCreate(&h2d_end[s]));
        CUDA_CHECK(cudaEventCreate(&comp_start[s]));
        CUDA_CHECK(cudaEventCreate(&comp_end[s]));
        CUDA_CHECK(cudaEventCreate(&d2h_start[s]));
        CUDA_CHECK(cudaEventCreate(&d2h_end[s]));
    }

    int block = 256;

    // Warm up
    compute<<<(N + 255) / 256, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch with per-phase events
    CUDA_CHECK(cudaEventRecord(globalStart));

    for (int s = 0; s < numStreams; s++) {
        int offset = s * chunkSize;
        int thisGrid = (chunkSize + block - 1) / block;

        CUDA_CHECK(cudaEventRecord(h2d_start[s], streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                    chunkBytes, cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaEventRecord(h2d_end[s], streams[s]));

        CUDA_CHECK(cudaEventRecord(comp_start[s], streams[s]));
        compute<<<thisGrid, block, 0, streams[s]>>>(d_data + offset, chunkSize);
        CUDA_CHECK(cudaEventRecord(comp_end[s], streams[s]));

        CUDA_CHECK(cudaEventRecord(d2h_start[s], streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                    chunkBytes, cudaMemcpyDeviceToHost, streams[s]));
        CUDA_CHECK(cudaEventRecord(d2h_end[s], streams[s]));
    }

    CUDA_CHECK(cudaEventRecord(globalEnd));
    CUDA_CHECK(cudaEventSynchronize(globalEnd));

    float totalMs;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, globalStart, globalEnd));

    // Print timeline
    printf("Stream  Phase      Start(ms)    End(ms)  Duration(ms)\n");
    printf("------  -----      ---------    -------  ------------\n");

    for (int s = 0; s < numStreams; s++) {
        float h2d_s, h2d_e, comp_s_val, comp_e_val, d2h_s, d2h_e;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_s, globalStart, h2d_start[s]));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_e, globalStart, h2d_end[s]));
        CUDA_CHECK(cudaEventElapsedTime(&comp_s_val, globalStart, comp_start[s]));
        CUDA_CHECK(cudaEventElapsedTime(&comp_e_val, globalStart, comp_end[s]));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_s, globalStart, d2h_start[s]));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_e, globalStart, d2h_end[s]));

        printf("  %d     H→D       %8.3f  %8.3f      %8.3f\n",
               s, h2d_s, h2d_e, h2d_e - h2d_s);
        printf("  %d     Compute   %8.3f  %8.3f      %8.3f\n",
               s, comp_s_val, comp_e_val, comp_e_val - comp_s_val);
        printf("  %d     D→H       %8.3f  %8.3f      %8.3f\n",
               s, d2h_s, d2h_e, d2h_e - d2h_s);
        printf("\n");
    }

    printf("Total wall time: %.3f ms\n\n", totalMs);

    // ASCII timeline visualization
    printf("=== Visual Timeline (approximate) ===\n\n");
    float scale = 60.0f / totalMs;  // chars per ms
    printf("Time:  ");
    for (int t = 0; t <= (int)(totalMs); t += (int)(totalMs / 10))
        printf("%-6d", t);
    printf("\n");

    for (int s = 0; s < numStreams; s++) {
        float h2d_s, h2d_e, comp_s_val, comp_e_val, d2h_s, d2h_e;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_s, globalStart, h2d_start[s]));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_e, globalStart, h2d_end[s]));
        CUDA_CHECK(cudaEventElapsedTime(&comp_s_val, globalStart, comp_start[s]));
        CUDA_CHECK(cudaEventElapsedTime(&comp_e_val, globalStart, comp_end[s]));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_s, globalStart, d2h_start[s]));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_e, globalStart, d2h_end[s]));

        printf("S%d:    ", s);
        int cols = (int)(totalMs * scale) + 1;
        for (int c = 0; c < cols; c++) {
            float t = c / scale;
            if (t >= h2d_s && t < h2d_e) printf("<");
            else if (t >= comp_s_val && t < comp_e_val) printf("=");
            else if (t >= d2h_s && t < d2h_e) printf(">");
            else printf(".");
        }
        printf("\n");
    }
    printf("\n  < = H→D    = = Compute    > = D→H    . = idle\n");

    // Cleanup
    for (int s = 0; s < numStreams; s++) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaEventDestroy(h2d_start[s]));
        CUDA_CHECK(cudaEventDestroy(h2d_end[s]));
        CUDA_CHECK(cudaEventDestroy(comp_start[s]));
        CUDA_CHECK(cudaEventDestroy(comp_end[s]));
        CUDA_CHECK(cudaEventDestroy(d2h_start[s]));
        CUDA_CHECK(cudaEventDestroy(d2h_end[s]));
    }
    CUDA_CHECK(cudaEventDestroy(globalStart));
    CUDA_CHECK(cudaEventDestroy(globalEnd));
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Compile and run

```bash
nvcc -O2 -o stream_timeline stream_timeline.cu
./stream_timeline
```

### Expected Output

```
=== Stream Timeline (N=4194304, 4 streams, chunk=4.0 MB) ===

Stream  Phase      Start(ms)    End(ms)  Duration(ms)
------  -----      ---------    -------  ------------
  0     H→D          0.000     0.342         0.342
  0     Compute      0.342     5.678         5.336
  0     D→H          5.678     6.020         0.342

  1     H→D          0.342     0.685         0.343
  1     Compute      0.685     6.015         5.330
  1     D→H          6.015     6.358         0.343

  2     H→D          0.685     1.028         0.343
  2     Compute      5.678     11.012        5.334
  2     D→H         11.012    11.355         0.343

  3     H→D          1.028     1.371         0.343
  3     Compute      6.015    11.350         5.335
  3     D→H         11.350    11.694         0.344

Total wall time: 11.694 ms

=== Visual Timeline (approximate) ===

S0:    <<===========================>>............................
S1:    .<<===========================>>...........................
S2:    ..<<..........................===========================>>
S3:    ...<<........................===========================.>>

  < = H→D    = = Compute    > = D→H    . = idle
```

> Notice how stream 1's H→D overlaps with stream 0's compute, and so on. The pipeline keeps all engines busy.

---

## Experiments

### Experiment 1: Vary compute intensity
Reduce the loop count in `heavyCompute` from 200 to 10. With very light compute, overlap helps less because copies dominate. Find the crossover.

### Experiment 2: Pageable vs pinned with streams
Try using `malloc` instead of `cudaMallocHost`. What happens to async transfers? (Spoiler: they silently fall back to synchronous.)

### Experiment 3: Stream callbacks
Add `cudaStreamAddCallback` to print a message when each stream finishes. This shows the real completion order.

### Experiment 4: Stream priorities
Create streams with `cudaStreamCreateWithPriority`. Give one stream high priority and others low. Does the high-priority stream finish first?

### Experiment 5: Multiple kernels per stream
Launch 3 different kernels per stream (e.g., preprocess → compute → postprocess). Does the GPU overlap kernels from different streams?

---

## What Just Happened?

1. **Streams enable concurrency.** Without streams, operations execute sequentially: H2D → compute → D2H → H2D → compute → D2H. With streams, operations from different streams can overlap: while stream 0 computes, stream 1 copies.

2. **Pinned memory is required for async transfers.** `cudaMemcpyAsync` with pageable memory silently degrades to synchronous behavior. Always use `cudaMallocHost` for stream-based pipelines.

3. **The GPU has separate engines.** Most GPUs have at least 2 copy engines (one for each direction) and multiple compute engines. Streams expose this hardware parallelism to the programmer.

4. **Diminishing returns after 4-8 streams.** Once all engines are busy, adding more streams just adds overhead without improving overlap. The sweet spot is usually 2-4 streams for simple pipelines.

---

## Key Insight

> **Streams turn a sequential pipeline (H2D → compute → D2H) into an overlapped pipeline where copies and compute happen simultaneously.** The speedup depends on how much copy time you can hide behind compute time. For compute-heavy kernels, 2-4 streams can nearly eliminate transfer overhead.

---

## Checkpoint Quiz

**Q1:** You have a workload with 10 ms of H2D, 50 ms of compute, and 10 ms of D2H. With perfect 4-stream overlap, what's the theoretical best total time?
<details><summary>Answer</summary>
Without overlap: 10 + 50 + 10 = 70 ms. With 4 streams, each chunk has 2.5 + 12.5 + 2.5 = 17.5 ms. With perfect overlap, the pipeline time ≈ first H2D (2.5) + all computes overlapped (12.5 × 4 / parallel ≈ 50) + last D2H (2.5) ≈ 55 ms. The practical speedup is ~1.27×. The overlap mainly hides the 5 ms of serial copy within the 50 ms compute window.
</details>

**Q2:** Why can't you use `malloc` (pageable memory) with `cudaMemcpyAsync`?
<details><summary>Answer</summary>
Pageable memory can be swapped to disk by the OS at any time. The GPU's DMA engine needs a fixed physical address to transfer data. With pageable memory, CUDA must first copy to an internal pinned buffer (blocking the CPU), then DMA from that buffer — making the "async" call effectively synchronous. Pinned memory (`cudaMallocHost`) guarantees the memory stays at a fixed physical address, enabling true async DMA.
</details>

**Q3:** Your GPU has 1 copy engine (not 2). Can you still benefit from streams?
<details><summary>Answer</summary>
Yes, but with reduced benefit. With 1 copy engine, H2D and D2H cannot overlap with each other, but copies can still overlap with compute. The timeline becomes: H2D₀ → [compute₀ + H2D₁] → [compute₁ + D2H₀] → D2H₁. You save ~1 copy time compared to fully sequential. With 2 copy engines, you could additionally overlap H2D and D2H for even more speedup.
</details>

---

*Next Lab: [Lab 09 — Reduction Optimization](Lab09_Reduction_Optimization.md)*
