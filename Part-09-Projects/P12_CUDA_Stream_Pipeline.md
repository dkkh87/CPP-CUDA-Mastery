# Project 12 — Multi-Stage Processing Pipeline with CUDA Streams

> **Difficulty:** 🟡 Intermediate
> **Estimated Time:** 3–4 hours
> **GPU Required:** Any CUDA-capable GPU (Compute Capability ≥ 3.5)

---

## Prerequisites

| Topic | Why It Matters |
|---|---|
| CUDA kernel launch syntax | You will write and launch multiple kernels per stream |
| Host / device memory management | `cudaMalloc`, `cudaFree`, `cudaMemcpy` basics |
| Pinned (page-locked) memory | Required for asynchronous transfers (`cudaMallocHost`) |
| Basic concurrency concepts | Streams overlap transfers and compute — you must reason about ordering |

---

## Learning Objectives

1. **Create and manage multiple CUDA streams** to overlap data transfers with kernel execution.
2. **Implement a staged pipeline** (Upload → Process1 → Process2 → Download) where different chunks execute different stages concurrently.
3. **Use CUDA events** to measure per-stage and total elapsed time accurately.
4. **Compare single-stream vs. multi-stream** execution and quantify the speedup.
5. **Read an Nsight Systems timeline** and identify transfer/compute overlap regions.

---

## Architecture Overview

### Pipeline Stages

```mermaid
flowchart LR
    A[Host Input] -->|cudaMemcpyAsync H→D| B[Stage 1: Scale + Bias]
    B -->|kernel| C[Stage 2: Smooth Filter]
    C -->|cudaMemcpyAsync D→H| D[Host Output]
```

### Stream Overlap Timeline (3 Streams, 4 Chunks)

Each row is a CUDA stream. Time flows left → right.

```mermaid
gantt
    title CUDA Stream Pipeline — 4 Chunks across 3 Streams
    dateFormat X
    axisFormat %s

    section Stream 0
    Upload C0   :a0, 0, 1
    Process1 C0 :b0, 1, 2
    Process2 C0 :c0, 2, 3
    Download C0 :d0, 3, 4
    Upload C3   :a3, 4, 5
    Process1 C3 :b3, 5, 6
    Process2 C3 :c3, 6, 7
    Download C3 :d3, 7, 8

    section Stream 1
    Upload C1   :a1, 1, 2
    Process1 C1 :b1, 2, 3
    Process2 C1 :c1, 3, 4
    Download C1 :d1, 4, 5

    section Stream 2
    Upload C2   :a2, 2, 3
    Process1 C2 :b2, 3, 4
    Process2 C2 :c2, 4, 5
    Download C2 :d2, 5, 6
```

> **Key insight:** While Stream 0 runs Process1 on Chunk 0, Stream 1 uploads
> Chunk 1 — three different stages can execute in parallel across streams.

---

## Step-by-Step Implementation

### Step 1 — Error Checking Macro

```cuda
// stream_pipeline.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
```

### Step 2 — Processing Kernels

**Stage 1** — element-wise scale + bias. **Stage 2** — 1-D smooth filter (stencil average).

```cuda
__global__ void scale_bias_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n, float scale, float bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale + bias;
    }
}

__global__ void smooth_filter_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int n, int radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        int count = 0;
        for (int offset = -radius; offset <= radius; ++offset) {
            int neighbor = idx + offset;
            if (neighbor >= 0 && neighbor < n) {
                sum += input[neighbor];
                ++count;
            }
        }
        output[idx] = sum / count;
    }
}
```

### Step 3 — Single-Stream Baseline

```cuda
void run_single_stream(const float* h_input, float* h_output, int N,
                       float scale, float bias, int radius)
{
    size_t bytes = N * sizeof(float);
    float *d_input, *d_inter, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_inter,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    // Upload
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Stage 1
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    scale_bias_kernel<<<blocks, threads>>>(d_input, d_inter, N, scale, bias);

    // Stage 2
    smooth_filter_kernel<<<blocks, threads>>>(d_inter, d_output, N, radius);

    // Download
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("[Single-stream]  Total: %.3f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_inter));
    CUDA_CHECK(cudaFree(d_output));
}
```

### Step 4 — Multi-Stream Overlapped Pipeline

Split data into `NUM_CHUNKS` pieces, distribute round-robin across `NUM_STREAMS`
streams. Each stream issues Upload → Process1 → Process2 → Download; independent
streams overlap on the GPU.

```cuda
void run_multi_stream(const float* h_input, float* h_output, int N,
                      float scale, float bias, int radius,
                      int num_streams, int num_chunks)
{
    size_t bytes = N * sizeof(float);
    int chunk_size = (N + num_chunks - 1) / num_chunks;

    float *d_input, *d_inter, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_inter,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    cudaStream_t* streams = new cudaStream_t[num_streams];
    for (int s = 0; s < num_streams; ++s)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Per-chunk stage events for detailed timing
    cudaEvent_t *ev_us = new cudaEvent_t[num_chunks], *ev_ue = new cudaEvent_t[num_chunks];
    cudaEvent_t *ev_1s = new cudaEvent_t[num_chunks], *ev_1e = new cudaEvent_t[num_chunks];
    cudaEvent_t *ev_2s = new cudaEvent_t[num_chunks], *ev_2e = new cudaEvent_t[num_chunks];
    cudaEvent_t *ev_ds = new cudaEvent_t[num_chunks], *ev_de = new cudaEvent_t[num_chunks];
    for (int c = 0; c < num_chunks; ++c) {
        CUDA_CHECK(cudaEventCreate(&ev_us[c])); CUDA_CHECK(cudaEventCreate(&ev_ue[c]));
        CUDA_CHECK(cudaEventCreate(&ev_1s[c])); CUDA_CHECK(cudaEventCreate(&ev_1e[c]));
        CUDA_CHECK(cudaEventCreate(&ev_2s[c])); CUDA_CHECK(cudaEventCreate(&ev_2e[c]));
        CUDA_CHECK(cudaEventCreate(&ev_ds[c])); CUDA_CHECK(cudaEventCreate(&ev_de[c]));
    }

    CUDA_CHECK(cudaEventRecord(ev_start, 0));

    int threads = 256;
    for (int c = 0; c < num_chunks; ++c) {
        int offset   = c * chunk_size;
        int cur_size = min(chunk_size, N - offset);
        if (cur_size <= 0) break;
        size_t cur_bytes = cur_size * sizeof(float);
        int blocks = (cur_size + threads - 1) / threads;
        cudaStream_t st = streams[c % num_streams];

        CUDA_CHECK(cudaEventRecord(ev_us[c], st));
        CUDA_CHECK(cudaMemcpyAsync(d_input + offset, h_input + offset,
                                   cur_bytes, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaEventRecord(ev_ue[c], st));

        CUDA_CHECK(cudaEventRecord(ev_1s[c], st));
        scale_bias_kernel<<<blocks, threads, 0, st>>>(
            d_input + offset, d_inter + offset, cur_size, scale, bias);
        CUDA_CHECK(cudaEventRecord(ev_1e[c], st));

        CUDA_CHECK(cudaEventRecord(ev_2s[c], st));
        smooth_filter_kernel<<<blocks, threads, 0, st>>>(
            d_inter + offset, d_output + offset, cur_size, radius);
        CUDA_CHECK(cudaEventRecord(ev_2e[c], st));

        CUDA_CHECK(cudaEventRecord(ev_ds[c], st));
        CUDA_CHECK(cudaMemcpyAsync(h_output + offset, d_output + offset,
                                   cur_bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaEventRecord(ev_de[c], st));
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    for (int s = 0; s < num_streams; ++s)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    printf("[Multi-stream]   Total: %.3f ms  (streams=%d, chunks=%d)\n",
           total_ms, num_streams, num_chunks);

    printf("\n  %-6s  %-10s %-10s %-10s %-10s\n",
           "Chunk", "Upload ms", "Stage1 ms", "Stage2 ms", "DLoad ms");
    for (int c = 0; c < num_chunks; ++c) {
        if (c * chunk_size >= N) break;
        float t_up, t_p1, t_p2, t_dl;
        CUDA_CHECK(cudaEventElapsedTime(&t_up, ev_us[c], ev_ue[c]));
        CUDA_CHECK(cudaEventElapsedTime(&t_p1, ev_1s[c], ev_1e[c]));
        CUDA_CHECK(cudaEventElapsedTime(&t_p2, ev_2s[c], ev_2e[c]));
        CUDA_CHECK(cudaEventElapsedTime(&t_dl, ev_ds[c], ev_de[c]));
        printf("  C%-5d  %-10.3f %-10.3f %-10.3f %-10.3f\n", c, t_up, t_p1, t_p2, t_dl);
    }

    for (int c = 0; c < num_chunks; ++c) {
        cudaEventDestroy(ev_us[c]); cudaEventDestroy(ev_ue[c]);
        cudaEventDestroy(ev_1s[c]); cudaEventDestroy(ev_1e[c]);
        cudaEventDestroy(ev_2s[c]); cudaEventDestroy(ev_2e[c]);
        cudaEventDestroy(ev_ds[c]); cudaEventDestroy(ev_de[c]);
    }
    delete[] ev_us; delete[] ev_ue; delete[] ev_1s; delete[] ev_1e;
    delete[] ev_2s; delete[] ev_2e; delete[] ev_ds; delete[] ev_de;
    CUDA_CHECK(cudaEventDestroy(ev_start));  CUDA_CHECK(cudaEventDestroy(ev_stop));
    for (int s = 0; s < num_streams; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    delete[] streams;
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_inter)); CUDA_CHECK(cudaFree(d_output));
}
```

### Step 5 — CPU Reference and Validation

```cuda
void cpu_reference(const float* input, float* output, int N,
                   float scale, float bias, int radius)
{
    float* temp = new float[N];
    for (int i = 0; i < N; ++i)
        temp[i] = input[i] * scale + bias;

    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        int count = 0;
        for (int off = -radius; off <= radius; ++off) {
            int nb = i + off;
            if (nb >= 0 && nb < N) { sum += temp[nb]; ++count; }
        }
        output[i] = sum / count;
    }
    delete[] temp;
}

bool validate(const float* gpu, const float* cpu, int N, float tol = 1e-4f)
{
    float max_err = 0.0f;
    int   err_idx = -1;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(gpu[i] - cpu[i]);
        if (diff > max_err) { max_err = diff; err_idx = i; }
    }
    if (max_err > tol) {
        printf("VALIDATION FAILED — max error %.6f at index %d "
               "(gpu=%.6f, cpu=%.6f)\n",
               max_err, err_idx, gpu[err_idx], cpu[err_idx]);
        return false;
    }
    printf("Validation PASSED (max error = %.2e)\n", max_err);
    return true;
}
```

### Step 6 — Main Driver

```cuda
int main(int argc, char** argv)
{
    const int    N     = (argc > 1) ? atoi(argv[1]) : 1 << 22;  // ~4M floats
    const int    NSTR  = (argc > 2) ? atoi(argv[2]) : 3;
    const int    NCHK  = (argc > 3) ? atoi(argv[3]) : 8;
    const float  SCALE = 2.0f;
    const float  BIAS  = 0.5f;
    const int    RAD   = 4;

    printf("N = %d (%.1f MB)  streams = %d  chunks = %d\n",
           N, N * sizeof(float) / (1024.0f * 1024.0f), NSTR, NCHK);

    // Pinned host memory — mandatory for async transfers
    float *h_input, *h_single, *h_multi, *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_single, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_multi,  N * sizeof(float)));
    h_ref = new float[N];

    // Initialize input
    srand(42);
    for (int i = 0; i < N; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference
    cpu_reference(h_input, h_ref, N, SCALE, BIAS, RAD);

    // GPU — single stream baseline
    run_single_stream(h_input, h_single, N, SCALE, BIAS, RAD);
    printf("Single-stream: ");
    validate(h_single, h_ref, N);

    // GPU — multi-stream pipeline
    run_multi_stream(h_input, h_multi, N, SCALE, BIAS, RAD, NSTR, NCHK);
    printf("Multi-stream:  ");
    validate(h_multi, h_ref, N);

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_single));
    CUDA_CHECK(cudaFreeHost(h_multi));
    delete[] h_ref;

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
```

---

## Build and Run

```bash
# Compile (adjust sm_75 to your GPU architecture)
nvcc -O2 -arch=sm_75 -o stream_pipeline stream_pipeline.cu

# Run with defaults (N=4M, 3 streams, 8 chunks)
./stream_pipeline

# Experiment with parameters
./stream_pipeline 16777216 4 16    # 16M floats, 4 streams, 16 chunks
./stream_pipeline 4194304  1 1     # single-stream equivalent via multi-stream path
```

---

## Testing Strategy

| Test | What It Validates |
|---|---|
| **Correctness** | `validate()` compares GPU output against CPU reference with tolerance 1e-4 |
| **Single vs. Multi output equality** | Both paths must produce bit-identical results for the same input |
| **Edge: N not divisible by chunks** | Last chunk is smaller — verify no out-of-bounds writes |
| **Edge: 1 stream, 1 chunk** | Multi-stream path degenerates to single-stream — must still be correct |
| **Edge: chunks > N** | Several chunks will have `cur_size <= 0` — code must skip them cleanly |
| **Nsight Compute** | Run kernels through `ncu` to verify no memory access violations |

### Quick Smoke Test

```bash
./stream_pipeline 1024 2 4        # minimal — catches indexing bugs
./stream_pipeline 33554432 3 12   # large — stresses overlap
```

---

## Performance Analysis

### Why Multi-Stream Is Faster

Single-stream is strictly sequential — `Upload + Process1 + Process2 + Download`.
Multi-stream overlaps stages across chunks so the wall-clock time approaches
`max(total_upload, total_compute, total_download)` instead of their sum:

```
Stream 0: [Up C0][P1 C0][P2 C0][Dl C0]          [Up C3]...
Stream 1:        [Up C1][P1 C1][P2 C1][Dl C1]
Stream 2:               [Up C2][P1 C2][P2 C2][Dl C2]
```

### Nsight Systems Timeline Analysis

Profile with Nsight Systems to visually confirm overlap:

```bash
nsys profile --trace=cuda,nvtx -o pipeline_report ./stream_pipeline 16777216 3 8
```

**What to look for:** (1) HtoD transfers stagger across streams, (2) kernel
launches from different streams interleave on the Compute row, (3) DtoH
downloads begin before all kernels finish, (4) each stream row shows a clean
4-stage sequence offset in time.

> **Pitfall:** Using pageable memory (`malloc`) instead of pinned memory
> (`cudaMallocHost`) silently serializes all transfers — the Nsight timeline
> will show zero concurrency.

### Tuning Parameters

| Parameter | Effect |
|---|---|
| `num_streams` | More streams → more overlap potential, but diminishing returns past 3–4 |
| `num_chunks` | More chunks → finer granularity, better overlap; too many → launch overhead |
| `N` (array size) | Larger arrays benefit more — transfer time dominates and overlap hides it |
| `radius` (filter) | Larger radius → more compute per element → overlap hides transfer cost better |

---

## Extensions and Challenges

### 🔵 Extension 1 — Host Callback Post-Processing

Use `cudaLaunchHostFunc` to trigger CPU-side work per chunk without explicit sync.

### 🔵 Extension 2 — Add a Third Kernel Stage

Insert a nonlinear activation (`tanhf`) between Stage 1 and Stage 2. Measure
whether a deeper pipeline benefits from more streams.

### 🔴 Extension 3 — Dynamic Chunk Scheduling

Replace round-robin with an atomic work-counter — each stream grabs the next
unprocessed chunk, mimicking real-world GPU task schedulers.

### 🔴 Extension 4 — Multi-GPU Pipeline

Extend to 2+ GPUs with `cudaMemcpyPeerAsync` for inter-GPU transfers.

---

## Key Takeaways

1. **Pinned memory is non-negotiable.** `cudaMemcpyAsync` only runs asynchronously
   with page-locked buffers. Pageable memory forces hidden synchronization.
2. **Streams are independent task queues.** Intra-stream order is guaranteed;
   inter-stream operations may overlap based on hardware resources.
3. **Events are the right timing tool.** `cudaEventElapsedTime` measures GPU-side
   time without host jitter — always prefer events over `std::chrono`.
4. **More streams ≠ always faster.** Diminishing returns past 3–4 streams due to
   limited copy engines and SM capacity.
5. **Chunk granularity matters.** 4–16 chunks is a good starting range — too few
   limits overlap, too many adds launch overhead.
6. **Nsight Systems is essential.** The timeline view is the only reliable way to
   confirm overlap is happening.

---

## References

- [CUDA Programming Guide — Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [CUDA Best Practices — Async Transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
