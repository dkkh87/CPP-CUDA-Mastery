# Lab 04: Shared Memory Speedup 🟡

| Detail | Value |
|---|---|
| **Difficulty** | 🟡 Intermediate |
| **Estimated Time** | 60–90 minutes |
| **Prerequisites** | Labs 01-03; understanding of CUDA memory hierarchy |
| **GPU Required** | Any NVIDIA GPU (Compute Capability 3.0+) |

---

## Objective

By the end of this lab you will:
- Implement matrix transpose using naive global memory access
- Implement an optimized version using shared memory with coalesced access
- Measure and compare performance of both versions
- Understand and observe shared memory bank conflicts
- Experiment with different tile sizes and their impact

---

## Setup

Create a working directory for this lab's shared memory experiments.

```bash
mkdir -p ~/cuda-labs/lab04 && cd ~/cuda-labs/lab04
```

### Background: Why Matrix Transpose Is Hard on GPUs

Transposing a matrix means writing column `j` of the input as row `j` of the output. If you read row-major (coalesced), you must write column-major (uncoalesced) — or vice versa. Shared memory lets you decouple the read and write patterns: read coalesced into shared memory, then write coalesced from shared memory.

---

## Step 1: Naive Matrix Transpose

Create `transpose.cu`:

```cuda
// transpose.cu — Naive vs Shared Memory Matrix Transpose
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define TILE_DIM 32
#define BLOCK_ROWS 8

// ==================== NAIVE TRANSPOSE ====================
// Reads are coalesced (row-major), writes are strided (column-major) → SLOW
__global__ void transposeNaive(const float *input, float *output, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((row + j) < height && col < width) {
            output[col * height + (row + j)] = input[(row + j) * width + col];
        }
    }
}

// ==================== SHARED MEMORY TRANSPOSE ====================
// Uses shared memory to convert uncoalesced writes into coalesced writes
__global__ void transposeShared(const float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts!

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory → shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((yIndex + j) < height && xIndex < width) {
            tile[threadIdx.y + j][threadIdx.x] = input[(yIndex + j) * width + xIndex];
        }
    }

    __syncthreads();

    // Coalesced write from shared memory → global memory
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;  // Note: swapped block indices
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((yIndex + j) < width && xIndex < height) {
            output[(yIndex + j) * height + xIndex] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ==================== SHARED MEMORY WITH BANK CONFLICTS ====================
// Same as above but WITHOUT the +1 padding — shows the impact of bank conflicts
__global__ void transposeSharedConflicts(const float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];  // NO +1 → bank conflicts!

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((yIndex + j) < height && xIndex < width) {
            tile[threadIdx.y + j][threadIdx.x] = input[(yIndex + j) * width + xIndex];
        }
    }

    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((yIndex + j) < width && xIndex < height) {
            output[(yIndex + j) * height + xIndex] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ==================== COPY KERNEL (BANDWIDTH REFERENCE) ====================
__global__ void copyKernel(const float *input, float *output, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((row + j) < height && col < width) {
            output[(row + j) * width + col] = input[(row + j) * width + col];
        }
    }
}

// ==================== VERIFICATION ====================
void transposeHost(const float *in, float *out, int w, int h) {
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            out[c * h + r] = in[r * w + c];
}

bool verify(const float *ref, const float *test, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - test[i]) > 1e-5f) {
            printf("  MISMATCH at %d: ref=%.4f got=%.4f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

// ==================== BENCHMARK ====================
float benchmark(void (*kernel)(const float*, float*, int, int),
                const float *d_in, float *d_out,
                int width, int height, int iterations) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
              (height + TILE_DIM - 1) / TILE_DIM);

    // Warm up
    kernel<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_in, d_out, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;
}

int main() {
    int W = 4096, H = 4096;
    size_t bytes = W * H * sizeof(float);
    int iterations = 100;

    printf("Matrix Transpose: %d × %d (%.0f MB)\n", W, H, bytes / (1024.0 * 1024.0));
    printf("Tile: %d×%d, Block: %d×%d threads\n\n", TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

    // Allocate
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    for (int i = 0; i < W * H; i++) h_in[i] = (float)(i % 1000) * 0.001f;
    transposeHost(h_in, h_ref, W, H);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Benchmark each version
    struct {
        const char *name;
        void (*kernel)(const float*, float*, int, int);
        bool isTranspose;
    } kernels[] = {
        {"Copy (baseline)",       copyKernel,               false},
        {"Naive transpose",       transposeNaive,            true},
        {"Shared (bank conflict)", transposeSharedConflicts, true},
        {"Shared (optimized)",    transposeShared,           true},
    };
    int numKernels = sizeof(kernels) / sizeof(kernels[0]);

    float copyBW = 0;
    printf("%-25s  %10s  %12s  %10s\n", "Kernel", "Time (ms)", "Bandwidth", "Speedup");
    printf("%-25s  %10s  %12s  %10s\n", "------", "---------", "---------", "-------");

    float naiveMs = 1.0f;
    for (int k = 0; k < numKernels; k++) {
        float ms = benchmark(kernels[k].kernel, d_in, d_out, W, H, iterations);
        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;  // read + write

        if (k == 0) copyBW = bw;
        if (k == 1) naiveMs = ms;

        // Verify correctness
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        bool correct;
        if (kernels[k].isTranspose) {
            correct = verify(h_ref, h_out, W * H);
        } else {
            correct = verify(h_in, h_out, W * H);
        }

        printf("%-25s  %10.3f  %9.1f GB/s  %9.2fx  %s\n",
               kernels[k].name, ms, bw,
               (k <= 1) ? 1.0f : naiveMs / ms,
               correct ? "✓" : "✗ WRONG");
    }

    printf("\nTheoretical peak memory bandwidth: check with nvidia-smi or deviceQuery\n");
    printf("Copy bandwidth: %.1f GB/s (this is our achievable ceiling)\n", copyBW);

    // Cleanup
    free(h_in); free(h_out); free(h_ref);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
```

### Compile and run

Compile and run to compare naive transpose, shared-memory transpose, and shared-memory with bank-conflict padding.

```bash
nvcc -O2 -o transpose transpose.cu
./transpose
```

### Expected Output

```
Matrix Transpose: 4096 × 4096 (64 MB)
Tile: 32×32, Block: 32×8 threads

Kernel                     Time (ms)     Bandwidth     Speedup
------                     ---------     ---------     -------
Copy (baseline)                0.198     648.5 GB/s      1.00x  ✓
Naive transpose                0.512     250.8 GB/s      1.00x  ✓
Shared (bank conflict)         0.248     517.7 GB/s      2.06x  ✓
Shared (optimized)             0.215     597.2 GB/s      2.38x  ✓
```

> The shared memory version is ~2-3× faster than naive. The `+1` padding to avoid bank conflicts gives another ~15% improvement.

---

## Step 2: Visualize Bank Conflicts

Create `bank_conflicts.cu`:

```cuda
// bank_conflicts.cu — Demonstrate shared memory bank conflicts
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N 1024
#define BLOCK_SIZE 256
#define ITERATIONS 10000

// No bank conflicts: stride-1 access
__global__ void noBankConflicts(float *output) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        smem[tid] = (float)tid;              // Write: stride 1 → no conflict
        __syncthreads();
        sum += smem[tid];                     // Read: stride 1 → no conflict
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sum;
}

// 2-way bank conflicts: stride-2 access
__global__ void twowayConflicts(float *output) {
    __shared__ float smem[BLOCK_SIZE * 2];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        smem[tid * 2] = (float)tid;          // Write: stride 2 → 2-way conflict
        __syncthreads();
        sum += smem[tid * 2];                 // Read: stride 2 → 2-way conflict
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sum;
}

// 32-way bank conflicts: stride-32 access (worst case!)
__global__ void fullConflicts(float *output) {
    __shared__ float smem[BLOCK_SIZE * 32];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        smem[tid * 32] = (float)tid;         // Write: stride 32 → 32-way conflict!
        __syncthreads();
        sum += smem[tid * 32];                // Read: stride 32 → 32-way conflict!
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sum;
}

int main() {
    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("=== Shared Memory Bank Conflict Impact ===\n");
    printf("Block size: %d, Iterations per thread: %d\n\n", BLOCK_SIZE, ITERATIONS);

    // Stride 1: no conflicts
    CUDA_CHECK(cudaEventRecord(start));
    noBankConflicts<<<1, BLOCK_SIZE>>>(d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t1; CUDA_CHECK(cudaEventElapsedTime(&t1, start, stop));
    printf("Stride  1 (no conflicts):   %8.3f ms  (baseline)\n", t1);

    // Stride 2: 2-way conflicts
    CUDA_CHECK(cudaEventRecord(start));
    twowayConflicts<<<1, BLOCK_SIZE>>>(d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t2; CUDA_CHECK(cudaEventElapsedTime(&t2, start, stop));
    printf("Stride  2 (2-way conflict): %8.3f ms  (%.2fx slower)\n", t2, t2/t1);

    // Stride 32: 32-way conflicts
    CUDA_CHECK(cudaEventRecord(start));
    fullConflicts<<<1, BLOCK_SIZE>>>(d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t32; CUDA_CHECK(cudaEventElapsedTime(&t32, start, stop));
    printf("Stride 32 (32-way conflict):%8.3f ms  (%.2fx slower)\n", t32, t32/t1);

    printf("\n→ Bank conflicts serialize shared memory accesses.\n");
    printf("→ 32-way conflict means 32 threads access the same bank sequentially!\n");
    printf("→ Fix: pad shared memory arrays with +1 to shift access patterns.\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
```

### Compile and run

Compile and run to measure the performance impact of different shared memory access strides — from conflict-free to worst-case 32-way bank conflicts.

```bash
nvcc -O2 -o bank_conflicts bank_conflicts.cu
./bank_conflicts
```

### Expected Output

```
=== Shared Memory Bank Conflict Impact ===
Block size: 256, Iterations per thread: 10000

Stride  1 (no conflicts):      0.182 ms  (baseline)
Stride  2 (2-way conflict):    0.348 ms  (1.91x slower)
Stride 32 (32-way conflict):   2.850 ms  (15.66x slower)
```

---

## Experiments

### Experiment 1: Tile size variation
Change `TILE_DIM` to 16 and 64 in `transpose.cu`. Recompile and compare bandwidth. Smaller tiles have lower shared memory usage but more blocks; larger tiles may exceed shared memory capacity.

### Experiment 2: Rectangular matrices
Test transpose with non-square matrices: 4096×1024, 1024×4096, 8192×2048. Does the speedup ratio change?

### Experiment 3: Remove the `+1` padding
In `transposeShared`, change `tile[TILE_DIM][TILE_DIM + 1]` to `tile[TILE_DIM][TILE_DIM]`. How much slower does it get? This is the bank conflict penalty.

### Experiment 4: Smaller matrices
Test with 256×256, 512×512, 1024×1024 matrices. At what size does shared memory optimization no longer help?

### Experiment 5: In-place style
Modify the naive kernel to add the result instead of overwriting: `output[...] += input[...]`. How does this affect performance? (Read-modify-write is even more sensitive to coalescing.)

---

## What Just Happened?

1. **Shared memory is a programmer-managed cache.** Unlike L1/L2 caches, you explicitly load data into shared memory and control when threads synchronize. This gives you precise control over memory access patterns.

2. **The key trick: decouple read and write patterns.** Transpose requires reading rows and writing columns (or vice versa). With shared memory, you read a tile coalesced, store it in shared memory, then write it back coalesced by transposing within the tile.

3. **Bank conflicts degrade shared memory performance.** Shared memory has 32 banks. When multiple threads in a warp access the same bank, accesses are serialized. The `+1` padding trick shifts each row by one element, ensuring column accesses hit different banks.

4. **The `__syncthreads()` barrier is essential.** Without it, some threads might read shared memory before other threads have finished writing to it. This is the #1 source of shared memory bugs.

---

## Key Insight

> **Shared memory lets you "rearrange" your data access patterns within a tile.** Read coalesced from global memory into shared memory, rearrange via index manipulation, then write coalesced back. This is the fundamental technique behind most GPU memory optimizations.

---

## Checkpoint Quiz

**Q1:** Why does the naive transpose have poor performance even though *reads* are coalesced?
<details><summary>Answer</summary>
Because the *writes* are strided (column-major access to a row-major array). When threads in a warp write to addresses that are `height` elements apart, each write goes to a different cache line. The GPU must issue up to 32 separate memory transactions instead of 1 coalesced transaction. Writes are just as important as reads for coalescing.
</details>

**Q2:** What is the purpose of the `+1` in `__shared__ float tile[TILE_DIM][TILE_DIM + 1]`?
<details><summary>Answer</summary>
It adds one element of padding to each row of the shared memory array. Without it, column accesses (`tile[threadIdx.x][j]`) would cause all 32 threads in a warp to access the same shared memory bank (since columns are 32 elements apart = 32 banks apart = same bank). The +1 shifts each row by one bank, so column accesses hit 32 different banks — no conflicts.
</details>

**Q3:** You have a 4096×4096 matrix. The copy kernel achieves 600 GB/s and your optimized transpose achieves 550 GB/s. Is there room for improvement?
<details><summary>Answer</summary>
The transpose achieves 550/600 = 91.7% of copy bandwidth, which is excellent. A transpose inherently does more work than a copy (the access pattern reorganization), so achieving >90% of copy bandwidth means there's very little room for improvement. The remaining gap is likely due to residual bank conflicts or instruction overhead. In practice, this is considered near-optimal.
</details>

---

*Next Lab: [Lab 05 — Warp Divergence Impact](Lab05_Warp_Divergence_Impact.md)*
