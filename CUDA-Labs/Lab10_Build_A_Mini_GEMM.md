# Lab 10: Build a Mini GEMM 🔴

| Detail | Value |
|---|---|
| **Difficulty** | 🔴 Advanced |
| **Estimated Time** | 90–120 minutes |
| **Prerequisites** | Labs 01-09; understanding of shared memory, tiling, and memory coalescing |
| **GPU Required** | NVIDIA GPU with Compute Capability 3.0+ |

---

## Objective

By the end of this lab you will:
- Implement matrix multiplication (GEMM) in three progressively optimized versions
- Understand why shared memory tiling is essential for GEMM
- Add register blocking for further speedup
- Compare your implementations against cuBLAS
- Appreciate why cuBLAS is still faster (and what tricks it uses)

---

## Setup

```bash
mkdir -p ~/cuda-labs/lab10 && cd ~/cuda-labs/lab10
```

### Background: Why GEMM Matters

General Matrix Multiply (GEMM): `C = A × B` where A is M×K, B is K×N, C is M×N.

GEMM is the core of deep learning (every fully-connected layer, attention mechanism, and convolution can be expressed as GEMM). It's also the benchmark for GPU performance — a well-optimized GEMM reaches >90% of theoretical peak FLOPS.

**Theoretical FLOPS:** 2×M×N×K (multiply + add per output element, K times).

---

## Step 1: Naive Matrix Multiply

Create `gemm.cu`:

```cuda
// gemm.cu — Matrix Multiply: Naive → Tiled → Register-Blocked → cuBLAS
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, stat); exit(1); \
    } \
} while(0)

// ============================================================
// VERSION 1: NAIVE — one thread per output element
// Each thread computes one element of C by iterating over K
// ============================================================
__global__ void gemm_naive(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================
// VERSION 2: SHARED MEMORY TILING
// Load tiles of A and B into shared memory, compute partial sums
// Reduces global memory access from O(K) to O(K/TILE_SIZE) per element
// ============================================================
#define TILE_SIZE 32

__global__ void gemm_tiled(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles of K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaboratively load tile of A into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Collaboratively load tile of B into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// VERSION 3: REGISTER BLOCKING (2×2 per thread)
// Each thread computes a 2×2 sub-tile of C
// Doubles arithmetic intensity without increasing shared memory traffic
// ============================================================
#define BM 64   // Block tile height
#define BN 64   // Block tile width
#define BK 16   // K-dimension tile size
#define TM 4    // Thread tile height (each thread computes TM×TN elements)
#define TN 4    // Thread tile width

__global__ void gemm_regblock(const float *A, const float *B, float *C,
                              int M, int N, int K) {
    // Block-level shared memory
    __shared__ float As[BK][BM];  // Transposed for better access
    __shared__ float Bs[BK][BN];

    // Thread position within the block
    int threadRow = threadIdx.x / (BN / TN);  // Which TM-row this thread handles
    int threadCol = threadIdx.x % (BN / TN);  // Which TN-col this thread handles

    // Global position of this block's output tile
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // Registers for the TM×TN output sub-tile
    float regC[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    // Number of threads in this block
    int numThreads = (BM / TM) * (BN / TN);

    // Iterate over K-dimension tiles
    for (int bk = 0; bk < K; bk += BK) {
        // Collaboratively load As and Bs
        // Each thread loads multiple elements to fill the shared memory tiles
        for (int loadOffset = 0; loadOffset < BK * BM; loadOffset += numThreads) {
            int idx = loadOffset + threadIdx.x;
            if (idx < BK * BM) {
                int loadRow = idx / BM;  // K dimension
                int loadCol = idx % BM;  // M dimension
                int gRow = blockRow + loadCol;
                int gCol = bk + loadRow;
                As[loadRow][loadCol] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            }
        }

        for (int loadOffset = 0; loadOffset < BK * BN; loadOffset += numThreads) {
            int idx = loadOffset + threadIdx.x;
            if (idx < BK * BN) {
                int loadRow = idx / BN;  // K dimension
                int loadCol = idx % BN;  // N dimension
                int gRow = bk + loadRow;
                int gCol = blockCol + loadCol;
                Bs[loadRow][loadCol] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
            }
        }

        __syncthreads();

        // Compute — each thread handles a TM×TN sub-tile
        for (int k = 0; k < BK; k++) {
            // Load A column into registers
            for (int tm = 0; tm < TM; tm++) {
                regA[tm] = As[k][threadRow * TM + tm];
            }
            // Load B row into registers
            for (int tn = 0; tn < TN; tn++) {
                regB[tn] = Bs[k][threadCol * TN + tn];
            }
            // Outer product
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    regC[tm][tn] += regA[tm] * regB[tn];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int gRow = blockRow + threadRow * TM + tm;
            int gCol = blockCol + threadCol * TN + tn;
            if (gRow < M && gCol < N) {
                C[gRow * N + gCol] = regC[tm][tn];
            }
        }
    }
}

// ============================================================
// HOST REFERENCE
// ============================================================
void gemm_host(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify(const float *ref, const float *test, int M, int N, float tolerance) {
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(ref[i] - test[i]);
        float maxVal = fmaxf(fabsf(ref[i]), fabsf(test[i]));
        if (diff > tolerance * maxVal && diff > 1e-5f) {
            if (errors < 3) {
                printf("  MISMATCH [%d]: ref=%.4f got=%.4f diff=%.6f\n",
                       i, ref[i], test[i], diff);
            }
            errors++;
        }
    }
    return errors == 0;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char **argv) {
    int M = 2048, N = 2048, K = 2048;
    if (argc > 1) M = N = K = atoi(argv[1]);

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    double flops = 2.0 * M * N * K;

    printf("=== Mini GEMM: %d × %d × %d ===\n", M, N, K);
    printf("FLOPs: %.2f GFLOP\n\n", flops / 1e9);

    // Allocate and initialize
    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_ref = (float *)malloc(bytesC);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) * 0.01f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) * 0.01f;

    // Host reference (only for small matrices)
    bool doVerify = (M <= 2048);
    if (doVerify) {
        printf("Computing host reference... ");
        fflush(stdout);
        gemm_host(h_A, h_B, h_ref, M, N, K);
        printf("done.\n\n");
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int runs = 20;

    printf("%-30s  %10s  %12s  %10s  %s\n",
           "Version", "Time (ms)", "GFLOPS", "Efficiency", "Correct");
    printf("%-30s  %10s  %12s  %10s  %s\n",
           "-------", "---------", "------", "----------", "-------");

    // ==================== V1: NAIVE ====================
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        double gflops = flops / (ms / 1000.0) / 1e9;

        bool correct = true;
        if (doVerify) {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
            correct = verify(h_ref, h_C, M, N, 0.01f);
        }

        printf("%-30s  %10.3f  %12.1f  %10s  %s\n",
               "V1: Naive (16×16)", ms, gflops, "-", correct ? "✓" : "✗");
    }

    // ==================== V2: TILED ====================
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        double gflops = flops / (ms / 1000.0) / 1e9;

        bool correct = true;
        if (doVerify) {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
            correct = verify(h_ref, h_C, M, N, 0.01f);
        }

        printf("%-30s  %10.3f  %12.1f  %10s  %s\n",
               "V2: Tiled (32×32)", ms, gflops, "-", correct ? "✓" : "✗");
    }

    // ==================== V3: REGISTER BLOCKED ====================
    {
        int numThreads = (BM / TM) * (BN / TN);  // 16*16 = 256
        dim3 block(numThreads);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        gemm_regblock<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            gemm_regblock<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        double gflops = flops / (ms / 1000.0) / 1e9;

        bool correct = true;
        if (doVerify) {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
            correct = verify(h_ref, h_C, M, N, 0.01f);
        }

        printf("%-30s  %10.3f  %12.1f  %10s  %s\n",
               "V3: Register blocked (4×4)", ms, gflops, "-", correct ? "✓" : "✗");
    }

    // ==================== cuBLAS ====================
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        float alpha = 1.0f, beta = 0.0f;

        // cuBLAS uses column-major, so we compute C^T = B^T × A^T
        // which gives us C in row-major
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        double gflops = flops / (ms / 1000.0) / 1e9;

        // Get theoretical peak
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        double peakGflops = prop.clockRate * 1e-6 * prop.multiProcessorCount * 128 * 2;
        // This is approximate — actual peak depends on architecture

        bool correct = true;
        if (doVerify) {
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
            correct = verify(h_ref, h_C, M, N, 0.01f);
        }

        printf("%-30s  %10.3f  %12.1f  %10s  %s\n",
               "cuBLAS", ms, gflops, "reference", correct ? "✓" : "✗");

        cublasDestroy(handle);
    }

    // ==================== ANALYSIS ====================
    printf("\n=== Why cuBLAS Is Faster ===\n");
    printf("cuBLAS uses techniques that are extremely hard to replicate:\n");
    printf("  1. Architecture-specific register blocking (8×8 or larger)\n");
    printf("  2. Double-buffered shared memory (load next tile while computing)\n");
    printf("  3. Software pipelining of global loads\n");
    printf("  4. Careful tuning per GPU architecture (different for Ampere vs Hopper)\n");
    printf("  5. Warp-level matrix operations (WMMA/Tensor Cores on newer GPUs)\n");
    printf("  6. Auto-tuning over hundreds of kernel variants\n");
    printf("\nYour optimized kernel is likely 30-60%% of cuBLAS — that's respectable!\n");
    printf("In production, always use cuBLAS/cuBLASLt for GEMM.\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
```

### Compile and run

```bash
nvcc -O2 -o gemm gemm.cu -lcublas
./gemm           # Default: 2048×2048
./gemm 1024      # Smaller
./gemm 4096      # Larger (takes longer)
```

### Expected Output

```
=== Mini GEMM: 2048 × 2048 × 2048 ===
FLOPs: 17.18 GFLOP

Computing host reference... done.

Version                         Time (ms)        GFLOPS   Efficiency  Correct
-------                         ---------        ------   ----------  -------
V1: Naive (16×16)                  25.432         675.5            -  ✓
V2: Tiled (32×32)                   4.231        4062.1            -  ✓
V3: Register blocked (4×4)          2.156        7970.3            -  ✓
cuBLAS                              0.892       19260.5    reference  ✓

=== Why cuBLAS Is Faster ===
cuBLAS uses techniques that are extremely hard to replicate:
  1. Architecture-specific register blocking (8×8 or larger)
  2. Double-buffered shared memory (load next tile while computing)
  3. Software pipelining of global loads
  4. Careful tuning per GPU architecture
  5. Warp-level matrix operations (WMMA/Tensor Cores)
  6. Auto-tuning over hundreds of kernel variants
```

---

## Step 2: Understanding the Optimization Journey

### Why V1 Is Slow: Memory Bandwidth Bottleneck

```
V1: Each output C[i][j] reads row A[i][:] and column B[:][j] from global memory
    For 2048×2048: each thread reads 2×2048 = 4096 floats from global memory
    Total reads: 2048² × 4096 = 17.2 billion float reads!
    At ~500 GB/s bandwidth: 17.2B × 4B / 500 GB/s = 137 ms (minimum)
```

### Why V2 Is 6× Faster: Data Reuse

```
V2: Each TILE_SIZE×TILE_SIZE tile of A and B is loaded ONCE into shared memory
    and reused TILE_SIZE times by different threads.
    Data reuse factor: TILE_SIZE = 32×
    Total global reads: 2 × (2048/32) × 2048² = 537M float reads (32× reduction!)
```

### Why V3 Is 2× Faster than V2: Register Reuse

```
V3: Each thread computes TM×TN = 4×4 = 16 output elements
    The A and B values loaded into registers are reused TN and TM times
    Register reuse: 4× over V2's approach
    Shared memory traffic also reduced by the same factor
```

---

## Experiments

### Experiment 1: Matrix size sweep
Run all versions for M=N=K from 512 to 8192 (powers of 2). Plot GFLOPS vs matrix size. At what size does each version "peak"?

### Experiment 2: Non-square matrices
Test with M=4096, N=128, K=4096 (tall-skinny × skinny-wide). How does the performance change compared to square matrices?

### Experiment 3: Tile size tuning
In V2, change `TILE_SIZE` from 32 to 16 and 64. In V3, change `TM×TN` from 4×4 to 2×2 and 8×8. Find the optimal configuration for your GPU.

### Experiment 4: Double precision
Change all `float` to `double` and `cublasSgemm` to `cublasDgemm`. How does the FP64/FP32 performance ratio compare to the theoretical ratio for your GPU?

### Experiment 5: Tensor Core GEMM
If your GPU supports Tensor Cores (Volta+), use `cublasGemmEx` with half precision. Compare FP16 Tensor Core GFLOPS with FP32 GFLOPS.

---

## What Just Happened?

1. **Naive → Tiled (6×): Shared memory eliminates redundant global memory access.** Each element of A and B is loaded from global memory once per tile instead of once per output element. This TILE_SIZE× reduction in memory traffic directly translates to speedup.

2. **Tiled → Register-blocked (2×): Registers eliminate redundant shared memory access.** Each thread computes multiple output elements, reusing values from registers. This reduces shared memory traffic and increases arithmetic intensity (FLOPs per byte loaded).

3. **Register-blocked → cuBLAS (2-3×): Engineering excellence.** cuBLAS uses larger register tiles (8×8 or 16×16), double-buffered shared memory loading (overlap next tile load with current tile compute), architecture-specific tuning, and Tensor Cores where available.

4. **The arithmetic intensity hierarchy:**
   - Naive: ~2 FLOPs per global byte → memory bound
   - Tiled: ~64 FLOPs per global byte → better
   - Register-blocked: ~256 FLOPs per global byte → compute bound
   - cuBLAS: ~1024+ FLOPs per global byte → peak efficiency

---

## Key Insight

> **GEMM optimization is a lesson in data reuse.** Every level of the memory hierarchy (registers → shared memory → L2 → global memory) is progressively slower and higher-bandwidth. The art of GPU programming is maximizing reuse at each level. Load once into shared memory, reuse 32×. Load once into registers, reuse 4-8×. cuBLAS takes this to the extreme with 100×+ reuse factors.

---

## Checkpoint Quiz

**Q1:** For a 4096×4096 GEMM, how many FLOPs are performed? If your GPU achieves 10 TFLOPS, what's the minimum possible compute time?
<details><summary>Answer</summary>
FLOPs = 2 × 4096 × 4096 × 4096 = 137.4 GFLOP. At 10 TFLOPS: 137.4 / 10,000 = 0.01374 seconds ≈ 13.7 ms. If your GPU's peak is 20 TFLOPS (e.g., RTX 3080): 137.4 / 20,000 ≈ 6.9 ms. cuBLAS should get close to this.
</details>

**Q2:** Why does tiling with TILE_SIZE=32 give exactly 32× reduction in global memory traffic compared to naive?
<details><summary>Answer</summary>
In the naive version, each thread reads an entire row of A and column of B (K elements each) to compute one output. With tiling, each tile of A (TILE_SIZE × TILE_SIZE) is loaded once from global memory and shared among TILE_SIZE threads (one for each column of the output tile). Similarly for B. So each element is loaded once and used TILE_SIZE = 32 times, reducing global traffic by 32×.
</details>

**Q3:** Your register-blocked GEMM achieves 8 TFLOPS on a GPU with 20 TFLOPS peak. What's the likely bottleneck, and what would you try next?
<details><summary>Answer</summary>
At 40% of peak, you're likely still limited by shared memory bandwidth or register spilling. Next steps: (1) Increase the register tile size (TM×TN from 4×4 to 8×8) to increase arithmetic intensity, but check register usage doesn't cause spilling. (2) Add double-buffering: use two shared memory buffers and load the next tile while computing the current one. (3) Tune BK to balance shared memory capacity vs. K-loop overhead. (4) Ensure loads from global memory are fully coalesced using vectorized loads (float4).
</details>

---

## Congratulations! 🎉

You've completed all 10 CUDA labs! Here's what you've mastered:

| Lab | Key Concept |
|-----|-------------|
| 01 | GPU programming basics, CPU vs GPU crossover |
| 02 | Thread indexing in 1D, 2D, 3D |
| 03 | Memory transfer costs, pinned vs pageable |
| 04 | Shared memory and bank conflicts |
| 05 | Warp divergence and how to avoid it |
| 06 | Occupancy: what it is and why max ≠ best |
| 07 | Memory coalescing and SoA vs AoS |
| 08 | Stream-based overlap of copy and compute |
| 09 | Parallel reduction: 5 optimization levels |
| 10 | GEMM: tiling, register blocking, vs cuBLAS |

**Next steps:** Profile real workloads with NVIDIA Nsight Compute, explore Tensor Core programming, and study the [CUTLASS](https://github.com/NVIDIA/cutlass) library for production-grade GEMM implementations.
