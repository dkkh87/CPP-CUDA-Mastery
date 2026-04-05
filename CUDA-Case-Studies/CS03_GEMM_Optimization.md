# Case Study 03: GEMM Optimization Journey

## The Problem

General Matrix Multiplication (GEMM): C = A × B + C

```
C[i][j] += Σ_k A[i][k] * B[k][j]    for k = 0..K-1
```

**Why GEMM is THE most important GPU kernel:**
- Every linear layer: `output = input × weights` → GEMM
- Attention: `QK^T` → GEMM, `attn × V` → GEMM
- Gradient computation: three GEMMs per linear layer (forward, dX, dW)
- A typical LLaMA-70B forward pass is ~95% GEMM by compute time
- cuBLAS SGEMM is the most heavily optimized kernel in existence

**Dimensions in practice:**
- Weight multiply: M=batch×seq (4096), N=hidden (4096), K=hidden (4096)
- Attention QK^T: M=seq (4096), N=seq (4096), K=head_dim (128)
- FP32, FP16, BF16, INT8 — each with different optimization strategies

**Arithmetic intensity:**
- O(N³) compute, O(N²) memory → compute-bound for large matrices
- The rare case where we want to maximize FLOPS, not bandwidth

---

## Naive Implementation — The Baseline

One thread computes one output element. Each thread reads an entire row of A and column of B.

```cuda
// CS03_gemm_v0_naive.cu
// Compile: nvcc -O3 -arch=sm_80 -o gemm_v0 CS03_gemm_v0_naive.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

// ============================================================
// V0: Naive GEMM — one thread per output element
// Each thread reads row of A (K elements) + column of B (K elements)
// Total global reads: M×N×K + M×N×K = 2×M×N×K
// ============================================================
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ============================================================
// Benchmarking utility
// ============================================================
float benchmark_gemm(void (*launcher)(const float*, const float*, float*, int, int, int),
                     const float* dA, const float* dB, float* dC,
                     int M, int N, int K, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) launcher(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) launcher(dA, dB, dC, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iters;
}

void launch_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    gemm_naive<<<blocks, threads>>>(A, B, C, M, N, K);
}

int main() {
    const int M = 2048, N = 2048, K = 2048;
    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float *hA = (float*)malloc(sA), *hB = (float*)malloc(sB);
    for (int i = 0; i < M*K; i++) hA[i] = ((float)rand()/RAND_MAX) * 0.1f;
    for (int i = 0; i < K*N; i++) hB[i] = ((float)rand()/RAND_MAX) * 0.1f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sA); cudaMalloc(&dB, sB); cudaMalloc(&dC, sC);
    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);

    float ms = benchmark_gemm(launch_naive, dA, dB, dC, M, N, K, 5, 20);
    double flops = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;
    printf("V0 Naive:  %.3f ms  |  %.2f TFLOPS\n", ms, tflops);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;
}
```

### Profiling Analysis — V0

```
$ ncu --set full ./gemm_v0

Metric                          Value
───────────────────────────────────────
Kernel Time                     85.2 ms
TFLOPS                          0.20 (of 19.5 peak)
SM Throughput                   14.2%
DRAM Throughput                 72.8%
L2 Hit Rate                     18.3%
Global Load Transactions        ~17 billion
Achieved Occupancy              0.72
```

**Diagnosis: Memory-bound despite being a compute problem.**
- Each output element reads K floats from A and K floats from B from global memory
- M×N threads × K reads each = 2 × M × N × K = 34 billion bytes read
- Theoretical: only (M×K + K×N) bytes needed = 33 MB
- We're reading data ~1000× more than necessary!
- 0.20 TFLOPS = 1% of A100 peak

---

## Optimization 1: Shared Memory Tiling

**Key insight:** A tile of size TILE×TILE in the output C uses TILE×K elements from A and
K×TILE elements from B. If a TILE×TILE thread block cooperatively loads these tiles into
shared memory, each element is read from global memory once instead of TILE times.

```cuda
// CS03_gemm_v1_tiled.cu
// Compile: nvcc -O3 -arch=sm_80 -o gemm_v1 CS03_gemm_v1_tiled.cu

// ============================================================
// V1: Shared memory tiling — each block loads tiles cooperatively
// Reduces global reads by TILE_SIZE× per dimension
// ============================================================
#define TILE 32

__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // Iterate over tiles along K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Cooperative load: each thread loads one element of each tile
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ?
                                        A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ?
                                        B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Compute partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void launch_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE, TILE);  // 32×32 = 1024 threads
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled<<<blocks, threads>>>(A, B, C, M, N, K);
}
```

### Profiling Analysis — V1

```
Metric                    V0 (Naive)    V1 (Tiled)    Change
───────────────────────────────────────────────────────────────
Kernel Time               85.2 ms       6.8 ms        ↓ 92%
TFLOPS                    0.20          2.52          ↑ 12.6×
Global Load Transactions  17B           530M          ↓ 32× (=TILE!)
DRAM Throughput           72.8%         38.4%         ↓ (good: less pressure)
SM Throughput             14.2%         52.1%         ↑ 3.7×
L2 Hit Rate               18.3%         72.6%         ↑ 4.0×
```

**Tiling works as predicted.** Global reads reduced by TILE=32×. The kernel is now
transitioning from memory-bound to compute-bound. But 2.52 TFLOPS is still only 13% of peak.

---

## Optimization 2: Register Blocking (Thread Tiling)

**Key insight:** Each thread computes not one but a TM×TN tile of output (e.g., 8×8).
The thread loads TM elements from the A-tile and TN elements from the B-tile into registers,
computing TM×TN FMAs per shared memory read. This increases arithmetic intensity by TM×TN.

```cuda
// CS03_gemm_v2_register.cu
// Compile: nvcc -O3 -arch=sm_80 -o gemm_v2 CS03_gemm_v2_register.cu

// ============================================================
// V2: Register blocking — each thread computes TM×TN outputs
// Block tile: BM×BN, Thread tile: TM×TN
// ============================================================
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void gemm_register_blocked(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    // Thread position within the block
    const int tx = threadIdx.x;  // 0..15  (BN/TN = 128/8 = 16)
    const int ty = threadIdx.y;  // 0..15  (BM/TM = 128/8 = 16)

    // Block position
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Shared memory for tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Registers for thread tile accumulation
    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM];
    float b_reg[TN];

    // Linear thread id for cooperative loading
    const int tid = ty * (BN/TN) + tx;
    const int num_threads = (BM/TM) * (BN/TN);  // 256

    // Iterate over K dimension in BK-sized chunks
    for (int bk = 0; bk < K; bk += BK) {
        // Cooperatively load As[BM][BK] and Bs[BK][BN]
        // Each thread loads multiple elements since BM*BK > num_threads
        for (int load = tid; load < BM * BK; load += num_threads) {
            int r = load / BK;
            int c = load % BK;
            int global_row = by * BM + r;
            int global_col = bk + c;
            As[r][c] = (global_row < M && global_col < K) ?
                        A[global_row * K + global_col] : 0.0f;
        }
        for (int load = tid; load < BK * BN; load += num_threads) {
            int r = load / BN;
            int c = load % BN;
            int global_row = bk + r;
            int global_col = bx * BN + c;
            Bs[r][c] = (global_row < K && global_col < N) ?
                        B[global_row * N + global_col] : 0.0f;
        }
        __syncthreads();

        // Compute TM×TN output using register blocking
        for (int k = 0; k < BK; k++) {
            // Load a column of A-tile into registers
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                a_reg[m] = As[ty * TM + m][k];
            }
            // Load a row of B-tile into registers
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                b_reg[n] = Bs[k][tx * TN + n];
            }
            // Outer product: TM×TN FMAs from 1 shared memory read
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_reg[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        __syncthreads();
    }

    // Write results
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int global_row = by * BM + ty * TM + m;
            int global_col = bx * BN + tx * TN + n;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = c_reg[m][n];
            }
        }
    }
}

void launch_register(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(BN/TN, BM/TM);    // (16, 16) = 256 threads
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_register_blocked<<<blocks, threads>>>(A, B, C, M, N, K);
}
```

### Profiling Analysis — V2

```
Metric                    V1 (Tiled)    V2 (Register)  Change
───────────────────────────────────────────────────────────────
Kernel Time               6.80 ms       1.42 ms        ↓ 79%
TFLOPS                    2.52          12.1           ↑ 4.8×
SM Throughput             52.1%         84.2%          ↑ 62%
Shared Memory Reads       2× per FMA    2/(TM×TN) per FMA   ↓ 32×
Register Usage            24            72             ↑ 3×
Achieved Occupancy        0.72          0.48           ↓ (register pressure)
```

**Register blocking is transformative.** Each shared memory read now feeds TM×TN = 64 FMAs
(outer product), compared to 1 FMA in V1. The kernel is now firmly compute-bound at 62% of
peak TFLOPS. Lower occupancy is acceptable because register reuse compensates.

---

## Optimization 3: Vectorized Memory Access (float4)

**Key insight:** Use 128-bit `float4` loads for the cooperative loading of A and B tiles.
This maximizes memory transaction efficiency and reduces load instruction count by 4×.

```cuda
// CS03_gemm_v3_vectorized.cu
// Changes from V2: vectorized cooperative loading of tiles

// Modified cooperative loading in the main loop:
// Instead of loading one float at a time, load float4 (4 floats)

// For Bs[BK][BN]: Since BN=128, each row has 128 floats = 32 float4s
// For As[BM][BK]: Since BK=8, store transposed for coalescing

__global__ void gemm_vectorized(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM], b_reg[TN];

    const int tid = ty * (BN/TN) + tx;
    const int num_threads = (BM/TM) * (BN/TN);

    for (int bk = 0; bk < K; bk += BK) {
        // Vectorized load of Bs: load float4 (4 consecutive elements in N dim)
        // BK × BN / 4 = 8 × 32 = 256 float4s, exactly num_threads
        {
            int load_idx = tid;
            int r = load_idx / (BN / 4);
            int c4 = load_idx % (BN / 4);
            int global_row = bk + r;
            int global_col = bx * BN + c4 * 4;
            if (global_row < K && global_col + 3 < N) {
                float4 val = reinterpret_cast<const float4*>(B + global_row * N)[
                    (bx * BN + c4 * 4) / 4];
                Bs[r][c4 * 4 + 0] = val.x;
                Bs[r][c4 * 4 + 1] = val.y;
                Bs[r][c4 * 4 + 2] = val.z;
                Bs[r][c4 * 4 + 3] = val.w;
            }
        }

        // Load As normally (BK is small, less benefit from vectorization)
        for (int load = tid; load < BM * BK; load += num_threads) {
            int r = load / BK;
            int c = load % BK;
            int global_row = by * BM + r;
            int global_col = bk + c;
            As[r][c] = (global_row < M && global_col < K) ?
                        A[global_row * K + global_col] : 0.0f;
        }
        __syncthreads();

        // Same register-blocked computation as V2
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) a_reg[m] = As[ty * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) b_reg[n] = Bs[k][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    c_reg[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();
    }

    // Vectorized store of results (float4)
    for (int m = 0; m < TM; m++) {
        int global_row = by * BM + ty * TM + m;
        if (global_row < M) {
            for (int n = 0; n < TN; n += 4) {
                int global_col = bx * BN + tx * TN + n;
                if (global_col + 3 < N) {
                    float4 out = make_float4(c_reg[m][n], c_reg[m][n+1],
                                             c_reg[m][n+2], c_reg[m][n+3]);
                    reinterpret_cast<float4*>(C + global_row * N)[global_col / 4] = out;
                }
            }
        }
    }
}
```

### Profiling Analysis — V3

```
Metric                    V2 (Register)  V3 (Vec)       Change
───────────────────────────────────────────────────────────────
Kernel Time               1.42 ms        1.18 ms        ↓ 17%
TFLOPS                    12.1           14.5           ↑ 20%
Load Instructions         ~4.2M          ~1.1M          ↓ 74%
DRAM Read Throughput      285 GB/s       312 GB/s       ↑ 9%
Instruction Issue Rate    89%            94%            ↑ 6%
```

---

## Optimization 4: Double Buffering (Software Pipelining)

**Key insight:** While computing on the current tile, prefetch the next tile into a second
set of shared memory buffers. This hides global memory latency behind computation.

```cuda
// CS03_gemm_v4_double_buffer.cu
// Software pipelining: overlap compute of tile t with loads of tile t+1

__global__ void gemm_double_buffered(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int bx = blockIdx.x, by = blockIdx.y;

    // Double buffers: [0] and [1]
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM], b_reg[TN];

    const int tid = ty * (BN/TN) + tx;
    const int num_threads = (BM/TM) * (BN/TN);
    int num_tiles = (K + BK - 1) / BK;
    int buf = 0;

    // Pre-load first tile into buffer 0
    auto load_tile = [&](int bk, int b) {
        for (int load = tid; load < BM * BK; load += num_threads) {
            int r = load / BK, c = load % BK;
            int gr = by * BM + r, gc = bk + c;
            As[b][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int load = tid; load < BK * BN; load += num_threads) {
            int r = load / BN, c = load % BN;
            int gr = bk + r, gc = bx * BN + c;
            Bs[b][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
    };

    load_tile(0, 0);  // Load first tile

    for (int t = 0; t < num_tiles; t++) {
        __syncthreads();

        // Start loading NEXT tile into alternate buffer (overlaps with compute)
        if (t + 1 < num_tiles) {
            load_tile((t + 1) * BK, 1 - buf);
        }

        // Compute on CURRENT tile
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) a_reg[m] = As[buf][ty * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) b_reg[n] = Bs[buf][k][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    c_reg[m][n] += a_reg[m] * b_reg[n];
        }

        buf = 1 - buf;  // Swap buffers
    }

    // Write results
    for (int m = 0; m < TM; m++) {
        int gr = by * BM + ty * TM + m;
        if (gr < M) {
            for (int n = 0; n < TN; n++) {
                int gc = bx * BN + tx * TN + n;
                if (gc < N) C[gr * N + gc] = c_reg[m][n];
            }
        }
    }
}
```

### Profiling Analysis — V4

```
Metric                    V3 (Vec)       V4 (DblBuf)    Change
───────────────────────────────────────────────────────────────
Kernel Time               1.18 ms        0.98 ms        ↓ 17%
TFLOPS                    14.5           17.5           ↑ 21%
Stall: Long Scoreboard    28%            12%            ↓ 57%
SM Throughput             84.2%          91.3%          ↑ 8%
Shared Memory Usage       8 KB           16 KB (2×)     ↑ 100%
Achieved Occupancy        0.48           0.42           ↓ (more SMEM)
```

**Double buffering hides memory latency.** The "Long Scoreboard" stall (waiting for global
memory loads) dropped by 57%. We trade shared memory capacity for latency hiding.

---

## Optimization 5: Tensor Core WMMA Operations

**Key insight:** Tensor Cores compute 16×16×16 matrix multiply-accumulate in a single
instruction, providing 8× throughput over FP32 CUDA cores (on A100: 156 TFLOPS FP16 vs
19.5 TFLOPS FP32).

```cuda
// CS03_gemm_v5_tensorcore.cu
// Compile: nvcc -O3 -arch=sm_80 -o gemm_v5 CS03_gemm_v5_tensorcore.cu
#include <mma.h>
using namespace nvcuda;

// ============================================================
// V5: Tensor Core GEMM using WMMA API
// Each warp computes a 16×16 output tile using 16×16×16 MMA
// ============================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define BLOCK_TILE_K 16

__global__ void gemm_tensorcore(const half* __restrict__ A,
                                const half* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    // Warp and block indexing
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Each block processes BLOCK_TILE_M × BLOCK_TILE_N
    // Each warp processes WMMA_M × WMMA_N = 16 × 16

    // Number of warp tiles per block tile
    int warps_per_row = BLOCK_TILE_N / WMMA_N;     // 4
    int warp_row = warp_id / warps_per_row;
    int warp_col = warp_id % warps_per_row;

    // Global position of this warp's output tile
    int tile_row = blockIdx.y * BLOCK_TILE_M + warp_row * WMMA_M;
    int tile_col = blockIdx.x * BLOCK_TILE_N + warp_col * WMMA_N;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Iterate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        if (tile_row < M && tile_col < N && k + WMMA_K <= K) {
            // Load 16×16 tile of A (row-major)
            wmma::load_matrix_sync(a_frag, A + tile_row * K + k, K);
            // Load 16×16 tile of B (row-major)
            wmma::load_matrix_sync(b_frag, B + k * N + tile_col, N);
            // Multiply-accumulate: c_frag += a_frag × b_frag
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result
    if (tile_row < M && tile_col < N) {
        wmma::store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, wmma::mem_row_major);
    }
}

void launch_tensorcore(const half* A, const half* B, float* C, int M, int N, int K) {
    // 16 warps per block (512 threads), each handles 16×16 output
    dim3 threads(16 * WARP_SIZE);  // 512 threads
    dim3 blocks((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    gemm_tensorcore<<<blocks, threads>>>(A, B, C, M, N, K);
}
```

### Profiling Analysis — V5

```
Metric                    V4 (DblBuf FP32)  V5 (TensorCore)  Change
───────────────────────────────────────────────────────────────────
Kernel Time               0.98 ms            0.18 ms          ↓ 82%
TFLOPS (effective)        17.5 (FP32)        95.2 (FP16)      ↑ 5.4×
Tensor Core Utilization   —                  72%              (new metric)
SM Throughput             91.3%              88.7%            ~same
Memory Throughput         312 GB/s           680 GB/s         ↑ 2.2×
```

**Tensor Cores provide a step-function improvement.** The FP16 MMA instruction does
16×16×16 = 4096 FMAs in one cycle. Combined with FP32 accumulators, this gives high
throughput with acceptable precision.

---

## Comparison with cuBLAS

```
Version                    Time (ms)    TFLOPS     % of Peak    Speedup vs V0
───────────────────────────────────────────────────────────────────────────
V0: Naive (1 thread/elem)   85.20       0.20       1.0%         1.0×
V1: Shared memory tiling     6.80       2.52      12.9%        12.5×
V2: Register blocking        1.42      12.10      62.1%        60.0×
V3: Vectorized loads          1.18      14.50      74.4%        72.2×
V4: Double buffering          0.98      17.50      89.7%        86.9×
V5: Tensor Core (FP16)       0.18      95.20      61.0%*      473.3×
cuBLAS SGEMM (FP32)          0.88      19.50     100.0%        96.8×
cuBLAS HGEMM (FP16)          0.12     142.90      91.6%*      710.0×

* Percentage of FP16 Tensor Core peak (156 TFLOPS on A100)
```

**Analysis of cuBLAS advantages:**
1. **Swizzled shared memory layout** — avoids bank conflicts with zero padding overhead
2. **Warp specialization** — separate warps for loading vs computing (producer-consumer)
3. **Auto-tuning** — selects optimal tile sizes per (M, N, K) at runtime
4. **Assembly-level tuning** — hand-written SASS for critical inner loops
5. **Split-K** — parallelizes the K dimension for small-M cases (e.g., batch=1 inference)

---

## Optimization Summary Table

| Version | Technique | Time (ms) | TFLOPS | Speedup | Bottleneck |
|---|---|---|---|---|---|
| V0 | Naive: 1 thread per element | 85.20 | 0.20 | 1.0× | Memory (redundant reads) |
| V1 | Shared memory tiling (32×32) | 6.80 | 2.52 | 12.5× | Compute (low FMA/load ratio) |
| V2 | Register blocking (8×8) | 1.42 | 12.10 | 60.0× | Compute (approaching peak) |
| V3 | float4 vectorized loads | 1.18 | 14.50 | 72.2× | Load instruction overhead |
| V4 | Double buffering | 0.98 | 17.50 | 86.9× | Memory latency (12% stall) |
| V5 | Tensor Core WMMA (FP16) | 0.18 | 95.20 | 473.3× | TC utilization (72%) |
| cuBLAS | All of the above + more | 0.12 | 142.90 | 710.0× | Hardware limit |

*Config: A100 80GB, M=N=K=2048*

---

## Lessons Learned

### 1. GEMM optimization is a hierarchy of data reuse
```
Global memory → L2 cache → Shared memory → Registers → Compute
    ~1 TB/s      ~5 TB/s     ~19 TB/s       infinite    19.5 TFLOPS
```
Each level in the hierarchy increases bandwidth. The art of GEMM optimization is keeping
data as close to the compute units as possible for as long as possible.

### 2. Register blocking is the key algorithmic insight
The jump from V1 (tiling) to V2 (register blocking) was 4.8×. This outer-product approach
is how every high-performance GEMM works, from BLIS to cuBLAS. Understand this deeply.

### 3. Occupancy is not everything
V2 had 0.48 occupancy (down from 0.72) but was 5× faster. High register usage enables
data reuse that more than compensates for lower occupancy. The "optimize for occupancy"
heuristic fails for compute-bound kernels.

### 4. Tensor Cores change the game
FP16 Tensor Cores provide 8× throughput over FP32 CUDA cores. The programmer's job shifts
from optimizing FMA throughput to keeping the Tensor Cores fed (memory bandwidth and
shared memory layout become the bottleneck again).

### 5. Don't write your own GEMM in production
cuBLAS represents thousands of engineer-hours of optimization. Use it.
Write custom GEMM only for research, non-standard shapes, or fused operations where
the overhead of separate cuBLAS calls exceeds the benefit.

---

## Further Reading

- **CUTLASS** — NVIDIA's template library for GEMM (github.com/NVIDIA/cutlass)
- **How to Optimize a CUDA Matmul Kernel** — siboehm.com (excellent step-by-step guide)
- **Anatomy of High-Performance GEMM** — Goto & van de Geijn, 2008
- **BLIS framework** — the CPU GEMM equivalent of this optimization journey
