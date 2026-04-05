# Case Study 02: LayerNorm Kernel Optimization Journey

## The Problem

Layer Normalization normalizes activations across the feature dimension:

```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

where: μ = mean(x),  σ² = var(x),  γ and β are learnable parameters
```

**Why it matters for AI/ML:**
- Present in **every transformer block** — both pre-norm and post-norm architectures
- LLaMA-70B has 80 layers × 2 LayerNorms per layer = 160 calls per forward pass
- During training, 3× more calls (forward + 2 backward passes = 480 calls per step)
- Feature dimension: 4096 (7B) to 8192 (70B) to 12288 (175B)
- Often fused with residual connections, dropout, or bias addition

**Computational profile:**
- Two reductions per row: mean (sum) and variance (sum of squares)
- Then an elementwise rescale: very low arithmetic intensity
- Memory-bandwidth bound — similar to softmax but with 2 extra parameter vectors (γ, β)

---

## Naive Implementation — The Baseline

Two separate passes: compute mean, then compute variance with the mean.

```cuda
// CS02_layernorm_v0_naive.cu
// Compile: nvcc -O3 -arch=sm_80 -o layernorm_v0 CS02_layernorm_v0_naive.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// ============================================================
// V0: Naive LayerNorm — one thread per row, two passes
// ============================================================
__global__ void layernorm_naive(const float* __restrict__ input,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                float* __restrict__ output,
                                int num_rows, int row_length,
                                float epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const float* x = input + row * row_length;
    float* y = output + row * row_length;
    float inv_n = 1.0f / (float)row_length;

    // Pass 1: Compute mean
    float sum = 0.0f;
    for (int i = 0; i < row_length; i++) {
        sum += x[i];
    }
    float mean = sum * inv_n;

    // Pass 2: Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < row_length; i++) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(var_sum * inv_n + epsilon);

    // Pass 3: Normalize and apply affine transform
    for (int i = 0; i < row_length; i++) {
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ============================================================
// Timing and validation
// ============================================================
void launch_naive(const float* in, const float* gamma, const float* beta,
                  float* out, int rows, int cols, float eps) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    layernorm_naive<<<blocks, threads>>>(in, gamma, beta, out, rows, cols, eps);
}

int main() {
    const int rows = 2048;     // batch_size × seq_len
    const int cols = 4096;     // hidden_dim (LLaMA-7B)
    const float eps = 1e-5f;
    const size_t data_size = rows * cols * sizeof(float);
    const size_t param_size = cols * sizeof(float);

    // Allocate and initialize
    float* h_in = (float*)malloc(data_size);
    float* h_gamma = (float*)malloc(param_size);
    float* h_beta = (float*)malloc(param_size);
    for (int i = 0; i < rows * cols; i++) h_in[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < cols; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

    float *d_in, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMalloc(&d_gamma, param_size);
    cudaMalloc(&d_beta, param_size);
    cudaMemcpy(d_in, h_in, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_size, cudaMemcpyHostToDevice);

    // Warmup + benchmark
    for (int i = 0; i < 10; i++)
        launch_naive(d_in, d_gamma, d_beta, d_out, rows, cols, eps);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        launch_naive(d_in, d_gamma, d_beta, d_out, rows, cols, eps);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop);
    ms /= 100.0f;
    float bw = (2.0f * data_size + 2 * param_size) / (ms * 1e-3f) / 1e9f;
    printf("V0 Naive:  %.3f ms  |  Bandwidth: %.1f GB/s\n", ms, bw);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_in); free(h_gamma); free(h_beta);
    return 0;
}
```

### Profiling Analysis — V0

```
$ ncu --set full ./layernorm_v0

Metric                          Value
───────────────────────────────────────
Kernel Time                     4.12 ms
SM Throughput                   6.8%
DRAM Throughput                 9.7%
Achieved Occupancy              0.38
Warp Execution Efficiency       3.1%
Global Load Efficiency          12.5% (uncoalesced)
```

**Diagnosis:** Same catastrophic pattern as naive softmax.
- 1 thread per row → 31/32 warp lanes idle
- Sequential row access → no coalescing
- Three passes → 3× data reads plus gamma/beta reads
- GPU is doing almost nothing useful

---

## Optimization 1: Welford's Online Algorithm

**Key insight:** Welford's algorithm computes mean and variance in a single numerically-stable
pass. It avoids catastrophic cancellation that the naive `E[x²] - (E[x])²` formula suffers from.

```cuda
// CS02_layernorm_v1_welford.cu
// Compile: nvcc -O3 -arch=sm_80 -o layernorm_v1 CS02_layernorm_v1_welford.cu

// ============================================================
// V1: Welford's online algorithm — single pass for mean + var
// ============================================================
__global__ void layernorm_welford(const float* __restrict__ input,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ output,
                                  int num_rows, int row_length,
                                  float epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const float* x = input + row * row_length;
    float* y = output + row * row_length;

    // Welford's online algorithm:
    //   For each new sample x_i:
    //     count += 1
    //     delta = x_i - mean
    //     mean += delta / count
    //     delta2 = x_i - mean   (using updated mean)
    //     M2 += delta * delta2
    //   variance = M2 / count
    float mean = 0.0f;
    float M2 = 0.0f;

    for (int i = 0; i < row_length; i++) {
        float val = x[i];
        float count = (float)(i + 1);
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    float variance = M2 / (float)row_length;
    float inv_std = rsqrtf(variance + epsilon);

    // Second pass: normalize + affine
    for (int i = 0; i < row_length; i++) {
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
```

### Profiling Analysis — V1

```
Metric                    V0 (Naive)    V1 (Welford)   Change
───────────────────────────────────────────────────────────────
Kernel Time               4.12 ms       3.25 ms        ↓ 21%
Global Memory Reads       3× data       2× data        ↓ 33%
Numerical Stability       Poor          Excellent       ✓
Warp Execution Eff.       3.1%          3.1%           — (same)
```

**Better algorithm, same parallelism problem.** The division in Welford's inner loop adds
compute cost, but saving a full pass more than compensates.

---

## Optimization 2: Warp-Level Parallel Reduction

**Key insight:** Assign one warp per row. Each thread computes partial sums, then reduce
with shuffles. Welford's algorithm generalizes to parallel: merge partial (count, mean, M2).

```cuda
// CS02_layernorm_v2_warp.cu
// Compile: nvcc -O3 -arch=sm_80 -o layernorm_v2 CS02_layernorm_v2_warp.cu

// ============================================================
// V2: Warp-parallel with Welford merge via shuffles
// ============================================================

// Parallel Welford merge: combine two partial statistics
__device__ __forceinline__ void welford_merge(float& mean_a, float& m2_a, float& count_a,
                                              float mean_b, float m2_b, float count_b) {
    float count_ab = count_a + count_b;
    if (count_ab == 0.0f) return;
    float delta = mean_b - mean_a;
    float new_mean = mean_a + delta * (count_b / count_ab);
    float new_m2 = m2_a + m2_b + delta * delta * (count_a * count_b / count_ab);
    mean_a = new_mean;
    m2_a = new_m2;
    count_a = count_ab;
}

// Warp-level Welford reduction
__device__ __forceinline__ void warp_welford_reduce(float& mean, float& m2, float& count) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_mean  = __shfl_xor_sync(0xFFFFFFFF, mean, offset);
        float other_m2    = __shfl_xor_sync(0xFFFFFFFF, m2, offset);
        float other_count = __shfl_xor_sync(0xFFFFFFFF, count, offset);
        welford_merge(mean, m2, count, other_mean, other_m2, other_count);
    }
}

__global__ void layernorm_warp(const float* __restrict__ input,
                               const float* __restrict__ gamma,
                               const float* __restrict__ beta,
                               float* __restrict__ output,
                               int num_rows, int row_length,
                               float epsilon) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_rows) return;

    const float* x = input + warp_id * row_length;
    float* y = output + warp_id * row_length;

    // Each lane computes Welford stats over its elements
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    float local_count = 0.0f;

    for (int i = lane_id; i < row_length; i += 32) {
        float val = x[i];
        local_count += 1.0f;
        float delta = val - local_mean;
        local_mean += delta / local_count;
        float delta2 = val - local_mean;
        local_m2 += delta * delta2;
    }

    // Merge all lanes' statistics via shuffle
    warp_welford_reduce(local_mean, local_m2, local_count);

    float inv_std = rsqrtf(local_m2 / local_count + epsilon);
    float mean = local_mean;

    // Normalize + affine in one coalesced pass
    for (int i = lane_id; i < row_length; i += 32) {
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

void launch_warp(const float* in, const float* gamma, const float* beta,
                 float* out, int rows, int cols, float eps) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (rows + warps_per_block - 1) / warps_per_block;
    layernorm_warp<<<blocks, threads>>>(in, gamma, beta, out, rows, cols, eps);
}
```

### Profiling Analysis — V2

```
Metric                    V1 (Welford)  V2 (Warp)     Change
───────────────────────────────────────────────────────────────
Kernel Time               3.25 ms       0.29 ms       ↓ 91%!
Warp Execution Eff.       3.1%          100%           ↑ 32×
DRAM Throughput           9.7%          71%            ↑ 7.3×
SM Throughput             6.8%          58%            ↑ 8.5×
Achieved Occupancy        0.38          0.84           ↑ 2.2×
```

**Same massive jump as softmax.** Warp-level parallelism is the single biggest win.

---

## Optimization 3: Fused LayerNorm + Residual + Dropout

**Key insight:** In transformers, LayerNorm is always preceded by a residual connection
and (during training) dropout. Three separate kernels:

```
dropout_out = dropout(sublayer_output, mask, p)
residual_out = input + dropout_out
layernorm_out = LayerNorm(residual_out)
```

Each reads and writes the full tensor to global memory. Fusing them into one kernel
reduces memory traffic by 3×.

```cuda
// CS02_layernorm_v3_fused.cu
// Compile: nvcc -O3 -arch=sm_80 -o layernorm_v3 CS02_layernorm_v3_fused.cu

// ============================================================
// V3: Fused LayerNorm + Residual Add + Dropout
// 3 kernels → 1 kernel, 3× less memory traffic
// ============================================================
__global__ void fused_residual_dropout_layernorm(
        const float* __restrict__ sublayer_out,  // output of sublayer (MHA or FFN)
        const float* __restrict__ residual,       // skip connection input
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        const uint8_t* __restrict__ dropout_mask,
        float* __restrict__ output,
        float* __restrict__ residual_out,  // Save for backward pass
        int num_rows, int row_length,
        float dropout_scale,   // 1.0 / (1.0 - dropout_prob)
        float epsilon) {

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_rows) return;

    const float* sub = sublayer_out + warp_id * row_length;
    const float* res = residual     + warp_id * row_length;
    const uint8_t* mask = dropout_mask + warp_id * row_length;
    float* out = output      + warp_id * row_length;
    float* res_save = residual_out + warp_id * row_length;

    // Pass 1: Fused dropout + residual + Welford stats
    float local_mean = 0.0f, local_m2 = 0.0f, local_count = 0.0f;

    for (int i = lane_id; i < row_length; i += 32) {
        // Fused: dropout → residual add → accumulate stats
        float dropped = sub[i] * (float)mask[i] * dropout_scale;
        float val = res[i] + dropped;
        res_save[i] = val;   // Save for backward

        // Welford accumulate
        local_count += 1.0f;
        float delta = val - local_mean;
        local_mean += delta / local_count;
        float delta2 = val - local_mean;
        local_m2 += delta * delta2;
    }

    // Warp-level Welford merge
    warp_welford_reduce(local_mean, local_m2, local_count);

    float inv_std = rsqrtf(local_m2 / local_count + epsilon);
    float mean = local_mean;

    // Pass 2: Normalize + affine (reads res_save from L2 cache — hopefully!)
    for (int i = lane_id; i < row_length; i += 32) {
        out[i] = (res_save[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
```

### Profiling Analysis — V3

```
Metric                    V2 (Warp)    V3 (Fused)     Change
───────────────────────────────────────────────────────────────
Total Time (3 kernels)    0.29+0.08+   0.22 ms        ↓ ~50%
                          0.10 = 0.47
DRAM Reads                3× tensor    1× tensor      ↓ 67%
DRAM Writes               3× tensor    2× tensor      ↓ 33%
Kernel Launches           3            1              ↓ 67%
L2 Hit Rate               28%          72%            ↑ 2.6×
```

**Fusion is crucial for memory-bound ops.** The residual_out write in pass 1 hits L2 cache
and is read back in pass 2 without ever going to DRAM (for row_length ≤ 8192 on A100).

---

## Optimization 4: Vectorized Loads + Register Blocking

**Key insight:** Combine float4 vectorized loads with register-level caching of gamma/beta
parameters to reduce instruction overhead and improve memory throughput.

```cuda
// CS02_layernorm_v4_vectorized.cu
// Compile: nvcc -O3 -arch=sm_80 -o layernorm_v4 CS02_layernorm_v4_vectorized.cu

// ============================================================
// V4: Vectorized float4 loads + register blocking for gamma/beta
// ============================================================
__global__ void layernorm_vectorized(const float* __restrict__ input,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ output,
                                     int num_rows, int row_length,
                                     float epsilon) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_rows) return;

    int vec_len = row_length / 4;
    const float4* x4 = reinterpret_cast<const float4*>(input + warp_id * row_length);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    float4* y4 = reinterpret_cast<float4*>(output + warp_id * row_length);

    // Pass 1: Vectorized mean + variance via parallel sum
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = lane_id; i < vec_len; i += 32) {
        float4 v = x4[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum    += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        local_sq_sum += __shfl_xor_sync(0xFFFFFFFF, local_sq_sum, offset);
    }

    float mean = local_sum / (float)row_length;
    float variance = local_sq_sum / (float)row_length - mean * mean;
    float inv_std = rsqrtf(variance + epsilon);

    // Pass 2: Vectorized normalize + affine
    for (int i = lane_id; i < vec_len; i += 32) {
        float4 v = x4[i];
        float4 g = g4[i];
        float4 b = b4[i];
        float4 out;
        out.x = (v.x - mean) * inv_std * g.x + b.x;
        out.y = (v.y - mean) * inv_std * g.y + b.y;
        out.z = (v.z - mean) * inv_std * g.z + b.z;
        out.w = (v.w - mean) * inv_std * g.w + b.w;
        y4[i] = out;
    }
}
```

### Profiling Analysis — V4

```
Metric                    V3 (Fused)    V4 (Vec)      Change
───────────────────────────────────────────────────────────────
Kernel Time (LN only)     0.22 ms       0.15 ms       ↓ 32%
Load Instructions         524,288       131,072       ↓ 75%
DRAM Throughput           71%           88%           ↑ 24%
L1 Hit Rate               45%           68%           ↑ 51%
Instruction Count         ~1.2M         ~620K         ↓ 48%
```

---

## Comparison with Apex / cuDNN LayerNorm

```
Version                          Time (ms)    Speedup vs V0    Bandwidth (GB/s)
───────────────────────────────────────────────────────────────────────────────
V0: Naive (1 thread/row)          4.120        1.0×              16
V1: Welford's algorithm            3.250        1.3×              20
V2: Warp shuffle reduction         0.290       14.2×             227
V3: Fused residual+dropout+LN      0.220*      18.7×             299
V4: Vectorized float4              0.150       27.5×             439
NVIDIA Apex fused_layer_norm       0.120       34.3×             548
cuDNN layernorm                    0.110       37.5×             598

* V3 measures full fused operation (residual + dropout + LN)
```

**What Apex / cuDNN do differently:**
1. **Block-level reduction** for hidden_dim ≥ 1024 — more parallelism than a single warp
2. **Cooperative groups** for flexible reduction sizes
3. **Mixed-precision support** — BF16 input, FP32 accumulation, BF16 output
4. **RMSNorm variant** — skips mean subtraction (used in LLaMA), saving one reduction
5. **Persistent thread blocks** — reuse blocks across rows without re-launch

---

## Optimization Summary Table

| Version | Technique | Time (ms) | Speedup | Bottleneck |
|---|---|---|---|---|
| V0 | Naive: 1 thread/row, 2 passes | 4.120 | 1.0× | Warp utilization (3.1%) |
| V1 | Welford's online algorithm | 3.250 | 1.3× | Still 1 thread/row |
| V2 | Warp shuffle + Welford merge | 0.290 | 14.2× | Memory bandwidth (71%) |
| V3 | Fused residual + dropout + LN | 0.220 | 18.7× | Memory bandwidth (71%) |
| V4 | Vectorized float4 loads | 0.150 | 27.5× | Memory bandwidth (88%) |
| Apex | Block reduction, mixed prec | 0.120 | 34.3× | Hardware BW limit |

*Config: A100 80GB, batch×seq=2048, hidden_dim=4096, FP32*

---

## Lessons Learned

### 1. Welford's algorithm is essential for numerical stability
The naive formula `var = E[x²] - E[x]²` suffers from catastrophic cancellation when the
mean is large relative to the variance. Welford's avoids this, and it parallelizes cleanly.

### 2. Fusion is the #1 optimization for training workloads
Fusing residual + dropout + LayerNorm eliminates 4× memory round-trips to DRAM.
In training, where these three ops always appear together, fusion delivers more speedup
than any single-kernel optimization.

### 3. Parameter broadcasting is cache-friendly
gamma and beta are small vectors (4096 floats = 16 KB) shared across all rows. After the
first warp reads them, they live in L1/L2 cache. This makes the affine transform essentially
free compared to the reduction.

### 4. RMSNorm saves one reduction
LLaMA uses RMSNorm (no mean subtraction): `RMSNorm(x) = x / √(mean(x²) + ε) * γ`.
One reduction instead of two → ~1.5× faster than LayerNorm for the same hidden_dim.

### 5. Block vs warp: depends on hidden_dim
- hidden_dim ≤ 1024 → warp-per-row is optimal (32 threads saturate memory BW)
- hidden_dim ≥ 4096 → block-per-row wins (256 threads reduce instruction pressure)
- Library kernels dispatch dynamically based on the dimension

### 6. Backward pass is harder to fuse
LayerNorm backward requires ∂L/∂γ and ∂L/∂β, which are reductions across the batch
dimension — orthogonal to the forward reduction. This typically needs a separate kernel.

---

## Connection to the AI/ML Stack

```
Transformer block (pre-norm):
    x_norm = LayerNorm(x)                   ← THIS CASE STUDY
    attn_out = MultiHeadAttention(x_norm)   ← CS04 (Attention)
    x = x + Dropout(attn_out)              ← Fused with LayerNorm above
    x_norm = LayerNorm(x)                   ← THIS CASE STUDY
    ffn_out = FFN(x_norm)                   ← CS03 (GEMM) + CS05 (Fusion)
    x = x + Dropout(ffn_out)               ← Fused with LayerNorm above
```

LayerNorm is the "glue" that enables deep stacking of transformer layers. Optimizing
it — especially through fusion — has a multiplicative impact on end-to-end performance.

---

## Further Reading

- **Welford's online algorithm** — B. P. Welford, 1962, Technometrics
- **NVIDIA Apex fused_layer_norm** — github.com/NVIDIA/apex
- **Megatron-LM layer norm kernel** — github.com/NVIDIA/Megatron-LM
- **Root Mean Square Layer Normalization** — Zhang & Sennrich, 2019 (RMSNorm paper)
