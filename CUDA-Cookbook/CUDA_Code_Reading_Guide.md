# CUDA Code Reading Guide — Understanding Production ML Kernels

> **Reading expert code is the fastest way to master CUDA.**
> This guide teaches you how to read real-world CUDA code from production
> frameworks. We describe patterns, explain techniques, and provide simplified
> illustrative examples. No copyrighted code is reproduced.

---

## 1. How to Read CUDA Code

### 1.1 The Launch-First Method

Start from the kernel launch — it's your roadmap:

```
my_kernel<<<grid, block, shmem_bytes, stream>>>(args...);
```

| Parameter | What it tells you |
|---|---|
| `gridDim` | How the problem is decomposed across the GPU |
| `blockDim` | How many threads cooperate within a block |
| `sharedMem` | Whether the kernel uses dynamic shared memory |
| `stream` | Whether this is part of an async pipeline |

**Step-by-step:** Read launch config → map arguments to memory types
(pointers = global, values = registers) → find thread-to-data mapping →
locate synchronization points → identify tiling/warp-level patterns.

### 1.2 Common Thread-to-Data Mappings

```cpp
// 1D mapping
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D mapping (matrices)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Grid-stride loop (handles arbitrary sizes)
for (int i = idx; i < n; i += blockDim.x * gridDim.x) { ... }
```

### 1.3 What to Spot

| Pattern | Clue |
|---|---|
| **Tiling** | Loops loading chunks into shared memory |
| **Coalescing** | Adjacent threads accessing adjacent addresses |
| **Warp primitives** | `__shfl_*`, `__ballot_sync`, `__reduce_*` |
| **Fusion** | Multiple logical operations in one kernel |
| **Vectorized loads** | `float4`, `int4`, `reinterpret_cast<float4*>` |

### 1.4 Naming Conventions

```
Function names:
  *_kernel        — The actual __global__ function
  *_launcher      — Host function that calculates grid/block and launches
  *_impl          — Implementation detail (often a template)
  *_forward       — Forward pass (training)
  *_backward      — Backward pass (gradient computation)

Variable names:
  tid / threadIdx — Thread index within block
  bid / blockIdx  — Block index within grid
  wid             — Warp index (tid / 32)
  lane            — Lane within warp (tid % 32)
  smem / s_*      — Shared memory pointers/arrays
  g_*             — Global memory pointers

Template parameters:
  BLOCK_SIZE      — Threads per block
  TILE_M/N/K      — Tile dimensions for each matrix axis
  UNROLL          — Unroll factor for inner loops
  VEC_SIZE        — Vector load width (1, 2, 4)
```

### 1.5 Reading Strategy Checklist

Use this every time you open a new CUDA file:

```
□ Find the kernel launch site (<<<...>>>)
□ Note grid dimensions — how is work decomposed?
□ Note block dimensions — how many threads cooperate?
□ Check shared memory allocation (static or dynamic?)
□ Read the kernel signature — what data goes in/out?
□ Find the thread-to-data mapping (first few lines of kernel)
□ Identify synchronization points (__syncthreads, etc.)
□ Look for tiling loops (load → sync → compute → sync → store)
□ Check for warp-level optimizations (__shfl, __ballot)
□ Note any template parameters — what is configurable?
```

---

## 2. PyTorch's CUDA Backend — Architecture Overview

### 2.1 Key Directories

```
pytorch/
├── aten/src/ATen/native/cuda/   ← CUDA kernels live here
├── c10/cuda/                    ← Allocator, streams, guards
├── torch/csrc/cuda/             ← CUDA-specific runtime
└── tools/autograd/              ← Code generation for derivatives
```

### 2.2 How torch.add() Becomes a CUDA Kernel

```
torch.add(a, b)                         ← Python
    │
    ▼
Dispatcher (routes by device + dtype)    ← C++
    │
    ├── CUDA key → at::native::add_cuda(...)
    │                    │
    │                    ▼
    │              gpu_kernel_with_scalars(...)
    │                    │
    │                    ▼
    │              vectorized_elementwise_kernel<<<...>>>
    └── AutogradCUDA key → Records op for backward pass
```

### 2.3 Finding the Kernel for Any Operation

1. Search `native_functions.yaml` for the operation name
2. Find the `CUDA:` dispatch entry — it names the C++ function
3. Search `aten/src/ATen/native/cuda/` for that function name
4. The function typically ends with a kernel launch or calls a template helper

### 2.4 The Elementwise Kernel Pattern

```cpp
// Simplified illustration of PyTorch's elementwise pattern
template <typename func_t>
__global__ void elementwise_kernel(int N, func_t f) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        f(i);  // Operation is a functor/lambda
    }
}

template <typename scalar_t>
void add_cuda_impl(Tensor& result, const Tensor& a, const Tensor& b) {
    int N = a.numel();
    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);
    elementwise_kernel<<<blocks, threads>>>(N,
        [=] __device__ (int i) {
            result.data_ptr<scalar_t>()[i] =
                a.data_ptr<scalar_t>()[i] + b.data_ptr<scalar_t>()[i];
        });
}
```

**Key design decisions:** grid-stride loop (one kernel for any size),
functor pattern (body decoupled from loop), template on `scalar_t`
(same kernel for float/double/half), `AT_DISPATCH_FLOATING_TYPES` macro
for runtime dtype dispatch.

### 2.5 Key Files to Study

| File area | What you'll learn |
|---|---|
| `native/cuda/Reduce.cuh` | Reduction patterns, warp shuffles |
| `native/cuda/Normalization.cu` | BatchNorm, shared memory reductions |
| `native/cuda/SoftMax.cu` | Softmax, online algorithms |
| `native/cuda/Loops.cuh` | Core elementwise loop infrastructure |
| `c10/cuda/CUDACachingAllocator.cpp` | Memory pooling strategy |

### 2.6 Tracing torch.sum() End-to-End

```
1. Python: torch.sum(x) → dispatch to sum_cuda
2. Implementation: reduction template that:
   a. Grid-stride loop loads elements into thread-local partial sums
   b. Warp-level reduction using __shfl_down_sync
   c. Block-level reduction via shared memory
   d. Final block results written to global; second pass if needed
3. Result: single-element tensor on GPU
```

---

## 3. CUTLASS — NVIDIA's Template Library for GEMM

### 3.1 Why CUTLASS Exists

| Approach | Performance | Flexibility | Complexity |
|---|---|---|---|
| cuBLAS | ~100% peak | Low | Low |
| CUTLASS | 90-100% peak | High (templates) | Medium |
| Hand-written | Variable | Total | Very high |

Use CUTLASS for custom data types (FP8, INT4), fused epilogues, non-standard
layouts, or learning how near-peak GEMM works.

### 3.2 Template Hierarchy

```
Device level  → Partitions across full GPU (grid of threadblocks)
    ▼
Threadblock   → Each block computes a tile of C (e.g., 128×128)
(CTA level)     Loads tiles of A, B into shared memory; iterates over K
    ▼
Warp level    → Each warp computes a sub-tile (e.g., 64×64)
                Uses tensor cores (wmma / mma instructions)
    ▼
Instruction   → Maps to hardware: mma.sync, ldmatrix (e.g., 16×8×16)
```

### 3.3 The K-Dimension Mainloop

```cpp
// Simplified CUTLASS-style tiled GEMM (illustrative)
template <int BM, int BN, int BK>
__global__ void gemm_tiled(
    const float* A, const float* B, float* C, int M, int N, int K
) {
    __shared__ float sA[BM][BK], sB[BK][BN];
    int brow = blockIdx.y * BM, bcol = blockIdx.x * BN;
    int row = threadIdx.y, col = threadIdx.x;
    float acc = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        // Collaborative load into shared memory
        sA[row][col] = ((brow+row)<M && (kt+col)<K)
            ? A[(brow+row)*K + kt+col] : 0.0f;
        sB[row][col] = ((kt+row)<K && (bcol+col)<N)
            ? B[(kt+row)*N + bcol+col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BK; ++k)
            acc += sA[row][k] * sB[k][col];
        __syncthreads();
    }

    if ((brow+row)<M && (bcol+col)<N)
        C[(brow+row)*N + bcol+col] = acc;
}
```

### 3.4 The Epilogue Concept

The epilogue handles post-GEMM operations while data is still in registers:

```cpp
// Without fusion: 3 kernels, 3× memory traffic
gemm<<<...>>>(A, B, C);
bias_add<<<...>>>(C, bias);
relu<<<...>>>(C);

// With epilogue fusion: 1 kernel, 1× memory traffic
// After accumulation, still in registers:
float result = alpha * accumulator + bias[col];
result = fmaxf(result, 0.0f);  // Fused ReLU
output[idx] = result;           // Single write
```

Epilogues are template parameters in CUTLASS — plug in any post-processing
without a separate kernel.

### 3.5 Key Design Patterns

- **Tile iterators** — abstract memory traversal, handle boundary conditions
- **Fragments** — register-resident arrays; each thread's piece of a tile
- **Predicated loads** — load zero for out-of-bounds, avoiding warp divergence
- **CuTe (CUTLASS 3.x)** — layout algebra with `Shape × Stride`, replacing
  manual index math. Look for `cute::make_layout`, `cute::tiled_divide`.

---

## 4. Flash Attention — Reading the Algorithm

### 4.1 The Core Idea

Standard attention materializes the full N×N matrix (O(N²) memory).
Flash Attention tiles the computation with online softmax — O(N) memory:

```
For each block of Q (outer loop):
  For each block of K, V (inner loop):
    1. Load Q_block, K_block, V_block into SRAM
    2. S_block = Q_block × K_block^T        (small tile)
    3. Update running softmax statistics
    4. Accumulate O_block += softmax(S_block) × V_block
  Write final O_block to HBM
```

### 4.2 Online Softmax Pattern

The crucial insight — computing softmax without seeing all values at once:

```cpp
// Online softmax accumulation — simplified illustrative pattern
float row_max = -INFINITY, row_sum = 0.0f;
float output[D] = {0};

for (int blk = 0; blk < num_kv_blocks; blk++) {
    float scores[BLOCK_SIZE];
    compute_qk_dot(q_row, k_block[blk], scores);

    float block_max = max_of(scores, BLOCK_SIZE);
    float new_max = fmaxf(row_max, block_max);

    // Rescale previous accumulated values
    float correction = expf(row_max - new_max);
    row_sum *= correction;
    for (int d = 0; d < D; d++) output[d] *= correction;

    // Accumulate new block
    for (int j = 0; j < BLOCK_SIZE; j++) {
        float p = expf(scores[j] - new_max);
        row_sum += p;
        for (int d = 0; d < D; d++)
            output[d] += p * v_block[blk][j][d];
    }
    row_max = new_max;
}

for (int d = 0; d < D; d++) output[d] /= row_sum;
```

### 4.3 Memory Hierarchy Strategy

```
HBM (global): Q, K, V, O tensors — large, slow
    ↕ (minimize these transfers)
SRAM (shared): Q/K/V block tiles — ~164 KB per SM
    ↕
Registers: Accumulators, softmax stats — fastest

Typical tile budget (A100):
  Q tile: 128×64×2B = 16KB    V tile: 128×64×2B = 16KB
  K tile: 128×64×2B = 16KB    S tile: 128×128×4B = 64KB
  Total: ~112KB (fits, room for double buffering)
```

### 4.4 Forward vs Backward Pass

```
Forward pass saves:
  - Output O
  - Row-wise log-sum-exp values (for backward recomputation)
  - RNG state (for dropout reproducibility)

Backward pass:
  - Recomputes attention weights on-the-fly (same tiling as forward)
  - Computes dQ, dK, dV using the recomputed weights
  - Two passes: one for dV and dK, one for dQ (needs transpose)
```

**Why recompute instead of save?** Storing the N×N attention matrix would
negate the memory savings. Recomputation is cheaper than the memory cost,
especially on modern GPUs where compute is cheaper than memory bandwidth.

### 4.5 What to Study

| Component | What to look for |
|---|---|
| Forward kernel | Double-loop tiling, online softmax |
| Backward kernel | Recomputation logic, dQ/dK/dV accumulation |
| Launch code | Block size selection based on head dimension |
| Masking | Causal mask applied inside the inner loop |

---

## 5. Triton — Python-to-GPU Compiler

### 5.1 How Triton Differs from CUDA

| Aspect | CUDA | Triton |
|---|---|---|
| Language | C++ | Python with decorators |
| Model | Thread-level | Block-level |
| Memory mgmt | Manual | Automatic |
| Sync | Manual `__syncthreads()` | Implicit |
| Tensor cores | Manual via wmma/mma | Automatic via `tl.dot` |

### 5.2 Block-Based Programming Model

```python
# CUDA thinks per-thread: "Thread i processes element i"
# Triton thinks per-block: "This program processes a BLOCK of elements"
offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
block = tl.load(input + offsets, mask=offsets < N)
tl.store(output + offsets, block * 2, mask=offsets < N)
```

### 5.3 When Triton vs CUDA

**Use Triton:** Prototyping, fused operations, auto-tuning needed,
~90% of handwritten CUDA performance is acceptable, multi-vendor portability.

**Use CUDA:** Precise shared memory control, custom warp-level patterns,
specific PTX instructions, irregular access patterns, last 5-10% perf.

### 5.4 Softmax: Triton vs CUDA Side by Side

**Triton (~15 lines):**

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(input_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    num = tl.exp(x - x_max)
    result = num / tl.sum(num, axis=0)
    tl.store(output_ptr + row * n_cols + offs, result, mask=mask)
```

**CUDA (~50 lines):**

```cpp
template <int BLOCK_SIZE>
__global__ void softmax_kernel(const float* in, float* out, int n_cols) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* row_in = in + row * n_cols;

    // Pass 1: find max (parallel reduction with warp shuffles)
    float tmax = -INFINITY;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE)
        tmax = fmaxf(tmax, row_in[i]);
    for (int off = 16; off > 0; off >>= 1)
        tmax = fmaxf(tmax, __shfl_down_sync(0xFFFFFFFF, tmax, off));
    __shared__ float smem[32];
    if (tid % 32 == 0) smem[tid/32] = tmax;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < BLOCK_SIZE/32) ? smem[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, off));
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: sum of exp (same reduction pattern)
    float tsum = 0.0f;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE)
        tsum += expf(row_in[i] - row_max);
    // ... (warp + block reduction for sum) ...
    __syncthreads();
    float row_sum = smem[0];

    // Pass 3: normalize
    for (int i = tid; i < n_cols; i += BLOCK_SIZE)
        out[row * n_cols + i] = expf(row_in[i] - row_max) / row_sum;
}
```

**Key difference:** Triton hides reduction, shared memory, and sync entirely.
CUDA gives full control over warp shuffles and memory layout.

---

## 6. Common Patterns You'll See in Expert Code

### 6.1 Double Buffering / Software Pipelining

Load next tile while computing on current tile — hides memory latency:

```cpp
__shared__ float buf[2][TILE_SIZE];
int cur = 0;
load_tile_async(buf[cur], global_ptr, 0);
__syncthreads();

for (int tile = 1; tile < num_tiles; tile++) {
    int nxt = 1 - cur;
    load_tile_async(buf[nxt], global_ptr, tile);  // Load next
    compute_on_tile(buf[cur]);                     // Compute current
    __syncthreads();
    cur = nxt;
}
compute_on_tile(buf[cur]);
```

### 6.2 Predicated Loads

Avoid warp divergence at boundaries — load zero for out-of-bounds:

```cpp
// Divergent (bad):       if (idx < N) val = data[idx]; else val = 0;
// Predicated (better):   float val = (idx < N) ? __ldg(&data[idx]) : 0.0f;
```

### 6.3 Warp Specialization

Different warps perform different roles within one block:

```cpp
int warp_id = threadIdx.x / 32;
if (warp_id < NUM_COMPUTE_WARPS) {
    // Compute warps: matrix multiply on shared memory data
    wait(barrier_loaded);
    warp_mma(shared_A, shared_B, accum);
    signal(barrier_computed);
} else {
    // Producer warps: load data from global to shared memory
    wait(barrier_computed);
    async_copy(shared_A, global_A + next_tile);
    signal(barrier_loaded);
}
```

### 6.4 Persistent Kernels

One kernel stays resident, processes work from a queue — eliminates launch overhead:

```cpp
__device__ int work_counter = 0;
__global__ void persistent_gemm(WorkItem* queue, int total) {
    while (true) {
        int idx = (threadIdx.x == 0) ? atomicAdd(&work_counter, 1) : 0;
        idx = __shfl_sync(0xFFFFFFFF, idx, 0);
        if (idx >= total) return;
        compute_tile(queue[idx]);  // Process one work item
    }
}
```

### 6.5 Epilogue Fusion

Combine post-GEMM operations (bias, activation, residual) into one write:

```
Without fusion: GEMM writes C → bias kernel reads/writes → ReLU reads/writes
With fusion:    GEMM applies bias+ReLU in registers → single write
Savings: 4096×4096 FP32 = 64MB; eliminates multiple 64MB round-trips
```

### 6.6 Quantization-Aware Kernels

Reduced precision (INT8/FP8) patterns to recognize:
- INT8 inputs with INT32 accumulators (no precision loss during accumulation)
- Scale factors as separate arrays (per-tensor, per-channel, per-group)
- Dequantize at the end: `float result = (float)int32_acc * scale_A * scale_B`
- `dp4a` instruction for INT8 dot products

---

## 7. Building Your Code Reading Practice

### 7.1 Recommended Repos (Progressive Difficulty)

**Tier 1 — Start here:** `NVIDIA/cuda-samples` (vectorAdd, reduction, matrixMul),
Triton tutorials (01-vector-add, 02-fused-softmax).

**Tier 2 — Production patterns:** `NVIDIA/CUTLASS` (examples/basic_gemm),
`pytorch/pytorch` (aten/src/ATen/native/cuda/).

**Tier 3 — Expert level:** `Dao-AILab/flash-attention`,
`NVIDIA/TensorRT-LLM`, `vllm-project/vllm` (PagedAttention).

### 7.2 Progressive Difficulty Path

```
L1: Elementwise  → Vector add, scalar multiply, fused multiply-add
L2: Reductions   → Sum, max, softmax, layer normalization
L3: Matrix ops   → Naive matmul → tiled → CUTLASS-style → tensor core
L4: Attention    → Standard → Flash → multi-head → paged
L5: Full model   → Fused QKV, quantized inference, speculative decoding
```

### 7.3 Modification Exercises

1. **Change block size** (32→1024) and measure with Nsight Compute
2. **Add shared memory tiling** to a naive kernel; compare performance
3. **Replace scalar loads** with `float4`; measure speedup
4. **Fuse two consecutive kernels** that read/write the same data

### 7.4 Profiling While Reading

```bash
nsys profile --stats=true ./my_program          # Timeline view
ncu --set full ./my_program                      # Kernel details

# Key metrics: Achieved Occupancy, Memory Throughput,
# Compute Throughput, Warp Stall Reasons
```

### 7.5 Cross-Reference Exercise

Compare the same operation across codebases — e.g., Layer Normalization in
PyTorch vs Apex vs Triton. Ask:

- How does each handle the reduction (sum, mean)?
- What block size does each choose, and why?
- How does each handle vectorized memory access?
- Which fuses the affine transform (γ, β)?
- Which handles mixed precision (FP16 input, FP32 accumulation)?

This exercise reveals that different implementations make different
trade-offs. Understanding *why* teaches you more than memorizing *how*.

### 7.6 Code Reading Journal

For each kernel you read, record:

```
Kernel:      [Name and source]
Purpose:     [What it computes]
Grid/Block:  [How work is decomposed]
Memory:      [What goes where — global, shared, registers]
Techniques:  [Patterns used and why]
Bound:       [Compute-bound or memory-bound?]
Learned:     [New insight or technique]
```

### 7.7 Reading Checklist

```
┌────────────────────────────────────────────────────────┐
│               CUDA KERNEL READING CHECKLIST             │
├────────────────────────────────────────────────────────┤
│ □ Launch config: grid, block, shared memory, stream    │
│ □ Data flow: input/output pointers, constants          │
│ □ Thread mapping: 1D/2D, grid-stride, indexing         │
│ □ Memory hierarchy: global → shared → registers        │
│ □ Synchronization: __syncthreads, __shfl, barriers     │
│ □ Tiling: load → sync → compute → sync → store?        │
│ □ Optimization: double buffering, predication, fusion  │
│ □ Performance: compute-bound or memory-bound?          │
│ □ Portability: hardcoded assumptions? Configurable?    │
└────────────────────────────────────────────────────────┘
```

### 7.8 Setting Up for Exploration

```bash
# Essential tools
pip install triton pynvml

# Clone repos for study
git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git
git clone --depth 1 https://github.com/NVIDIA/cutlass.git

# Navigation (ripgrep for fast CUDA search)
rg "__global__" --type cpp        # Find all kernels
rg "<<<.*>>>" --glob "*.cu"       # Find all launch sites

# Verify setup
nvcc --version && nvidia-smi
```

---

## Key Insights

```
1. Memory access pattern matters more than computation
2. Warps execute in lockstep — think per-warp, not per-thread
3. Shared memory is programmer-managed L1 cache
4. Kernel fusion eliminates memory round-trips (often 5-10× gain)
5. Templates enable compile-time optimization (unrolling, scheduling)
6. Reduction is the fundamental building block (softmax, norm, loss)
```

**Start today:** Pick one kernel from any repo above. Spend 30 minutes with
the reading checklist. Write down what you learn. Repeat daily with harder
kernels. Within a month, you'll read CUDA code fluently.
