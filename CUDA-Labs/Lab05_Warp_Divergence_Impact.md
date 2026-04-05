# Lab 05: Warp Divergence Impact 🟡

| Detail | Value |
|---|---|
| **Difficulty** | 🟡 Intermediate |
| **Estimated Time** | 50–70 minutes |
| **Prerequisites** | Labs 01-04; understanding of warps (32 threads executing in lockstep) |
| **GPU Required** | Any NVIDIA GPU |

---

## Objective

By the end of this lab you will:
- Understand what warp divergence is and why it hurts performance
- Write kernels with different branching patterns and measure the cost
- Use `printf` to see which threads execute which code paths
- Quantify divergence impact: 2× to 32× slowdown depending on pattern
- Learn strategies to minimize divergence

---

## Setup

Create a working directory for this lab's warp divergence experiments.

```bash
mkdir -p ~/cuda-labs/lab05 && cd ~/cuda-labs/lab05
```

### Background: What Is Warp Divergence?

A warp is 32 threads that execute the **same instruction** at the **same time** (SIMT — Single Instruction, Multiple Threads). When threads in a warp take different branches of an `if/else`, the GPU must execute **both paths sequentially**, masking inactive threads. This is called **warp divergence**.

```
if (condition) {
    path_A();  // Only threads with condition==true execute
} else {
    path_B();  // Only threads with condition==false execute
}
// Both paths run — inactive threads are masked (doing no useful work)
```

---

## Step 1: See Divergence in Action

Create `diverge_visual.cu`:

```cuda
// diverge_visual.cu — Visualize which threads take which branch
#include <cstdio>

// No divergence: all threads in a warp take the same path
__global__ void noDivergence() {
    int tid = threadIdx.x;
    int warpId = tid / 32;

    if (warpId == 0) {
        // All 32 threads in warp 0 go here
        if (tid < 8) printf("Warp 0, Thread %2d: path A\n", tid);
    } else {
        // All 32 threads in warp 1 go here
        if (tid >= 32 && tid < 40) printf("Warp 1, Thread %2d: path B\n", tid);
    }
}

// Divergence: threads WITHIN the same warp take different paths
__global__ void withDivergence() {
    int tid = threadIdx.x;

    if (tid % 2 == 0) {
        // Even threads go here
        if (tid < 8) printf("Thread %2d: path A (even)\n", tid);
    } else {
        // Odd threads go here
        if (tid < 8) printf("Thread %2d: path B (odd)\n", tid);
    }
}

int main() {
    printf("=== No Divergence (branch on warp ID) ===\n");
    printf("All threads in each warp take the same path:\n");
    noDivergence<<<1, 64>>>();
    cudaDeviceSynchronize();

    printf("\n=== With Divergence (branch on thread ID) ===\n");
    printf("Threads in the SAME warp take different paths:\n");
    withDivergence<<<1, 64>>>();
    cudaDeviceSynchronize();

    printf("\n→ In the divergent case, each warp executes BOTH paths\n");
    printf("  (masking half the threads each time) = 2× the work!\n");

    return 0;
}
```

### Compile and run

Compile and run to see which threads take which branch in divergent versus non-divergent kernels.

```bash
nvcc -o diverge_visual diverge_visual.cu
./diverge_visual
```

### Expected Output

```
=== No Divergence (branch on warp ID) ===
All threads in each warp take the same path:
Warp 0, Thread  0: path A
Warp 0, Thread  1: path A
...
Warp 1, Thread 32: path B
...

=== With Divergence (branch on thread ID) ===
Threads in the SAME warp take different paths:
Thread  0: path A (even)
Thread  1: path B (odd)
Thread  2: path A (even)
Thread  3: path B (odd)
...
```

---

## Step 2: Measure Divergence Cost

Create `diverge_bench.cu`:

```cuda
// diverge_bench.cu — Quantify the performance cost of warp divergence
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 22)  // 4M elements
#define ITERS 100

// Baseline: no branching at all
__global__ void noBranch(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        for (int k = 0; k < ITERS; k++) {
            val = val * 1.001f + 0.001f;
        }
        data[i] = val;
    }
}

// No divergence: branch on warp (all threads in warp take same path)
__global__ void warpBranch(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        int warpLane = threadIdx.x % 32;

        if (warpLane < 32) {  // Always true — all threads take same path
            for (int k = 0; k < ITERS; k++) {
                val = val * 1.001f + 0.001f;
            }
        } else {
            for (int k = 0; k < ITERS; k++) {
                val = val * 0.999f - 0.001f;
            }
        }
        data[i] = val;
    }
}

// 50% divergence: half the warp takes each path
__global__ void halfDiverge(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        int warpLane = threadIdx.x % 32;

        if (warpLane < 16) {  // Half the warp goes each way
            for (int k = 0; k < ITERS; k++) {
                val = val * 1.001f + 0.001f;
            }
        } else {
            for (int k = 0; k < ITERS; k++) {
                val = val * 0.999f - 0.001f;
            }
        }
        data[i] = val;
    }
}

// Full divergence: every other thread takes a different path
__global__ void fullDiverge(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];

        if (threadIdx.x % 2 == 0) {  // Even/odd split
            for (int k = 0; k < ITERS; k++) {
                val = val * 1.001f + 0.001f;
            }
        } else {
            for (int k = 0; k < ITERS; k++) {
                val = val * 0.999f - 0.001f;
            }
        }
        data[i] = val;
    }
}

// Multi-way divergence: 4 different paths within a warp
__global__ void fourWayDiverge(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        int quarter = (threadIdx.x % 32) / 8;  // 0, 1, 2, or 3

        switch (quarter) {
            case 0:
                for (int k = 0; k < ITERS; k++) val = val * 1.001f + 0.001f;
                break;
            case 1:
                for (int k = 0; k < ITERS; k++) val = val * 0.999f - 0.001f;
                break;
            case 2:
                for (int k = 0; k < ITERS; k++) val = val * 1.002f + 0.002f;
                break;
            case 3:
                for (int k = 0; k < ITERS; k++) val = val * 0.998f - 0.002f;
                break;
        }
        data[i] = val;
    }
}

// Asymmetric divergence: 1 thread does heavy work, 31 do light work
__global__ void asymmetricDiverge(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];

        if (threadIdx.x % 32 == 0) {  // Only 1 thread per warp
            for (int k = 0; k < ITERS * 10; k++) {  // 10× more work
                val = val * 1.001f + 0.001f;
            }
        } else {
            for (int k = 0; k < ITERS; k++) {
                val = val * 0.999f - 0.001f;
            }
        }
        data[i] = val;
    }
}

float benchKernel(void (*kernel)(float*, int), float *d_data, int n, const char *name) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid = (n + block - 1) / block;

    // Warm up
    kernel<<<grid, block>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int runs = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        kernel<<<grid, block>>>(d_data, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= runs;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

int main() {
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    printf("=== Warp Divergence Performance Impact ===\n");
    printf("N = %d, %d iterations per thread per kernel\n\n", N, ITERS);

    struct { const char *name; void (*kernel)(float*, int); } tests[] = {
        {"No branching",            noBranch},
        {"Branch (no divergence)",  warpBranch},
        {"50%% divergence (16/16)", halfDiverge},
        {"100%% divergence (even/odd)", fullDiverge},
        {"4-way divergence",        fourWayDiverge},
        {"Asymmetric (1 vs 31)",    asymmetricDiverge},
    };
    int numTests = sizeof(tests) / sizeof(tests[0]);

    float baseline = 0;
    printf("%-30s  %10s  %10s\n", "Kernel", "Time (ms)", "Slowdown");
    printf("%-30s  %10s  %10s\n", "------", "---------", "--------");

    for (int t = 0; t < numTests; t++) {
        float ms = benchKernel(tests[t].kernel, d_data, N, tests[t].name);
        if (t == 0) baseline = ms;
        printf("%-30s  %10.3f  %9.2fx\n", tests[t].name, ms, ms / baseline);
    }

    printf("\n=== Analysis ===\n");
    printf("• 'No divergence' ≈ 'No branching': branch cost is zero when uniform\n");
    printf("• 50%% divergence ≈ 2× slowdown: both paths execute, half threads masked\n");
    printf("• Asymmetric: ALL 31 idle threads wait for the 1 busy thread!\n");

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Compile and run

Compile and run to measure the performance cost of different divergence patterns — from zero divergence to asymmetric workloads.

```bash
nvcc -O2 -o diverge_bench diverge_bench.cu
./diverge_bench
```

### Expected Output

```
=== Warp Divergence Performance Impact ===
N = 4194304, 100 iterations per thread per kernel

Kernel                          Time (ms)    Slowdown
------                          ---------    --------
No branching                        0.862      1.00x
Branch (no divergence)              0.863      1.00x
50% divergence (16/16)              1.712      1.99x
100% divergence (even/odd)          1.718      1.99x
4-way divergence                    3.421      3.97x
Asymmetric (1 vs 31)                8.534      9.90x
```

---

## Step 3: Fixing Divergence — Data Reorganization

Create `diverge_fix.cu`:

```cuda
// diverge_fix.cu — Convert divergent code to non-divergent
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 22)
#define ITERS 100

// DIVERGENT: branch on data value (unpredictable)
__global__ void processDataDivergent(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = input[i];
        if (val > 0.5f) {
            for (int k = 0; k < ITERS; k++)
                val = val * 1.001f + sinf(val) * 0.001f;
        } else {
            for (int k = 0; k < ITERS; k++)
                val = val * 0.999f - cosf(val) * 0.001f;
        }
        output[i] = val;
    }
}

// NON-DIVERGENT: sort data so consecutive elements take the same branch
__global__ void processDataSorted(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = input[i];
        if (val > 0.5f) {
            for (int k = 0; k < ITERS; k++)
                val = val * 1.001f + sinf(val) * 0.001f;
        } else {
            for (int k = 0; k < ITERS; k++)
                val = val * 0.999f - cosf(val) * 0.001f;
        }
        output[i] = val;
    }
}

// BRANCHLESS: use math to avoid the branch entirely
__global__ void processDataBranchless(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = input[i];
        float selector = (val > 0.5f) ? 1.0f : -1.0f;
        float factor = 1.0f + selector * 0.001f;
        for (int k = 0; k < ITERS; k++) {
            val = val * factor + selector * sinf(val) * 0.001f;
        }
        output[i] = val;
    }
}

int compare_float(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_random = (float *)malloc(bytes);
    float *h_sorted = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; i++) h_random[i] = (float)rand() / RAND_MAX;
    memcpy(h_sorted, h_random, bytes);
    qsort(h_sorted, N, sizeof(float), compare_float);

    float *d_random, *d_sorted, *d_out;
    CUDA_CHECK(cudaMalloc(&d_random, bytes));
    CUDA_CHECK(cudaMalloc(&d_sorted, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_random, h_random, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sorted, h_sorted, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid = (N + block - 1) / block;
    int runs = 20;

    printf("=== Fixing Warp Divergence ===\n");
    printf("N = %d, data is random [0,1), branch on val > 0.5\n\n", N);

    // Divergent (random data)
    processDataDivergent<<<grid, block>>>(d_random, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        processDataDivergent<<<grid, block>>>(d_random, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float divergent_ms;
    CUDA_CHECK(cudaEventElapsedTime(&divergent_ms, start, stop));
    divergent_ms /= runs;

    // Same kernel, sorted data (no divergence)
    processDataSorted<<<grid, block>>>(d_sorted, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        processDataSorted<<<grid, block>>>(d_sorted, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float sorted_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sorted_ms, start, stop));
    sorted_ms /= runs;

    // Branchless version (random data)
    processDataBranchless<<<grid, block>>>(d_random, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        processDataBranchless<<<grid, block>>>(d_random, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float branchless_ms;
    CUDA_CHECK(cudaEventElapsedTime(&branchless_ms, start, stop));
    branchless_ms /= runs;

    printf("%-35s  %8.3f ms  (baseline)\n", "Divergent (random data)", divergent_ms);
    printf("%-35s  %8.3f ms  (%.2fx speedup)\n", "Same code, sorted data", sorted_ms, divergent_ms / sorted_ms);
    printf("%-35s  %8.3f ms  (%.2fx speedup)\n", "Branchless (random data)", branchless_ms, divergent_ms / branchless_ms);

    printf("\n=== Strategies to Avoid Divergence ===\n");
    printf("1. Sort/partition data so adjacent elements take the same branch\n");
    printf("2. Replace branches with branchless arithmetic (select, min, max)\n");
    printf("3. Assign work by warp, not by thread (branch on warpId)\n");
    printf("4. Use separate kernels for different categories of data\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_random));
    CUDA_CHECK(cudaFree(d_sorted));
    CUDA_CHECK(cudaFree(d_out));
    free(h_random); free(h_sorted);
    return 0;
}
```

### Compile and run

Compile and run to compare three strategies for eliminating warp divergence: random data (divergent), sorted data (non-divergent), and branchless arithmetic.

```bash
nvcc -O2 -o diverge_fix diverge_fix.cu
./diverge_fix
```

### Expected Output

```
=== Fixing Warp Divergence ===
N = 4194304, data is random [0,1), branch on val > 0.5

Divergent (random data)              12.345 ms  (baseline)
Same code, sorted data                7.213 ms  (1.71x speedup)
Branchless (random data)              6.892 ms  (1.79x speedup)

=== Strategies to Avoid Divergence ===
1. Sort/partition data so adjacent elements take the same branch
2. Replace branches with branchless arithmetic (select, min, max)
3. Assign work by warp, not by thread (branch on warpId)
4. Use separate kernels for different categories of data
```

---

## Experiments

### Experiment 1: Varying divergence ratio
Instead of 50/50 split (`val > 0.5`), try 90/10 (`val > 0.1`) and 99/1 (`val > 0.01`). How does the ratio affect performance? Even 1% divergence still causes some serialization.

### Experiment 2: Short vs long branches
Replace the 100-iteration loop with a single operation in both branches. Does divergence still matter for very short branches?

### Experiment 3: Nested divergence
Add nested if/else within each branch. How does 2-level divergence compare to 1-level?

### Experiment 4: Warp-aware partitioning
Modify the kernel so threads process data in warp-sized chunks that share the same branch. Compare performance to random assignment.

### Experiment 5: Compiler optimization
Compile with `-O0` vs `-O2` vs `-O3`. Does the compiler eliminate any divergence automatically?

---

## What Just Happened?

1. **Warp divergence is a SIMT penalty.** When threads in a warp disagree on a branch, the GPU executes both paths sequentially, masking inactive threads. With 2 paths, you get ~2× slowdown. With 4 paths, ~4× slowdown.

2. **Divergence only matters within a warp.** If all 32 threads in a warp agree (even if different warps take different paths), there's zero divergence penalty. This is why sorting data or partitioning by warp eliminates the cost.

3. **Branchless code avoids divergence entirely.** By using arithmetic (multiply by 0 or 1, conditional assignment) instead of if/else, all threads execute the same instructions. The cost is slightly more ALU work, but no serialization.

4. **Asymmetric workloads are worst.** If 1 thread in a warp does 10× more work than the other 31, all 31 threads are idle waiting for that 1 thread. The warp completes at the speed of the slowest thread.

---

## Key Insight

> **Divergence cost = (number of unique paths taken within a warp) × (cost of one path).** A warp is only as fast as its slowest thread. Design your data layout and kernel logic so that adjacent threads (same warp) do the same thing.

---

## Checkpoint Quiz

**Q1:** A warp of 32 threads hits an `if/else`. 30 threads take the `if` path and 2 take the `else` path. How many times slower is this compared to no divergence?
<details><summary>Answer</summary>
Approximately 2× slower. Even though only 2 threads take the `else` path, the entire warp must still execute both code paths. The `if` path runs with 30 threads active (2 masked), then the `else` path runs with 2 threads active (30 masked). The total time is the sum of both path execution times, regardless of how many threads are active in each.
</details>

**Q2:** You have a kernel that processes two categories of data (type A and type B), mixed randomly in the input array. What are two strategies to eliminate divergence?
<details><summary>Answer</summary>
1. **Sort/partition the data** before processing: group all type-A elements together and all type-B elements together. Consecutive warps will then process the same type without divergence.
2. **Launch two separate kernels**: one for type-A elements and one for type-B elements. Each kernel has no branching on type. Use a compaction step first to separate the two types.
</details>

**Q3:** Does warp divergence affect GPU performance if different blocks take different branches?
<details><summary>Answer</summary>
No. Divergence only matters within a single warp (32 threads that execute in lockstep). Different blocks are scheduled independently on different SMs. If block 0 takes path A and block 1 takes path B, there is zero divergence penalty — each block's warps are unanimous in their branch choice.
</details>

---

*Next Lab: [Lab 06 — Occupancy Experiments](Lab06_Occupancy_Experiments.md)*
