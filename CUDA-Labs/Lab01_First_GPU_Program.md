# Lab 01: Your Very First GPU Program 🟢

| Detail | Value |
|---|---|
| **Difficulty** | 🟢 Beginner |
| **Estimated Time** | 45–60 minutes |
| **Prerequisites** | Basic C/C++, a Linux system with an NVIDIA GPU, CUDA Toolkit installed |
| **GPU Required** | Any NVIDIA GPU (Compute Capability 3.0+) |

---

## Objective

By the end of this lab you will:
- Write, compile, and run your first CUDA kernel
- Perform vector addition on the GPU
- Time CPU vs GPU execution and compare
- Find the **crossover point** where the GPU starts beating the CPU
- Understand the basic CUDA programming model: host, device, kernels, grids, blocks, threads

---

## Setup

### Verify your environment

```bash
# Check CUDA compiler is available
nvcc --version

# Check your GPU
nvidia-smi

# Create a working directory
mkdir -p ~/cuda-labs/lab01 && cd ~/cuda-labs/lab01
```

You should see your GPU model and CUDA version. If `nvcc` is not found, ensure the CUDA Toolkit is installed and `PATH` includes `/usr/local/cuda/bin`.

---

## Step 1: Hello from the GPU

Every CUDA journey begins with a kernel that proves the GPU is alive. Create `hello_gpu.cu`:

```cuda
// hello_gpu.cu — Your very first CUDA program
#include <cstdio>

// __global__ marks this function as a CUDA kernel — callable from host, runs on device
__global__ void helloKernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU thread %d (block %d, thread-in-block %d)\n",
           tid, blockIdx.x, threadIdx.x);
}

int main() {
    printf("Launching kernel with 2 blocks of 4 threads each...\n\n");

    // Launch: <<<numBlocks, threadsPerBlock>>>
    helloKernel<<<2, 4>>>();

    // Wait for GPU to finish before the program exits
    cudaDeviceSynchronize();

    printf("\nDone!\n");
    return 0;
}
```

### Compile and run

```bash
nvcc -o hello_gpu hello_gpu.cu
./hello_gpu
```

### Expected Output

```
Launching kernel with 2 blocks of 4 threads each...

Hello from GPU thread 0 (block 0, thread-in-block 0)
Hello from GPU thread 1 (block 0, thread-in-block 1)
Hello from GPU thread 2 (block 0, thread-in-block 2)
Hello from GPU thread 3 (block 0, thread-in-block 3)
Hello from GPU thread 4 (block 1, thread-in-block 0)
Hello from GPU thread 5 (block 1, thread-in-block 1)
Hello from GPU thread 6 (block 1, thread-in-block 2)
Hello from GPU thread 7 (block 1, thread-in-block 3)

Done!
```

> **Note:** Thread print order may vary — GPUs execute threads in parallel, so ordering is not guaranteed.

---

## Step 2: Vector Addition — CPU Version

Now let's do real work. We'll add two arrays element-by-element. First, the CPU baseline.

Create `vecadd.cu`:

```cuda
// vecadd.cu — CPU vs GPU vector addition with timing
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// ===================== CPU VERSION =====================
void vectorAddCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ===================== GPU VERSION =====================
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// ===================== ERROR CHECKING =====================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ===================== TIMING HELPERS =====================
double cpuTimer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ===================== MAIN =====================
int main(int argc, char **argv) {
    int N = 1 << 20;  // ~1 million elements (default)
    if (argc > 1) N = atoi(argv[1]);

    size_t bytes = N * sizeof(float);
    printf("Vector Addition: N = %d (%zu MB)\n\n", N, bytes / (1024 * 1024));

    // --- Allocate host memory ---
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c_cpu = (float *)malloc(bytes);
    float *h_c_gpu = (float *)malloc(bytes);

    // --- Initialize with random data ---
    srand(42);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // ==================== CPU TIMING ====================
    double t0 = cpuTimer();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    double t1 = cpuTimer();
    double cpuTime = (t1 - t0) * 1000.0;  // ms
    printf("CPU time:          %8.3f ms\n", cpuTime);

    // ==================== GPU TIMING ====================
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Create CUDA events for precise GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Time the ENTIRE GPU pipeline: copy + compute + copy back ---
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTotalMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTotalMs, start, stop));
    printf("GPU time (total):  %8.3f ms  (includes memory transfers)\n", gpuTotalMs);

    // --- Time ONLY the kernel ---
    CUDA_CHECK(cudaEventRecord(start));
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuKernelMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuKernelMs, start, stop));
    printf("GPU time (kernel): %8.3f ms  (compute only)\n", gpuKernelMs);

    // --- Speedup ---
    printf("\nSpeedup (total):   %6.2fx\n", cpuTime / gpuTotalMs);
    printf("Speedup (kernel):  %6.2fx\n", cpuTime / gpuKernelMs);

    // ==================== VERIFY RESULTS ====================
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f) {
            errors++;
            if (errors <= 5) {
                printf("MISMATCH at i=%d: CPU=%.6f GPU=%.6f\n",
                       i, h_c_cpu[i], h_c_gpu[i]);
            }
        }
    }
    printf("\nVerification: %s (%d errors out of %d)\n",
           errors == 0 ? "PASSED ✓" : "FAILED ✗", errors, N);

    // ==================== CLEANUP ====================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);

    return 0;
}
```

### Compile and run

```bash
nvcc -O2 -o vecadd vecadd.cu
./vecadd            # Default: 1M elements
./vecadd 1000       # 1K elements
./vecadd 10000000   # 10M elements
```

### Expected Output (approximate, varies by hardware)

```
Vector Addition: N = 1048576 (4 MB)

CPU time:            1.850 ms
GPU time (total):    0.942 ms  (includes memory transfers)
GPU time (kernel):   0.031 ms  (compute only)

Speedup (total):    1.96x
Speedup (kernel):  59.68x

Verification: PASSED ✓
```

---

## Step 3: Find the Crossover Point

Create `crossover.cu` — a sweep that tests many sizes:

```cuda
// crossover.cu — Find the N where GPU beats CPU
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void vectorAddCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

__global__ void vectorAddGPU(const float *a, const float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

double cpuTimer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("%-12s  %10s  %10s  %10s  %s\n",
           "N", "CPU (ms)", "GPU (ms)", "Speedup", "Winner");
    printf("%-12s  %10s  %10s  %10s  %s\n",
           "---", "---", "---", "---", "---");

    // Warm up the GPU
    float *d_warmup;
    CUDA_CHECK(cudaMalloc(&d_warmup, 1024));
    CUDA_CHECK(cudaFree(d_warmup));

    int sizes[] = {
        1000, 5000, 10000, 50000, 100000,
        500000, 1000000, 5000000, 10000000, 50000000
    };
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < numSizes; s++) {
        int N = sizes[s];
        size_t bytes = N * sizeof(float);

        float *h_a = (float *)malloc(bytes);
        float *h_b = (float *)malloc(bytes);
        float *h_c = (float *)malloc(bytes);

        for (int i = 0; i < N; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
        }

        // CPU timing (average of 3 runs)
        double cpuTotal = 0;
        for (int r = 0; r < 3; r++) {
            double t0 = cpuTimer();
            vectorAddCPU(h_a, h_b, h_c, N);
            double t1 = cpuTimer();
            cpuTotal += (t1 - t0);
        }
        double cpuMs = (cpuTotal / 3.0) * 1000.0;

        // GPU timing (including transfers, average of 3 runs)
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float gpuTotal = 0;
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        for (int r = 0; r < 3; r++) {
            CUDA_CHECK(cudaEventRecord(start));
            CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
            vectorAddGPU<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
            CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            gpuTotal += ms;
        }
        float gpuMs = gpuTotal / 3.0f;
        float speedup = cpuMs / gpuMs;

        printf("%-12d  %10.3f  %10.3f  %10.2fx  %s\n",
               N, cpuMs, gpuMs, speedup,
               speedup > 1.0f ? "GPU ✓" : "CPU ✓");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        free(h_a); free(h_b); free(h_c);
    }

    return 0;
}
```

### Compile and run

```bash
nvcc -O2 -o crossover crossover.cu
./crossover
```

### Expected Output (approximate)

```
N              CPU (ms)    GPU (ms)     Speedup  Winner
---            ---         ---          ---      ---
1000              0.002       0.045       0.04x  CPU ✓
5000              0.009       0.047       0.19x  CPU ✓
10000             0.018       0.050       0.36x  CPU ✓
50000             0.089       0.072       1.24x  GPU ✓
100000            0.178       0.094       1.89x  GPU ✓
500000            0.890       0.213       4.18x  GPU ✓
1000000           1.783       0.387       4.61x  GPU ✓
5000000           8.923       1.647       5.42x  GPU ✓
10000000         17.851       3.188       5.60x  GPU ✓
50000000         89.167      15.562       5.73x  GPU ✓
```

The crossover typically happens between N=10,000 and N=100,000 depending on your GPU.

---

## Experiments

### Experiment 1: Change the block size
In `vecadd.cu`, change `threadsPerBlock` from 256 to 32, 64, 128, 512, and 1024. How does it affect kernel time? Total time?

### Experiment 2: More work per element
Replace `c[i] = a[i] + b[i]` with:
```cuda
c[i] = sinf(a[i]) * cosf(b[i]) + sqrtf(a[i] * a[i] + b[i] * b[i]);
```
How does the crossover point change when there's more compute per element?

### Experiment 3: Double precision
Change all `float` to `double`. How does this affect GPU speedup? (Consumer GPUs have much less FP64 throughput.)

### Experiment 4: Remove the boundary check
In the kernel, remove `if (i < n)`. Run with N that isn't a multiple of the block size. What happens?

### Experiment 5: Multiple kernel launches
Launch the kernel 100 times in a loop (without re-copying data). What's the per-launch overhead?

---

## What Just Happened?

1. **Small arrays favor the CPU** because the GPU has fixed overhead: driver setup, kernel launch (~5-10μs), and memory transfer over PCIe. For tiny arrays, this overhead dwarfs the computation.

2. **Large arrays favor the GPU** because the GPU has thousands of cores working in parallel. Once the array is big enough to keep all those cores busy, the GPU's raw throughput wins decisively.

3. **Memory transfer is the bottleneck.** Notice the kernel-only time is tiny compared to the total GPU time. Most time is spent copying data over PCIe (~12 GB/s) versus GPU memory bandwidth (~500+ GB/s). This is the fundamental challenge of GPU programming.

4. **The crossover point** depends on your specific GPU and CPU. Faster GPUs have lower overhead and cross over sooner. The key insight is that the crossover exists — GPUs are not universally faster.

---

## Key Insight

> **The GPU is not a magic "go faster" button.** There is a minimum problem size below which the CPU wins. Your job as a GPU programmer is to (a) make the problem big enough, (b) minimize data transfer, and (c) give each thread enough work.

---

## Checkpoint Quiz

**Q1:** Why does the GPU kernel time seem so much faster than the total GPU time?
<details><summary>Answer</summary>
The total GPU time includes cudaMemcpy transfers over PCIe. The kernel itself runs in GPU memory at ~500+ GB/s bandwidth, but transferring data between host and device is limited to PCIe bandwidth (~12-32 GB/s). For simple operations like vector addition, the transfer time dominates.
</details>

**Q2:** What happens if you launch more threads than there are elements (i.e., `blocksPerGrid * threadsPerBlock > N`)?
<details><summary>Answer</summary>
The extra threads enter the kernel but the `if (i < n)` guard causes them to do nothing and return. Without this guard, those threads would read/write out-of-bounds memory, causing undefined behavior or crashes.
</details>

**Q3:** You have 100 elements to add. Should you use the GPU?
<details><summary>Answer</summary>
No. The kernel launch overhead alone (~5-10μs) plus two cudaMemcpy calls (~20-50μs) far exceeds the CPU time for 100 additions (~0.1μs). The GPU needs thousands to millions of elements before the parallelism advantage offsets the fixed costs.
</details>

---

*Next Lab: [Lab 02 — Thread Indexing Playground](Lab02_Thread_Indexing_Playground.md)*
