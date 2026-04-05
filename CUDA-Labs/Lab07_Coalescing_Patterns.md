# Lab 07: Coalescing Patterns 🟡

| Detail | Value |
|---|---|
| **Difficulty** | 🟡 Intermediate |
| **Estimated Time** | 60–80 minutes |
| **Prerequisites** | Labs 01-06; understanding of GPU memory hierarchy |
| **GPU Required** | Any NVIDIA GPU (Compute Capability 3.0+) |

---

## Objective

By the end of this lab you will:
- Write kernels with sequential, strided, and random access patterns
- Measure effective memory throughput for each pattern
- Compare Structure-of-Arrays (SoA) vs Array-of-Structures (AoS)
- Quantify the 10-30× penalty of uncoalesced memory access
- Understand why memory coalescing is the single most important GPU optimization

---

## Setup

```bash
mkdir -p ~/cuda-labs/lab07 && cd ~/cuda-labs/lab07
```

### Background: What Is Coalescing?

When 32 threads in a warp access memory, the hardware combines (coalesces) their requests into as few memory transactions as possible. If threads access **consecutive** addresses, a single 128-byte transaction serves all 32 threads. If threads access **scattered** addresses, up to 32 separate transactions are needed — each fetching a full cache line of which only 4 bytes are used.

---

## Step 1: Access Pattern Benchmark

Create `coalesce_bench.cu`:

```cuda
// coalesce_bench.cu — Measure bandwidth for different access patterns
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 24)  // 16M elements = 64 MB

// Pattern 1: COALESCED — sequential access (stride 1)
__global__ void sequential(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}

// Pattern 2: STRIDED — threads access every Kth element
template<int STRIDE>
__global__ void strided(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (i * STRIDE) % n;
    if (i < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

// Pattern 3: RANDOM — each thread accesses a random location
__global__ void randomAccess(const float *input, float *output,
                             const int *indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = indices[i];
        output[idx] = input[idx] * 2.0f;
    }
}

// Pattern 4: OFFSET — threads access with a fixed offset from aligned
template<int OFFSET>
__global__ void offsetAccess(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i + OFFSET;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

float benchmark(void (*kernel)(const float*, float*, int),
                const float *d_in, float *d_out, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid = (n + block - 1) / block;

    kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    int runs = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / runs;
}

int main() {
    size_t bytes = N * sizeof(float);
    printf("=== Memory Coalescing Patterns ===\n");
    printf("N = %d elements (%.0f MB)\n\n", N, bytes / (1024.0 * 1024.0));

    // Allocate
    float *d_in, *d_out;
    int *d_indices;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));

    // Initialize input
    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Create random permutation for random access
    int *h_indices = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_indices[i] = i;
    srand(42);
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = h_indices[i]; h_indices[i] = h_indices[j]; h_indices[j] = tmp;
    }
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));

    // Benchmark
    printf("%-25s  %10s  %12s  %10s\n",
           "Pattern", "Time (ms)", "Bandwidth", "Efficiency");
    printf("%-25s  %10s  %12s  %10s\n",
           "-------", "---------", "---------", "----------");

    // Sequential (coalesced)
    float seqMs = benchmark(sequential, d_in, d_out, N);
    float peakBW = 2.0f * bytes / (seqMs / 1000.0f) / 1e9;
    printf("%-25s  %10.3f  %9.1f GB/s  %9.1f%%\n",
           "Sequential (stride 1)", seqMs, peakBW, 100.0f);

    // Strided patterns
    int strides[] = {2, 4, 8, 16, 32};
    for (int s = 0; s < 5; s++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int block = 256;
        int grid = (N + block - 1) / block;
        int stride = strides[s];

        // Use function pointer trick: benchmark each stride
        // We need to call the template directly
        auto *kernelFn = (stride == 2) ? strided<2> :
                         (stride == 4) ? strided<4> :
                         (stride == 8) ? strided<8> :
                         (stride == 16) ? strided<16> : strided<32>;

        kernelFn<<<grid, block>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        int runs = 50;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            kernelFn<<<grid, block>>>(d_in, d_out, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;
        char name[64];
        snprintf(name, sizeof(name), "Stride %d", stride);
        printf("%-25s  %10.3f  %9.1f GB/s  %9.1f%%\n",
               name, ms, bw, (bw / peakBW) * 100.0f);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // Random access
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int block = 256, grid = (N + block - 1) / block;
        randomAccess<<<grid, block>>>(d_in, d_out, d_indices, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        int runs = 20;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++)
            randomAccess<<<grid, block>>>(d_in, d_out, d_indices, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= runs;

        float bw = 2.0f * bytes / (ms / 1000.0f) / 1e9;
        printf("%-25s  %10.3f  %9.1f GB/s  %9.1f%%\n",
               "Random", ms, bw, (bw / peakBW) * 100.0f);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    free(h_in); free(h_indices);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_indices));
    return 0;
}
```

### Compile and run

```bash
nvcc -O2 -o coalesce_bench coalesce_bench.cu
./coalesce_bench
```

### Expected Output

```
=== Memory Coalescing Patterns ===
N = 16777216 elements (64 MB)

Pattern                    Time (ms)     Bandwidth   Efficiency
-------                    ---------     ---------   ----------
Sequential (stride 1)         0.398     322.5 GB/s      100.0%
Stride 2                      0.692     185.5 GB/s       57.5%
Stride 4                      1.243     103.3 GB/s       32.0%
Stride 8                      2.385      53.8 GB/s       16.7%
Stride 16                     4.612      27.8 GB/s        8.6%
Stride 32                     8.934      14.4 GB/s        4.5%
Random                        12.456     10.3 GB/s        3.2%
```

> Stride-32 and random access achieve only 3-5% of coalesced bandwidth — a 20-30× penalty!

---

## Step 2: Structure of Arrays vs Array of Structures

Create `soa_vs_aos.cu`:

```cuda
// soa_vs_aos.cu — Why data layout matters enormously on GPUs
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

#define N (1 << 22)  // 4M particles

// ==================== Array of Structures (AoS) ====================
struct ParticleAoS {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float mass;
    float charge;       // 8 floats = 32 bytes per particle
};

__global__ void updateAoS(ParticleAoS *particles, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

// ==================== Structure of Arrays (SoA) ====================
struct ParticleSoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
    float *charge;
};

__global__ void updateSoA(float *x, float *y, float *z,
                          const float *vx, const float *vy, const float *vz,
                          int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

int main() {
    printf("=== SoA vs AoS: Data Layout Impact on GPU Performance ===\n");
    printf("N = %d particles\n\n", N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid = (N + block - 1) / block;
    int runs = 50;
    float dt = 0.001f;

    // ==================== AoS ====================
    size_t aosBytes = N * sizeof(ParticleAoS);
    printf("AoS size: %.0f MB (%zu bytes per particle, %d fields)\n",
           aosBytes / (1024.0 * 1024.0), sizeof(ParticleAoS), 8);

    ParticleAoS *h_aos = (ParticleAoS *)malloc(aosBytes);
    for (int i = 0; i < N; i++) {
        h_aos[i] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f, 1.0f, -1.0f};
    }

    ParticleAoS *d_aos;
    CUDA_CHECK(cudaMalloc(&d_aos, aosBytes));
    CUDA_CHECK(cudaMemcpy(d_aos, h_aos, aosBytes, cudaMemcpyHostToDevice));

    updateAoS<<<grid, block>>>(d_aos, N, dt);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        updateAoS<<<grid, block>>>(d_aos, N, dt);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float aosMs;
    CUDA_CHECK(cudaEventElapsedTime(&aosMs, start, stop));
    aosMs /= runs;

    // Calculate effective bandwidth: we read 6 floats and write 3 floats per particle
    float aosBW = (float)N * 6 * sizeof(float) / (aosMs / 1000.0f) / 1e9;
    // But AoS actually loads entire structs due to coalescing issues
    float aosActualBW = (float)aosBytes / (aosMs / 1000.0f) / 1e9;

    // ==================== SoA ====================
    size_t fieldBytes = N * sizeof(float);
    printf("SoA size: %.0f MB (%zu bytes per field × 8 fields)\n",
           8 * fieldBytes / (1024.0 * 1024.0), fieldBytes);

    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    CUDA_CHECK(cudaMalloc(&d_x, fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_y, fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_z, fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_vx, fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_vy, fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_vz, fieldBytes));

    // Initialize SoA
    float *h_field = (float *)malloc(fieldBytes);
    for (int i = 0; i < N; i++) h_field[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_x, h_field, fieldBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_field, fieldBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_field, fieldBytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < N; i++) h_field[i] = 0.1f;
    CUDA_CHECK(cudaMemcpy(d_vx, h_field, fieldBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_field, fieldBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_field, fieldBytes, cudaMemcpyHostToDevice));

    updateSoA<<<grid, block>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, N, dt);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++)
        updateSoA<<<grid, block>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, N, dt);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float soaMs;
    CUDA_CHECK(cudaEventElapsedTime(&soaMs, start, stop));
    soaMs /= runs;

    float soaBW = (float)N * 6 * sizeof(float) / (soaMs / 1000.0f) / 1e9;

    // ==================== Results ====================
    printf("\n%-15s  %10s  %12s  %10s\n", "Layout", "Time (ms)", "Eff. BW", "Speedup");
    printf("%-15s  %10s  %12s  %10s\n", "------", "---------", "-------", "-------");
    printf("%-15s  %10.3f  %9.1f GB/s  %9.1fx\n", "AoS", aosMs, aosBW, 1.0f);
    printf("%-15s  %10.3f  %9.1f GB/s  %9.1fx\n", "SoA", soaMs, soaBW, aosMs / soaMs);

    printf("\n=== Why SoA Wins ===\n");
    printf("AoS: Thread 0 reads particle[0].x, thread 1 reads particle[1].x\n");
    printf("     These are 32 bytes apart (sizeof(Particle)) → stride-8 access!\n");
    printf("     Each cache line (128B) serves only 4 threads → 87.5%% waste\n\n");
    printf("SoA: Thread 0 reads x[0], thread 1 reads x[1]\n");
    printf("     These are 4 bytes apart (sizeof(float)) → perfectly coalesced!\n");
    printf("     Each cache line (128B) serves 32 threads → 0%% waste\n");

    // Cleanup
    free(h_aos); free(h_field);
    CUDA_CHECK(cudaFree(d_aos));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
```

### Compile and run

```bash
nvcc -O2 -o soa_vs_aos soa_vs_aos.cu
./soa_vs_aos
```

### Expected Output

```
=== SoA vs AoS: Data Layout Impact on GPU Performance ===
N = 4194304 particles

AoS size: 128 MB (32 bytes per particle, 8 fields)
SoA size: 128 MB (16777216 bytes per field × 8 fields)

Layout           Time (ms)       Eff. BW     Speedup
------           ---------       -------     -------
AoS                  0.952      105.2 GB/s       1.0x
SoA                  0.187      535.8 GB/s       5.1x

=== Why SoA Wins ===
AoS: Thread 0 reads particle[0].x, thread 1 reads particle[1].x
     These are 32 bytes apart (sizeof(Particle)) → stride-8 access!
     Each cache line (128B) serves only 4 threads → 87.5% waste

SoA: Thread 0 reads x[0], thread 1 reads x[1]
     These are 4 bytes apart (sizeof(float)) → perfectly coalesced!
     Each cache line (128B) serves 32 threads → 0% waste
```

---

## Experiments

### Experiment 1: Vary the struct size
Add more fields to `ParticleAoS` (16 floats = 64 bytes, 32 floats = 128 bytes). How does the AoS/SoA gap change with larger strides?

### Experiment 2: Use only some fields
Modify `updateAoS` to use only `x` and `vx` (1 of 8 fields). The AoS version loads the entire 32-byte struct just to read 8 bytes — what's the bandwidth waste?

### Experiment 3: AoSoA (hybrid layout)
Try an Array-of-Structures-of-Arrays layout where you group 32 x-values, then 32 y-values, etc. This gives coalesced access within a warp while keeping related data nearby.

### Experiment 4: Read-only `__ldg()` intrinsic
Replace reads with `__ldg(&input[i])` in the strided access kernel. Does the read-only cache help with uncoalesced reads?

### Experiment 5: Global memory alignment
Use `cudaMallocPitch` for 2D data. How does proper alignment affect coalescing for 2D access patterns?

---

## What Just Happened?

1. **Coalescing is king.** The difference between stride-1 (coalesced) and stride-32 or random access is 20-30×. No other single optimization comes close to this impact.

2. **AoS kills GPU performance.** When threads access the same field of adjacent structures, they're striding by the structure size. A struct with 8 floats means stride-8 access — each cache line is only 12.5% utilized.

3. **SoA enables perfect coalescing.** When each field is a separate contiguous array, threads accessing adjacent elements get stride-1 access — 100% cache line utilization.

4. **The GPU memory system is designed for coalesced access.** Memory controllers serve 128-byte aligned cache lines. When a warp's 32 threads each need 4 bytes from the same cache line, one transaction serves all of them. With scattered access, 32 transactions are needed.

---

## Key Insight

> **If you remember one thing from all CUDA optimization: make adjacent threads access adjacent memory addresses.** This single principle — memory coalescing — determines more of your kernel's performance than any other factor. When in doubt, restructure your data layout before touching your compute logic.

---

## Checkpoint Quiz

**Q1:** 32 threads in a warp each read a `float` (4 bytes). In the best case, how many 128-byte cache line transactions are needed? In the worst case?
<details><summary>Answer</summary>
**Best case (coalesced):** 32 × 4 = 128 bytes = exactly 1 cache line transaction.
**Worst case (all different cache lines):** 32 transactions, each loading 128 bytes but using only 4 bytes = 4096 bytes loaded for 128 bytes used = 3.125% utilization.
</details>

**Q2:** You have a `float3` (12 bytes) array. Thread `i` accesses `arr[i]`. Is this coalesced?
<details><summary>Answer</summary>
Partially. `float3` is 12 bytes, which is not a power of 2. Thread 0 accesses bytes 0-11, thread 1 accesses bytes 12-23, etc. These cross cache line boundaries unpredictably. The fix: use `float4` (16 bytes, power of 2) and ignore the 4th component, or use SoA layout with separate x, y, z arrays.
</details>

**Q3:** Your kernel reads from array A (coalesced) and writes to array B (stride-16). Which would help more: optimizing the read pattern or the write pattern?
<details><summary>Answer</summary>
Optimizing the write pattern. The reads are already coalesced (efficient). The stride-16 writes waste ~94% of each cache line's capacity. Making writes coalesced (e.g., using shared memory as an intermediary, like in matrix transpose) would dramatically improve bandwidth utilization. Always fix the most uncoalesced access first.
</details>

---

*Next Lab: [Lab 08 — Stream Overlap](Lab08_Stream_Overlap.md)*
