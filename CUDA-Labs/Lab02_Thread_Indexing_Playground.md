# Lab 02: Thread Indexing Playground 🟢

| Detail | Value |
|---|---|
| **Difficulty** | 🟢 Beginner |
| **Estimated Time** | 45–60 minutes |
| **Prerequisites** | Lab 01 complete, basic understanding of CUDA kernel launch |
| **GPU Required** | Any NVIDIA GPU |

---

## Objective

By the end of this lab you will:
- Understand the relationship between `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`
- Map 1D, 2D, and 3D thread grids to data
- Visualize how CUDA organizes threads into blocks and grids
- Debug common grid-sizing mistakes
- Build intuition for choosing grid/block dimensions

---

## Setup

```bash
mkdir -p ~/cuda-labs/lab02 && cd ~/cuda-labs/lab02
```

---

## Step 1: Visualize 1D Thread Indexing

Create `index_1d.cu`:

```cuda
// index_1d.cu — See exactly how 1D thread indices map
#include <cstdio>

__global__ void show1DIndex() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d → Global ID %d\n",
           blockIdx.x, threadIdx.x, globalId);
}

int main() {
    printf("=== Grid: 3 blocks × 4 threads/block = 12 threads ===\n\n");
    show1DIndex<<<3, 4>>>();
    cudaDeviceSynchronize();

    printf("\n=== Grid: 2 blocks × 8 threads/block = 16 threads ===\n\n");
    show1DIndex<<<2, 8>>>();
    cudaDeviceSynchronize();

    return 0;
}
```

### Compile and run

```bash
nvcc -o index_1d index_1d.cu
./index_1d
```

### Expected Output (order may vary within each launch)

```
=== Grid: 3 blocks × 4 threads/block = 12 threads ===

Block 0, Thread 0 → Global ID 0
Block 0, Thread 1 → Global ID 1
Block 0, Thread 2 → Global ID 2
Block 0, Thread 3 → Global ID 3
Block 1, Thread 0 → Global ID 4
Block 1, Thread 1 → Global ID 5
Block 1, Thread 2 → Global ID 6
Block 1, Thread 3 → Global ID 7
Block 2, Thread 0 → Global ID 8
Block 2, Thread 1 → Global ID 9
Block 2, Thread 2 → Global ID 10
Block 2, Thread 3 → Global ID 11

=== Grid: 2 blocks × 8 threads/block = 16 threads ===

Block 0, Thread 0 → Global ID 0
...
Block 1, Thread 7 → Global ID 15
```

### The formula

```
globalId = blockIdx.x * blockDim.x + threadIdx.x
```

Think of it like addresses: `blockIdx.x` is the street number, `blockDim.x` is how many houses per street, `threadIdx.x` is the house number.

---

## Step 2: 2D Thread Indexing

Images, matrices, and 2D data use 2D grids. Create `index_2d.cu`:

```cuda
// index_2d.cu — 2D grid and block indexing
#include <cstdio>

__global__ void show2DIndex() {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Linearized index for accessing 1D arrays with 2D layout
    int width = gridDim.x * blockDim.x;
    int linearId = row * width + col;

    printf("Block(%d,%d) Thread(%d,%d) → Row %d, Col %d → Linear %d\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           row, col, linearId);
}

__global__ void fillMatrix(int *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        // Store the global thread ID so we can see the mapping
        matrix[idx] = idx;
    }
}

int main() {
    // --- Part 1: Visualize small grid ---
    printf("=== 2D Grid: (2×2) blocks of (2×2) threads = 4×4 grid ===\n\n");
    dim3 blocks(2, 2);     // 2 blocks in x, 2 blocks in y
    dim3 threads(2, 2);    // 2 threads in x, 2 threads in y
    show2DIndex<<<blocks, threads>>>();
    cudaDeviceSynchronize();

    // --- Part 2: Fill and display a matrix ---
    printf("\n=== Fill a 6×4 matrix with thread IDs ===\n\n");
    int W = 6, H = 4;
    int *d_mat, *h_mat;
    h_mat = (int *)malloc(W * H * sizeof(int));
    cudaMalloc(&d_mat, W * H * sizeof(int));

    dim3 blk(3, 2);   // 3 blocks in x, 2 in y
    dim3 thr(2, 2);   // 2 threads in x, 2 in y → covers 6×4
    fillMatrix<<<blk, thr>>>(d_mat, W, H);
    cudaMemcpy(h_mat, d_mat, W * H * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Matrix (row, col) → linear index:\n");
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            printf("%3d ", h_mat[r * W + c]);
        }
        printf("\n");
    }

    cudaFree(d_mat);
    free(h_mat);
    return 0;
}
```

### Compile and run

```bash
nvcc -o index_2d index_2d.cu
./index_2d
```

### Expected Output

```
=== 2D Grid: (2×2) blocks of (2×2) threads = 4×4 grid ===

Block(0,0) Thread(0,0) → Row 0, Col 0 → Linear 0
Block(0,0) Thread(1,0) → Row 0, Col 1 → Linear 1
Block(0,0) Thread(0,1) → Row 1, Col 0 → Linear 4
Block(0,0) Thread(1,1) → Row 1, Col 1 → Linear 5
...

=== Fill a 6×4 matrix with thread IDs ===

Matrix (row, col) → linear index:
  0   1   2   3   4   5
  6   7   8   9  10  11
 12  13  14  15  16  17
 18  19  20  21  22  23
```

---

## Step 3: 3D Thread Indexing

3D grids are used for volumetric data (medical imaging, physics simulations). Create `index_3d.cu`:

```cuda
// index_3d.cu — 3D grid indexing
#include <cstdio>

__global__ void show3DIndex() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int width  = gridDim.x * blockDim.x;
    int height = gridDim.y * blockDim.y;

    int linearId = z * (width * height) + y * width + x;

    printf("(%d,%d,%d) → Linear %d\n", x, y, z, linearId);
}

int main() {
    printf("=== 3D Grid: (2×2×2) blocks of (2×2×2) threads = 4×4×4 ===\n\n");
    dim3 blocks(2, 2, 2);
    dim3 threads(2, 2, 2);
    show3DIndex<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Compile and run

```bash
nvcc -o index_3d index_3d.cu
./index_3d
```

You'll see 64 lines (4×4×4 = 64 threads), each with its (x,y,z) coordinate and linearized index.

---

## Step 4: What Happens with Wrong Grid Dimensions?

Create `grid_mistakes.cu` to explore common errors:

```cuda
// grid_mistakes.cu — Exploring grid dimension mistakes
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s (code %d)\n", cudaGetErrorString(err), err); \
    } \
} while(0)

__global__ void writeArray(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i * 10;
}

__global__ void writeArrayNoBoundsCheck(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally NO bounds check — dangerous!
    arr[i] = i * 10;
}

int main() {
    int N = 10;
    int *d_arr;
    int h_arr[32] = {0};
    cudaMalloc(&d_arr, 32 * sizeof(int));

    // --- Mistake 1: Too few threads ---
    printf("=== Mistake 1: Too few threads (only 4 for 10 elements) ===\n");
    cudaMemset(d_arr, 0, 32 * sizeof(int));
    writeArray<<<1, 4>>>(d_arr, N);  // Only 4 threads for 10 elements!
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("arr[%d] = %d\n", i, h_arr[i]);
    printf("→ Elements 4-9 are ZERO — not enough threads!\n\n");

    // --- Mistake 2: Too many threads (but with bounds check) ---
    printf("=== Correct: More threads than elements (with bounds check) ===\n");
    cudaMemset(d_arr, 0, 32 * sizeof(int));
    writeArray<<<1, 16>>>(d_arr, N);  // 16 threads for 10 elements — OK with bounds check
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("arr[%d] = %d\n", i, h_arr[i]);
    printf("→ Works correctly! Extra threads do nothing.\n\n");

    // --- The correct formula ---
    printf("=== The correct grid-size formula ===\n");
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("N=%d, threadsPerBlock=%d → blocks=%d, total threads=%d\n",
           N, threadsPerBlock, blocks, blocks * threadsPerBlock);
    printf("Formula: blocks = (N + threadsPerBlock - 1) / threadsPerBlock\n");
    printf("This guarantees at least N threads are launched.\n\n");

    // --- Block size limits ---
    printf("=== Block size limits ===\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max block dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions:  (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("\n=== Trying to exceed max threads per block ===\n");
    writeArray<<<1, 2048>>>(d_arr, N);  // Likely exceeds max (usually 1024)
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_arr);
    return 0;
}
```

### Compile and run

```bash
nvcc -o grid_mistakes grid_mistakes.cu
./grid_mistakes
```

### Expected Output

```
=== Mistake 1: Too few threads (only 4 for 10 elements) ===
arr[0] = 0
arr[1] = 10
arr[2] = 20
arr[3] = 30
arr[4] = 0
arr[5] = 0
...
→ Elements 4-9 are ZERO — not enough threads!

=== Correct: More threads than elements (with bounds check) ===
arr[0] = 0
arr[1] = 10
...
arr[9] = 90
→ Works correctly! Extra threads do nothing.

=== Block size limits ===
Max threads per block: 1024
Max block dimensions: (1024, 1024, 64)
Max grid dimensions:  (2147483647, 65535, 65535)

=== Trying to exceed max threads per block ===
CUDA error: invalid configuration argument (code 9)
```

---

## Step 5: Mapping Threads to Real Data

Create `thread_mapping.cu` — a practical example processing an image-like 2D array:

```cuda
// thread_mapping.cu — Map 2D threads to a "grayscale image"
#include <cstdio>
#include <cuda_runtime.h>

// Simulate inverting a grayscale image
__global__ void invertImage(unsigned char *img, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        img[idx] = 255 - img[idx];
    }
}

int main() {
    int W = 16, H = 8;
    size_t bytes = W * H * sizeof(unsigned char);

    unsigned char *h_img = (unsigned char *)malloc(bytes);
    unsigned char *d_img;
    cudaMalloc(&d_img, bytes);

    // Create a gradient image
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            h_img[r * W + c] = (unsigned char)(r * 16 + c * 8);

    printf("Original image (first 4 rows):\n");
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < W; c++) printf("%3d ", h_img[r * W + c]);
        printf("\n");
    }

    cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice);

    // Choose block size: 8×8 = 64 threads per block
    dim3 blockSize(8, 8);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x,
                  (H + blockSize.y - 1) / blockSize.y);
    printf("\nGrid: (%d, %d) blocks of (%d, %d) threads\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    invertImage<<<gridSize, blockSize>>>(d_img, W, H);

    cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost);

    printf("\nInverted image (first 4 rows):\n");
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < W; c++) printf("%3d ", h_img[r * W + c]);
        printf("\n");
    }

    cudaFree(d_img);
    free(h_img);
    return 0;
}
```

### Compile and run

```bash
nvcc -o thread_mapping thread_mapping.cu
./thread_mapping
```

### Expected Output

```
Original image (first 4 rows):
  0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120
 16  24  32  40  48  56  64  72  80  88  96 104 112 120 128 136
 32  40  48  56  64  72  80  88  96 104 112 120 128 136 144 152
 48  56  64  72  80  88  96 104 112 120 128 136 144 152 160 168

Grid: (2, 1) blocks of (8, 8) threads

Inverted image (first 4 rows):
255 247 239 231 223 215 207 199 191 183 175 167 159 151 143 135
239 231 223 215 207 199 191 183 175 167 159 151 143 135 127 119
223 215 207 199 191 183 175 167 159 151 143 135 127 119 111 103
207 199 191 183 175 167 159 151 143 135 127 119 111 103  95  87
```

---

## Experiments

### Experiment 1: Warp alignment
In `index_1d.cu`, change the block size to 17. What happens? How many threads are "wasted" per block? (Hint: warps are 32 threads.)

### Experiment 2: Non-square blocks
In `index_2d.cu`, try block sizes of `dim3(32, 1)`, `dim3(16, 2)`, `dim3(8, 4)`, `dim3(4, 8)`, `dim3(1, 32)`. All have 32 threads. How does the thread-to-data mapping change?

### Experiment 3: Exceeding limits
Try launching a kernel with `dim3(1024, 1024)` threads per block (1M threads per block). What error do you get? What's the actual limit?

### Experiment 4: gridDim and blockDim inside a kernel
Add a kernel that prints `gridDim.x`, `gridDim.y`, `blockDim.x`, `blockDim.y` from thread (0,0) only. Verify these match your launch configuration.

### Experiment 5: Non-divisible dimensions
Use a 7×5 matrix with block size (4, 4). How many total threads are launched? How many actually do work? What percentage is "waste"?

---

## What Just Happened?

1. **Thread indexing is just arithmetic.** The global thread ID is computed from block index, block size, and thread index. There's no magic — it's the same as computing array indices from row/column numbers.

2. **`dim3` is syntactic sugar.** When you write `<<<4, 256>>>`, CUDA interprets these as `dim3(4, 1, 1)` and `dim3(256, 1, 1)`. You only need 2D or 3D when your data is naturally multi-dimensional.

3. **You must handle edge cases.** When your data size isn't a perfect multiple of your block size, you'll launch extra threads. The `if (i < n)` bounds check is not optional — without it, you get out-of-bounds memory access.

4. **Block size affects performance** but doesn't change correctness (as long as you have enough threads). The optimal block size depends on the GPU architecture — 128 or 256 is usually a safe starting point.

---

## Key Insight

> **Every thread computes its own unique global index from `blockIdx`, `blockDim`, and `threadIdx`.** This index is how you map threads to data. Get this formula wrong and your kernel produces garbage. Get it right and the rest is just writing the computation.

---

## Checkpoint Quiz

**Q1:** You have a 1000×1000 matrix and use block size `dim3(16, 16)`. How many blocks do you need in each dimension?
<details><summary>Answer</summary>
`gridDim.x = (1000 + 16 - 1) / 16 = 63` blocks in x.
`gridDim.y = (1000 + 16 - 1) / 16 = 63` blocks in y.
Total: 63 × 63 = 3,969 blocks. Total threads: 63×16 × 63×16 = 1,008 × 1,008 = 1,016,064. Of those, only 1,000,000 do useful work. The extra 16,064 threads (~1.6%) are masked by bounds checks.
</details>

**Q2:** What is the maximum number of threads per block on modern NVIDIA GPUs?
<details><summary>Answer</summary>
1024 threads per block. This is a hardware limit. You can distribute them in any shape (e.g., 1024×1×1 or 32×32×1 or 16×16×4) as long as the product doesn't exceed 1024.
</details>

**Q3:** You launch `kernel<<<dim3(2,3), dim3(4,5)>>>()`. What are `gridDim`, `blockDim`, and how many total threads are launched?
<details><summary>Answer</summary>
`gridDim = (2, 3, 1)`, `blockDim = (4, 5, 1)`. Total threads = 2×3 blocks × 4×5 threads/block = 6 × 20 = 120 threads.
</details>

---

*Next Lab: [Lab 03 — Memory Transfer Costs](Lab03_Memory_Transfer_Costs.md)*
