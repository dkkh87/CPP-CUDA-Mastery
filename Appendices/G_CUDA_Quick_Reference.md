# Appendix G — CUDA Quick Reference

> **Print-friendly, one-stop cheat sheet for every essential CUDA API and concept.**
> Pin it to your wall. Keep it open while coding. Never Google the same thing twice.

---

## Table of Contents

| # | Section | Page |
|---|---------|------|
| 1 | [Kernel Launch Syntax](#1-kernel-launch-syntax) | — |
| 2 | [Thread Hierarchy](#2-thread-hierarchy-quick-reference) | — |
| 3 | [Memory Management API](#3-memory-management-api) | — |
| 4 | [Memory Types](#4-memory-types-quick-reference) | — |
| 5 | [Synchronization API](#5-synchronization-api) | — |
| 6 | [Warp Intrinsics](#6-warp-intrinsics) | — |
| 7 | [Stream API](#7-stream-api) | — |
| 8 | [Event API](#8-event-api) | — |
| 9 | [Error Handling](#9-error-handling) | — |
| 10 | [Device Query](#10-device-query) | — |
| 11 | [Compilation Quick Reference](#11-compilation-quick-reference) | — |
| 12 | [Common Patterns Cheat Sheet](#12-common-patterns-cheat-sheet) | — |
| 13 | [Performance Numbers at a Glance](#13-performance-numbers-at-a-glance) | — |
| 14 | [Common Error Messages & Fixes](#14-common-error-messages--fixes) | — |

---

## 1. Kernel Launch Syntax

### Declaration Qualifiers

| Qualifier | Called From | Runs On |
|-----------|-----------|---------|
| `__global__` | Host (or device with dynamic parallelism) | Device |
| `__device__` | Device only | Device |
| `__host__` | Host only | Host |
| `__host__ __device__` | Both | Both (compiled twice) |

### Launch Configuration

```cuda
__global__ void myKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}

// Launch syntax: kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args)
int N = 1000000;
int blockSize = 256;
int gridSize  = (N + blockSize - 1) / blockSize;  // = 3907

myKernel<<<gridSize, blockSize>>>(d_data, N);
myKernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);  // with stream
```

### Thread Indexing Formulas

**1D Grid of 1D Blocks (most common)**
```
globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
totalThreads = gridDim.x * blockDim.x;
```
*Example:* Grid = 4 blocks, Block = 256 threads → thread (block=2, tid=100) → globalIdx = 2×256 + 100 = **612**

**2D Grid of 2D Blocks (images, matrices)**
```
col = blockIdx.x * blockDim.x + threadIdx.x;
row = blockIdx.y * blockDim.y + threadIdx.y;
linearIdx = row * width + col;
```
*Example:* 1920×1080 image, blocks of 16×16 → grid = (120, 68) → pixel at block(5,3), thread(7,11) → col = 5×16+7 = **87**, row = 3×16+11 = **59**

**3D Grid of 3D Blocks (volumes, physics)**
```
x = blockIdx.x * blockDim.x + threadIdx.x;
y = blockIdx.y * blockDim.y + threadIdx.y;
z = blockIdx.z * blockDim.z + threadIdx.z;
linearIdx = z * (width * height) + y * width + x;
```

### Grid-Stride Loop Pattern

Handles arrays larger than total thread count; enables kernel reuse with any grid size:

```cuda
__global__ void processArray(float* data, int N) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        data[i] = sqrtf(data[i]);
    }
}

// Can launch with any grid size — the loop handles the rest
processArray<<<128, 256>>>(d_data, 10000000);  // 128×256 = 32768 threads process 10M elements
```

---

## 2. Thread Hierarchy Quick Reference

### Built-in Variables

| Variable | Type | Meaning | Range |
|----------|------|---------|-------|
| `threadIdx.x/y/z` | `uint3` | Thread index within block | 0 … blockDim-1 |
| `blockIdx.x/y/z` | `uint3` | Block index within grid | 0 … gridDim-1 |
| `blockDim.x/y/z` | `dim3` | Block dimensions (threads per block) | Set at launch |
| `gridDim.x/y/z` | `dim3` | Grid dimensions (blocks per grid) | Set at launch |
| `warpSize` | `int` | Threads per warp | Always **32** |

### Hardware Limits

| Dimension/Resource | Maximum Value |
|--------------------|---------------|
| Threads per block (total) | **1024** |
| Block dim x | 1024 |
| Block dim y | 1024 |
| Block dim z | 64 |
| Grid dim x | 2³¹ − 1 (2,147,483,647) |
| Grid dim y | 65,535 |
| Grid dim z | 65,535 |
| Warp size | 32 (fixed, all architectures) |
| Warps per SM | 32–64 (arch-dependent) |
| Blocks per SM | 16–32 (arch-dependent) |

### dim3 Usage

```cuda
dim3 blockSize(16, 16);       // 256 threads per block (2D)
dim3 gridSize(
    (width  + 15) / 16,
    (height + 15) / 16
);
kernel<<<gridSize, blockSize>>>(args);

dim3 blockSize3D(8, 8, 8);    // 512 threads per block (3D)
```

### Visual Hierarchy

```
Grid
├── Block (0,0)          ← blockIdx
│   ├── Thread (0,0)     ← threadIdx
│   ├── Thread (1,0)
│   ├── ...
│   └── Thread (15,15)
├── Block (1,0)
│   ├── Warp 0: threads 0–31    ← SIMT execution unit
│   ├── Warp 1: threads 32–63
│   └── ...
└── Block (gridDim.x-1, gridDim.y-1)
```

---

## 3. Memory Management API

### Allocation & Deallocation

| Function | What It Does | Signature |
|----------|-------------|-----------|
| `cudaMalloc` | Allocate device memory | `cudaMalloc(void** ptr, size_t size)` |
| `cudaFree` | Free device memory | `cudaFree(void* ptr)` |
| `cudaMallocHost` | Allocate pinned host memory (page-locked) | `cudaMallocHost(void** ptr, size_t size)` |
| `cudaFreeHost` | Free pinned host memory | `cudaFreeHost(void* ptr)` |
| `cudaMallocManaged` | Allocate Unified Memory (auto-migrating) | `cudaMallocManaged(void** ptr, size_t size)` |
| `cudaMallocPitch` | Allocate 2D device memory (aligned rows) | `cudaMallocPitch(void** ptr, size_t* pitch, size_t w, size_t h)` |
| `cudaMalloc3D` | Allocate 3D device memory | `cudaMalloc3D(cudaPitchedPtr* ptr, cudaExtent extent)` |

### Data Transfer

| Function | What It Does | Usage Pattern |
|----------|-------------|---------------|
| `cudaMemcpy` | Synchronous copy (blocks host) | `cudaMemcpy(dst, src, size, kind)` |
| `cudaMemcpyAsync` | Asynchronous copy (non-blocking) | `cudaMemcpyAsync(dst, src, size, kind, stream)` |
| `cudaMemset` | Fill device memory with byte value | `cudaMemset(ptr, value, size)` |
| `cudaMemsetAsync` | Async fill device memory | `cudaMemsetAsync(ptr, value, size, stream)` |
| `cudaMemcpy2D` | Copy 2D pitched memory | `cudaMemcpy2D(dst, dpitch, src, spitch, w, h, kind)` |

### cudaMemcpy Direction Constants

| Constant | Direction |
|----------|-----------|
| `cudaMemcpyHostToDevice` | Host → Device (H2D) |
| `cudaMemcpyDeviceToHost` | Device → Host (D2H) |
| `cudaMemcpyDeviceToDevice` | Device → Device (D2D) |
| `cudaMemcpyHostToHost` | Host → Host |
| `cudaMemcpyDefault` | Auto-detect (Unified Memory) |

### Unified Memory Hints

| Function | What It Does | Usage |
|----------|-------------|-------|
| `cudaMemPrefetchAsync` | Prefetch pages to device/host | `cudaMemPrefetchAsync(ptr, size, deviceId, stream)` |
| `cudaMemAdvise` | Hint access pattern to driver | `cudaMemAdvise(ptr, size, advice, device)` |

**Advice flags for `cudaMemAdvise`:**
| Flag | Meaning |
|------|---------|
| `cudaMemAdviseSetReadMostly` | Data mostly read — replicate across GPUs |
| `cudaMemAdviseSetPreferredLocation` | Prefer this device for residency |
| `cudaMemAdviseSetAccessedBy` | Create direct mapping for this device |

### Typical Workflow

```cuda
float *d_data;
size_t bytes = N * sizeof(float);

cudaMalloc(&d_data, bytes);                              // 1. Allocate
cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice); // 2. Upload
myKernel<<<grid, block>>>(d_data, N);                    // 3. Compute
cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost); // 4. Download
cudaFree(d_data);                                        // 5. Free
```

---

## 4. Memory Types Quick Reference

| Type | Scope | Lifetime | Speed | Declaration | Typical Size |
|------|-------|----------|-------|-------------|-------------|
| **Registers** | Thread | Thread | ★★★★★ Fastest | Automatic variables | 255 per thread (32-bit) |
| **Local** | Thread | Thread | ★★ (spills to DRAM) | Compiler-managed spill | Up to stack limit |
| **Shared** | Block | Block | ★★★★ ~100× faster than global | `__shared__ float s[256];` | 48–228 KB per SM |
| **Global** | Grid + Host | Application | ★★ Slowest on-chip path | `cudaMalloc` / `__device__` | Device VRAM (GBs) |
| **Constant** | Grid (read-only) | Application | ★★★★ Cached, broadcast | `__constant__ float c[1024];` | 64 KB total |
| **Texture** | Grid (read-only) | Application | ★★★★ Cached, spatial locality | Texture objects / `tex2D` | Device VRAM |

### Memory Qualifiers

```cuda
__device__   float globalVar;            // Global memory (device-lifetime)
__constant__ float constData[1024];      // Constant memory (64 KB max)
__shared__   float tile[32][32];         // Shared memory (per-block)

// Dynamic shared memory (size set at launch)
extern __shared__ float dynamicShared[];
kernel<<<grid, block, sharedBytes>>>(args);
```

### Shared Memory Bank Conflicts

- **32 banks**, 4 bytes per bank, successive 4-byte words map to successive banks
- **Conflict:** two threads in same warp access different addresses in same bank → serialized
- **Broadcast:** multiple threads read same address in same bank → no conflict
- **Padding trick:** `__shared__ float tile[32][33];` adds 1 padding column to avoid conflicts

---

## 5. Synchronization API

### Block-Level Synchronization

| Function | What It Does | Where |
|----------|-------------|-------|
| `__syncthreads()` | Barrier for all threads in a block | Device code only |
| `__syncthreads_count(pred)` | Barrier + count threads where pred ≠ 0 | Device code only |
| `__syncthreads_and(pred)` | Barrier + returns non-zero if all preds non-zero | Device code only |
| `__syncthreads_or(pred)` | Barrier + returns non-zero if any pred non-zero | Device code only |

### Warp-Level Synchronization

| Function | What It Does |
|----------|-------------|
| `__syncwarp(mask)` | Synchronize threads in warp specified by mask |

### Host-Level Synchronization

| Function | What It Does |
|----------|-------------|
| `cudaDeviceSynchronize()` | Block host until ALL device work completes |
| `cudaStreamSynchronize(stream)` | Block host until all work in stream completes |
| `cudaEventSynchronize(event)` | Block host until event is recorded |
| `cudaStreamWaitEvent(stream, event)` | Make stream wait for event (device-side) |

### Atomic Operations

All atomics return the **old** value at the target address.

| Function | Operation | Supported Types |
|----------|-----------|-----------------|
| `atomicAdd(addr, val)` | `*addr += val` | int, unsigned, float, double (cc≥6.0) |
| `atomicSub(addr, val)` | `*addr -= val` | int, unsigned |
| `atomicMin(addr, val)` | `*addr = min(*addr, val)` | int, unsigned |
| `atomicMax(addr, val)` | `*addr = max(*addr, val)` | int, unsigned |
| `atomicExch(addr, val)` | `*addr = val` | int, unsigned, float |
| `atomicInc(addr, val)` | `*addr = (*addr >= val) ? 0 : *addr+1` | unsigned |
| `atomicDec(addr, val)` | `*addr = (*addr==0 \|\| *addr>val) ? val : *addr-1` | unsigned |
| `atomicCAS(addr, compare, val)` | `if (*addr == compare) *addr = val` | int, unsigned, unsigned long long |
| `atomicAnd(addr, val)` | `*addr &= val` | int, unsigned |
| `atomicOr(addr, val)` | `*addr \|= val` | int, unsigned |
| `atomicXor(addr, val)` | `*addr ^= val` | int, unsigned |

### Memory Fences

| Function | Scope |
|----------|-------|
| `__threadfence_block()` | Writes visible to all threads in block |
| `__threadfence()` | Writes visible to all threads in device |
| `__threadfence_system()` | Writes visible to all threads + host |

---

## 6. Warp Intrinsics

> All warp intrinsics require compute capability ≥ 7.0 (Volta+).
> The `mask` parameter specifies participating threads. Use `0xFFFFFFFF` for full warp.

### Shuffle Operations

Exchange data between threads in a warp **without shared memory**.

| Function | What It Does | Example Use |
|----------|-------------|-------------|
| `__shfl_sync(mask, val, srcLane)` | Get `val` from thread `srcLane` | Broadcast one value |
| `__shfl_up_sync(mask, val, delta)` | Get `val` from thread `laneId - delta` | Inclusive prefix scan |
| `__shfl_down_sync(mask, val, delta)` | Get `val` from thread `laneId + delta` | Warp reduction |
| `__shfl_xor_sync(mask, val, laneMask)` | Get `val` from thread `laneId ^ laneMask` | Butterfly reduction |

```cuda
// Warp reduction using __shfl_down_sync
float val = threadData;
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
// After loop: thread 0 holds the sum of all 32 threads
```

### Vote Operations

| Function | What It Does | Return |
|----------|-------------|--------|
| `__ballot_sync(mask, pred)` | Each bit = whether that thread's pred ≠ 0 | `unsigned int` bitmask |
| `__any_sync(mask, pred)` | Is any thread's pred ≠ 0? | `int` (0 or 1) |
| `__all_sync(mask, pred)` | Are all threads' preds ≠ 0? | `int` (0 or 1) |
| `__activemask()` | Bitmask of currently active threads | `unsigned int` |

```cuda
unsigned mask = __ballot_sync(0xFFFFFFFF, data[tid] > threshold);
int count = __popc(mask);  // population count — number of threads that matched
```

### Match Operations (cc ≥ 7.0)

| Function | What It Does |
|----------|-------------|
| `__match_any_sync(mask, val)` | Bitmask of threads with same `val` |
| `__match_all_sync(mask, val, &pred)` | Bitmask if ALL have same `val`; sets `pred` |

---

## 7. Stream API

Streams enable **concurrent execution** of kernels, memcpy, and other operations.

| Function | What It Does |
|----------|-------------|
| `cudaStreamCreate(&stream)` | Create a new stream |
| `cudaStreamCreateWithFlags(&stream, flags)` | Create with flags (`cudaStreamNonBlocking`) |
| `cudaStreamCreateWithPriority(&stream, flags, priority)` | Create with scheduling priority |
| `cudaStreamDestroy(stream)` | Destroy a stream |
| `cudaStreamSynchronize(stream)` | Block host until stream completes |
| `cudaStreamQuery(stream)` | Check if stream is done (non-blocking) |
| `cudaStreamWaitEvent(stream, event, flags)` | Insert dependency on event |
| `cudaStreamGetPriority(stream, &priority)` | Get stream priority |
| `cudaDeviceGetStreamPriorityRange(&lo, &hi)` | Get valid priority range |

### Stream Usage Pattern

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Pipeline: overlap compute on s1 with transfer on s2
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, s1);
kernel<<<grid, block, 0, s1>>>(d_a);
cudaMemcpyAsync(h_a, d_a, size, cudaMemcpyDeviceToHost, s1);

cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, s2);
kernel<<<grid, block, 0, s2>>>(d_b);
cudaMemcpyAsync(h_b, d_b, size, cudaMemcpyDeviceToHost, s2);

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);
cudaStreamDestroy(s1);
cudaStreamDestroy(s2);
```

> **Key rule:** Use `cudaMallocHost` (pinned memory) for async transfers — pageable memory silently falls back to synchronous.

---

## 8. Event API

Events are lightweight GPU timestamps used for **timing** and **inter-stream synchronization**.

| Function | What It Does |
|----------|-------------|
| `cudaEventCreate(&event)` | Create an event |
| `cudaEventCreateWithFlags(&event, flags)` | Create with flags (e.g., `cudaEventDisableTiming`) |
| `cudaEventRecord(event, stream)` | Record event into stream's work queue |
| `cudaEventSynchronize(event)` | Block host until event completes |
| `cudaEventQuery(event)` | Check if event completed (non-blocking) |
| `cudaEventElapsedTime(&ms, start, stop)` | Milliseconds between two events |
| `cudaEventDestroy(event)` | Destroy an event |

### Timing Pattern

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel took %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### Inter-Stream Dependency

```cuda
cudaEventRecord(event, stream1);       // Event placed in stream1
cudaStreamWaitEvent(stream2, event);   // stream2 waits for event
// stream2 work only starts after stream1 reaches the event
```

---

## 9. Error Handling

### Error Functions

| Function | What It Does |
|----------|-------------|
| `cudaGetLastError()` | Return last error and **reset** it to `cudaSuccess` |
| `cudaPeekAtLastError()` | Return last error **without** resetting |
| `cudaGetErrorString(err)` | Human-readable error description |
| `cudaGetErrorName(err)` | Error enum name as string |

### CUDA_CHECK Macro (Complete Implementation)

```cuda
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s: %s\n",             \
                    __FILE__, __LINE__,                                    \
                    cudaGetErrorName(err), cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, bytes));
CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

// For kernel launches (error is asynchronous)
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());          // Check launch config errors
CUDA_CHECK(cudaDeviceSynchronize());     // Check execution errors
```

### Error Handling Best Practices

```
✓ Check EVERY CUDA API call in development
✓ Check cudaGetLastError() after every kernel launch
✓ Use cudaDeviceSynchronize() after kernels during debugging
✗ Don't call cudaDeviceSynchronize() in production hot paths (kills concurrency)
```

---

## 10. Device Query

### Device Management Functions

| Function | What It Does |
|----------|-------------|
| `cudaGetDeviceCount(&count)` | Number of CUDA-capable GPUs |
| `cudaGetDevice(&deviceId)` | Get current device ID |
| `cudaSetDevice(deviceId)` | Set active GPU for subsequent calls |
| `cudaGetDeviceProperties(&props, deviceId)` | Query full device properties |
| `cudaDeviceReset()` | Reset device and release resources |
| `cudaDeviceGetAttribute(&val, attr, dev)` | Query single attribute |

### Key cudaDeviceProp Fields

| Property | Type | Meaning |
|----------|------|---------|
| `name` | `char[256]` | GPU name (e.g., "NVIDIA A100") |
| `totalGlobalMem` | `size_t` | Total VRAM in bytes |
| `sharedMemPerBlock` | `size_t` | Max shared memory per block |
| `sharedMemPerMultiprocessor` | `size_t` | Max shared memory per SM |
| `regsPerBlock` | `int` | 32-bit registers per block |
| `warpSize` | `int` | Threads per warp (always 32) |
| `maxThreadsPerBlock` | `int` | Max threads per block (1024) |
| `maxThreadsDim[3]` | `int[3]` | Max block dimensions |
| `maxGridSize[3]` | `int[3]` | Max grid dimensions |
| `multiProcessorCount` | `int` | Number of SMs |
| `clockRate` | `int` | Core clock in kHz |
| `memoryClockRate` | `int` | Memory clock in kHz |
| `memoryBusWidth` | `int` | Memory bus width in bits |
| `major` / `minor` | `int` | Compute capability |
| `concurrentKernels` | `int` | Concurrent kernel support |
| `asyncEngineCount` | `int` | Number of copy engines |
| `managedMemory` | `int` | Unified Memory support |

### Device Query Snippet

```cuda
int devCount;
cudaGetDeviceCount(&devCount);
for (int d = 0; d < devCount; d++) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, d);
    printf("GPU %d: %s (cc %d.%d) — %d SMs, %.1f GB VRAM\n",
           d, p.name, p.major, p.minor,
           p.multiProcessorCount,
           p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}
```

---

## 11. Compilation Quick Reference

### nvcc Essential Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-arch=sm_XX` | Target virtual architecture | `-arch=sm_80` |
| `-code=sm_XX` | Target real architecture (SASS) | `-code=sm_80` |
| `-gencode` | Multi-arch fat binary | See below |
| `-O2` / `-O3` | Optimization level | `-O2` (default for release) |
| `-G` | Device debug info (disables optimizations) | Debug builds only |
| `-g` | Host debug info | `-g` |
| `-lineinfo` | Line info without disabling optimizations | Production profiling |
| `--ptxas-options=-v` | Show register/shared memory usage | Always useful |
| `-Xcompiler` | Pass flag to host compiler | `-Xcompiler -Wall` |
| `-std=c++17` | C++ standard | `-std=c++17` |
| `--expt-relaxed-constexpr` | Allow constexpr in device code | Templates |
| `--extended-lambda` | Allow `__device__` lambdas | Modern CUDA |
| `-maxrregcount=N` | Limit registers per thread | Increase occupancy |
| `-use_fast_math` | Fast (less precise) math intrinsics | Performance |

### Multi-Architecture Compilation

```bash
# Fat binary targeting V100 + A100 + H100
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     -o my_program my_program.cu

# Last line with code=compute_XX includes PTX for forward compatibility
```

### Compute Capability Reference

| GPU | Compute Capability | Architecture |
|-----|-------------------|--------------|
| GTX 1080 Ti | 6.1 | Pascal |
| Titan V | 7.0 | Volta |
| RTX 2080 Ti | 7.5 | Turing |
| A100 | 8.0 | Ampere |
| RTX 3090 | 8.6 | Ampere |
| RTX 4090 | 8.9 | Ada Lovelace |
| H100 | 9.0 | Hopper |
| B200 | 10.0 | Blackwell |

### Build Examples

```bash
# Debug build
nvcc -G -g -lineinfo -arch=sm_80 -o debug_prog program.cu

# Release build with register info
nvcc -O2 -arch=sm_80 --ptxas-options=-v -o program program.cu

# Separate compilation (multi-file projects)
nvcc -dc -arch=sm_80 file1.cu -o file1.o
nvcc -dc -arch=sm_80 file2.cu -o file2.o
nvcc -arch=sm_80 file1.o file2.o -o program
```

---

## 12. Common Patterns Cheat Sheet

### Grid-Stride Loop

```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    output[i] = process(input[i]);
```

### 2D Indexing (Images/Matrices)

```cuda
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (col < width && row < height)
    output[row * width + col] = input[row * width + col] * 2.0f;
```

### Warp Reduction (Sum)

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;  // result valid in lane 0
}
```

### Block Reduction via Shared Memory

```cuda
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // one slot per warp
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;  // result valid in thread 0
}
```

### Shared Memory Tiled Matrix Multiply

```cuda
__global__ void matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

### Bounds-Checked Error Wrap

```cuda
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 13. Performance Numbers at a Glance

### Memory Hierarchy Bandwidth & Latency

| Memory Level | Bandwidth | Latency (cycles) | Cached? |
|-------------|-----------|-------------------|---------|
| Registers | ~20 TB/s (per SM) | 0–1 | N/A |
| Shared Memory | ~19 TB/s (per SM) | ~20–30 | N/A |
| L1 Cache | ~19 TB/s (per SM) | ~30 | Yes |
| L2 Cache | ~6 TB/s (A100) | ~200 | Yes |
| HBM (Global) | ~2 TB/s (A100) | ~400–800 | Via L1/L2 |
| PCIe 4.0 x16 | ~32 GB/s | 1000s+ | No |
| PCIe 5.0 x16 | ~64 GB/s | 1000s+ | No |
| NVLink 4.0 | ~900 GB/s (total) | ~hundreds | No |

### Key GPU Specifications

| Spec | A100 (80 GB) | H100 (80 GB) | B200 (192 GB) |
|------|-------------|--------------|---------------|
| Architecture | Ampere | Hopper | Blackwell |
| Compute Capability | 8.0 | 9.0 | 10.0 |
| SMs | 108 | 132 | 192 |
| FP32 Cores | 6912 | 16896 | 18432 |
| Tensor Cores | 432 (3rd gen) | 528 (4th gen) | 768 (5th gen) |
| HBM Bandwidth | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| HBM Capacity | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| FP16 Tensor TFLOPS | 312 | 990 | 2250 |
| FP32 TFLOPS | 19.5 | 67 | 90 |
| TDP | 300W | 700W | 1000W |
| L2 Cache | 40 MB | 50 MB | 96 MB |
| Shared Mem / SM | 164 KB | 228 KB | 228 KB |
| NVLink | 600 GB/s | 900 GB/s | 1800 GB/s |

### Rules of Thumb

```
► GPU arithmetic: ~100 TFLOPS (FP32) — usually NOT the bottleneck
► HBM bandwidth: ~2–8 TB/s — this IS usually the bottleneck
► Arithmetic intensity = FLOPs / Bytes loaded → aim for > 10
► Occupancy sweet spot: 50–75% is usually sufficient
► Shared memory: use it when data is reused ≥ 2× within a block
► Warp divergence: branches with < 32 threads cost ~2× wallclock
► Coalesced access: consecutive threads → consecutive addresses → 1 transaction
► Uncoalesced access: random pattern → up to 32× more transactions
```

---

## 14. Common Error Messages & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `cudaErrorInvalidConfiguration` | Launch config exceeds device limits (e.g., >1024 threads/block) | Reduce block size; check `maxThreadsPerBlock` |
| `cudaErrorInvalidValue` | Bad argument (NULL pointer, negative size, invalid direction) | Validate all arguments; ensure prior `cudaMalloc` succeeded |
| `cudaErrorMemoryAllocation` / `cudaErrorOutOfMemory` | Device out of VRAM | Reduce allocation sizes; free unused memory; check for leaks |
| `cudaErrorIllegalAddress` | Kernel accessed invalid memory (out-of-bounds or freed) | Add bounds checks; verify array sizes; run with `compute-sanitizer` |
| `cudaErrorLaunchTimeout` | Kernel ran too long on display GPU (WDDM timeout) | Shorten kernel; use TCC mode; or use a non-display GPU |
| `cudaErrorInvalidDeviceFunction` | Kernel compiled for wrong compute capability | Recompile with correct `-arch=sm_XX` matching your GPU |
| `cudaErrorNoDevice` | No CUDA GPU found | Verify GPU installed; check driver with `nvidia-smi` |
| `cudaErrorInsufficientDriver` | Driver too old for CUDA toolkit version | Update NVIDIA driver to match toolkit |
| `cudaErrorMisalignedAddress` | Unaligned memory access in kernel | Ensure pointers are aligned; avoid casts that break alignment |
| `cudaErrorAssert` | Device-side `assert()` failed | Debug kernel logic; check index calculations |

### Debugging Toolkit

```bash
# Memory error detection
compute-sanitizer ./my_program

# Race condition detection
compute-sanitizer --tool racecheck ./my_program

# Memory leak detection
compute-sanitizer --tool memcheck --leak-check full ./my_program

# Profiling
nsys profile ./my_program                # System-wide timeline
ncu --set full ./my_program              # Kernel-level analysis
```

---

## Quick Index — "How Do I …?"

| Task | Go To |
|------|-------|
| Launch a kernel? | [§1](#1-kernel-launch-syntax) |
| Calculate thread ID? | [§1 Thread Indexing](#thread-indexing-formulas) |
| Allocate GPU memory? | [§3](#3-memory-management-api) |
| Copy data to/from GPU? | [§3 Data Transfer](#data-transfer) |
| Use shared memory? | [§4](#4-memory-types-quick-reference) |
| Synchronize threads? | [§5](#5-synchronization-api) |
| Do a warp reduction? | [§6](#6-warp-intrinsics) |
| Overlap compute & transfer? | [§7](#7-stream-api) |
| Time a kernel? | [§8](#8-event-api) |
| Handle errors properly? | [§9](#9-error-handling) |
| Query GPU specs? | [§10](#10-device-query) |
| Compile for multiple GPUs? | [§11](#11-compilation-quick-reference) |
| Write a tiled matrix multiply? | [§12](#12-common-patterns-cheat-sheet) |
| Figure out if I'm memory-bound? | [§13](#13-performance-numbers-at-a-glance) |
| Debug a crash? | [§14](#14-common-error-messages--fixes) |

---

*Appendix G — CUDA Quick Reference • CPP-CUDA-Mastery*
