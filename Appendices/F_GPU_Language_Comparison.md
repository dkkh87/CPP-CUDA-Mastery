# Appendix F — From CUDA to Other GPU Languages

> Side-by-side comparison of GPU programming models with code examples.
> Covers CUDA, OpenCL, HIP, SYCL, Metal, Vulkan Compute, and Triton.

---

## 1. CUDA vs OpenCL

### Overview

| Aspect          | CUDA                          | OpenCL                           |
|-----------------|-------------------------------|----------------------------------|
| **Vendor**      | NVIDIA only                   | Khronos (multi-vendor)           |
| **Hardware**    | NVIDIA GPUs                   | Any GPU, CPU, FPGA, DSP         |
| **Language**    | C/C++ extensions              | C99 kernel language + host API   |
| **Compiler**    | nvcc (NVIDIA)                 | Runtime compilation (ICD)        |
| **Ecosystem**   | Rich (cuBLAS, cuDNN, etc.)    | Limited libraries                |
| **Performance** | Best on NVIDIA                | 5-15% slower on NVIDIA typically |
| **Learning**    | Easier (simpler API)          | Harder (verbose setup)           |

### Vector Add Comparison

**CUDA:**
```cuda
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Launch
vecAdd<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);
```

**OpenCL:**
```c
// Kernel (string or .cl file)
const char* kernelSrc = R"(
__kernel void vecAdd(__global const float* A,
                     __global const float* B,
                     __global float* C, int N) {
    int i = get_global_id(0);
    if (i < N) C[i] = A[i] + B[i];
}
)";

// Host code (significantly more verbose)
cl_platform_id platform;
clGetPlatformIDs(1, &platform, NULL);
cl_device_id device;
clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, NULL);

cl_program program = clCreateProgramWithSource(ctx, 1, &kernelSrc, NULL, NULL);
clBuildProgram(program, 1, &device, NULL, NULL, NULL);
cl_kernel kernel = clCreateKernel(program, "vecAdd", NULL);

cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY, N*sizeof(float), NULL, NULL);
// ... create d_B, d_C similarly

clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, N*sizeof(float), h_A, 0, NULL, NULL);
clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
// ... set other args

size_t globalSize = ((N + 255) / 256) * 256;
size_t localSize = 256;
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, N*sizeof(float), h_C, 0, NULL, NULL);
```

**Verdict:** CUDA is dramatically simpler. OpenCL provides portability at the cost of verbosity. OpenCL has largely been superseded by SYCL/Vulkan Compute for new projects.

---

## 2. CUDA vs HIP (AMD)

### Overview

| Aspect          | CUDA                          | HIP                              |
|-----------------|-------------------------------|----------------------------------|
| **Vendor**      | NVIDIA                        | AMD (also runs on NVIDIA)        |
| **Hardware**    | NVIDIA GPUs                   | AMD + NVIDIA GPUs                |
| **Language**    | CUDA C++                      | Nearly identical to CUDA         |
| **Compiler**    | nvcc                          | hipcc (wraps clang or nvcc)      |
| **Conversion**  | —                             | hipify-clang (automated)         |
| **Performance** | Native                        | Native on AMD, ~same on NVIDIA   |

### hipify Conversion Tool

```bash
# Automated conversion
hipify-clang cuda_source.cu -o hip_source.cpp

# Perl-based (simpler, less accurate)
hipify-perl cuda_source.cu > hip_source.cpp

# Check what would change
hipify-clang --print-stats cuda_source.cu
```

### Vector Add Comparison

**CUDA:**
```cuda
#include <cuda_runtime.h>

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    vecAdd<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

**HIP (nearly identical):**
```cpp
#include <hip/hip_runtime.h>

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));
    hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, N * sizeof(float), hipMemcpyHostToDevice);
    vecAdd<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);
    hipMemcpy(h_C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
}
```

### API Mapping Table

| CUDA                        | HIP                           | Notes                    |
|-----------------------------|-------------------------------|--------------------------|
| `cudaMalloc`                | `hipMalloc`                   | Identical semantics      |
| `cudaFree`                  | `hipFree`                     |                          |
| `cudaMemcpy`                | `hipMemcpy`                   |                          |
| `cudaMemcpyAsync`           | `hipMemcpyAsync`              |                          |
| `cudaMallocHost`            | `hipHostMalloc`               | Different name           |
| `cudaDeviceSynchronize`     | `hipDeviceSynchronize`        |                          |
| `cudaGetDeviceCount`        | `hipGetDeviceCount`           |                          |
| `cudaSetDevice`             | `hipSetDevice`                |                          |
| `cudaStreamCreate`          | `hipStreamCreate`             |                          |
| `cudaEventCreate`           | `hipEventCreate`              |                          |
| `cudaEventRecord`           | `hipEventRecord`              |                          |
| `cudaEventSynchronize`      | `hipEventSynchronize`         |                          |
| `cudaGetLastError`          | `hipGetLastError`             |                          |
| `cudaMemcpyHostToDevice`    | `hipMemcpyHostToDevice`       |                          |
| `cudaMemcpyDeviceToHost`    | `hipMemcpyDeviceToHost`       |                          |
| `__shared__`                | `__shared__`                  | Same keyword             |
| `__syncthreads()`           | `__syncthreads()`             | Same function            |
| `atomicAdd`                 | `atomicAdd`                   | Same function            |
| `__shfl_down_sync`          | `__shfl_down`                 | Different API (no mask)  |
| `cooperative_groups`        | Partial support               | Incomplete on HIP        |
| `cublas*`                   | `hipblas*` / `rocblas*`       | Different naming         |
| `cudnn*`                    | `miopen*`                     | Different API            |
| `nccl*`                     | `rccl*`                       | Compatible API           |

### Portability Guide

```cpp
// Write portable code using preprocessor
#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
#else
    #include <cuda_runtime.h>
#endif

// Or use HIP everywhere (it maps to CUDA on NVIDIA)
// Compile with: hipcc --platform nvidia (NVIDIA) or hipcc (AMD)
```

**Verdict:** HIP is the best path for CUDA portability. ~95% of CUDA code converts automatically. Remaining 5% requires manual work (warp intrinsics, inline PTX, vendor-specific libraries).

---

## 3. CUDA vs SYCL (Intel DPC++)

### Overview

| Aspect          | CUDA                          | SYCL / DPC++                     |
|-----------------|-------------------------------|----------------------------------|
| **Standard**    | Proprietary                   | Khronos open standard            |
| **Vendor**      | NVIDIA                        | Intel (primary), multi-vendor    |
| **Language**    | C++ extensions (`.cu`)        | Standard C++ (single-source)     |
| **Compiler**    | nvcc                          | icpx, clang with SYCL support   |
| **Memory**      | Explicit + Unified Memory     | Buffers/Accessors + USM          |
| **Hardware**    | NVIDIA GPUs                   | Intel, NVIDIA, AMD (with plugins)|

### Vector Add Comparison

**CUDA:**
```cuda
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

**SYCL (Buffer/Accessor model):**
```cpp
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q{sycl::gpu_selector_v};

    sycl::buffer<float> bufA(h_A, N);
    sycl::buffer<float> bufB(h_B, N);
    sycl::buffer<float> bufC(h_C, N);

    q.submit([&](sycl::handler& h) {
        auto A = bufA.get_access<sycl::access::mode::read>(h);
        auto B = bufB.get_access<sycl::access::mode::read>(h);
        auto C = bufC.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            C[i] = A[i] + B[i];
        });
    });
    // Data automatically copied back when buffers go out of scope
}
```

**SYCL (USM — more CUDA-like):**
```cpp
#include <sycl/sycl.hpp>

int main() {
    sycl::queue q{sycl::gpu_selector_v};

    float* d_A = sycl::malloc_device<float>(N, q);
    float* d_B = sycl::malloc_device<float>(N, q);
    float* d_C = sycl::malloc_device<float>(N, q);

    q.memcpy(d_A, h_A, N * sizeof(float));
    q.memcpy(d_B, h_B, N * sizeof(float));
    q.wait();

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_C[i] = d_A[i] + d_B[i];
    }).wait();

    q.memcpy(h_C, d_C, N * sizeof(float));
    q.wait();

    sycl::free(d_A, q); sycl::free(d_B, q); sycl::free(d_C, q);
}
```

### Key Differences

| Feature              | CUDA                     | SYCL                          |
|----------------------|--------------------------|-------------------------------|
| Kernel definition    | `__global__` function    | Lambda in `parallel_for`      |
| Thread indexing      | `threadIdx`, `blockIdx`  | `sycl::id`, `sycl::nd_item`  |
| Shared memory        | `__shared__` keyword     | `sycl::local_accessor`        |
| Synchronization      | `__syncthreads()`        | `group_barrier()`             |
| Atomics              | `atomicAdd`, etc.        | `sycl::atomic_ref`            |
| Streams              | `cudaStream_t`           | `sycl::queue` (in-order/OOO) |
| Error handling       | Error codes              | Exceptions (async handler)    |

### Conversion Tool: SYCLomatic (dpct)

```bash
# Convert CUDA to SYCL
dpct --in-root=cuda_src --out-root=sycl_src

# Generates SYCL code with DPCT helper functions
# Manual cleanup usually needed for optimal code
```

---

## 4. CUDA vs Metal (Apple)

### Overview

| Aspect          | CUDA                          | Metal Compute                    |
|-----------------|-------------------------------|----------------------------------|
| **Vendor**      | NVIDIA                        | Apple                            |
| **Hardware**    | NVIDIA GPUs                   | Apple GPU (M-series, A-series)   |
| **Language**    | CUDA C++                      | Metal Shading Language (MSL)     |
| **Memory**      | Separate CPU/GPU memory       | Unified memory architecture      |
| **Framework**   | CUDA Runtime                  | Metal framework (Objective-C/Swift) |

### Vector Add Comparison

**Metal Shader (MSL):**
```metal
// Compute shader file: vecadd.metal
#include <metal_stdlib>
using namespace metal;

kernel void vecAdd(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C       [[buffer(2)]],
                   constant uint& N      [[buffer(3)]],
                   uint i                [[thread_position_in_grid]]) {
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

**Metal Host Code (Swift):**
```swift
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = device.makeDefaultLibrary()!
let function = library.makeFunction(name: "vecAdd")!
let pipeline = try device.makeComputePipelineState(function: function)

// Create buffers (shared memory — no explicit copy!)
let bufA = device.makeBuffer(bytes: hostA, length: N * 4, options: .storageModeShared)!
let bufB = device.makeBuffer(bytes: hostB, length: N * 4, options: .storageModeShared)!
let bufC = device.makeBuffer(length: N * 4, options: .storageModeShared)!

let cmdBuffer = queue.makeCommandBuffer()!
let encoder = cmdBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(bufA, offset: 0, index: 0)
encoder.setBuffer(bufB, offset: 0, index: 1)
encoder.setBuffer(bufC, offset: 0, index: 2)

var n = UInt32(N)
encoder.setBytes(&n, length: 4, index: 3)

let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
let gridSize = MTLSize(width: N, height: 1, depth: 1)
encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
encoder.endEncoding()
cmdBuffer.commit()
cmdBuffer.waitUntilCompleted()
```

### Key Differences

| Concept         | CUDA                    | Metal                           |
|-----------------|-------------------------|---------------------------------|
| Kernel          | `__global__` function   | `kernel` function               |
| Thread ID       | `threadIdx + blockIdx`  | `thread_position_in_grid`       |
| Block size      | `blockDim`              | `threads_per_threadgroup`       |
| Shared memory   | `__shared__`            | `threadgroup` address space     |
| Synchronization | `__syncthreads()`       | `threadgroup_barrier()`         |
| Memory          | cudaMalloc + cudaMemcpy | Shared buffers (zero-copy)      |
| Launch          | `<<<grid, block>>>`    | Command encoder + dispatch      |

---

## 5. CUDA vs Vulkan Compute

### Overview

| Aspect          | CUDA                          | Vulkan Compute                   |
|-----------------|-------------------------------|----------------------------------|
| **Vendor**      | NVIDIA                        | Khronos (multi-vendor)           |
| **Primary use** | General compute               | Graphics + compute               |
| **Language**    | CUDA C++                      | GLSL/HLSL compiled to SPIR-V    |
| **Overhead**    | Low                           | Very low (explicit control)      |
| **Complexity**  | Moderate                      | Very high (verbose)              |
| **When to use** | Pure compute                  | Mixed graphics + compute         |

### Vulkan Compute Pipeline (Simplified)

```
1. Create VkInstance, VkPhysicalDevice, VkDevice
2. Create VkBuffer for input/output data
3. Allocate VkDeviceMemory, bind to buffers
4. Create VkShaderModule from SPIR-V
5. Create VkDescriptorSetLayout, VkPipelineLayout
6. Create VkComputePipeline
7. Allocate VkDescriptorSet, update with buffer bindings
8. Create VkCommandPool, VkCommandBuffer
9. Record commands: bind pipeline → bind descriptors → dispatch
10. Submit to VkQueue, wait for VkFence
```

**When Graphics Helps:** Vulkan Compute is worthwhile when your compute results feed directly into a graphics pipeline (visualizations, simulations with rendering, image processing for display). For pure compute, CUDA or HIP is simpler.

---

## 6. CUDA vs Triton (Python)

### Overview

| Aspect          | CUDA                          | Triton                           |
|-----------------|-------------------------------|----------------------------------|
| **Language**    | C/C++                         | Python                           |
| **Abstraction** | Low-level                     | High-level (tile-based)          |
| **Vendor**      | NVIDIA                        | OpenAI (multi-backend)           |
| **Target**      | NVIDIA GPUs                   | NVIDIA, AMD (experimental)       |
| **Performance** | Maximum (hand-tuned)          | 80-95% of CUDA for most kernels  |
| **Productivity**| Days for complex kernels      | Hours for equivalent kernels     |

### Vector Add Comparison

**CUDA:**
```cuda
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

**Triton:**
```python
import triton
import triton.language as tl

@triton.jit
def vecAdd_kernel(A_ptr, B_ptr, C_ptr, N,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    tl.store(C_ptr + offsets, a + b, mask=mask)

# Launch
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
vecAdd_kernel[grid](A, B, C, N, BLOCK_SIZE=1024)
```

### When Triton Is Enough vs When You Need CUDA

**Use Triton when:**
- Implementing custom attention mechanisms, activations, or loss functions
- Prototyping GPU kernels quickly
- Performance within 80-95% of CUDA is acceptable
- Working in a Python/PyTorch ecosystem
- Team doesn't have CUDA expertise

**Use raw CUDA when:**
- Maximum performance is critical (last 5-20%)
- Fine-grained control over shared memory, registers needed
- Multi-GPU communication (NCCL integration)
- Interfacing with C++ codebases
- Hardware-specific optimizations (Tensor Cores, warp-level)
- Writing GPU libraries that will be widely deployed

### Triton Capabilities

| Feature                  | Triton Support          | CUDA Equivalent           |
|--------------------------|-------------------------|---------------------------|
| Thread-level control     | No (tile-level)         | Full control              |
| Shared memory            | Implicit (auto-managed) | Explicit `__shared__`     |
| Memory coalescing        | Automatic               | Manual (SoA layout)       |
| Warp shuffles            | No                      | `__shfl_*_sync`           |
| Tensor Cores             | `tl.dot()` (automatic)  | `wmma` / `mma.sync`      |
| Auto-tuning              | Built-in (`triton.autotune`) | Manual                |
| Multi-GPU                | Via PyTorch              | NCCL / manual             |
| Inline assembly          | No                      | PTX `asm()`               |

---

## 7. Comprehensive Comparison Table

| Feature                 | CUDA     | OpenCL   | HIP      | SYCL     | Metal    | Vulkan   | Triton   |
|-------------------------|----------|----------|----------|----------|----------|----------|----------|
| **Vendor lock-in**      | NVIDIA   | None     | Low      | None     | Apple    | None     | Low      |
| **NVIDIA perf**         | ★★★★★   | ★★★★    | ★★★★★   | ★★★★    | —        | ★★★★    | ★★★★    |
| **AMD perf**            | —        | ★★★★    | ★★★★★   | ★★★★    | —        | ★★★★    | ★★★     |
| **Intel perf**          | —        | ★★★     | —        | ★★★★★   | —        | ★★★★    | —        |
| **Apple perf**          | —        | —        | —        | —        | ★★★★★   | ★★★★    | —        |
| **Ecosystem richness**  | ★★★★★   | ★★      | ★★★     | ★★★     | ★★★     | ★★★★    | ★★★     |
| **Ease of learning**    | ★★★★    | ★★      | ★★★★    | ★★★     | ★★★     | ★        | ★★★★★   |
| **Code portability**    | ★        | ★★★★    | ★★★★    | ★★★★★   | ★        | ★★★★    | ★★★     |
| **Debugging tools**     | ★★★★★   | ★★      | ★★★     | ★★★     | ★★★★    | ★★★     | ★★      |
| **Documentation**       | ★★★★★   | ★★★     | ★★★     | ★★★     | ★★★★    | ★★★★    | ★★★     |
| **Production maturity** | ★★★★★   | ★★★★    | ★★★★    | ★★★     | ★★★★    | ★★★★★   | ★★★     |
| **ML framework support**| ★★★★★   | ★       | ★★★★    | ★★      | ★★★     | ★        | ★★★★    |

---

## 8. Portability Strategies

### Strategy 1: Write in HIP (NVIDIA + AMD)

```
Source code (.cpp / .hip)
        │
    ┌───┴───┐
    │       │
  hipcc   hipcc --platform nvidia
    │       │
  AMD GPU  NVIDIA GPU (via CUDA backend)
```

**Pros:** Minimal code changes from CUDA, good performance on both vendors.
**Cons:** No Intel/Apple support, HIP lags behind latest CUDA features.

### Strategy 2: Write in SYCL (True Portability)

```
Source code (.cpp, standard C++)
        │
    ┌───┼───────┬──────┐
    │   │       │      │
  icpx  clang  hipSYCL AdaptiveCpp
    │   │       │      │
  Intel NVIDIA  AMD    CPU fallback
```

**Pros:** Open standard, single-source C++, widest hardware coverage.
**Cons:** Smaller ecosystem, not always optimal performance, less mature tooling.

### Strategy 3: Abstraction Libraries

| Library          | Description                           | Supported Backends       |
|------------------|---------------------------------------|--------------------------|
| **Kokkos**       | Performance-portable C++ library      | CUDA, HIP, SYCL, OpenMP |
| **RAJA**         | LLNL's portability layer              | CUDA, HIP, OpenMP       |
| **Alpaka**       | Abstraction for parallel hierarchies  | CUDA, HIP, OpenMP, TBB  |
| **stdpar**       | C++ parallel algorithms (nvc++)       | CUDA, CPU multicore     |
| **hpx**          | C++ standard library for concurrency  | CUDA, CPU                |

### Strategy 4: Python + Triton (ML-Focused)

```python
# Write kernels in Triton (Python)
# Backend handles NVIDIA/AMD compilation
# Best for ML custom operators

@triton.jit
def my_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, tl.exp(x), mask=mask)
```

### Strategy 5: Conditional Compilation

```cpp
// Unified header approach
#if defined(__CUDA_ARCH__)
    #define GPU_GLOBAL __global__
    #define GPU_DEVICE __device__
    #define GPU_SHARED __shared__
    #define SYNC_THREADS() __syncthreads()
#elif defined(__HIP_DEVICE_COMPILE__)
    #define GPU_GLOBAL __global__
    #define GPU_DEVICE __device__
    #define GPU_SHARED __shared__
    #define SYNC_THREADS() __syncthreads()
#elif defined(__SYCL_DEVICE_ONLY__)
    // SYCL uses different paradigm — not 1:1 mappable
#endif
```

---

## 9. Vector Add in Every Language (Complete)

### Summary Table

| Language     | Kernel LoC | Host LoC | Total LoC | Boilerplate |
|-------------|-----------|---------|-----------|-------------|
| CUDA        | 4         | 12      | 16        | Low         |
| HIP         | 4         | 12      | 16        | Low         |
| OpenCL      | 4         | 40+     | 44+       | Very High   |
| SYCL (USM)  | 3         | 15      | 18        | Moderate    |
| SYCL (Buf)  | 3         | 12      | 15        | Moderate    |
| Metal       | 5         | 25      | 30        | High        |
| Vulkan      | 10 (GLSL) | 100+    | 110+      | Extreme     |
| Triton      | 6         | 3       | 9         | Very Low    |

### Vulkan Compute Shader (GLSL → SPIR-V)

```glsl
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) buffer B { float b[]; };
layout(set = 0, binding = 2) buffer C { float c[]; };

layout(push_constant) uniform Params { uint N; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
```

```bash
# Compile GLSL to SPIR-V
glslangValidator -V vecadd.comp -o vecadd.spv
```

---

## 10. Decision Matrix: Which Language Should You Use?

```
START
│
├── Are you targeting NVIDIA GPUs exclusively?
│   ├── Yes, and need maximum performance → CUDA
│   ├── Yes, but want Python productivity → Triton
│   └── Yes, but might support AMD later → HIP
│
├── Do you need to target AMD GPUs?
│   ├── Primary target → HIP (native AMD, works on NVIDIA)
│   ├── Along with Intel → SYCL
│   └── Along with NVIDIA → HIP
│
├── Do you need to target Intel GPUs?
│   └── Yes → SYCL / DPC++
│
├── Are you on Apple Silicon?
│   ├── ML workloads → MLX (Python) or Metal (MSL)
│   └── Graphics + compute → Metal
│
├── Do you need mixed graphics + compute?
│   └── Yes → Vulkan Compute (or Metal on Apple)
│
├── Are you writing ML custom operators?
│   ├── PyTorch ecosystem → Triton (easiest)
│   ├── Need max performance → CUDA
│   └── Need portability → SYCL or HIP
│
└── Do you need maximum portability across all vendors?
    ├── C++ codebase → SYCL or Kokkos
    ├── Python codebase → Triton
    └── HPC / scientific → Kokkos or RAJA
```

---

*Appendix F — From CUDA to Other GPU Languages — Part of the CPP-CUDA-Mastery series*
