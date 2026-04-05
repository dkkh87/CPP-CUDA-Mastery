# Appendix I — Glossary

> A comprehensive glossary of 130+ terms used throughout the C++ & CUDA Mastery chapters.
> Terms are alphabetically ordered. Each entry follows the format:
>
> **Term** — Definition. *(See: relevant chapter/part)*

---

## A

**ABI (Application Binary Interface)** — The low-level contract between compiled binaries specifying calling conventions, name mangling, vtable layout, and data alignment. Changing a class's private members can break ABI even when the public API is unchanged. *(See: Part 3 — Advanced C++)*

**Abstract Class** — A class containing at least one pure virtual function (`= 0`), making it impossible to instantiate directly. Subclasses must override all pure virtuals to become concrete types. *(See: Part 1 — C++ Foundations)*

**Aggregate** — A class or array with no user-declared constructors, no private/protected non-static data members, no virtual functions, and no base classes (pre-C++17 rules relaxed slightly in later standards). Aggregates support brace-enclosed initialization. *(See: Part 2 — Intermediate C++)*

**AllReduce** — A collective communication primitive that reduces values (e.g., sum, max) across all participating processes and distributes the result back to every participant. Critical for synchronizing gradients in distributed training. *(See: Part 8 — CUDA AI/ML)*

**Allocator** — An object that encapsulates a memory allocation strategy for STL containers. Custom allocators enable arena allocation, pool allocation, or GPU-pinned memory without changing container code. *(See: Part 3 — Advanced C++)*

**Asynchronous Memcpy** — A CUDA memory transfer (`cudaMemcpyAsync`) executed on a stream, allowing the CPU to continue issuing work and enabling overlap of data transfer with kernel execution on different streams. *(See: Part 6 — CUDA Foundations)*

**Attention** — A mechanism in neural networks that computes weighted relevance scores between elements of a sequence, allowing the model to focus on the most informative parts of its input. Foundation of the Transformer architecture. *(See: Part 8 — CUDA AI/ML)*

**Auto** — A C++11 keyword that instructs the compiler to deduce a variable's type from its initializer expression. Reduces verbosity and prevents accidental implicit conversions. *(See: Part 2 — Intermediate C++)*

**Autograd** — An automatic differentiation engine (e.g., PyTorch's `torch.autograd`) that records operations on tensors and replays them in reverse to compute gradients. Enables backpropagation without hand-written derivative code. *(See: Part 8 — CUDA AI/ML)*

**Atomic Operation** — An indivisible read-modify-write operation (e.g., `atomicAdd`, `std::atomic::fetch_add`) that completes without interruption, preventing data races in concurrent code. On GPUs, atomics operate on global or shared memory. *(See: Part 5 — Systems Programming, Part 6 — CUDA Foundations)*

## B

**Barrier** — A synchronization point where all participating threads (or processes) must arrive before any may proceed. In CUDA, `__syncthreads()` is a block-level barrier; `cooperative_groups` extends barriers to grid scope. *(See: Part 6 — CUDA Foundations)*

**Backpropagation** — The algorithm that computes the gradient of a loss function with respect to each weight in a neural network by applying the chain rule layer-by-layer from output to input. *(See: Part 8 — CUDA AI/ML)*

**Bank Conflict** — A situation in GPU shared memory where two or more threads in the same warp access different addresses that map to the same memory bank, forcing serialized access and reducing throughput. *(See: Part 6 — CUDA Foundations)*

**Batch Normalization** — A technique that normalizes the inputs to each layer across the mini-batch, reducing internal covariate shift and typically accelerating training convergence. *(See: Part 8 — CUDA AI/ML)*

**BLAS (Basic Linear Algebra Subprograms)** — A standardized set of low-level routines for common linear algebra operations (dot products, matrix multiply, etc.). GPU-accelerated implementations include cuBLAS. *(See: Part 7 — CUDA Advanced)*

**Block (Thread Block)** — A group of CUDA threads that execute on a single Streaming Multiprocessor, share shared memory, and can synchronize via `__syncthreads()`. Blocks are the primary unit of cooperative parallelism. *(See: Part 6 — CUDA Foundations)*

**BF16 (bfloat16)** — A 16-bit floating-point format with the same 8-bit exponent as FP32 but only 7 bits of mantissa. Maintains FP32's dynamic range at half the memory, widely used in training. *(See: Part 7 — CUDA Advanced)*

## C

**Cache Line** — The smallest unit of data transferred between main memory and CPU cache, typically 64 bytes on x86. Misaligned or scattered accesses waste bandwidth by loading unused bytes. *(See: Part 5 — Systems Programming)*

**Coalescing (Memory Coalescing)** — The GPU hardware's ability to merge multiple threads' memory accesses into fewer, wider transactions when addresses are contiguous and aligned. Essential for high global-memory throughput. *(See: Part 6 — CUDA Foundations)*

**Compute Capability** — A version number (e.g., 8.0, 9.0) that identifies a GPU's architecture and supported features—maximum threads per block, shared memory size, Tensor Core availability, etc. *(See: Part 6 — CUDA Foundations)*

**Concepts (C++20)** — Named compile-time predicates that constrain template parameters, replacing SFINAE-based techniques with readable, first-class syntax: `template<std::integral T>`. *(See: Part 4 — Modern C++ Evolution)*

**Const** — A qualifier that marks a variable, pointer, or member function as immutable. `const` correctness helps the compiler optimize and prevents accidental mutation. *(See: Part 1 — C++ Foundations)*

**Constant Memory** — A read-only GPU memory region (typically 64 KB) cached aggressively on-chip. Ideal for small, frequently-read lookup tables broadcast to all threads uniformly. *(See: Part 6 — CUDA Foundations)*

**Consteval (C++20)** — A specifier that forces a function to be evaluated at compile time; if it cannot be, the program is ill-formed. Stricter than `constexpr`. *(See: Part 4 — Modern C++ Evolution)*

**Constexpr** — A specifier indicating that a function or variable *can* be evaluated at compile time. Since C++14, `constexpr` functions may contain loops, branches, and local variables. *(See: Part 2 — Intermediate C++)*

**Convolution** — A mathematical operation that slides a small filter (kernel) over input data, computing element-wise products and sums. Foundation of CNNs for image recognition, often implemented as an optimized GEMM on GPUs. *(See: Part 8 — CUDA AI/ML)*

**Cooperative Groups** — A CUDA programming model (CC ≥ 6.0) that generalizes thread synchronization beyond the block level, enabling grid-wide, multi-GPU, and dynamically-sized group synchronization. *(See: Part 7 — CUDA Advanced)*

**Copy Elision** — A compiler optimization that constructs an object directly in its destination, eliminating copy or move constructors. Mandatory in C++17 for certain prvalue cases (guaranteed copy elision). *(See: Part 2 — Intermediate C++)*

**Coroutine (C++20)** — A function that can suspend execution (`co_await`, `co_yield`, `co_return`) and resume later, enabling lazy generators, async I/O, and cooperative multitasking without callback spaghetti. *(See: Part 4 — Modern C++ Evolution)*

**CRTP (Curiously Recurring Template Pattern)** — A C++ idiom where a class derives from a template instantiated with itself: `class Derived : Base<Derived>`. Enables static polymorphism with zero virtual-call overhead. *(See: Part 3 — Advanced C++)*

**cuBLAS** — NVIDIA's GPU-accelerated implementation of BLAS, providing highly tuned routines for matrix multiplication (GEMM), triangular solve, and other dense linear algebra operations. *(See: Part 7 — CUDA Advanced)*

**CUDA Core** — A scalar floating-point/integer execution unit inside a Streaming Multiprocessor. Modern GPUs contain thousands of CUDA cores executing in SIMT fashion. *(See: Part 6 — CUDA Foundations)*

**CUDA Graph** — A mechanism that captures a sequence of CUDA operations (kernels, memcpys) into a graph object that can be launched repeatedly with minimal CPU overhead, eliminating per-launch latency. *(See: Part 7 — CUDA Advanced)*

**cuDNN (CUDA Deep Neural Network library)** — NVIDIA's library of highly optimized primitives for deep learning: convolutions, pooling, normalization, activation functions, and RNNs. *(See: Part 8 — CUDA AI/ML)*

**cuFFT** — NVIDIA's GPU-accelerated Fast Fourier Transform library, supporting 1D, 2D, and 3D transforms in single, double, and half precision. Used in signal processing, image filtering, and scientific computing. *(See: Part 7 — CUDA Advanced)*

**cuRAND** — NVIDIA's library for GPU-accelerated random number generation supporting multiple algorithms (XORWOW, Philox, MRG32k3a). Provides both host API for bulk generation and device API for per-thread generation inside kernels. *(See: Part 7 — CUDA Advanced)*

**CUTLASS** — NVIDIA's open-source C++ template library for high-performance GEMM and convolution kernels, designed to be customizable and composable at the warp and thread level. *(See: Part 7 — CUDA Advanced)*

## D

**Deadlock** — A state where two or more threads are each waiting for the other to release a resource, causing all of them to block indefinitely. Proper lock ordering and lock-free algorithms prevent deadlocks. *(See: Part 5 — Systems Programming)*

**Dangling Reference** — A reference (or pointer) that refers to an object whose lifetime has ended. Accessing it is undefined behavior. Common sources include returning a reference to a local variable. *(See: Part 1 — C++ Foundations)*

**Data Parallelism** — A distributed training strategy where each device holds a full model replica and processes a different data shard; gradients are synchronized via AllReduce after each step. *(See: Part 8 — CUDA AI/ML)*

**Data Race** — A bug where two or more threads access the same memory location concurrently, at least one access is a write, and there is no synchronization between them. Data races produce undefined behavior in both C++ and CUDA. *(See: Part 5 — Systems Programming)*

**Decltype** — A C++11 keyword that yields the declared type of an expression without evaluating it. Used in template code and trailing return types to preserve exact type information. *(See: Part 2 — Intermediate C++)*

**Device** — In CUDA terminology, the GPU and its memory. Contrasted with *host* (the CPU). Device code is compiled by `nvcc` and runs on the GPU. *(See: Part 6 — CUDA Foundations)*

**DRAM (Dynamic Random-Access Memory)** — Volatile memory that stores data in capacitors requiring periodic refresh. Both CPU main memory (DDR) and GPU global memory (HBM/GDDR) are DRAM-based. *(See: Part 5 — Systems Programming)*

**Dynamic Parallelism** — A CUDA feature (CC ≥ 3.5) allowing a kernel running on the GPU to launch child kernels without returning to the CPU, enabling recursive and adaptive algorithms on-device. *(See: Part 7 — CUDA Advanced)*

## E

**Exception Safety** — A guarantee about program state when an exception is thrown. *Basic*: no leaks, invariants intact. *Strong*: operation is rolled back (commit-or-rollback). *Nothrow*: operation never throws. *(See: Part 2 — Intermediate C++)*

**Executor** — An abstraction (proposed for C++26, available in libraries like Asio) that represents a policy for *where* and *how* work is executed—thread pools, GPU streams, or inline. *(See: Part 4 — Modern C++ Evolution)*

**Expression Template** — A C++ metaprogramming technique that encodes arithmetic expressions as nested template types, deferring evaluation to eliminate temporaries—commonly used in linear algebra libraries like Eigen. *(See: Part 3 — Advanced C++)*

## F

**False Sharing** — A performance pathology where threads on different cores modify independent variables that reside on the same cache line, causing repeated cache invalidation and inter-core traffic. *(See: Part 5 — Systems Programming)*

**Flash Attention** — An I/O-aware attention algorithm that tiles the computation to fit in SRAM (shared memory), dramatically reducing HBM reads/writes and enabling longer sequences without quadratic memory growth. *(See: Part 8 — CUDA AI/ML)*

**FLOPS (Floating-Point Operations Per Second)** — A measure of computational throughput. Modern GPUs achieve hundreds of TFLOPS (teraFLOPS) on Tensor Cores. Distinguish from FLOP (a single operation). *(See: Part 7 — CUDA Advanced)*

**Fused Kernel** — A single GPU kernel that combines multiple logically separate operations (e.g., bias + activation + dropout) to avoid intermediate memory round-trips and kernel launch overhead. *(See: Part 7 — CUDA Advanced)*

**Forward Declaration** — Declaring a class, function, or type name without defining it, allowing files to reference names without including full definitions. Reduces compile-time dependencies and speeds up builds. *(See: Part 1 — C++ Foundations)*

**FP8 (8-bit Floating Point)** — An ultra-low-precision format (E4M3 or E5M2 variants) introduced on Hopper GPUs for inference and training, offering 2× throughput over FP16 with careful loss scaling. *(See: Part 7 — CUDA Advanced)*

**FP16 (Half Precision)** — IEEE 754 16-bit floating-point with 5 exponent bits and 10 mantissa bits. Commonly used in mixed-precision training; smaller range than BF16 but higher precision. *(See: Part 7 — CUDA Advanced)*

**Functor (Function Object)** — Any object that overloads `operator()`, making it callable like a function. Functors can carry state and are commonly used as predicates and comparators in STL algorithms. *(See: Part 1 — C++ Foundations)*

## G

**GEMM (General Matrix Multiply)** — The core linear algebra operation `C = αAB + βC`. Most deep learning workloads (fully-connected layers, convolutions, attention) reduce to batched GEMMs on GPUs. *(See: Part 7 — CUDA Advanced)*

**Global Memory** — The GPU's main DRAM (HBM or GDDR), accessible by all threads across all blocks. High bandwidth but high latency (~400–600 cycles). Must be accessed with coalesced patterns for performance. *(See: Part 6 — CUDA Foundations)*

**Gradient** — The vector of partial derivatives of a loss function with respect to model parameters. Gradients guide weight updates during optimization (e.g., SGD, Adam). *(See: Part 8 — CUDA AI/ML)*

**Grid** — The top-level organizational unit of a CUDA kernel launch, composed of one or more thread blocks arranged in 1D, 2D, or 3D. A grid maps to the entire problem domain. *(See: Part 6 — CUDA Foundations)*

**Grid-Stride Loop** — A CUDA programming pattern where each thread processes multiple elements by striding through the data in increments of `gridDim.x * blockDim.x`, ensuring correct behavior regardless of launch configuration. *(See: Part 6 — CUDA Foundations)*

## H

**HBM (High Bandwidth Memory)** — A 3D-stacked DRAM technology used in modern GPUs (e.g., A100: 2 TB/s, H100: 3.35 TB/s). Provides massive bandwidth but is still the primary bottleneck for memory-bound kernels. *(See: Part 7 — CUDA Advanced)*

**Header Guard** — A preprocessor idiom (`#ifndef`/`#define`/`#endif` or `#pragma once`) preventing multiple inclusions of the same header file within a single translation unit. *(See: Part 1 — C++ Foundations)*

**Heap** — The region of memory used for dynamic allocation (`new`, `malloc`). Objects on the heap persist until explicitly freed and have non-deterministic allocation latency. Contrast with *stack*. *(See: Part 1 — C++ Foundations)*

**Host** — In CUDA terminology, the CPU and its system memory. Host code is compiled by the standard C++ compiler; `nvcc` separates host and device code during compilation. *(See: Part 6 — CUDA Foundations)*

## I

**Inference** — The process of running a trained neural network on new inputs to produce predictions, as opposed to training. Inference optimizations include quantization, pruning, and batching. *(See: Part 8 — CUDA AI/ML)*

**Inline** — A suggestion to the compiler to substitute a function's body at the call site, eliminating call overhead. In modern C++, `inline` primarily affects linkage (permitting definitions in headers). *(See: Part 1 — C++ Foundations)*

**ILP (Instruction-Level Parallelism)** — The ability to execute multiple independent instructions simultaneously within a single thread or warp. On GPUs, ILP can compensate for low occupancy by keeping execution units busy with independent operations. *(See: Part 7 — CUDA Advanced)*

**Iterator** — An object that provides sequential access to elements in a container, abstracting pointer-like traversal. The STL defines five iterator categories: input, output, forward, bidirectional, and random-access. *(See: Part 1 — C++ Foundations)*

## K

**Kernel** — A function declared with `__global__` that executes on the GPU. Launched from the host with the `<<<grid, block>>>` syntax, each invocation spawns thousands to millions of threads. *(See: Part 6 — CUDA Foundations)*

**Kernel Fusion** — See *Fused Kernel*. The compiler or programmer combines multiple operations into a single kernel to minimize global memory traffic and launch overhead. *(See: Part 7 — CUDA Advanced)*

## L

**L1 Cache (GPU)** — A per-SM on-chip cache (configurable with shared memory on some architectures) that automatically caches global memory accesses. Typical size: 128–256 KB per SM on Ampere/Hopper. *(See: Part 7 — CUDA Advanced)*

**L2 Cache (GPU)** — A device-wide cache shared by all SMs. Larger than L1 (e.g., 40 MB on A100, 50 MB on H100) and configurable for persistence policies via `cudaAccessPolicyWindow`. *(See: Part 7 — CUDA Advanced)*

**Lambda** — An anonymous function object defined inline with capture syntax: `[captures](params){ body }`. Lambdas can capture local variables by value or reference and are the modern replacement for most functors. *(See: Part 2 — Intermediate C++)*

**Layer Normalization** — A normalization technique that normalizes across the feature dimension within a single sample, rather than across the batch. Preferred over batch normalization in Transformers and RNNs. *(See: Part 8 — CUDA AI/ML)*

**Linkage** — Determines a name's visibility across translation units. *Internal linkage* (e.g., `static`, anonymous namespace) restricts a symbol to its translation unit. *External linkage* makes it accessible from other units. *(See: Part 1 — C++ Foundations)*

**Local Memory (GPU)** — Per-thread private memory that spills to DRAM when registers are exhausted. Despite the name "local," it resides in global memory and is slow—avoid excessive register pressure to prevent spilling. *(See: Part 6 — CUDA Foundations)*

**Lvalue** — An expression that designates a persistent object with an identifiable memory address. Named variables, dereferenced pointers, and array elements are lvalues. Contrast with *rvalue*. *(See: Part 2 — Intermediate C++)*

**Loss Scaling** — A mixed-precision training technique that multiplies the loss by a large factor before backpropagation to prevent small FP16 gradients from underflowing to zero, then unscales before the optimizer step. *(See: Part 8 — CUDA AI/ML)*

## M

**Mixed Precision** — A training/inference technique that uses lower-precision formats (FP16/BF16) for computation while keeping a master copy of weights in FP32 for numerical stability. Typically 2× speedup on Tensor Cores. *(See: Part 8 — CUDA AI/ML)*

**Model Parallelism** — A distributed training strategy that partitions a single model across multiple devices—each device holds a subset of layers or parameters—enabling models too large to fit in one GPU's memory. *(See: Part 8 — CUDA AI/ML)*

**Move Semantics** — C++11 feature allowing resources (heap memory, file handles) to be *transferred* from one object to another via move constructors and move assignment operators, avoiding expensive deep copies. *(See: Part 2 — Intermediate C++)*

**Memory Fence** — An instruction that enforces ordering constraints on memory operations, ensuring that reads/writes before the fence are visible to other threads before those after it. CUDA provides `__threadfence()`, `__threadfence_block()`, and `__threadfence_system()`. *(See: Part 7 — CUDA Advanced)*

**Memory Pool (cudaMallocAsync)** — A CUDA feature that pre-allocates a pool of device memory and services allocation/free requests from it, avoiding the latency of `cudaMalloc`/`cudaFree` on each call. *(See: Part 7 — CUDA Advanced)*

## N

**NCCL (NVIDIA Collective Communications Library)** — A library providing optimized multi-GPU and multi-node collective operations (AllReduce, Broadcast, AllGather) over NVLink, PCIe, and InfiniBand. *(See: Part 8 — CUDA AI/ML)*

**NRVO / RVO (Named/Return Value Optimization)** — Compiler optimizations that construct a function's return value directly in the caller's destination object, eliding the copy/move. RVO is guaranteed in C++17 for prvalues; NRVO remains optional but nearly universal. *(See: Part 2 — Intermediate C++)*

**Nsight Compute** — NVIDIA's interactive GPU kernel profiler providing detailed performance metrics: occupancy, memory throughput, instruction mix, warp stall reasons, and roofline analysis. *(See: Part 7 — CUDA Advanced)*

**Nsight Systems** — NVIDIA's system-wide performance analysis tool that visualizes CPU/GPU interactions, kernel timelines, memory transfers, API calls, and pipeline bubbles across the entire application. *(See: Part 7 — CUDA Advanced)*

**NUMA (Non-Uniform Memory Access)** — A memory architecture where access latency depends on the distance between a processor and the memory bank. NUMA-aware allocation is critical for multi-socket CPU and multi-GPU systems. *(See: Part 5 — Systems Programming)*

**NVLink** — NVIDIA's high-speed GPU-to-GPU interconnect providing significantly higher bandwidth than PCIe (e.g., 900 GB/s bidirectional on NVLink 4.0 / Hopper). Enables fast P2P communication and unified memory access. *(See: Part 7 — CUDA Advanced)*

**nvcc** — NVIDIA's CUDA compiler driver that separates host code (forwarded to the host C++ compiler) from device code (compiled to PTX/SASS). Supports `-arch`, `-gencode`, and various optimization flags. *(See: Part 6 — CUDA Foundations)*

**NVSwitch** — A high-radix switch chip that connects multiple GPUs via NVLink in an all-to-all topology (e.g., 8 GPUs in DGX systems), providing full bisection bandwidth without PCIe bottlenecks. *(See: Part 7 — CUDA Advanced)*

## O

**Object Slicing** — The loss of derived-class data when a derived object is copied into a base-class variable by value. Only the base portion is retained; virtual dispatch on the copy calls base implementations. *(See: Part 1 — C++ Foundations)*

**Occupancy** — The ratio of active warps to the maximum warps an SM can support. Higher occupancy helps hide memory latency but is not always necessary for peak performance—register usage and ILP matter too. *(See: Part 6 — CUDA Foundations)*

**Occupancy Calculator** — An NVIDIA-provided tool (spreadsheet or `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API) that determines optimal block size given a kernel's register and shared memory usage. *(See: Part 6 — CUDA Foundations)*

**ODR (One Definition Rule)** — A C++ rule stating that every entity (variable, function, class) must have exactly one definition across all translation units. Violations cause undefined behavior, often manifesting as linker errors or silent corruption. *(See: Part 1 — C++ Foundations)*

**Overload Resolution** — The compiler process of selecting the best-matching function from a set of overloaded candidates based on argument types, conversions, and template specialization ranking. *(See: Part 2 — Intermediate C++)*

## P

**P2P (Peer-to-Peer)** — Direct GPU-to-GPU memory access over NVLink or PCIe without staging through host memory. Enabled via `cudaDeviceEnablePeerAccess()`. *(See: Part 7 — CUDA Advanced)*

**PCIe (Peripheral Component Interconnect Express)** — The standard high-speed serial bus connecting GPUs to the CPU. PCIe 4.0 provides ~32 GB/s per direction (×16); PCIe 5.0 doubles that. Often the bottleneck for host↔device transfers. *(See: Part 6 — CUDA Foundations)*

**PIMPL (Pointer to Implementation)** — A C++ idiom that hides a class's private members behind an opaque pointer, reducing compile-time dependencies and preserving ABI stability across library versions. *(See: Part 3 — Advanced C++)*

**Pinned Memory (Page-Locked Memory)** — Host memory allocated with `cudaMallocHost()` that cannot be paged out to disk by the OS. Enables faster and asynchronous host↔device transfers, at the cost of reducing available system memory. *(See: Part 6 — CUDA Foundations)*

**Pipeline Parallelism** — A distributed training strategy that assigns different layers of a model to different devices and overlaps forward/backward passes of successive micro-batches to keep all devices busy. *(See: Part 8 — CUDA AI/ML)*

**PMR (Polymorphic Memory Resource)** — A C++17 framework (`std::pmr`) that decouples containers from their allocation strategy via a virtual `memory_resource` base class. Enables runtime-switchable allocators without changing container types. *(See: Part 3 — Advanced C++)*

**PTX (Parallel Thread Execution)** — NVIDIA's intermediate virtual instruction set. CUDA code compiles to PTX first, then the driver's JIT compiler lowers PTX to device-specific SASS at load time. *(See: Part 7 — CUDA Advanced)*

## Q

**Quantization** — The process of reducing a model's numerical precision (e.g., FP32 → INT8 or FP8) to shrink model size and increase inference throughput, with minimal accuracy loss when calibrated properly. *(See: Part 8 — CUDA AI/ML)*

## R

**RAII (Resource Acquisition Is Initialization)** — A C++ idiom where resource ownership is tied to object lifetime: acquire in the constructor, release in the destructor. Guarantees cleanup even when exceptions occur. Smart pointers, lock guards, and file handles all use RAII. *(See: Part 1 — C++ Foundations)*

**Range (C++20)** — A generalization of iterator pairs into a single object that represents a sequence. Ranges enable lazy, composable pipelines: `views::filter(...) | views::transform(...)`. *(See: Part 4 — Modern C++ Evolution)*

**Register (GPU)** — The fastest storage on the GPU, private to each thread. A typical SM has 65,536 32-bit registers shared among all active threads; high register usage per thread reduces occupancy. *(See: Part 6 — CUDA Foundations)*

**Reduction** — A parallel primitive that combines all elements of an array into a single value using an associative operator (sum, max, etc.). Efficient GPU reductions use shared memory, warp shuffles, and multi-pass techniques. *(See: Part 6 — CUDA Foundations)*

**Roofline Model** — A visual performance model that plots attainable FLOPS against operational intensity (FLOPS/byte), identifying whether a kernel is compute-bound or memory-bound. Nsight Compute can generate roofline charts. *(See: Part 7 — CUDA Advanced)*

**Rvalue** — An expression that denotes a temporary or expiring value with no persistent address. Rvalue references (`T&&`) enable move semantics and perfect forwarding. *(See: Part 2 — Intermediate C++)*

## S

**SASS (Shader Assembly)** — The native machine code instruction set for NVIDIA GPUs. SASS is architecture-specific (unlike PTX) and is what actually executes on the hardware. Viewable via `cuobjdump` or Nsight Compute. *(See: Part 7 — CUDA Advanced)*

**SFINAE (Substitution Failure Is Not An Error)** — A C++ template rule: if substituting template arguments into a function signature produces an invalid type, that overload is silently removed from the candidate set rather than causing a compile error. Superseded by Concepts in C++20. *(See: Part 3 — Advanced C++)*

**Shared Memory (GPU)** — A fast, programmer-managed on-chip memory (up to 228 KB per SM on Hopper) shared by all threads in a block. Used for inter-thread communication, tiling, and reduction. Organized into 32 banks. *(See: Part 6 — CUDA Foundations)*

**SIMD (Single Instruction, Multiple Data)** — A CPU execution model where one instruction operates on multiple data elements simultaneously (e.g., SSE, AVX-512). The GPU's SIMT model is a related but distinct concept. *(See: Part 5 — Systems Programming)*

**SM (Streaming Multiprocessor)** — The fundamental processing unit of an NVIDIA GPU, containing CUDA cores, Tensor Cores, register files, shared memory, warp schedulers, and L1 cache. A GPU has tens to hundreds of SMs. *(See: Part 6 — CUDA Foundations)*

**Smart Pointer** — A RAII wrapper around a raw pointer that automatically manages the pointed-to object's lifetime. `std::unique_ptr` provides exclusive ownership; `std::shared_ptr` provides shared ownership with reference counting. *(See: Part 1 — C++ Foundations)*

**Softmax** — A function that converts a vector of real numbers into a probability distribution: `softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)`. Used as the final activation in classification and within attention mechanisms. *(See: Part 8 — CUDA AI/ML)*

**SIMT (Single Instruction, Multiple Threads)** — The GPU execution model where threads in a warp execute the same instruction on different data, similar to CPU SIMD but with independent thread state and divergence handling. *(See: Part 6 — CUDA Foundations)*

**Spill (Register Spill)** — When a kernel uses more registers than available per thread, the excess is "spilled" to local memory (DRAM), severely degrading performance. Monitor with `--ptxas-options=-v` or Nsight Compute. *(See: Part 7 — CUDA Advanced)*

**Stack** — The region of memory used for local variables and function call frames, managed automatically via push/pop. Stack allocation is fast (single pointer adjustment) but limited in size (typically 1–8 MB). *(See: Part 1 — C++ Foundations)*

**STL (Standard Template Library)** — The portion of the C++ Standard Library providing generic containers (`vector`, `map`), iterators, and algorithms (`sort`, `find`). Template-based for type safety and zero-overhead abstraction. *(See: Part 1 — C++ Foundations)*

**Stream (CUDA)** — A sequence of GPU operations (kernels, memcpys) that execute in order. Operations on *different* streams can execute concurrently, enabling overlap of compute and data transfer. The default stream (stream 0) synchronizes with all other streams unless `--default-stream per-thread` is used. *(See: Part 6 — CUDA Foundations)*

## T

**Template Metaprogramming (TMP)** — A technique that uses C++ templates to perform computation at compile time—type manipulation, conditional compilation, and compile-time loops—producing highly optimized, specialized code with no runtime cost. *(See: Part 3 — Advanced C++)*

**Tensor** — A multi-dimensional array that generalizes scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions. The fundamental data structure in deep learning frameworks (PyTorch, TensorFlow). *(See: Part 8 — CUDA AI/ML)*

**Tensor Core** — Specialized hardware units on NVIDIA GPUs (Volta and later) that perform small matrix multiply-accumulate operations (e.g., 4×4×4) in a single clock cycle, delivering an order of magnitude more FLOPS than CUDA cores for supported precisions. *(See: Part 7 — CUDA Advanced)*

**TensorRT** — NVIDIA's high-performance deep learning inference optimizer and runtime. It applies layer fusion, kernel auto-tuning, precision calibration (FP16/INT8), and memory optimization to maximize GPU utilization at inference time. *(See: Part 8 — CUDA AI/ML)*

**Texture Memory** — A GPU memory region optimized for 2D spatial locality, offering hardware interpolation and boundary handling. Historically important for graphics; in compute, L1/L2 caches often suffice, but texture fetches remain useful for image-processing kernels. *(See: Part 6 — CUDA Foundations)*

**TF32 (TensorFloat-32)** — An internal Tensor Core format on Ampere+ GPUs with 8 exponent bits (like FP32) and 10 mantissa bits (like FP16). Provides near-FP32 accuracy at FP16-like speed, enabled by default for `torch.matmul`. *(See: Part 7 — CUDA Advanced)*

**Thread (CUDA)** — The finest-grained unit of execution on a GPU. Each thread has its own registers and local memory, executes the same kernel code, and is identified by `threadIdx` within its block. *(See: Part 6 — CUDA Foundations)*

**Thread Safety** — The property that a function or data structure behaves correctly when accessed concurrently by multiple threads, without data races or corruption. Achieved via mutexes, atomics, or lock-free designs. *(See: Part 5 — Systems Programming)*

**Throughput-Oriented Architecture** — A design philosophy where the GPU maximizes aggregate throughput across thousands of threads rather than minimizing single-thread latency. This contrasts with CPUs, which are latency-oriented. *(See: Part 6 — CUDA Foundations)*

**Thrust** — A CUDA C++ template library modeled after the STL, providing high-level parallel primitives (`sort`, `reduce`, `scan`, `transform`) that automatically dispatch to GPU or CPU backends. *(See: Part 7 — CUDA Advanced)*

**Tiling** — An optimization strategy that partitions data into small tiles (blocks) that fit in fast on-chip memory (shared memory, L1 cache). Tiling improves data locality and is fundamental to efficient GEMM, convolution, and attention kernels. *(See: Part 7 — CUDA Advanced)*

**Translation Unit** — The result of preprocessing a single source file: the file itself plus all `#include`d headers, after macro expansion. Each translation unit is compiled independently; the linker combines them. *(See: Part 1 — C++ Foundations)*

**Type Erasure** — A design pattern that hides concrete types behind a uniform interface without requiring a common base class. `std::function`, `std::any`, and `std::variant` are standard-library examples. *(See: Part 3 — Advanced C++)*

## U

**UB (Undefined Behavior)** — Code whose behavior the C++ standard imposes no requirements on. The compiler may assume UB never occurs, enabling aggressive optimizations that can produce surprising results. Common UBs: signed overflow, null dereference, data races, out-of-bounds access. *(See: Part 1 — C++ Foundations)*

**Unified Memory** — A CUDA feature that creates a single managed memory pool accessible by both CPU and GPU. The driver migrates pages on demand, simplifying programming at the potential cost of migration overhead. *(See: Part 6 — CUDA Foundations)*

## V

**Variadic Template** — A template that accepts a variable number of type or value parameters using parameter packs (`template<typename... Args>`). Foundation for `std::tuple`, `std::variant`, and perfect-forwarding `make_unique`. *(See: Part 3 — Advanced C++)*

**Value Category** — The classification of every C++ expression as either an *lvalue*, *xvalue*, or *prvalue* (collectively *glvalue* or *rvalue*). Value categories govern which operations (copy, move, bind-to-reference) are valid. *(See: Part 2 — Intermediate C++)*

**Virtual Table (vtable)** — A compiler-generated lookup table of function pointers used to dispatch virtual function calls at runtime. Each polymorphic class has its own vtable; each object holds a hidden pointer (vptr) to its class's table. *(See: Part 1 — C++ Foundations)*

## W

**Warm-up (GPU)** — The practice of running a kernel once before benchmarking to trigger JIT compilation (PTX → SASS), load caches, and stabilize GPU clock frequencies. Without warm-up, the first kernel launch is misleadingly slow. *(See: Part 7 — CUDA Advanced)*

**Warp** — A group of 32 threads that execute in lock-step on a single SM partition. The warp is the GPU's fundamental scheduling unit; all threads in a warp execute the same instruction simultaneously (SIMT). *(See: Part 6 — CUDA Foundations)*

**Warp Divergence** — A condition where threads within a warp follow different control-flow paths (e.g., if/else branches), causing the hardware to serialize both paths and reducing effective parallelism. *(See: Part 6 — CUDA Foundations)*

**Warp Scheduler** — Hardware logic on each SM that selects eligible warps for execution every cycle. Modern SMs have multiple warp schedulers (e.g., 4 per SM on Ampere), enabling instruction-level parallelism across warps. *(See: Part 6 — CUDA Foundations)*

**Warp Shuffle** — Intrinsic functions (`__shfl_sync`, `__shfl_xor_sync`, etc.) that allow threads within a warp to directly read each other's registers without going through shared memory. Extremely fast for reductions and prefix sums. *(See: Part 7 — CUDA Advanced)*

**WMMA (Warp Matrix Multiply-Accumulate)** — A CUDA C++ API (`nvcuda::wmma`) that exposes Tensor Core operations at the warp level, enabling matrix fragments to be loaded, multiplied, and stored with a few intrinsic calls. *(See: Part 7 — CUDA Advanced)*

## Z

**Zero-Copy Memory** — Host memory mapped into the GPU's address space, allowing the GPU to access it directly over PCIe without an explicit `cudaMemcpy`. Useful for small or infrequently accessed data, but bandwidth is limited by the PCIe bus. *(See: Part 6 — CUDA Foundations)*

---

## Quick-Reference: Precision Formats

| Format | Bits | Exponent | Mantissa | Typical Use |
|--------|------|----------|----------|-------------|
| FP32 | 32 | 8 | 23 | Master weights, CPU math |
| TF32 | 19* | 8 | 10 | Tensor Core internal (Ampere+) |
| FP16 | 16 | 5 | 10 | Mixed-precision training/inference |
| BF16 | 16 | 8 | 7 | Training (wider dynamic range) |
| FP8 E4M3 | 8 | 4 | 3 | Inference (Hopper+) |
| FP8 E5M2 | 8 | 5 | 2 | Training gradients (Hopper+) |
| INT8 | 8 | — | — | Post-training quantization |

*\*TF32 is stored as 32-bit internally but only 19 bits are significant during Tensor Core computation.*

---

## Quick-Reference: GPU Memory Hierarchy

| Memory | Scope | Speed | Size (typical) |
|--------|-------|-------|-----------------|
| Registers | Thread | ~1 cycle | 256 KB/SM |
| Shared Memory | Block | ~20–30 cycles | 48–228 KB/SM |
| L1 Cache | SM | ~30 cycles | 128–256 KB/SM |
| L2 Cache | Device | ~200 cycles | 6–50 MB |
| Global (HBM) | Device | ~400–600 cycles | 16–80 GB |
| Constant | Device (cached) | ~1–100 cycles* | 64 KB |

*\*Constant memory is fast when all threads in a warp access the same address (broadcast); serialized otherwise.*

---

## Chapter Cross-Reference Index

| Part | Key Glossary Terms |
|------|-------------------|
| Part 1 — C++ Foundations | ABI, Abstract Class, Const, Dangling Reference, Functor, Header Guard, Heap, Inline, Iterator, Linkage, ODR, Object Slicing, RAII, Smart Pointer, Stack, STL, Translation Unit, UB, vtable |
| Part 2 — Intermediate | Aggregate, Auto, Constexpr, Copy Elision, Decltype, Exception Safety, Lambda, Lvalue/Rvalue, Move Semantics, NRVO/RVO, Overload Resolution, Value Category |
| Part 3 — Advanced C++ | Allocator, CRTP, Expression Template, PIMPL, PMR, SFINAE, Template Metaprogramming, Type Erasure |
| Part 4 — Modern C++ | Concepts, Consteval, Range |
| Part 5 — Systems | Cache Line, DRAM, False Sharing, NUMA, SIMD, Thread Safety |
| Part 6 — CUDA Foundations | Bank Conflict, Block, Coalescing, Compute Capability, Constant Memory, Device, Global Memory, Grid, Host, Kernel, Local Memory, Occupancy, PCIe, Register, Shared Memory, SM, Stream, Thread, Unified Memory, Warp, Warp Divergence |
| Part 7 — CUDA Advanced | BF16/FP16/FP8/TF32, Cooperative Groups, CUDA Core, CUDA Graph, cuBLAS, CUTLASS, Dynamic Parallelism, FLOPS, GEMM, HBM, L1/L2 Cache, Nsight Compute, Nsight Systems, NVLink, NVSwitch, P2P, PTX, SASS, Tensor Core, Thrust, Tiling, Warp Shuffle, WMMA |
| Part 8 — CUDA AI/ML | AllReduce, Attention, Autograd, Backpropagation, Batch Norm, Convolution, cuDNN, Data/Model/Pipeline Parallelism, Flash Attention, Gradient, Inference, Layer Norm, Mixed Precision, NCCL, Quantization, Softmax, Tensor, TensorRT |

---

*Last updated: 2025. Contributions welcome—open a PR to add missing terms.*
