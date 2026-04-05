# Appendix E — GPU Architecture Timeline & Comparison

> Every NVIDIA GPU architecture from Tesla to Blackwell, plus AMD, Intel, and Apple alternatives.
> Includes detailed specs, key innovations, and a decision tree for GPU selection.

---

## 1. NVIDIA Architecture Generations

### Tesla (2006) — The Beginning of GPU Compute

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2006                                       |
| **Compute Capability** | 1.0, 1.1, 1.3                          |
| **Key GPUs**       | G80 (GeForce 8800 GTX), Tesla C870/C1060   |
| **Process**        | 90nm → 65nm → 55nm                        |
| **Transistors**    | 681M (G80)                                 |
| **Key Innovation** | First CUDA-capable GPU, unified shaders    |

**What Changed:**
- Replaced fixed-function vertex/pixel shaders with unified shader processors
- Introduced CUDA programming model (C-like language for GPU compute)
- Single-precision (FP32) compute only — no FP64 until GT200 (compute 1.3)
- Shared memory introduced (16 KB per SM)
- Warp size established at 32 threads
- Memory coalescing rules were strict (half-warp aligned, sequential)

**Limitations:** No true cache hierarchy, limited error handling, primitive memory model, no double precision initially.

---

### Fermi (2010) — The First True Compute GPU

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2010                                       |
| **Compute Capability** | 2.0, 2.1                               |
| **Key GPUs**       | GF100/GF110 (Tesla C2050/C2070, GTX 480)  |
| **Process**        | 40nm                                       |
| **Transistors**    | 3.0B                                       |
| **SMs**            | 16 (512 CUDA cores)                        |
| **Key Innovation** | L1/L2 cache hierarchy, ECC memory          |

**What Changed:**
- First GPU with a real cache hierarchy: L1 (16/48 KB configurable with shared mem) + L2 (768 KB)
- ECC memory support for scientific/HPC reliability
- Unified address space (shared, global, local in one space)
- True FP64 at half-rate of FP32
- Concurrent kernel execution (up to 16 kernels)
- Faster context switching and atomic operations
- C++ support improved (virtual functions, try/catch on device)

**Impact:** Made GPUs viable for HPC. Fermi was the architecture that convinced the scientific community.

---

### Kepler (2012) — Dynamic Parallelism & Efficiency

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2012                                       |
| **Compute Capability** | 3.0, 3.2, 3.5, 3.7                     |
| **Key GPUs**       | GK104 (GTX 680), GK110 (Tesla K40, K80)   |
| **Process**        | 28nm                                       |
| **Transistors**    | 7.1B (GK110)                               |
| **SMs (SMX)**      | 15 (2880 CUDA cores, GK110)               |
| **Key Innovation** | Dynamic parallelism, Hyper-Q, GPU Boost    |

**What Changed:**
- **Dynamic Parallelism**: GPU kernels can launch other kernels (recursive algorithms, adaptive refinement)
- **Hyper-Q**: 32 simultaneous hardware work queues (was 1) — enables fine-grained task parallelism
- **GPU Boost**: Dynamic clock scaling based on thermal headroom
- Warp shuffle instructions (`__shfl`) for intra-warp communication
- Read-only data cache (texture path for constants)
- 255 registers per thread (up from 63)
- K80 = two GK210 dies on one card (dual-GPU)

**Impact:** Enabled more complex algorithms directly on GPU without CPU orchestration.

---

### Maxwell (2014) — Energy Efficiency Revolution

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2014                                       |
| **Compute Capability** | 5.0, 5.2, 5.3                          |
| **Key GPUs**       | GM204 (GTX 980), GM200 (Titan X, M40)     |
| **Process**        | 28nm                                       |
| **Transistors**    | 8.0B (GM200)                               |
| **SMs (SMM)**      | 24 (3072 CUDA cores, GM200)               |
| **Key Innovation** | 2x perf/watt over Kepler                  |

**What Changed:**
- Redesigned SM (SMM) — more efficient scheduling, larger shared memory (96 KB)
- Focus on energy efficiency (mobile GPUs, laptops)
- Reduced FP64 throughput (1/32 of FP32) — not for HPC
- Improved memory compression (lossless, reduced bandwidth needs)
- Native shared memory atomics
- Better occupancy with fewer registers per SM

**Impact:** Made GPU computing viable in power-constrained environments. Gaming-focused architecture with limited HPC use.

---

### Pascal (2016) — NVLink, Unified Memory, HBM

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2016                                       |
| **Compute Capability** | 6.0, 6.1, 6.2                          |
| **Key GPUs**       | GP100 (Tesla P100), GP104 (GTX 1080)      |
| **Process**        | 16nm FinFET                                |
| **Transistors**    | 15.3B (GP100)                              |
| **SMs**            | 56 (3584 CUDA cores, P100)                |
| **Memory**         | 16 GB HBM2 (P100), 720 GB/s               |
| **Key Innovation** | NVLink 1.0, unified memory, FP16 compute  |

**What Changed:**
- **NVLink 1.0**: 160 GB/s GPU-to-GPU (5x PCIe 3.0)
- **HBM2**: First GPU with High Bandwidth Memory (720 GB/s)
- **Unified Memory**: Page migration engine — automatic data movement between CPU and GPU
- **FP16 (half precision)**: 2x throughput for deep learning
- Compute preemption: pixel-level preemption for better multi-tasking
- Page faulting and memory oversubscription

**Impact:** The deep learning revolution GPU. P100 was the workhorse for training neural networks. Unified memory simplified programming.

---

### Volta (2017) — Tensor Cores Change Everything

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2017                                       |
| **Compute Capability** | 7.0                                    |
| **Key GPUs**       | GV100 (Tesla V100, Titan V)                |
| **Process**        | 12nm                                       |
| **Transistors**    | 21.1B                                      |
| **SMs**            | 80 (5120 CUDA cores)                       |
| **Tensor Cores**   | 640 (1st generation)                       |
| **Memory**         | 32 GB HBM2, 900 GB/s                      |
| **Key Innovation** | Tensor Cores, independent thread scheduling|

**What Changed:**
- **Tensor Cores**: Mixed-precision matrix multiply-accumulate (4×4 FP16 → FP32 accumulate). Up to 125 TFLOPS
- **Independent Thread Scheduling**: Each thread has its own program counter and call stack. Enables fine-grained synchronization, fixes warp-level assumptions
- **NVLink 2.0**: 300 GB/s
- Cooperative groups API for flexible thread synchronization
- Unified L1 cache + shared memory (128 KB configurable)
- HBM2 with 900 GB/s bandwidth

**Impact:** Defined the modern AI training era. Tensor Cores made mixed-precision training practical, delivering 3x speedup over Pascal for deep learning.

---

### Turing (2018) — Ray Tracing & Inference

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2018                                       |
| **Compute Capability** | 7.5                                    |
| **Key GPUs**       | TU102 (RTX 2080 Ti), TU104 (T4)           |
| **Process**        | 12nm                                       |
| **Transistors**    | 18.6B (TU102)                              |
| **SMs**            | 72 (4608 CUDA cores, TU102)               |
| **Tensor Cores**   | 576 (2nd gen) — added INT8, INT4           |
| **Key Innovation** | RT Cores, INT8/INT4 inference              |

**What Changed:**
- **RT Cores**: Hardware ray tracing acceleration (BVH traversal)
- **2nd gen Tensor Cores**: Added INT8 (for inference) and INT4 support
- **T4**: Inference-optimized GPU (75W TDP, INT8 at 130 TOPS)
- Mesh shading pipeline
- Variable rate shading
- Turing was primarily a gaming architecture with important inference implications

**Impact:** Brought real-time ray tracing to consumers. T4 became the most widely deployed inference GPU in cloud.

---

### Ampere (2020) — TF32, MIG, 3rd Gen Tensor Cores

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2020                                       |
| **Compute Capability** | 8.0 (A100), 8.6 (RTX 3090), 8.9 (L40) |
| **Key GPUs**       | GA100 (A100), GA102 (RTX 3090, A6000)     |
| **Process**        | 7nm (TSMC) for A100, Samsung 8nm for GA102|
| **Transistors**    | 54.2B (GA100)                              |
| **SMs**            | 108 (6912 CUDA cores, A100)               |
| **Tensor Cores**   | 432 (3rd gen)                              |
| **Memory**         | 80 GB HBM2e, 2,039 GB/s (A100)            |
| **Key Innovation** | TF32, MIG, sparsity, 3rd gen Tensor Cores |

**What Changed:**
- **TF32 (TensorFloat-32)**: 19-bit format (8-bit exponent, 10-bit mantissa) — FP32-like range with TF32 Tensor Core throughput. Transparent to code using FP32
- **MIG (Multi-Instance GPU)**: Partition A100 into up to 7 isolated GPU instances
- **3rd gen Tensor Cores**: BF16 support, TF32, structural sparsity (2:4 pattern for 2x throughput)
- **NVLink 3.0**: 600 GB/s
- **NVSwitch**: All-to-all GPU fabric in DGX systems
- L2 cache: 40 MB (up from 6 MB) with residency control
- Asynchronous copy (global → shared without using registers)
- PCIe Gen4 support

**Impact:** The workhorse of the LLM revolution. A100 trained GPT-3, PaLM, LLaMA, and most major models. MIG enabled efficient cloud inference.

---

### Hopper (2022) — Transformer Engine & FP8

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2022                                       |
| **Compute Capability** | 9.0                                    |
| **Key GPUs**       | GH100 (H100 SXM, H100 PCIe)              |
| **Process**        | 4nm (TSMC)                                 |
| **Transistors**    | 80B                                        |
| **SMs**            | 132 (16,896 CUDA cores)                   |
| **Tensor Cores**   | 528 (4th gen)                              |
| **Memory**         | 80 GB HBM3, 3,350 GB/s                    |
| **Key Innovation** | Transformer Engine, FP8, Thread Block Clusters |

**What Changed:**
- **Transformer Engine (TE)**: Automatic per-layer FP8/FP16 selection with dynamic scaling. 6x transformer throughput over A100
- **FP8**: Two formats — E4M3 (more precision) and E5M2 (more range). 1,979 TFLOPS
- **Thread Block Clusters**: Group up to 16 blocks, enabling distributed shared memory (DSMEM) across blocks
- **DPX Instructions**: Hardware-accelerated dynamic programming (Smith-Waterman, Floyd-Warshall)
- **NVLink 4.0**: 900 GB/s (7th generation NVLink)
- **Asynchronous execution**: TMA (Tensor Memory Accelerator) for async bulk data movement
- **Confidential computing**: Hardware-based GPU security
- L2 cache: 50 MB
- PCIe Gen5 support

**H200 Variant (2024):**
- Same GH100 die, upgraded memory: 141 GB HBM3e at 4,800 GB/s
- 1.4-1.9x inference speedup over H100 (more memory for KV cache)

**Impact:** Specifically designed for transformer workloads. H100/H200 are the current gold standard for LLM training and inference.

---

### Blackwell (2024) — Two-Die Compute Monster

| Attribute           | Details                                    |
|--------------------|--------------------------------------------|
| **Year**           | 2024                                       |
| **Compute Capability** | 10.0                                   |
| **Key GPUs**       | GB100 (B200, B100), GB202 (RTX 5090)      |
| **Process**        | 4nm (TSMC), chiplet design                 |
| **Transistors**    | 208B (two dies)                            |
| **SMs**            | 160 (20,480 CUDA cores)                   |
| **Tensor Cores**   | 640 (5th gen)                              |
| **Memory**         | 192 GB HBM3e (B200), 8,000 GB/s           |
| **Key Innovation** | Dual-die, NVLink 5.0, FP4, 5th gen TCs    |

**What Changed:**
- **Dual-die design**: Two compute dies connected via 10 TB/s chip-to-chip interconnect — acts as a single GPU
- **5th gen Tensor Cores**: FP4 support (9,000 TFLOPS!), improved FP8 (4,500 TFLOPS)
- **NVLink 5.0**: 1,800 GB/s (doubled from Hopper)
- **2nd gen Transformer Engine**: Better FP8 handling, micro-tensor scaling
- **Decompression engine**: Hardware LZ4/Snappy decompression for data loading
- **RAS (Reliability)**: Enhanced error detection and correction
- **Confidential computing v2**: Full VM-level GPU security

**B300 Variant (2025):**
- 288 GB HBM3e, 8,000 GB/s
- Higher TDP (1200W vs 1000W)
- ~10,000 TFLOPS FP4

**Impact:** Designed for trillion-parameter model training and massive inference workloads. The architectural shift to chiplets mirrors AMD CPU strategy.

---

## 2. Architecture Comparison Table

| Architecture | Year | Process | Transistors | SMs  | CUDA Cores | Tensor Cores | Memory        | BW (GB/s) | Key Innovation                  |
|-------------|------|---------|-------------|------|------------|-------------|---------------|-----------|----------------------------------|
| Tesla       | 2006 | 90nm    | 681M        | 16   | 128        | —           | 768 MB GDDR3  | 86        | First CUDA GPU                   |
| Fermi       | 2010 | 40nm    | 3.0B        | 16   | 512        | —           | 6 GB GDDR5    | 177       | L1/L2 cache, ECC                |
| Kepler      | 2012 | 28nm    | 7.1B        | 15   | 2880       | —           | 12 GB GDDR5   | 288       | Dynamic parallelism              |
| Maxwell     | 2014 | 28nm    | 8.0B        | 24   | 3072       | —           | 12 GB GDDR5   | 336       | Energy efficiency                |
| Pascal      | 2016 | 16nm    | 15.3B       | 56   | 3584       | —           | 16 GB HBM2    | 720       | NVLink, HBM2, FP16              |
| Volta       | 2017 | 12nm    | 21.1B       | 80   | 5120       | 640 (1st)   | 32 GB HBM2    | 900       | Tensor Cores                     |
| Turing      | 2018 | 12nm    | 18.6B       | 72   | 4608       | 576 (2nd)   | 11 GB GDDR6   | 616       | RT Cores, INT8                   |
| Ampere      | 2020 | 7nm     | 54.2B       | 108  | 6912       | 432 (3rd)   | 80 GB HBM2e   | 2,039     | TF32, MIG, sparsity             |
| Hopper      | 2022 | 4nm     | 80B         | 132  | 16,896     | 528 (4th)   | 80 GB HBM3    | 3,350     | Transformer Engine, FP8          |
| Blackwell   | 2024 | 4nm     | 208B        | 160  | 20,480     | 640 (5th)   | 192 GB HBM3e  | 8,000     | Dual-die, NVLink 5.0, FP4       |

---

## 3. Compute Capability Quick Reference

| CC    | Architecture | Max Threads/Block | Max Blocks/SM | Registers/SM | Shared Mem/SM | Warps/SM |
|-------|-------------|-------------------|---------------|-------------|---------------|----------|
| 1.x   | Tesla       | 512               | 8             | 8K-16K      | 16 KB         | 24-32    |
| 2.x   | Fermi       | 1024              | 8             | 32K         | 48 KB         | 48       |
| 3.x   | Kepler      | 1024              | 16            | 64K         | 48 KB         | 64       |
| 5.x   | Maxwell     | 1024              | 32            | 64K         | 96 KB         | 64       |
| 6.x   | Pascal      | 1024              | 32            | 64K         | 96 KB         | 64       |
| 7.0   | Volta       | 1024              | 32            | 64K         | 96 KB (configurable) | 64 |
| 7.5   | Turing      | 1024              | 16            | 64K         | 64 KB         | 32       |
| 8.0   | Ampere      | 1024              | 32            | 64K         | 164 KB        | 64       |
| 8.6   | Ampere (GA102) | 1024           | 16            | 64K         | 100 KB        | 48       |
| 9.0   | Hopper      | 1024              | 32            | 64K         | 228 KB        | 64       |
| 10.0  | Blackwell   | 1024              | 32            | 64K         | 228 KB        | 64       |

---

## 4. Alternative GPU Compute Platforms

### AMD ROCm / HIP

| Aspect            | Details                                          |
|-------------------|--------------------------------------------------|
| **Platform**      | ROCm (Radeon Open Compute)                       |
| **Language**      | HIP (Heterogeneous Interface for Portability)    |
| **Compiler**      | hipcc (wraps clang)                              |
| **Current GPUs**  | MI300X (192 GB HBM3, 5.3 TB/s)                  |
| **Ecosystem**     | hipBLAS, hipFFT, MIOpen (cuDNN equivalent)       |

**Key Points:**
- HIP is syntactically nearly identical to CUDA — most code ports with minimal changes
- `hipify-clang`: automated source-to-source translation from CUDA to HIP
- ROCm supports PyTorch and JAX natively
- MI300X competitive with H100 for LLM inference (more memory, similar bandwidth)
- Weaker ecosystem: fewer libraries, less documentation, smaller community
- `hipify-perl` for quick conversion, `hipify-clang` for production conversion
- Key limitation: no Tensor Core equivalent (CDNA matrix cores are less mature)

**API Mapping (Selection):**
| CUDA               | HIP                    |
|---------------------|------------------------|
| `cudaMalloc`        | `hipMalloc`            |
| `cudaMemcpy`        | `hipMemcpy`            |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaStream_t`      | `hipStream_t`          |
| `__shared__`        | `__shared__`           |
| `__syncthreads()`   | `__syncthreads()`      |
| `atomicAdd`         | `atomicAdd`            |

---

### Intel oneAPI / SYCL / DPC++

| Aspect            | Details                                          |
|-------------------|--------------------------------------------------|
| **Platform**      | oneAPI                                           |
| **Language**      | DPC++ (Data Parallel C++), SYCL 2020 compliant   |
| **Compiler**      | icpx (Intel DPC++/C++ Compiler)                  |
| **Current GPUs**  | Intel Data Center Max (Ponte Vecchio), Arc        |
| **Ecosystem**     | oneMKL, oneDNN, oneDAL                           |

**Key Points:**
- SYCL is an open standard (Khronos Group) — not vendor-locked
- DPC++ is Intel's SYCL implementation with extensions
- Can target CPUs, GPUs, and FPGAs from a single codebase
- Codeplay's oneAPI plugins add NVIDIA and AMD GPU support
- Good for heterogeneous computing, weaker for pure GPU performance
- SYCLomatic tool converts CUDA to SYCL/DPC++

---

### Apple Metal Compute

| Aspect            | Details                                          |
|-------------------|--------------------------------------------------|
| **Platform**      | Metal                                            |
| **Language**      | Metal Shading Language (MSL), C++14-based         |
| **Framework**     | Metal Performance Shaders (MPS), MLX              |
| **Current GPUs**  | M3 Ultra (76-core GPU, 192 GB unified memory)    |
| **Ecosystem**     | MPS, MLX (ML framework), Core ML                 |

**Key Points:**
- Unified memory architecture — CPU and GPU share the same memory pool
- No explicit data transfers needed (major advantage for some workloads)
- MSL is C++14-based with GPU-specific extensions
- MPS provides optimized kernels for ML operations
- MLX (Apple's ML framework) optimized for Apple Silicon
- Limited to Apple hardware — no server/cloud deployment
- GPU families define feature sets (Apple1 through Apple9+)
- Compute command encoder dispatches threadgroups (similar to CUDA blocks)

---

## 5. GPU Selection Decision Tree

```
What is your primary workload?
│
├── Training large models (>10B parameters)
│   ├── Budget: Enterprise → H100/H200 SXM (NVLink required)
│   ├── Budget: Maximum  → B200/B300 (latest, most TFLOPS)
│   └── AMD alternative → MI300X (competitive, better memory)
│
├── Training medium models (1-10B parameters)
│   ├── Research lab    → A100 80GB (proven, cost-effective)
│   ├── Cloud/short-term → H100 instances (Lambda, CoreWeave)
│   └── On-premise     → A100 or H100 PCIe
│
├── Inference (serving models)
│   ├── Latency-critical → H100/H200 (best per-token latency)
│   ├── Throughput-focused → A100 or L40S (cost per token)
│   ├── Edge/mobile    → Jetson Orin, T4
│   └── Apple ecosystem → M-series Macs with MLX
│
├── HPC / Scientific Computing
│   ├── FP64 critical  → A100, H100 (strong FP64)
│   ├── Not FP64 critical → H100 (TF32/FP32 is huge)
│   └── Multi-node     → H100 SXM with NVLink + InfiniBand
│
├── Development / Prototyping
│   ├── Local machine  → RTX 4090 (24GB, great perf/$)
│   ├── Cloud dev      → A100 instances (spot pricing)
│   └── Learning CUDA  → Any GTX/RTX GPU (even laptop)
│
├── Computer Vision / Graphics
│   ├── Training       → A100 / H100
│   ├── Inference + rendering → RTX 4090 / L40 (RT + Tensor)
│   └── Video processing → T4 (NVENC/NVDEC)
│
└── Cross-platform / Portable code
    ├── NVIDIA + AMD   → HIP (write once, target both)
    ├── Any vendor      → SYCL/DPC++ (open standard)
    ├── Python-centric → Triton (portable GPU kernels)
    └── Apple only     → Metal / MLX
```

---

## 6. Trends & Future Directions

### Architectural Trends
1. **Chiplet designs**: Blackwell's dual-die approach will expand — expect 4+ die GPUs
2. **Specialized engines**: More fixed-function units (Transformer Engine, DPX, decompression)
3. **Memory bandwidth**: HBM4 expected to deliver 6-8 TB/s per stack
4. **Interconnect**: NVLink scaling continues; CXL for memory disaggregation
5. **Power**: TDPs rising (400W → 700W → 1000W → 1200W) — cooling is the constraint

### Compute Format Trends
| Era          | Primary Format | Training        | Inference        |
|-------------|----------------|-----------------|------------------|
| 2016-2018   | FP32           | FP32            | FP32             |
| 2018-2020   | FP16           | Mixed (FP16+FP32) | FP16/INT8      |
| 2020-2022   | BF16/TF32      | BF16 + FP32     | INT8             |
| 2022-2024   | FP8            | FP8 + BF16      | FP8/INT8         |
| 2024+       | FP4            | FP8 + BF16      | FP4/INT4         |
| Future      | Custom formats | Microscaling    | 1-2 bit (binary) |

### Market Landscape (2024-2025)
- **NVIDIA**: 80%+ market share in AI training, dominant ecosystem
- **AMD**: Competitive hardware (MI300X), rapidly improving software (ROCm 6+)
- **Intel**: Struggling in discrete GPU but strong in CPU inference (AMX)
- **Apple**: Dominant in consumer ML, no server presence
- **Custom silicon**: Google TPU, AWS Trainium/Inferentia, Microsoft Maia
- **Startups**: Groq (LPU), Cerebras (wafer-scale), SambaNova, Graphcore (acquired)

---

*Appendix E — GPU Architecture Timeline & Comparison — Part of the CPP-CUDA-Mastery series*
