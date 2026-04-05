# 🚀 C++ & CUDA Mastery — From Beginner to GPU Programming Expert

A comprehensive, self-contained learning repository covering **modern C++ (C++11 → C++26)** and
**NVIDIA CUDA GPU programming** from first principles to production AI/ML pipelines. Every chapter
pairs deep conceptual explanations with production-grade code examples, performance annotations,
and exercises — so you can go from writing your first `Hello, World!` to implementing Flash
Attention on Tensor Cores.

---

## 📊 Repository at a Glance

| Metric | Count |
|---|---|
| **Total files** | 114 |
| **Total lines** | 80 000+ |
| **Chapters** | 70 |
| **Hands-on projects** | 15 + 2 capstones |
| **CUDA Labs** | 10 |
| **CUDA Cookbook & Guides** | 5 |
| **CUDA Optimization Case Studies** | 5 |
| **Appendices** | 6 |

---

## 🗺️ Learning Roadmap

```mermaid
graph LR
    A["Part 1\nC++ Foundations"] --> B["Part 2\nC++ Intermediate"]
    B --> C["Part 3\nC++ Advanced"]
    C --> D["Part 4\nModern C++ Evolution"]
    D --> E["Part 5\nSystems Programming"]
    E --> F["Part 6\nCUDA Foundations"]
    F --> G["Part 7\nCUDA Advanced"]
    G --> H["Part 8\nCUDA for AI/ML"]
    H --> I["Part 9\nProjects & Capstones"]

    style A fill:#4CAF50,color:#fff
    style B fill:#4CAF50,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#FF9800,color:#fff
    style E fill:#FF9800,color:#fff
    style F fill:#2196F3,color:#fff
    style G fill:#2196F3,color:#fff
    style H fill:#9C27B0,color:#fff
    style I fill:#F44336,color:#fff
```

> **Green** = C++ core · **Orange** = advanced C++ · **Blue** = CUDA · **Purple** = AI/ML · **Red** = projects

---

## 📚 Parts Overview

| # | Part | Chapters | Focus |
|---|---|---|---|
| 1 | [C++ Foundations](#part-1--c-foundations) | Ch 01 – 12 | Variables, control flow, pointers, memory, error handling |
| 2 | [C++ Intermediate](#part-2--c-intermediate) | Ch 13 – 22 | OOP, STL, smart pointers, templates, move semantics |
| 3 | [C++ Advanced](#part-3--c-advanced) | Ch 23 – 32 | TMP, concurrency, lock-free, design patterns, allocators |
| 4 | [Modern C++ Evolution](#part-4--modern-c-evolution) | Ch 33 – 38 | C++11 → C++26 standard-by-standard deep-dive |
| 5 | [Systems Programming](#part-5--systems-programming) | Ch 39 – 43 | Memory architecture, perf engineering, OS, networking |
| 6 | [CUDA Foundations](#part-6--cuda-foundations) | Ch 44 – 53 | GPU architecture, kernels, memory hierarchy, warps |
| 7 | [CUDA Advanced](#part-7--cuda-advanced) | Ch 54 – 63 | Streams, multi-GPU, graphs, Tensor Cores, profiling |
| 8 | [CUDA for AI/ML](#part-8--cuda-for-aiml) | Ch 64 – 70 | cuBLAS, cuDNN, TensorRT, Flash Attention, PyTorch C++ |
| 9 | [Projects & Capstones](#part-9--projects--capstones) | P01 – P15, C01 – C02 | 15 guided projects + 2 full capstones |
| — | [CUDA Labs](#-cuda-hands-on-labs) | Lab 01 – 10 | Step-by-step guided GPU experiments |
| — | [CUDA Cookbook](#-cuda-cookbook--guides) | 5 guides | Patterns, anti-patterns, benchmarks, mindset, code reading |
| — | [CUDA Case Studies](#-cuda-optimization-case-studies) | CS 01 – 05 | Softmax, LayerNorm, GEMM, Attention, Fusion |
| — | [Appendices](#appendices) | A – F | Interview guide, perf cheat sheet, CMake, debugging |

---

## Part 1 — C++ Foundations

> **12 chapters** · From zero to confident with pointers, memory, and error handling.

| # | Chapter | File |
|---|---|---|
| 01 | Hello C++ & Your First Program | [01_Hello_CPP.md](Part-01-CPP-Foundations/01_Hello_CPP.md) |
| 02 | Variables, Types & Memory Layout | [02_Variables_Types_Memory.md](Part-01-CPP-Foundations/02_Variables_Types_Memory.md) |
| 03 | Operators & Expressions | [03_Operators_Expressions.md](Part-01-CPP-Foundations/03_Operators_Expressions.md) |
| 04 | Control Flow | [04_Control_Flow.md](Part-01-CPP-Foundations/04_Control_Flow.md) |
| 05 | Functions: The Building Blocks | [05_Functions.md](Part-01-CPP-Foundations/05_Functions.md) |
| 06 | Arrays, Strings & C-Style Legacy | [06_Arrays_Strings.md](Part-01-CPP-Foundations/06_Arrays_Strings.md) |
| 07 | Pointers Deep Dive | [07_Pointers_Deep_Dive.md](Part-01-CPP-Foundations/07_Pointers_Deep_Dive.md) |
| 08 | References & Value Categories | [08_References_Value_Categories.md](Part-01-CPP-Foundations/08_References_Value_Categories.md) |
| 09 | Dynamic Memory | [09_Dynamic_Memory.md](Part-01-CPP-Foundations/09_Dynamic_Memory.md) |
| 10 | Structs & Enums | [10_Structs_Enums.md](Part-01-CPP-Foundations/10_Structs_Enums.md) |
| 11 | Namespaces & Headers | [11_Namespaces_Headers.md](Part-01-CPP-Foundations/11_Namespaces_Headers.md) |
| 12 | Error Handling | [12_Error_Handling.md](Part-01-CPP-Foundations/12_Error_Handling.md) |

---

## Part 2 — C++ Intermediate

> **10 chapters** · OOP, the STL, templates, and modern ownership patterns.

| # | Chapter | File |
|---|---|---|
| 13 | Object-Oriented Programming: Classes | [13_OOP_Classes.md](Part-02-CPP-Intermediate/13_OOP_Classes.md) |
| 14 | Inheritance & Polymorphism | [14_Inheritance_Polymorphism.md](Part-02-CPP-Intermediate/14_Inheritance_Polymorphism.md) |
| 15 | Operator Overloading | [15_Operator_Overloading.md](Part-02-CPP-Intermediate/15_Operator_Overloading.md) |
| 16 | Smart Pointers & Ownership | [16_Smart_Pointers.md](Part-02-CPP-Intermediate/16_Smart_Pointers.md) |
| 17 | STL Containers Deep-Dive | [17_STL_Containers.md](Part-02-CPP-Intermediate/17_STL_Containers.md) |
| 18 | STL Algorithms & Iterators | [18_STL_Algorithms.md](Part-02-CPP-Intermediate/18_STL_Algorithms.md) |
| 19 | Lambda Expressions & Functional C++ | [19_Lambdas_Functional.md](Part-02-CPP-Intermediate/19_Lambdas_Functional.md) |
| 20 | Move Semantics & Perfect Forwarding | [20_Move_Semantics.md](Part-02-CPP-Intermediate/20_Move_Semantics.md) |
| 21 | Templates: Generic Programming | [21_Templates.md](Part-02-CPP-Intermediate/21_Templates.md) |
| 22 | File I/O & Serialization | [22_File_IO.md](Part-02-CPP-Intermediate/22_File_IO.md) |

---

## Part 3 — C++ Advanced

> **10 chapters** · Template metaprogramming, concurrency, lock-free data structures, and design patterns.

| # | Chapter | File |
|---|---|---|
| 23 | Template Metaprogramming | [23_Template_Metaprogramming.md](Part-03-CPP-Advanced/23_Template_Metaprogramming.md) |
| 24 | Concepts & Constraints (C++20) | [24_Concepts_Constraints.md](Part-03-CPP-Advanced/24_Concepts_Constraints.md) |
| 25 | Concurrency & Multithreading | [25_Concurrency.md](Part-03-CPP-Advanced/25_Concurrency.md) |
| 26 | Async & Parallel Patterns | [26_Async_Parallel.md](Part-03-CPP-Advanced/26_Async_Parallel.md) |
| 27 | Memory Model & Lock-Free Programming | [27_Memory_Model_Lock_Free.md](Part-03-CPP-Advanced/27_Memory_Model_Lock_Free.md) |
| 28 | Design Patterns in Modern C++ | [28_Design_Patterns.md](Part-03-CPP-Advanced/28_Design_Patterns.md) |
| 29 | Compile-Time Programming | [29_Compile_Time_Programming.md](Part-03-CPP-Advanced/29_Compile_Time_Programming.md) |
| 30 | Type Erasure & Polymorphism Patterns | [30_Type_Erasure.md](Part-03-CPP-Advanced/30_Type_Erasure.md) |
| 31 | Custom Allocators & Memory Pools | [31_Custom_Allocators.md](Part-03-CPP-Advanced/31_Custom_Allocators.md) |
| 32 | Build Systems & Tooling | [32_Build_Systems_Tooling.md](Part-03-CPP-Advanced/32_Build_Systems_Tooling.md) |

---

## Part 4 — Modern C++ Evolution

> **6 chapters** · A standard-by-standard tour from C++11 through C++26.

| # | Chapter | File |
|---|---|---|
| 33 | C++11/14: The Modern Revolution | [33_CPP11_14_Revolution.md](Part-04-Modern-CPP-Evolution/33_CPP11_14_Revolution.md) |
| 34 | C++17: Practical Enhancements | [34_CPP17_Enhancements.md](Part-04-Modern-CPP-Evolution/34_CPP17_Enhancements.md) |
| 35 | C++20: The Big Four | [35_CPP20_Big_Four.md](Part-04-Modern-CPP-Evolution/35_CPP20_Big_Four.md) |
| 36 | C++23: Refinements and Additions | [36_CPP23_Refinements.md](Part-04-Modern-CPP-Evolution/36_CPP23_Refinements.md) |
| 37 | C++26: The Bleeding Edge | [37_CPP26_Bleeding_Edge.md](Part-04-Modern-CPP-Evolution/37_CPP26_Bleeding_Edge.md) |
| 38 | C++ Standards Comparison & Migration Guide | [38_Standards_Comparison.md](Part-04-Modern-CPP-Evolution/38_Standards_Comparison.md) |

---

## Part 5 — Systems Programming

> **5 chapters** · Hardware-aware performance engineering, OS interaction, and cross-language interop.

| # | Chapter | File |
|---|---|---|
| 39 | Memory Architecture Deep-Dive | [39_Memory_Architecture.md](Part-05-Systems-Programming/39_Memory_Architecture.md) |
| 40 | Performance Engineering | [40_Performance_Engineering.md](Part-05-Systems-Programming/40_Performance_Engineering.md) |
| 41 | OS Interaction & System Calls | [41_OS_Interaction.md](Part-05-Systems-Programming/41_OS_Interaction.md) |
| 42 | Networking in C++ | [42_Networking.md](Part-05-Systems-Programming/42_Networking.md) |
| 43 | Interop: C, Python, Rust | [43_Interop.md](Part-05-Systems-Programming/43_Interop.md) |

---

## Part 6 — CUDA Foundations

> **10 chapters** · GPU architecture, the CUDA programming model, memory hierarchy, and warp-level primitives.

| # | Chapter | File |
|---|---|---|
| 44 | GPU Architecture — From Transistors to Tensor Cores | [44_GPU_Architecture.md](Part-06-CUDA-Foundations/44_GPU_Architecture.md) |
| 45 | CUDA Programming Model | [45_CUDA_Programming_Model.md](Part-06-CUDA-Foundations/45_CUDA_Programming_Model.md) |
| 46 | Your First CUDA Kernels — Hands-On | [46_First_CUDA_Kernels.md](Part-06-CUDA-Foundations/46_First_CUDA_Kernels.md) |
| 47 | CUDA Memory Architecture — The Complete Picture | [47_CUDA_Memory_Architecture.md](Part-06-CUDA-Foundations/47_CUDA_Memory_Architecture.md) |
| 48 | Warps, Threads & Execution Deep-Dive | [48_Warps_Threads_Execution.md](Part-06-CUDA-Foundations/48_Warps_Threads_Execution.md) |
| 49 | Shared Memory Mastery | [49_Shared_Memory_Mastery.md](Part-06-CUDA-Foundations/49_Shared_Memory_Mastery.md) |
| 50 | CUDA Memory Management Patterns | [50_Memory_Management_Patterns.md](Part-06-CUDA-Foundations/50_Memory_Management_Patterns.md) |
| 51 | Error Handling, Debugging & Validation | [51_Error_Handling_Debugging.md](Part-06-CUDA-Foundations/51_Error_Handling_Debugging.md) |
| 52 | Warp-Level Primitives | [52_Warp_Level_Programming.md](Part-06-CUDA-Foundations/52_Warp_Level_Programming.md) |
| 53 | CUDA Compilation Pipeline & Toolchain | [53_CUDA_Compilation.md](Part-06-CUDA-Foundations/53_CUDA_Compilation.md) |

---

## Part 7 — CUDA Advanced

> **10 chapters** · Streams, multi-GPU, CUDA Graphs, Tensor Cores, and parallel algorithm primitives.

| # | Chapter | File |
|---|---|---|
| 54 | Streams & Async Execution | [54_Streams_Concurrency.md](Part-07-CUDA-Advanced/54_Streams_Concurrency.md) |
| 55 | Performance Optimization: Memory | [55_Memory_Optimization.md](Part-07-CUDA-Advanced/55_Memory_Optimization.md) |
| 56 | Performance Optimization: Compute | [56_Compute_Optimization.md](Part-07-CUDA-Advanced/56_Compute_Optimization.md) |
| 57 | Profiling with Nsight: Hands-On | [57_Profiling_Nsight.md](Part-07-CUDA-Advanced/57_Profiling_Nsight.md) |
| 58 | Multi-GPU Programming | [58_Multi_GPU.md](Part-07-CUDA-Advanced/58_Multi_GPU.md) |
| 59 | CUDA Graphs — Launch Optimization | [59_CUDA_Graphs.md](Part-07-CUDA-Advanced/59_CUDA_Graphs.md) |
| 60 | Cooperative Groups & Advanced Synchronization | [60_Cooperative_Groups.md](Part-07-CUDA-Advanced/60_Cooperative_Groups.md) |
| 61 | Distributed GPU Computing | [61_Distributed_GPU.md](Part-07-CUDA-Advanced/61_Distributed_GPU.md) |
| 62 | Mixed Precision & Tensor Core Programming | [62_Mixed_Precision_Tensor_Cores.md](Part-07-CUDA-Advanced/62_Mixed_Precision_Tensor_Cores.md) |
| 63 | Parallel Primitives — Reduction, Scan, Sort | [63_Parallel_Primitives.md](Part-07-CUDA-Advanced/63_Parallel_Primitives.md) |

---

## Part 8 — CUDA for AI/ML

> **7 chapters** · GPU-accelerated ML libraries, custom kernels, and production inference pipelines.

| # | Chapter | File |
|---|---|---|
| 64 | cuBLAS: GPU Linear Algebra for ML | [64_cuBLAS.md](Part-08-CUDA-AI-ML/64_cuBLAS.md) |
| 65 | cuDNN — Deep Learning Primitives | [65_cuDNN.md](Part-08-CUDA-AI-ML/65_cuDNN.md) |
| 66 | Writing Custom CUDA Kernels for ML | [66_Custom_CUDA_Kernels_ML.md](Part-08-CUDA-AI-ML/66_Custom_CUDA_Kernels_ML.md) |
| 67 | PyTorch C++ Extensions & LibTorch | [67_PyTorch_CPP_Extensions.md](Part-08-CUDA-AI-ML/67_PyTorch_CPP_Extensions.md) |
| 68 | TensorRT — Production Inference Optimization | [68_TensorRT.md](Part-08-CUDA-AI-ML/68_TensorRT.md) |
| 69 | Flash Attention: Architecture Deep-Dive | [69_Flash_Attention.md](Part-08-CUDA-AI-ML/69_Flash_Attention.md) |
| 70 | The Full GPU Pipeline: Training to Deployment | [70_Full_GPU_Pipeline.md](Part-08-CUDA-AI-ML/70_Full_GPU_Pipeline.md) |

---

## Part 9 — Projects & Capstones

> **15 guided projects + 2 capstones** · Build real systems that reinforce every concept.

### Guided Projects

| # | Project | File |
|---|---|---|
| P01 | Build a JSON Parser from Scratch | [P01_JSON_Parser.md](Part-09-Projects/P01_JSON_Parser.md) |
| P02 | STL-Compatible Skip List Container | [P02_Skip_List_Container.md](Part-09-Projects/P02_Skip_List_Container.md) |
| P03 | Lock-Free Thread Pool with C++20/23 | [P03_Thread_Pool.md](Part-09-Projects/P03_Thread_Pool.md) |
| P04 | Async HTTP/1.1 Server with io_uring | [P04_HTTP_Server.md](Part-09-Projects/P04_HTTP_Server.md) |
| P05 | High-Performance Math Library with Expression Templates | [P05_Expression_Templates.md](Part-09-Projects/P05_Expression_Templates.md) |
| P06 | GPU Vector Operations with CUDA | [P06_CUDA_Vector_Ops.md](Part-09-Projects/P06_CUDA_Vector_Ops.md) |
| P07 | Parallel Histogram Computation on the GPU | [P07_CUDA_Histogram.md](Part-09-Projects/P07_CUDA_Histogram.md) |
| P08 | GPU Image Processing: Convolution Filters in CUDA | [P08_CUDA_Image_Filters.md](Part-09-Projects/P08_CUDA_Image_Filters.md) |
| P09 | CUDA Matrix Multiplication — From Naive to Tensor Cores | [P09_CUDA_MatMul_Optimization.md](Part-09-Projects/P09_CUDA_MatMul_Optimization.md) |
| P10 | Blelloch Parallel Prefix Scan & Applications | [P10_CUDA_Prefix_Scan.md](Part-09-Projects/P10_CUDA_Prefix_Scan.md) |
| P11 | N-Body Gravity Simulation on GPU | [P11_CUDA_NBody.md](Part-09-Projects/P11_CUDA_NBody.md) |
| P12 | Multi-Stage Processing Pipeline with CUDA Streams | [P12_CUDA_Stream_Pipeline.md](Part-09-Projects/P12_CUDA_Stream_Pipeline.md) |
| P13 | GPU Radix Sort from Scratch | [P13_CUDA_Radix_Sort.md](Part-09-Projects/P13_CUDA_Radix_Sort.md) |
| P14 | Multi-GPU Parallel Reduction with NCCL | [P14_Multi_GPU_Reduction.md](Part-09-Projects/P14_Multi_GPU_Reduction.md) |
| P15 | Neural Network from Scratch in CUDA (No Libraries) | [P15_CUDA_Neural_Net.md](Part-09-Projects/P15_CUDA_Neural_Net.md) |

### Capstone Projects

| # | Capstone | File |
|---|---|---|
| C01 | High-Frequency Trading Engine in Modern C++ | [C01_HFT_Engine.md](Part-09-Projects/C01_HFT_Engine.md) |
| C02 | Real-Time CUDA Ray Tracer with AI Denoising | [C02_CUDA_Ray_Tracer.md](Part-09-Projects/C02_CUDA_Ray_Tracer.md) |

---

## 🧪 CUDA Hands-On Labs

> **10 guided labs** · Run code, observe results, experiment with changes. The fastest way to build GPU intuition.

| # | Lab | Difficulty | File |
|---|-----|-----------|------|
| 01 | Your First GPU Program | 🟢 Beginner | [Lab01_First_GPU_Program.md](CUDA-Labs/Lab01_First_GPU_Program.md) |
| 02 | Thread Indexing Playground | 🟢 Beginner | [Lab02_Thread_Indexing_Playground.md](CUDA-Labs/Lab02_Thread_Indexing_Playground.md) |
| 03 | Memory Transfer Costs | 🟢 Beginner | [Lab03_Memory_Transfer_Costs.md](CUDA-Labs/Lab03_Memory_Transfer_Costs.md) |
| 04 | Shared Memory Speedup | 🟡 Intermediate | [Lab04_Shared_Memory_Speedup.md](CUDA-Labs/Lab04_Shared_Memory_Speedup.md) |
| 05 | Warp Divergence Impact | 🟡 Intermediate | [Lab05_Warp_Divergence_Impact.md](CUDA-Labs/Lab05_Warp_Divergence_Impact.md) |
| 06 | Occupancy Experiments | 🟡 Intermediate | [Lab06_Occupancy_Experiments.md](CUDA-Labs/Lab06_Occupancy_Experiments.md) |
| 07 | Coalescing Patterns | 🟡 Intermediate | [Lab07_Coalescing_Patterns.md](CUDA-Labs/Lab07_Coalescing_Patterns.md) |
| 08 | Stream Overlap | 🔴 Advanced | [Lab08_Stream_Overlap.md](CUDA-Labs/Lab08_Stream_Overlap.md) |
| 09 | Reduction Optimization Journey | 🔴 Advanced | [Lab09_Reduction_Optimization.md](CUDA-Labs/Lab09_Reduction_Optimization.md) |
| 10 | Build a Mini GEMM | 🔴 Advanced | [Lab10_Build_A_Mini_GEMM.md](CUDA-Labs/Lab10_Build_A_Mini_GEMM.md) |

---

## 📖 CUDA Cookbook & Guides

> **5 practical references** · Patterns, anti-patterns, benchmarks, and the parallel-thinking mindset shift.

| Guide | What It Is | File |
|-------|-----------|------|
| 🧩 Patterns Cookbook | 30 copy-paste CUDA patterns ("I need X → here's the code") | [CUDA_Patterns_Cookbook.md](CUDA-Cookbook/CUDA_Patterns_Cookbook.md) |
| ❌ Anti-Patterns | 20 common mistakes with before/after code | [CUDA_Anti_Patterns.md](CUDA-Cookbook/CUDA_Anti_Patterns.md) |
| 🧠 Think Parallel | CPU → GPU mindset shift with 10 worked conversions | [Think_Parallel.md](CUDA-Cookbook/Think_Parallel.md) |
| 📊 Micro-Benchmarks | 12 programs isolating specific GPU behaviors | [CUDA_Micro_Benchmarks.md](CUDA-Cookbook/CUDA_Micro_Benchmarks.md) |
| 📚 Code Reading Guide | How to read PyTorch/CUTLASS/FlashAttention CUDA code | [CUDA_Code_Reading_Guide.md](CUDA-Cookbook/CUDA_Code_Reading_Guide.md) |

---

## 🔬 CUDA Optimization Case Studies

> **5 deep-dives** · Take a real ML kernel from naive to expert, with profiling at every step.

| # | Kernel | Optimization Journey | File |
|---|--------|---------------------|------|
| CS01 | Softmax | Naive → online softmax → warp shuffle → vectorized → fused | [CS01_Softmax_Optimization.md](CUDA-Case-Studies/CS01_Softmax_Optimization.md) |
| CS02 | LayerNorm | Two-pass → Welford → warp reduction → fused residual+dropout | [CS02_LayerNorm_Optimization.md](CUDA-Case-Studies/CS02_LayerNorm_Optimization.md) |
| CS03 | GEMM | Naive → tiled → register blocking → vectorized → Tensor Cores | [CS03_GEMM_Optimization.md](CUDA-Case-Studies/CS03_GEMM_Optimization.md) |
| CS04 | Attention | Full N×N matrix → fused softmax → tiled → Flash Attention | [CS04_Attention_Optimization.md](CUDA-Case-Studies/CS04_Attention_Optimization.md) |
| CS05 | Elementwise Fusion | 3 kernels → 1 fused kernel → vectorized → template fusion | [CS05_Elementwise_Fusion.md](CUDA-Case-Studies/CS05_Elementwise_Fusion.md) |

---

## 🎯 Recommended CUDA Mastery Reading Order

> For someone who knows C++ but is new to CUDA — the optimal path through all CUDA content:

| Phase | What to Read | Goal |
|-------|-------------|------|
| **1. Architecture** | Ch 44 (GPU Architecture) → Appendix E (Timeline) | Understand the hardware |
| **2. First Kernels** | Ch 45-46 → Lab 01-02 | Write and run your first GPU code |
| **3. Memory Model** | Ch 47-48 → Lab 03 → Cookbook: Think Parallel | Understand memory hierarchy & warps |
| **4. Shared Memory** | Ch 49 → Lab 04 → Lab 07 | Master tiling and coalescing |
| **5. Debugging** | Ch 51 → Anti-Patterns guide | Learn what NOT to do |
| **6. Warp-Level** | Ch 52 → Lab 05-06 | Warp primitives & occupancy |
| **7. Optimization** | Ch 55-56 → Lab 09 → Micro-Benchmarks | Memory & compute optimization |
| **8. Profiling** | Ch 57 → Appendix D | Profile like a pro with Nsight |
| **9. Streams & Graphs** | Ch 54, 59 → Lab 08 | Async execution & launch optimization |
| **10. Case Studies** | CS01-CS05 (Softmax → GEMM → Attention) | See optimization applied to real kernels |
| **11. Multi-GPU** | Ch 58, 60-61 | Scale beyond one GPU |
| **12. Tensor Cores** | Ch 62 → Lab 10 | Mixed precision & matrix hardware |
| **13. AI/ML Libraries** | Ch 64-70 | cuBLAS, cuDNN, TensorRT, Flash Attention |
| **14. Code Reading** | Code Reading Guide | Study production CUDA from PyTorch/CUTLASS |
| **15. Build Projects** | P06 → P07 → P08 → P09 → P11 → P15 → C02 | Apply everything (beginner → capstone) |
| **16. Portability** | Appendix F | CUDA vs HIP vs SYCL vs Triton |

---

## Appendices

> **6 reference appendices** · Quick-reference material you'll reach for again and again.

| ID | Title | File |
|---|---|---|
| A | C++ & CUDA Interview Mega-Guide | [A_Interview_Mega_Guide.md](Appendices/A_Interview_Mega_Guide.md) |
| B | Numbers Every Programmer Should Know | [B_Performance_Cheat_Sheet.md](Appendices/B_Performance_Cheat_Sheet.md) |
| C | CMake & Build System Cookbook | [C_CMake_Build_Cookbook.md](Appendices/C_CMake_Build_Cookbook.md) |
| D | Debugging & Profiling Toolkit Reference | [D_Debugging_Profiling_Toolkit.md](Appendices/D_Debugging_Profiling_Toolkit.md) |
| E | GPU Architecture Timeline & Comparison | [E_GPU_Architecture_Timeline.md](Appendices/E_GPU_Architecture_Timeline.md) |
| F | From CUDA to Other GPU Languages | [F_GPU_Language_Comparison.md](Appendices/F_GPU_Language_Comparison.md) |

---

## 🏁 Getting Started

| Your level | Start here | Path |
|---|---|---|
| **Complete beginner** | [Chapter 01](Part-01-CPP-Foundations/01_Hello_CPP.md) | Part 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 |
| **Know C++ basics** | [Chapter 13](Part-02-CPP-Intermediate/13_OOP_Classes.md) | Part 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 |
| **Experienced C++ dev** | [Chapter 44](Part-06-CUDA-Foundations/44_GPU_Architecture.md) | Part 6 → 7 → 8 → 9 (refer to Parts 3-5 as needed) |
| **Know CUDA basics** | [Chapter 54](Part-07-CUDA-Advanced/54_Streams_Concurrency.md) | Part 7 → 8 → 9 |
| **Interview prep** | [Appendix A](Appendices/A_Interview_Mega_Guide.md) | Appendices A & B + skim project chapters |

---

## 📈 Repository Stats

```
Parts ............ 9
Chapters ......... 70  (Ch 01 – Ch 70)
Projects ......... 15  (P01 – P15)
Capstones ........ 2   (C01 – C02)
CUDA Labs ........ 10  (Lab 01 – Lab 10)
CUDA Cookbook ..... 5   (Patterns, Anti-Patterns, Think Parallel, Benchmarks, Code Reading)
Case Studies ..... 5   (CS01 – CS05: Softmax, LayerNorm, GEMM, Attention, Fusion)
Appendices ....... 6   (A – F)
Total files ...... 114
Total lines ...... 80 000+
```

---

> **Tip:** Star ⭐ this repo to bookmark your progress, and use GitHub's built-in
> table-of-contents (☰ icon on any `.md` file) to navigate within chapters.
