# Appendix B — Numbers Every Programmer Should Know

> Performance reference: latencies, bandwidths, GPU specs, and estimation tools.
> Keep this open during design reviews, interviews, and optimization sessions.

---

## 1. CPU Latency Reference Table

| Level              | Latency         | Approx Cycles (4 GHz) | Size (Typical)   |
|--------------------|-----------------|----------------------|-------------------|
| Register           | ~0.25 ns        | 1                    | Few KB total      |
| L1 Cache Hit       | ~1 ns           | 4                    | 32-64 KB/core     |
| L2 Cache Hit       | ~4 ns           | 16                   | 256 KB-1 MB/core  |
| L3 Cache Hit       | ~12 ns          | 48                   | 8-64 MB shared    |
| DRAM (Main Memory) | ~100 ns         | 400                  | 16-512 GB         |
| SSD (NVMe random)  | ~100 µs         | 400,000              | 0.5-8 TB          |
| SSD (NVMe seq)     | ~10 µs          | 40,000               | —                 |
| HDD (random)       | ~10 ms          | 40,000,000           | 1-20 TB           |
| Network (same DC)  | ~0.5-1 ms       | 2-4 M                | —                 |
| Network (cross DC) | ~10-100 ms      | 40-400 M             | —                 |
| Network (cross-continent) | ~100-200 ms | 400-800 M         | —                 |

### Key Ratios to Memorize
- **L1 → DRAM**: 100x slower
- **DRAM → SSD**: 1,000x slower
- **SSD → HDD**: 100x slower
- **DRAM → Network**: 5,000-10,000x slower

### Other CPU Latencies
| Operation                | Latency        |
|--------------------------|----------------|
| Branch misprediction     | ~15-20 cycles  |
| Mutex lock/unlock (uncontended) | ~25 ns  |
| Mutex lock (contended)   | ~1-10 µs       |
| System call (getpid)     | ~100-200 ns    |
| Context switch           | ~1-10 µs       |
| Thread creation          | ~10-25 µs      |
| Process creation (fork)  | ~100 µs-1 ms   |
| TLB miss                 | ~10-30 ns      |
| Page fault (minor)       | ~1-5 µs        |
| Page fault (major)       | ~1-10 ms       |

---

## 2. GPU Latency Reference Table

| Level                | Latency (cycles) | Latency (ns @ 1.5 GHz) | Size             |
|----------------------|-------------------|-------------------------|------------------|
| Registers            | 0                 | 0                       | 256 KB/SM        |
| Shared Memory        | ~20-30 cycles     | ~15-20 ns               | 48-228 KB/SM     |
| L1 Cache             | ~33 cycles        | ~22 ns                  | 128-256 KB/SM    |
| L2 Cache             | ~200 cycles       | ~130 ns                 | 40-60 MB         |
| HBM (Global Memory)  | ~400-600 cycles   | ~300-400 ns             | 40-192 GB        |

### GPU Operation Latencies
| Operation                    | Latency           |
|------------------------------|-------------------|
| Warp shuffle                 | ~5 cycles         |
| `__syncthreads()`            | ~10-20 cycles     |
| Atomic (shared memory)       | ~20-50 cycles     |
| Atomic (global, uncontended) | ~200-500 cycles   |
| Atomic (global, contended)   | ~1000+ cycles     |
| Kernel launch overhead       | ~5-10 µs          |
| CUDA Graph launch overhead   | ~1-3 µs           |
| `cudaMemcpy` setup (small)   | ~10-20 µs         |
| `cudaDeviceSynchronize`      | ~5-50 µs          |

---

## 3. Bandwidth Reference Table

### Memory Bandwidth
| Technology          | Bandwidth       | Notes                         |
|---------------------|-----------------|-------------------------------|
| DDR4-3200           | ~51 GB/s        | Dual-channel desktop          |
| DDR5-5600           | ~90 GB/s        | Dual-channel, latest gen      |
| DDR5-6400 (server)  | ~410 GB/s       | 8-channel server              |
| GDDR6X              | ~1 TB/s         | RTX 4090                      |
| HBM2e               | ~2 TB/s         | A100 (2,039 GB/s)             |
| HBM3                | ~3.35 TB/s      | H100 (3,350 GB/s)             |
| HBM3e               | ~4.8 TB/s       | H200 (4,800 GB/s)             |
| HBM3e (B200)        | ~8 TB/s         | B200 (8,000 GB/s)             |

### Interconnect Bandwidth
| Technology          | Bandwidth (per direction) | Notes                    |
|---------------------|---------------------------|--------------------------|
| PCIe 4.0 x16        | ~32 GB/s                  | Bidirectional: 64 GB/s   |
| PCIe 5.0 x16        | ~64 GB/s                  | Bidirectional: 128 GB/s  |
| PCIe 6.0 x16        | ~128 GB/s                 | Available 2025+          |
| NVLink 3.0 (A100)   | ~600 GB/s                 | 12 links                 |
| NVLink 4.0 (H100)   | ~900 GB/s                 | 18 links                 |
| NVLink 5.0 (B200)   | ~1,800 GB/s               | 18 links, doubled BW     |
| NVSwitch (H100)     | ~900 GB/s/GPU             | All-to-all within node   |
| InfiniBand NDR      | ~400 Gb/s (50 GB/s)       | Per port                 |
| InfiniBand NDR200   | ~400 Gb/s (50 GB/s)       | Per port                 |
| Ethernet (100G)     | ~12.5 GB/s                | Per port                 |
| Ethernet (400G)     | ~50 GB/s                  | Per port                 |
| CXL 2.0             | ~64 GB/s                  | PCIe 5.0 based           |

### Storage Bandwidth
| Technology          | Sequential Read  | Sequential Write | Random IOPS |
|---------------------|-----------------|------------------|-------------|
| HDD (7200 RPM)     | ~200 MB/s       | ~200 MB/s        | ~100        |
| SATA SSD            | ~550 MB/s       | ~520 MB/s        | ~90K        |
| NVMe SSD (Gen4)    | ~7 GB/s         | ~5 GB/s          | ~1M         |
| NVMe SSD (Gen5)    | ~14 GB/s        | ~12 GB/s         | ~2M         |
| Intel Optane        | ~2.5 GB/s       | ~2.2 GB/s        | ~550K       |

---

## 4. NVIDIA GPU Specifications Comparison

### Data Center GPUs

| Spec                | A100 (2020)    | H100 SXM (2022) | H200 (2024)   | B200 (2024)   | B300 (2025)   |
|---------------------|----------------|------------------|---------------|---------------|---------------|
| Architecture        | Ampere         | Hopper           | Hopper        | Blackwell     | Blackwell     |
| Process Node        | 7nm (TSMC)     | 4nm (TSMC)       | 4nm (TSMC)    | 4nm (TSMC)    | 4nm (TSMC)    |
| Transistors         | 54.2B          | 80B              | 80B           | 208B          | 208B          |
| SMs                 | 108            | 132              | 132           | 160 (2 dies)  | 160 (2 dies)  |
| CUDA Cores          | 6,912          | 16,896           | 16,896        | 20,480        | 20,480        |
| Tensor Cores        | 432 (3rd gen)  | 528 (4th gen)    | 528 (4th gen) | 640 (5th gen) | 640 (5th gen) |
| Memory              | 80 GB HBM2e   | 80 GB HBM3      | 141 GB HBM3e | 192 GB HBM3e | 288 GB HBM3e |
| Memory BW           | 2,039 GB/s    | 3,350 GB/s       | 4,800 GB/s   | 8,000 GB/s   | 8,000 GB/s   |
| NVLink BW           | 600 GB/s      | 900 GB/s         | 900 GB/s     | 1,800 GB/s   | 1,800 GB/s   |
| TDP                 | 400W           | 700W             | 700W          | 1000W         | 1200W         |
| Compute Capability  | 8.0            | 9.0              | 9.0           | 10.0          | 10.0          |

### Theoretical Performance (TFLOPS)

| Precision     | A100   | H100 SXM | H200   | B200     | B300     |
|---------------|--------|----------|--------|----------|----------|
| FP64          | 9.7    | 33.5     | 33.5   | 40       | 45       |
| FP32          | 19.5   | 67       | 67     | 80       | 90       |
| TF32 (Tensor) | 156    | 495      | 495    | 1,000    | 1,250    |
| FP16 (Tensor) | 312    | 989      | 989    | 2,250    | 2,500    |
| BF16 (Tensor) | 312    | 989      | 989    | 2,250    | 2,500    |
| FP8 (Tensor)  | —      | 1,979    | 1,979  | 4,500    | 5,000    |
| INT8 (Tensor) | 624    | 1,979    | 1,979  | 4,500    | 5,000    |
| FP4 (Tensor)  | —      | —        | —      | 9,000    | 10,000   |

### Consumer GPUs (for reference)

| Spec                | RTX 3090  | RTX 4090  | RTX 5090      |
|---------------------|-----------|-----------|---------------|
| Architecture        | Ampere    | Ada       | Blackwell     |
| CUDA Cores          | 10,496    | 16,384    | 21,760        |
| Memory              | 24GB GDDR6X | 24GB GDDR6X | 32GB GDDR7  |
| Memory BW           | 936 GB/s  | 1,008 GB/s | 1,792 GB/s   |
| FP32 TFLOPS         | 35.6      | 82.6      | 105           |

---

## 5. Big-O Complexity for GPU Algorithms

| Algorithm               | Work       | Span (Depth) | Notes                           |
|--------------------------|-----------|--------------|----------------------------------|
| Vector Add               | O(N)      | O(1)         | Embarrassingly parallel          |
| Reduction (sum/max)      | O(N)      | O(log N)     | Tree reduction                   |
| Prefix Scan (inclusive)   | O(N)      | O(log N)     | Blelloch or Hillis-Steele        |
| Compact / Stream Compaction | O(N)   | O(log N)     | Scan + scatter                   |
| Radix Sort               | O(N·k)   | O(k·log N)   | k = bits, highly parallel        |
| Merge Sort               | O(N log N)| O(log² N)    | GPU-friendly divide-and-conquer  |
| GEMM (N×N)               | O(N³)    | O(N)         | Tiled, Tensor Core accelerated   |
| Convolution (2D, k×k)    | O(N²k²)  | O(1)         | im2col + GEMM or direct          |
| FFT (1D)                 | O(N log N)| O(log N)     | Cooley-Tukey butterfly           |
| SpMV (sparse mat-vec)    | O(nnz)   | O(N)         | Irregular access patterns        |
| Histogram                | O(N)      | O(1)         | Atomic-based or privatized       |
| BFS (graph)              | O(V+E)   | O(D)         | D = diameter, level-synchronous  |

### Practical Throughput Expectations

| Operation         | A100 Throughput        | H100 Throughput       |
|-------------------|------------------------|-----------------------|
| GEMM FP16 (4k²)  | ~250 TFLOPS            | ~800 TFLOPS           |
| Reduction FP32    | ~1.5 TB/s effective    | ~2.5 TB/s effective   |
| Memory copy       | ~1.9 TB/s              | ~3.1 TB/s             |
| Sort (32-bit)     | ~5B keys/sec           | ~12B keys/sec         |

---

## 6. Roofline Model

### What Is It?
The roofline model plots achievable performance (FLOPS) vs arithmetic intensity
(FLOPS/byte). Performance is bounded by:

```
Attainable FLOPS = min(Peak FLOPS, Peak Bandwidth × Arithmetic Intensity)
```

### How to Compute Arithmetic Intensity

```
AI = Total FLOP / Total Bytes Transferred (from memory)
```

### Worked Example: Vector Add (FP32)

```
Per element: 1 ADD (1 FLOP), 2 reads + 1 write (12 bytes)
AI = 1 / 12 = 0.083 FLOP/byte
H100: 3,350 GB/s × 0.083 = 278 GFLOPS (memory-bound)
H100 peak FP32: 67 TFLOPS
→ Vector add is severely memory-bound (0.4% compute utilization)
```

### Worked Example: GEMM FP16 (N=4096)

```
FLOPs: 2 × N³ = 2 × 4096³ = 137.4 TFLOPS
Data: 3 × N² × 2 bytes = 3 × 16M × 2 = 96 MB
AI = 137.4 × 10¹² / (96 × 10⁶) = 1,431,655 FLOP/byte
→ Extremely compute-bound at any sane bandwidth
H100 achievable: ~800 TFLOPS (limited by peak compute, not bandwidth)
```

### Ridge Point (where compute meets bandwidth)

```
Ridge AI = Peak FLOPS / Peak Bandwidth

H100: 67 TFLOPS (FP32) / 3,350 GB/s = 20 FLOP/byte
H100: 989 TFLOPS (FP16 TC) / 3,350 GB/s = 295 FLOP/byte
```

**Interpretation**: Operations with AI below the ridge point are memory-bound;
above it, they are compute-bound.

---

## 7. Cost Estimates

### Cloud GPU Pricing (approximate, on-demand, 2024-2025)

| GPU        | AWS (p-series)  | GCP            | Azure          | Lambda Labs     |
|------------|-----------------|----------------|----------------|-----------------|
| A100 80GB  | ~$3.00/hr       | ~$2.90/hr      | ~$3.40/hr      | ~$1.10/hr       |
| H100 80GB  | ~$8.00/hr       | ~$8.20/hr      | ~$8.00/hr      | ~$2.50/hr       |
| H200       | ~$10.00/hr      | —              | —              | ~$3.00/hr       |
| RTX 4090   | —               | —              | —              | ~$0.50/hr       |

### Training Cost Estimates

| Model Size   | GPUs          | Time        | Estimated Cost (on-demand) |
|--------------|---------------|-------------|----------------------------|
| 7B (LLaMA-2) | 32× A100 80GB | ~21 days   | ~$50K-100K                 |
| 13B          | 64× A100      | ~30 days    | ~$150K-300K                |
| 70B          | 256× A100     | ~45 days    | ~$1M-2M                   |
| 175B (GPT-3) | 1024× A100   | ~34 days    | ~$4M-8M                   |
| 405B (LLaMA-3.1) | 16K H100  | ~54 days   | ~$60M+                    |

### Inference Cost Rules of Thumb
- **Tokens/$ (H100, FP16)**: ~1M tokens per $1 for 70B model
- **GPU utilization target**: >70% to be cost-effective
- **Spot vs on-demand**: 60-80% savings with interruption risk
- **Reserved instances**: 30-60% savings with 1-3 year commitment

---

## 8. Quick Conversions & Rules of Thumb

### FLOPS ↔ Tokens ↔ Images

```
Training 1 token (Transformer):     ~6 × num_params FLOPs (forward + backward)
Inference 1 token (auto-regressive): ~2 × num_params FLOPs (forward only)

Example: 70B model inference
  = 2 × 70 × 10⁹ = 140 GFLOPS per token
  H100 FP16 Tensor: ~989 TFLOPS
  Theoretical max: 989 / 0.140 ≈ 7,064 tokens/sec
  Practical (50-60% util): ~3,500-4,200 tokens/sec
```

### Memory Estimation for LLMs

```
Model weights:         num_params × bytes_per_param
  FP32: 70B × 4 = 280 GB
  FP16: 70B × 2 = 140 GB (1× H200 or 2× H100)
  INT8: 70B × 1 = 70 GB  (1× H100)
  INT4: 70B × 0.5 = 35 GB (1× A100)

KV Cache per token:    2 × num_layers × hidden_dim × bytes × num_heads
  LLaMA-70B FP16: ~2.5 MB per token
  2048 tokens context: ~5 GB

Activation memory:     Batch × SeqLen × HiddenDim × NumLayers × ~10 bytes
```

### Power & Cooling
```
1 GPU server (8× H100):     ~10 kW
1 rack (4 servers):          ~40 kW
1000 GPU cluster:            ~1.25 MW + cooling overhead
PUE (Power Usage Effectiveness): 1.1-1.5×
Total facility power: IT power × PUE
```

### Networking
```
AllReduce bandwidth requirement (data parallelism):
  = 2 × model_size × (N-1)/N  per training step
  70B FP16 = 2 × 140 GB × (8-1)/8 ≈ 245 GB per step

Time to AllReduce on NVLink 4.0 (900 GB/s):
  = 245 / 900 ≈ 0.27 seconds (intra-node)

Time on InfiniBand NDR (50 GB/s):
  = 245 / 50 ≈ 4.9 seconds (inter-node, naive)
  Ring AllReduce: 245 / (50 × 0.8 efficiency) ≈ 6.1 seconds
```

---

## 9. Benchmark Reference Points

### CPU Single-Core
| Operation               | Throughput (Skylake)    |
|--------------------------|------------------------|
| Integer ADD              | ~4 Gops/s              |
| FP64 multiply           | ~2 GFLOPs              |
| FP32 FMA (AVX2)         | ~64 GFLOPs             |
| Memory read (sequential) | ~20 GB/s               |
| Memory read (random)     | ~1 GB/s (pointer chase) |
| strlen (1KB string)      | ~5 GB/s                |

### Network Operations
| Operation                | Latency / Throughput   |
|--------------------------|------------------------|
| TCP connect (loopback)   | ~30-50 µs              |
| HTTP request (localhost)  | ~100-500 µs            |
| Redis GET (localhost)    | ~30-80 µs              |
| PostgreSQL query (simple) | ~200-500 µs           |
| gRPC call (same datacenter) | ~0.5-2 ms           |

---

## 10. Dimensional Analysis Cheat Sheet

### Common Prefixes
| Prefix | Value    | Example                          |
|--------|----------|----------------------------------|
| nano   | 10⁻⁹    | 1 ns = L1 cache access           |
| micro  | 10⁻⁶    | 100 µs = SSD read                |
| milli  | 10⁻³    | 10 ms = HDD seek                 |
| kilo   | 10³     | 1 KB = half a printed page       |
| mega   | 10⁶     | 1 MB = 1 second of MP3           |
| giga   | 10⁹     | 1 GB = ~500K pages of text       |
| tera   | 10¹²    | 1 TB = ~500 hours of video       |
| peta   | 10¹⁵    | 1 PB = ~1M GB                    |
| exa    | 10¹⁸    | 1 EB = all words ever spoken     |

### Quick Math
```
2¹⁰ ≈ 1,000 (1 KB)
2²⁰ ≈ 1,000,000 (1 MB)
2³⁰ ≈ 1,000,000,000 (1 GB)
2⁴⁰ ≈ 1,000,000,000,000 (1 TB)

1 day = 86,400 seconds ≈ 10⁵ seconds
1 year ≈ 3.15 × 10⁷ seconds ≈ π × 10⁷ seconds

Requests at 1000 QPS for 1 day = 86.4M requests
1 GB/s for 1 hour = 3.6 TB transferred
```

---

*Appendix B — Numbers Every Programmer Should Know — Part of the CPP-CUDA-Mastery series*
