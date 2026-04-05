# Appendix J — GPU Buying Guide for CUDA Learners

> **TL;DR** — Use Google Colab (free T4) to start today. If you want your own hardware,
> a used RTX 3060 12 GB (~$200) is the sweet spot for learning. Everything in this repo
> runs on either of those options.

---

## 1. Do You Even Need a GPU to Start?

**No.** You do not need to buy anything to begin this curriculum.

| Concern | Reality |
|---------|---------|
| "I need a GPU before I can start" | Chapters 1–43 are pure C++ — your laptop CPU is fine |
| "My GPU is too old" | Any NVIDIA GPU from ~2016 onward runs the CUDA chapters |
| "I can't afford hardware" | Google Colab gives you a free T4 (16 GB VRAM) right now |

**Recommended path:**

1. Start Chapters 1–43 on your laptop (C++ only, no GPU required).
2. When you reach Chapter 44 (CUDA basics), open Google Colab — zero setup.
3. Only consider buying a GPU once you know you enjoy CUDA programming.

> 💡 *Don't let hardware be the reason you don't start. Open Colab, run a kernel,
> and decide later.*

---

## 2. Free GPU Options (Best for Beginners)

Every lab in this repo was tested on a free-tier GPU. Here are your options:

| Platform | GPU | VRAM | Free Tier | Session Limit | Best For |
|----------|-----|------|-----------|---------------|----------|
| **Google Colab** | T4 | 16 GB | ✅ Free | ~12 hr sessions | All labs in this repo |
| **Kaggle Notebooks** | P100 / T4 | 16 GB | ✅ 30 hr/week | ~9 hr sessions | Longer experiments |
| **Lightning AI Studios** | T4 | 16 GB | ✅ Free credits | Varies | Full Linux env |
| **Paperspace Gradient** | M4000 / Free GPU | 8 GB | ✅ Community | 6 hr sessions | Notebooks |

### Google Colab — The Default Choice

Colab is the fastest way to run CUDA C++ with zero local setup.

**Running CUDA C++ on Colab:**

```python
# Cell 1 — Write your CUDA source file
%%writefile vector_add.cu
#include <cstdio>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // ... allocate, launch, verify (see Chapter 44)
    printf("Vector addition complete!\n");
    return 0;
}
```

```python
# Cell 2 — Compile and run
!nvcc -o vector_add vector_add.cu && ./vector_add
```

```python
# Cell 3 — Check what GPU Colab gave you
!nvidia-smi
```

**Colab tips:**

- Go to *Runtime → Change runtime type → T4 GPU* before running.
- Save your `.cu` files to Google Drive so you don't lose work.
- Use `!nvcc --version` to check the CUDA toolkit version.
- Free tier sessions disconnect after ~90 min of idle time.

### Kaggle Notebooks

- 30 hours of GPU per week (resets weekly).
- Persistent storage for datasets.
- Same `%%writefile` + `!nvcc` workflow as Colab.

### Lightning AI Studios

- Full Linux environment with terminal access.
- Free GPU credits on signup.
- Closer to a "real" development machine than a notebook.

---

## 3. Cloud GPU Rental (Pay-per-Hour)

When free tiers aren't enough — training larger models, multi-GPU experiments,
or profiling on specific architectures.

| Provider | GPU Options | Approx $/hr | Best For |
|----------|------------|-------------|----------|
| **Lambda Cloud** | A100, H100 | $1.10 – $2.50 | Dedicated instances, ML focus |
| **RunPod** | A100, H100, 4090 | $0.40 – $2.50 | Flexible on-demand, templates |
| **Vast.ai** | Community GPUs | $0.15 – $1.50 | Cheapest option, variable |
| **AWS EC2** | P3 (V100), P4 (A100), P5 (H100) | $3.00 – $12.00+ | Enterprise, spot pricing |
| **GCP Compute** | T4, A100, H100 | $0.35 – $8.00+ | Deep integration with Vertex AI |
| **Azure** | T4, A100, H100 | $0.50 – $10.00+ | Enterprise, Azure ecosystem |

> ⚠️ *Prices change frequently. Check provider websites for current rates.*

### Saving Money on Cloud GPUs

- **Spot / preemptible instances** on AWS, GCP, or Azure run 60–80% cheaper.
  Trade-off: your instance can be reclaimed with little warning.
- **Use checkpointing** — save model state frequently so preemption doesn't
  waste hours of compute.
- **Right-size your instance** — a T4 at $0.35/hr beats an A100 at $3/hr when
  you're debugging a 10-line kernel.
- **Set billing alerts** — every cloud provider lets you set budget caps. Do it
  on day one.

---

## 4. Buying a GPU for Your Desktop

If you write CUDA code regularly, owning a GPU pays for itself quickly and
gives you the fastest iteration cycle (no uploads, no session limits).

### Budget Tiers

| Budget | GPU | VRAM | CUDA Cores | Arch | Good For |
|--------|-----|------|------------|------|----------|
| **$150–200** | RTX 3060 12 GB *(used)* | 12 GB | 3,584 | Ampere | ⭐ Best value for learning |
| **$250–350** | RTX 3060 12 GB *(new)* | 12 GB | 3,584 | Ampere | All labs, small models |
| **$350–500** | RTX 4060 Ti 16 GB | 16 GB | 4,352 | Ada | Good compute, efficient |
| **$600–900** | RTX 4070 Ti Super | 16 GB | 8,448 | Ada | Serious CUDA development |
| **$1,000–1,200** | RTX 4080 Super | 16 GB | 10,240 | Ada | Professional workloads |
| **$1,500–2,000** | RTX 4090 | 24 GB | 16,384 | Ada | Top consumer GPU |
| **$300–500** *(used)* | RTX 3090 24 GB | 24 GB | 10,496 | Ampere | Amazing value if used |

### The Sweet-Spot Pick: RTX 3060 12 GB

For a learner following this curriculum, the RTX 3060 12 GB is hard to beat:

- **12 GB VRAM** — more than the RTX 3070 or 3080 (both 8–10 GB).
- **Ampere architecture** — supports TF32, BF16, 3rd-gen Tensor Cores.
- **Compute capability 8.6** — runs every example in this repo.
- **Low power draw** — 170 W TDP, doesn't need a massive PSU.
- **Available used for ~$150–200** — the price of two months of cloud rental.

### The Used-Market Gem: RTX 3090 24 GB

If you can find one for $400–500 used:

- **24 GB VRAM** — matches professional A10G cards.
- **10,496 CUDA cores** — serious compute.
- **Caveat:** draws 350 W — make sure your PSU and cooling can handle it.

### Key Specs That Matter for CUDA

Not all specs are equally important for CUDA programming. Here's what to
prioritize:

| Spec | Why It Matters | What to Look For |
|------|---------------|-----------------|
| **VRAM** | Determines the largest data/model you can fit on the GPU | ≥ 8 GB minimum; 12–16 GB ideal; 24 GB for large models |
| **CUDA Cores** | Raw parallel compute throughput | More is better, but architecture matters too |
| **Memory Bandwidth** | How fast data moves between VRAM and compute units | Higher bandwidth = faster memory-bound kernels |
| **Compute Capability** | Determines which CUDA features are available | ≥ 7.0 for Tensor Cores; ≥ 8.0 for TF32, BF16 |
| **TDP / Power** | Determines PSU, cooling, and electricity cost | Budget GPUs: 75–170 W; Flagships: 300–450 W |

**Common trap:** Don't compare CUDA core counts across architectures.
A 4060 Ti with 4,352 Ada cores often outperforms a 3070 with 5,888 Ampere cores
due to architectural improvements.

### System Requirements Checklist

Before buying, verify your system can support the GPU:

- [ ] **PCIe slot**: x16 PCIe 3.0 or newer (all modern motherboards).
- [ ] **Power supply**: Meets GPU's TDP + ~200 W for the rest of the system.
- [ ] **Physical space**: Measure your case — modern GPUs are 2.5–3.5 slots thick.
- [ ] **CPU**: Any modern CPU works; CUDA offloads computation to the GPU.
- [ ] **RAM**: At least 16 GB system RAM; 32 GB is comfortable.
- [ ] **OS**: Linux (Ubuntu/Fedora) for best CUDA support; Windows works too.

---

## 5. What GPU Does This Repo Require?

Different chapters need different levels of hardware. Here's the breakdown:

| Chapters | Topic | Minimum GPU | Compute Capability |
|----------|-------|-------------|-------------------|
| 1–43 | C++ Foundations | **None** (CPU only) | — |
| 44–53 | CUDA Basics | Any CUDA GPU (even GTX 1050) | ≥ 5.0 |
| 54–57 | Memory, Streams, Atomics | Any recent GPU | ≥ 6.0 |
| 58 | Multi-GPU | 2+ GPUs or cloud | ≥ 6.0 |
| 59–61 | Optimization, Libraries | Recommended Volta+ | ≥ 7.0 |
| 62 | Tensor Cores | **Volta+** (V100, RTX 2000+) | ≥ 7.0 |
| 63 | Mixed Precision | Turing+ | ≥ 7.5 |
| 64–68 | ML Kernels, Profiling | Any modern GPU | ≥ 7.0 |
| 69 | Flash Attention | **Ampere+ recommended** | ≥ 8.0 |
| 70 | Capstone Project | T4 (Colab) works | ≥ 7.5 |

### What the Free Tier Covers

| Platform | Compute Capability | Covers Chapters |
|----------|--------------------|-----------------|
| Colab T4 | 7.5 (Turing) | 44–68, 70 (all but Flash Attention's best path) |
| Kaggle P100 | 6.0 (Pascal) | 44–58, partial 59+ |
| Kaggle T4 | 7.5 (Turing) | Same as Colab |

> ✅ **Bottom line:** A free Colab T4 handles every chapter except the most
> advanced Ampere-specific optimizations. You can complete the entire curriculum
> without spending a dollar.

---

## 6. Multi-GPU & Server Options

Some chapters (Ch 58 — Multi-GPU, Ch 61 — Distributed) benefit from multiple
GPUs. Here's how to access them:

### Cloud Multi-GPU Instances

| Provider | Instance | GPUs | Approx $/hr |
|----------|----------|------|-------------|
| Lambda Cloud | 8×A100 | 8 | ~$10 |
| RunPod | 2-8×A100/H100 | 2–8 | $5–$25 |
| AWS p4d.24xlarge | 8×A100 | 8 | ~$32 (on-demand) |
| GCP a2-megagpu-16g | 16×A100 | 16 | ~$40 |

> 💡 *For learning multi-GPU programming, 2 GPUs is enough. You don't need 8.*

### Building a Multi-GPU Workstation

Only consider this if you're doing multi-GPU work professionally:

- **Motherboard**: Needs 2+ PCIe x16 slots with proper spacing.
- **PSU**: 1000–1600 W for 2–4 GPUs.
- **Cooling**: Open-air case or blower-style GPUs to manage thermals.
- **NVLink**: Optional — consumer GPUs (RTX 30xx/40xx) don't support NVLink.
  Multi-GPU communication uses PCIe, which is fine for learning.

For this curriculum, cloud multi-GPU instances are more practical and
cost-effective than building a multi-GPU desktop.

---

## 7. Compute Capability Reference Table

Every NVIDIA GPU has a **compute capability** (CC) version that determines which
CUDA features it supports. Check your GPU's CC with:

```bash
nvidia-smi                           # shows GPU model
nvcc --list-gpu-arch                 # shows supported architectures
```

Or programmatically:

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Compute capability: %d.%d\n", prop.major, prop.minor);
```

### Full Reference

| CC | Architecture | Year | Example GPUs | Key CUDA Features |
|----|-------------|------|-------------|-------------------|
| **5.0** | Maxwell | 2014 | GTX 750 Ti, GTX 9xx | Unified memory, dynamic parallelism |
| **5.2** | Maxwell | 2015 | GTX 980 Ti, Titan X | Improved shared memory |
| **6.0** | Pascal | 2016 | P100, GP100 | FP16 (half precision), unified memory improvements |
| **6.1** | Pascal | 2016 | GTX 10xx, Titan Xp | INT8 inference, faster atomics |
| **7.0** | Volta | 2017 | V100, Titan V | ⭐ First Tensor Cores, independent thread scheduling |
| **7.5** | Turing | 2018 | RTX 20xx, T4 | RT cores, INT8/INT4 Tensor Cores, mixed precision |
| **8.0** | Ampere | 2020 | A100 | ⭐ TF32, 3rd-gen Tensor Cores, BF16, sparsity |
| **8.6** | Ampere | 2021 | RTX 30xx, A10 | Same as 8.0 with consumer optimizations |
| **8.9** | Ada Lovelace | 2022 | RTX 40xx, L40 | FP8, 4th-gen Tensor Cores, shader execution reorder |
| **9.0** | Hopper | 2023 | H100, H200 | ⭐ FP8, TMA (Tensor Memory Accelerator), DPX |
| **10.0** | Blackwell | 2024 | B200, GB200 | 5th-gen Tensor Cores, FP4, next-gen NVLink |

### Minimum CC for Key Features

| Feature | Minimum CC | Notes |
|---------|-----------|-------|
| Basic CUDA | 5.0+ | Any Maxwell or newer |
| Cooperative Groups | 6.0+ | Pascal and newer |
| Tensor Cores (FP16) | 7.0+ | Volta and newer |
| Mixed-Precision Training | 7.0+ | Volta and newer |
| TF32 (Tensor Float 32) | 8.0+ | Ampere and newer |
| BF16 (BFloat16) | 8.0+ | Ampere and newer |
| FP8 (Transformer Engine) | 8.9+ | Ada / Hopper and newer |
| TMA (Tensor Memory Accel.) | 9.0+ | Hopper only |
| Thread Block Clusters | 9.0+ | Hopper only |

---

## 8. Laptop GPUs — A Note

Laptop GPUs (e.g., RTX 4060 Laptop) have lower TDP and fewer CUDA cores than
their desktop equivalents. They work for CUDA learning but expect:

- **30–50% lower performance** than the desktop version.
- **Thermal throttling** on sustained workloads.
- **Shared VRAM** — some laptop GPUs share memory with the system.

A laptop with an RTX 3060 Laptop GPU (6 GB) can run most examples in this repo
but won't match the desktop RTX 3060's 12 GB VRAM. If buying a laptop for CUDA,
prioritize VRAM (get the 8 GB+ variant if available).

---

## 9. The Bottom Line

### Just Starting?

> **Use Google Colab.** Free, works today, no setup, no risk.
>
> Open Colab → Runtime → T4 GPU → Write CUDA → Run.

### Want Your Own GPU?

> **RTX 3060 12 GB used (~$150–200)** is the sweet spot.
>
> 12 GB VRAM, Ampere architecture, compute capability 8.6.
> Runs every example in this repo. Costs less than a month of cloud H100 rental.

### Serious About CUDA Development?

> **RTX 4090 ($1,600)** for local work, or **rent H100s** on Lambda/RunPod
> for the biggest workloads.
>
> 24 GB VRAM, 16,384 CUDA cores, Ada architecture.
> Or skip the upfront cost and pay $2/hr for an H100.

### Decision Flowchart

```
Start here
    │
    ├─ "I want to try CUDA"
    │       └─→ Google Colab (free T4) — start in 30 seconds
    │
    ├─ "I want to practice regularly"
    │       └─→ RTX 3060 12GB used (~$200) — best value
    │
    ├─ "I'm building ML models too"
    │       └─→ RTX 4070 Ti Super 16GB (~$800) — compute + VRAM
    │
    ├─ "I need the best consumer GPU"
    │       └─→ RTX 4090 24GB (~$1,600) — nothing beats it
    │
    └─ "I need H100 / multi-GPU"
            └─→ Cloud rental (Lambda, RunPod, AWS) — pay per hour
```

---

## Quick Links

| Resource | URL |
|----------|-----|
| Google Colab | https://colab.research.google.com |
| Kaggle Notebooks | https://www.kaggle.com/code |
| Lambda Cloud | https://lambdalabs.com/service/gpu-cloud |
| RunPod | https://www.runpod.io |
| Vast.ai | https://vast.ai |
| NVIDIA CUDA GPUs list | https://developer.nvidia.com/cuda-gpus |
| CUDA Toolkit downloads | https://developer.nvidia.com/cuda-downloads |
| This repo's setup guide | `../25_Environment_Setup.md` |

---

*Last updated: 2025. GPU prices and availability change frequently — verify
current pricing before purchasing.*
