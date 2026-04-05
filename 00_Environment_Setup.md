# Chapter 0 — Environment Setup: C++ and CUDA Development

> **Read this first.** Every chapter in this series assumes you have a working C++ compiler
> and (for GPU chapters) a functioning CUDA toolkit. Spend 30 minutes here now to save
> hours of debugging later.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [C++ Toolchain Setup](#2-c-toolchain-setup)
3. [CUDA Toolkit Installation](#3-cuda-toolkit-installation)
4. [Free GPU Access (No GPU? No Problem!)](#4-free-gpu-access-no-gpu-no-problem)
5. [IDE Setup](#5-ide-setup)
6. [First Build Test](#6-first-build-test)
7. [Docker Development Environment](#7-docker-development-environment)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### What You Need

| Requirement | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 16 GB+ |
| Disk space | 10 GB free | 30 GB+ free |
| Terminal skills | `cd`, `ls`, `mkdir` | Shell scripting basics |
| Text editor | Any | VS Code |
| GPU (for CUDA) | NVIDIA Kepler+ (CC 3.5) | RTX 3060+ (CC 8.6) |

> **No NVIDIA GPU?** Skip to [Section 4](#4-free-gpu-access-no-gpu-no-problem) — you can
> still learn CUDA using free cloud GPUs.

### Operating System Options

| OS | C++ Support | CUDA Support | Notes |
|---|---|---|---|
| **Ubuntu 22.04/24.04** | ✅ Full | ✅ Full | Recommended — fewest headaches |
| **Fedora / Arch** | ✅ Full | ✅ Full | Slightly more manual CUDA setup |
| **Windows 11 + WSL2** | ✅ Full | ✅ Full | Best of both worlds |
| **Windows (native)** | ✅ Full | ⚠️ Partial | Visual Studio required for CUDA |
| **macOS (Apple Silicon)** | ✅ Full | ❌ None | C++ only — no CUDA support |
| **macOS (Intel)** | ✅ Full | ❌ None | NVIDIA dropped macOS CUDA in 2019 |

**Bottom line:** Use Linux or WSL2 for the smoothest experience.

---

## 2. C++ Toolchain Setup

### Linux (Ubuntu / Debian)

```bash
# Update package lists
sudo apt update

# Install GCC, G++, make, and core build tools
sudo apt install -y build-essential

# Install Clang (optional but useful for a second compiler opinion)
sudo apt install -y clang

# Install CMake (build system generator)
sudo apt install -y cmake

# Install GDB (debugger)
sudo apt install -y gdb
```

Verify everything is installed:

```bash
g++ --version        # Should show GCC 11+ on Ubuntu 22.04
clang++ --version    # Should show Clang 14+
cmake --version      # Should show 3.22+
gdb --version        # Should show 12+
make --version       # Should show GNU Make 4+
```

**Want a newer GCC?** Ubuntu PPAs let you install newer versions side-by-side:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-13 g++-13

# Set as default (optional)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
```

### Windows

#### Option A: WSL2 (Recommended)

WSL2 gives you a real Linux kernel inside Windows — best for CUDA development.

```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu-22.04

# Restart your computer, then open Ubuntu from the Start menu
# Follow the Linux instructions above inside WSL
```

#### Option B: Visual Studio with C++ Workload

1. Download [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/)
2. In the installer, select **"Desktop development with C++"**
3. This installs MSVC, CMake, and a debugger
4. Open "Developer PowerShell for VS" to use `cl.exe` from the command line

#### Option C: MSYS2 / MinGW

```powershell
# Download and install MSYS2 from https://www.msys2.org/
# In the MSYS2 terminal:
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-gdb
```

Add `C:\msys64\mingw64\bin` to your system PATH.

### macOS

```bash
# Install Xcode Command Line Tools (includes Apple Clang)
xcode-select --install

# Verify
clang++ --version    # Apple clang 15+

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake and optionally LLVM (for a standard Clang)
brew install cmake
brew install llvm    # Optional: gives you a non-Apple clang

# If using Homebrew LLVM, add to your shell profile:
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

> **macOS note:** Apple Clang supports C++20 and C++23 features well. You do NOT need
> GCC unless you specifically need GNU extensions.

---

## 3. CUDA Toolkit Installation

### Linux (Ubuntu — Step by Step)

#### Step 1: Verify You Have an NVIDIA GPU

```bash
lspci | grep -i nvidia
# Should output something like: NVIDIA Corporation GA106 [GeForce RTX 3060]
```

If this shows nothing, you don't have an NVIDIA GPU — skip to [Section 4](#4-free-gpu-access-no-gpu-no-problem).

#### Step 2: Install the NVIDIA Driver

```bash
# Ubuntu's recommended driver installer
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify after reboot
nvidia-smi
# Should show driver version, GPU name, CUDA version supported
```

#### Step 3: Install the CUDA Toolkit

Visit [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
and select: Linux → x86_64 → Ubuntu → 22.04 → deb (network).

Or install directly from the command line:

```bash
# Add the NVIDIA package repository (Ubuntu 22.04 example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install the CUDA toolkit (this does NOT overwrite your display driver)
sudo apt install -y cuda-toolkit-12-6
```

#### Step 4: Set PATH and Environment Variables

Add these to your `~/.bashrc` (or `~/.zshrc`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then reload:

```bash
source ~/.bashrc
```

#### Step 5: Verify

```bash
nvcc --version
# Should show: Cuda compilation tools, release 12.6

nvidia-smi
# Should show your GPU, driver version, and CUDA version

# Quick compilation test
echo '#include <cstdio>
__global__ void hello() { printf("Hello from GPU thread %d\\n", threadIdx.x); }
int main() { hello<<<1,8>>>(); cudaDeviceSynchronize(); }' > test_cuda.cu
nvcc test_cuda.cu -o test_cuda && ./test_cuda
rm -f test_cuda test_cuda.cu
```

#### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `nvcc: command not found` | PATH not set | Add `/usr/local/cuda/bin` to PATH |
| `nvidia-smi` shows nothing | Driver not installed | Run `sudo ubuntu-drivers autoinstall` |
| "CUDA driver version is insufficient" | Driver too old for toolkit | Update driver or install older toolkit |
| `libcuda.so not found` | Missing library path | Add `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH` |

### WSL2

#### Prerequisites

- Windows 11 (or Windows 10 build 21H2+)
- An NVIDIA GPU with driver version 525.60+ **installed on Windows** (not inside WSL)

> **Key insight:** In WSL2, the **Windows host driver** provides GPU access.
> You only install the **CUDA toolkit** inside WSL — never the driver.

#### Installation

```bash
# Inside your WSL2 Ubuntu terminal:

# 1. DO NOT install nvidia-driver inside WSL — the host driver handles it

# 2. Install CUDA toolkit (same as native Linux)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# 3. Set PATH (same as native Linux)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 4. Verify GPU access
nvidia-smi          # Should show your Windows GPU
nvcc --version      # Should show CUDA 12.6
```

---

## 4. Free GPU Access (No GPU? No Problem!)

You can learn and practice CUDA on free cloud GPUs. Here are your best options:

### Google Colab (Easiest)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → T4 GPU**
4. Use the `%%cuda` magic (requires the `nvcc4jupyter` plugin):

```python
# Run this cell first to install the CUDA plugin
!pip install nvcc4jupyter
%load_ext nvcc4jupyter
```

```c
%%cuda
#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**Limits:** ~12 hours per session, usage quotas, T4 GPU (16 GB VRAM).

### Kaggle Notebooks

1. Go to [kaggle.com](https://www.kaggle.com), create an account
2. Create a new notebook
3. **Settings → Accelerator → GPU T4 ×2** (or P100)
4. Same `%%cuda` magic works here

**Limits:** 30 hours/week of GPU, P100 or T4 ×2.

### Paid Options (When You Need More)

| Provider | GPU | Price (approx.) | Best For |
|---|---|---|---|
| **Lambda Cloud** | H100 80GB | ~$2.50/hr | Serious training runs |
| **AWS (EC2 Spot)** | T4 / A10G / A100 | $0.30–3.00/hr | Flexible, wide GPU range |
| **GCP (Preemptible)** | T4 / A100 / H100 | $0.35–3.50/hr | TPU access too |
| **Azure (Spot)** | T4 / A100 | $0.40–3.00/hr | Enterprise integration |
| **Vast.ai** | Various | $0.10–2.00/hr | Cheapest per-hour |
| **RunPod** | A100 / H100 | $0.40–2.50/hr | Community GPU cloud |

### Comparison at a Glance

| Platform | Free Tier | GPU Type | Session Limit | Storage |
|---|---|---|---|---|
| **Google Colab** | ✅ Yes | T4 (16 GB) | ~12 hours | 100 GB disk |
| **Kaggle** | ✅ Yes | T4×2 or P100 | 30 hrs/week | 70 GB disk |
| **Lambda Cloud** | ❌ No | H100 (80 GB) | Unlimited | Persistent |
| **AWS Spot** | ❌ No | T4–H100 | Unlimited | EBS volumes |

---

## 5. IDE Setup

### VS Code (Recommended)

VS Code with the right extensions gives you IntelliSense, debugging, and integrated
terminal — all free.

#### Essential Extensions

Install these from the VS Code extensions marketplace (Ctrl+Shift+X):

| Extension | Publisher | Purpose |
|---|---|---|
| **C/C++** | Microsoft | IntelliSense, debugging, code navigation |
| **CMake Tools** | Microsoft | CMake configure, build, and debug |
| **Nsight Visual Studio Code Edition** | NVIDIA | CUDA debugging and profiling |
| **clangd** | LLVM | Alternative to Microsoft C/C++ — faster |

> **Pick one:** Use either the Microsoft C/C++ extension or clangd, not both.
> clangd is faster but requires a `compile_commands.json` (CMake generates this).

#### settings.json

Open with: Ctrl+Shift+P → "Preferences: Open User Settings (JSON)"

```jsonc
{
    // C++ standard
    "C_Cpp.default.cppStandard": "c++20",
    "C_Cpp.default.cStandard": "c17",

    // Include paths for CUDA headers
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/**",
        "/usr/local/cuda/include"
    ],

    // CMake configuration
    "cmake.configureOnOpen": true,
    "cmake.generator": "Ninja",

    // Format on save (optional but recommended)
    "editor.formatOnSave": true,
    "C_Cpp.clang_format_fallbackStyle": "Google",

    // Associate .cu and .cuh files with C++
    "files.associations": {
        "*.cu": "cuda-cpp",
        "*.cuh": "cuda-cpp"
    }
}
```

#### tasks.json (Build Tasks)

Create `.vscode/tasks.json` in your project:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build C++ (g++)",
            "type": "shell",
            "command": "g++",
            "args": [
                "-std=c++20",
                "-Wall", "-Wextra", "-O2",
                "${file}",
                "-o", "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": "build",
            "problemMatcher": ["$gcc"]
        },
        {
            "label": "Build CUDA (nvcc)",
            "type": "shell",
            "command": "nvcc",
            "args": [
                "-std=c++20",
                "-O2",
                "${file}",
                "-o", "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "group": "build",
            "problemMatcher": ["$gcc"]
        }
    ]
}
```

#### launch.json (Debugging)

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug C++",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "cwd": "${workspaceFolder}",
            "environment": [],
            "MIMode": "gdb",
            "preLaunchTask": "Build C++ (g++)"
        },
        {
            "name": "Debug CUDA (cuda-gdb)",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "cwd": "${workspaceFolder}",
            "environment": [],
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/local/cuda/bin/cuda-gdb",
            "preLaunchTask": "Build CUDA (nvcc)"
        }
    ]
}
```

### CLion

JetBrains CLion has excellent CMake integration and supports CUDA out of the box.

1. Install CLion from [jetbrains.com/clion](https://www.jetbrains.com/clion/)
2. Open your CMake project — CLion auto-detects `CMakeLists.txt`
3. For CUDA: CLion recognizes `.cu` files when `enable_language(CUDA)` is in CMakeLists
4. Set the C++ standard in **Settings → Build → CMake → CMake options**: `-DCMAKE_CUDA_STANDARD=20`

### Terminal-Only Development

#### Makefile for C++ Projects

```makefile
# ---- C++ Makefile Template ----
CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2
LDFLAGS  :=

SRC_DIR := src
OBJ_DIR := build
BIN     := main

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN)
```

#### Makefile for CUDA Projects

```makefile
# ---- CUDA Makefile Template ----
NVCC     := nvcc
CXXFLAGS := -std=c++20 -O2
CUFLAGS  := -arch=sm_86    # Change to match your GPU (sm_75 for T4, sm_80 for A100)
LDFLAGS  :=

SRC_DIR := src
OBJ_DIR := build
BIN     := main

CU_SRCS  := $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_OBJS  := $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CPP_OBJS := $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJS     := $(CU_OBJS) $(CPP_OBJS)

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) $(CUFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN)
```

**Common GPU architectures for the `-arch` flag:**

| GPU | Compute Capability | Flag |
|---|---|---|
| T4 | 7.5 | `-arch=sm_75` |
| RTX 3060/3090 | 8.6 | `-arch=sm_86` |
| A100 | 8.0 | `-arch=sm_80` |
| H100 | 9.0 | `-arch=sm_90` |
| RTX 4090 | 8.9 | `-arch=sm_89` |
| RTX 5090 | 10.0 | `-arch=sm_100` |

---

## 6. First Build Test

### Test 1: C++ Hello World

Create `hello.cpp`:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    std::cout << "=== C++ Environment Test ===\n";

    // Test C++20 features
    std::vector<int> nums = {5, 3, 8, 1, 9, 2, 7};

    // Ranges (C++20)
    std::ranges::sort(nums);

    std::cout << "Sorted: ";
    for (int n : nums) std::cout << n << " ";
    std::cout << "\n";

    // Structured bindings, auto
    auto sum = std::accumulate(nums.begin(), nums.end(), 0);
    std::cout << "Sum: " << sum << "\n";

    std::cout << "C++ standard: " << __cplusplus << "\n";
    std::cout << "✅ C++ toolchain is working!\n";
    return 0;
}
```

Build and run:

```bash
g++ -std=c++20 -Wall -O2 hello.cpp -o hello && ./hello
```

Expected output:

```
=== C++ Environment Test ===
Sorted: 1 2 3 5 7 8 9
Sum: 35
C++ standard: 202002
✅ C++ toolchain is working!
```

### Test 2: CUDA Hello World (GPU Info Query)

Create `hello_cuda.cu`:

```cuda
#include <cstdio>

__global__ void helloKernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 8) {
        printf("  Thread %d says hello from the GPU!\n", tid);
    }
}

int main() {
    printf("=== CUDA Environment Test ===\n");

    // Query GPU properties
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA devices found: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory:      %.1f GB\n", prop.totalGlobalMem / 1e9);
        printf("  SM count:           %d\n", prop.multiProcessorCount);
        printf("  Max threads/block:  %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size:          %d\n", prop.warpSize);
        printf("  Clock rate:         %.0f MHz\n", prop.clockRate / 1e3);
        printf("\n");
    }

    // Launch a simple kernel
    printf("Launching kernel with 1 block × 8 threads:\n");
    helloKernel<<<1, 8>>>();
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\n✅ CUDA toolchain is working!\n");
    return 0;
}
```

Build and run:

```bash
nvcc -std=c++20 -O2 hello_cuda.cu -o hello_cuda && ./hello_cuda
```

### Test 3: CMake Project with C++ and CUDA

This is the project structure you will use throughout this series:

```
my_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   └── kernels.cu
└── include/
    └── kernels.cuh
```

**CMakeLists.txt** — complete template:

```cmake
cmake_minimum_required(VERSION 3.24)
project(MyProject LANGUAGES CXX CUDA)

# ---- Standards ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- CUDA architecture (auto-detect or specify) ----
# "native" auto-detects your GPU; replace with "86" for RTX 3060, etc.
set(CMAKE_CUDA_ARCHITECTURES native)

# ---- Source files ----
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/kernels.cu
)

# ---- Include directories ----
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ---- Compiler warnings ----
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-relaxed-constexpr>
)
```

**include/kernels.cuh:**

```cuda
#pragma once

void launchHello(int blocks, int threads);
```

**src/kernels.cu:**

```cuda
#include "kernels.cuh"
#include <cstdio>

__global__ void helloKernel(int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        printf("  GPU thread %d reporting in\n", tid);
    }
}

void launchHello(int blocks, int threads) {
    helloKernel<<<blocks, threads>>>(blocks * threads);
    cudaDeviceSynchronize();
}
```

**src/main.cpp:**

```cpp
#include <iostream>
#include "kernels.cuh"

int main() {
    std::cout << "=== CMake + CUDA Project Test ===\n";
    std::cout << "Launching 2 blocks × 4 threads:\n";
    launchHello(2, 4);
    std::cout << "✅ CMake project is working!\n";
    return 0;
}
```

**Build and run:**

```bash
mkdir build && cd build
cmake .. -G Ninja        # or just: cmake ..
cmake --build .
./MyProject
```

---

## 7. Docker Development Environment

Docker ensures everyone gets the same environment — no more "works on my machine."

### Dockerfile

```dockerfile
# Use NVIDIA's official CUDA base image
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    gdb \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
```

### docker-compose.yml

```yaml
services:
  dev:
    build: .
    volumes:
      - .:/workspace
    working_dir: /workspace
    # GPU passthrough — requires nvidia-container-toolkit
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # Keep the container running for interactive use
    stdin_open: true
    tty: true
```

### Setup Steps

```bash
# 1. Install nvidia-container-toolkit (one-time setup on host)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Build and start the container
docker compose up -d --build

# 3. Attach to it
docker compose exec dev bash

# 4. Verify GPU access inside the container
nvidia-smi
nvcc --version
```

### VS Code Dev Containers

For a seamless Docker-based workflow in VS Code:

1. Install the **Dev Containers** extension
2. Create `.devcontainer/devcontainer.json`:

```json
{
    "name": "C++ CUDA Dev",
    "dockerComposeFile": "../docker-compose.yml",
    "service": "dev",
    "workspaceFolder": "/workspace",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-vscode.cpptools",
                "ms-vscode.cmake-tools",
                "nvidia.nsight-vscode-edition"
            ],
            "settings": {
                "C_Cpp.default.cppStandard": "c++20"
            }
        }
    }
}
```

3. Ctrl+Shift+P → **"Dev Containers: Reopen in Container"**

---

## 8. Troubleshooting

### "nvcc: command not found"

**Cause:** CUDA's `bin/` directory is not in your PATH.

```bash
# Check if nvcc exists
ls /usr/local/cuda/bin/nvcc

# Fix: Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# If cuda is installed but /usr/local/cuda doesn't exist, find it:
find / -name "nvcc" -type f 2>/dev/null
```

### "no CUDA-capable device is detected"

**Cause:** NVIDIA driver is not installed or GPU is not recognized.

```bash
# Check if the driver is loaded
nvidia-smi

# If nvidia-smi fails, install the driver:
sudo apt install -y nvidia-driver-550
sudo reboot

# In WSL2: Make sure you have the NVIDIA GPU driver installed on Windows
# (not inside WSL). Download from: https://www.nvidia.com/Download/index.aspx
```

### CUDA Toolkit vs Driver Compatibility

The CUDA toolkit requires a minimum driver version. Mismatches cause runtime errors.

| CUDA Toolkit | Minimum Driver (Linux) | Minimum Driver (Windows) |
|---|---|---|
| CUDA 12.6 | 560.28+ | 560.70+ |
| CUDA 12.4 | 550.54+ | 551.78+ |
| CUDA 12.2 | 535.54+ | 536.25+ |
| CUDA 12.0 | 525.60+ | 527.41+ |
| CUDA 11.8 | 520.61+ | 522.06+ |

**Rule of thumb:** Keep your driver as new as possible. A newer driver always supports
older CUDA toolkits (forward compatibility).

```bash
# Check your current driver version
nvidia-smi | head -3

# Check what CUDA version your driver supports (top-right of nvidia-smi output)
# This is the MAX CUDA version your driver supports, not the installed toolkit version
```

### Compilation Errors with C++ Standard Flags

```bash
# "error: 'ranges' is not a member of 'std'"
# Fix: You need C++20. Make sure you pass -std=c++20
g++ -std=c++20 my_file.cpp -o my_file

# nvcc doesn't support all C++20 features in device code
# For host-only C++20 features in .cu files:
nvcc -std=c++20 --expt-relaxed-constexpr my_file.cu -o my_file

# "error: parameter packs not expanded with '...'"
# This is a common nvcc template issue. Try:
nvcc -std=c++20 --expt-extended-lambda my_file.cu -o my_file
```

### CMake Can't Find CUDA

```bash
# "Could not find CUDA toolkit"
# Fix: Tell CMake where CUDA is
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Or set the environment variable
export CUDACXX=/usr/local/cuda/bin/nvcc
```

### Linker Errors: "undefined reference to cudaMalloc"

```bash
# You're compiling .cu files with g++ instead of nvcc
# Fix: Use nvcc for .cu files, or link with -lcudart

# With g++ (not recommended):
g++ -std=c++20 my_file.cpp -o my_file -L/usr/local/cuda/lib64 -lcudart

# With CMake: Make sure your project has CUDA as a language
# project(MyProject LANGUAGES CXX CUDA)   ← This is required
```

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT CHEAT SHEET                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Compile C++:     g++ -std=c++20 -O2 file.cpp -o out        │
│  Compile CUDA:    nvcc -std=c++20 -O2 file.cu -o out        │
│  CMake build:     mkdir build && cd build && cmake .. &&     │
│                   cmake --build .                            │
│                                                              │
│  Check GPU:       nvidia-smi                                 │
│  Check CUDA:      nvcc --version                             │
│  Check compiler:  g++ --version                              │
│                                                              │
│  GPU arch flags:  T4=sm_75  A100=sm_80  RTX3090=sm_86       │
│                   H100=sm_90  RTX4090=sm_89                  │
│                                                              │
│  Docker GPU:      docker run --gpus all nvidia/cuda:12.6...  │
│                                                              │
│  No GPU?          → Google Colab (free T4)                   │
│                   → Kaggle (free P100/T4×2)                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## What's Next?

Your environment is ready. Move on to:

- **[Chapter 01 — C++ Fundamentals](01_CPP_Fundamentals.md)** — if you're new to C++
- **[Chapter XX — Your First CUDA Kernel](XX_First_CUDA_Kernel.md)** — if you're ready for GPU programming

> **Tip:** Bookmark this page. You'll come back here when setting up on a new machine
> or when something breaks.

---

*Environment setup guide — Part of the C++ & CUDA Mastery series.*
