# Appendix C — CMake & Build System Cookbook

> Recipes for building C++ and CUDA projects with modern CMake, packaging, CI/CD, and containers.

---

## 1. Minimal C++ Project

### Directory Structure

```
my-project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   └── utils.cpp
├── include/
│   └── my-project/
│       └── utils.h
└── tests/
    └── test_utils.cpp
```

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)
project(my-project
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Example C++ project"
)

# Global settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library target
add_library(mylib
    src/utils.cpp
)
target_include_directories(mylib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_options(mylib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Executable target
add_executable(my-app src/main.cpp)
target_link_libraries(my-app PRIVATE mylib)

# Testing
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### tests/CMakeLists.txt

```cmake
find_package(GTest REQUIRED)

add_executable(unit_tests test_utils.cpp)
target_link_libraries(unit_tests PRIVATE mylib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)
```

---

## 2. CUDA Project Setup

### CMakeLists.txt for CUDA

```cmake
cmake_minimum_required(VERSION 3.24)
project(cuda-project
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture (or set explicitly)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;90")  # Volta, Ampere, Hopper
endif()

# CUDA library
add_library(cuda_kernels
    src/kernels/vector_add.cu
    src/kernels/reduction.cu
    src/kernels/matmul.cu
)
target_include_directories(cuda_kernels PUBLIC include)
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Host executable
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE cuda_kernels)

# Optional: link cuBLAS, cuDNN
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
)
```

### Setting Compute Capabilities

```cmake
# Single architecture
set(CMAKE_CUDA_ARCHITECTURES 90)  # Hopper only

# Multiple architectures (fat binary)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Native (detect current GPU)
set(CMAKE_CUDA_ARCHITECTURES native)  # CMake 3.24+

# All major architectures
set(CMAKE_CUDA_ARCHITECTURES all)     # CMake 3.24+
set(CMAKE_CUDA_ARCHITECTURES all-major)

# PTX for forward compatibility
set(CMAKE_CUDA_ARCHITECTURES "80-real;90-real;90-virtual")
```

### Mixed C++/CUDA Source Files

```cmake
# When .cpp files contain CUDA code via includes
set_source_files_properties(src/hybrid.cpp PROPERTIES LANGUAGE CUDA)

# Or use file extension convention
# .cu files are automatically compiled with nvcc
# .cpp files are compiled with the C++ compiler
```

---

## 3. Modern CMake Patterns

### Target-Based Design (No Global Variables)

```cmake
# BAD — pollutes global scope
include_directories(${SOME_DIR})
add_definitions(-DFOO=1)
link_libraries(somelib)

# GOOD — scoped to target
target_include_directories(mylib PUBLIC ${SOME_DIR})
target_compile_definitions(mylib PRIVATE FOO=1)
target_link_libraries(mylib PUBLIC somelib)
```

### INTERFACE Libraries (Header-Only)

```cmake
add_library(header_only INTERFACE)
target_include_directories(header_only INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(header_only INTERFACE cxx_std_20)
```

### Generator Expressions

```cmake
target_compile_options(mylib PRIVATE
    # Different flags per compiler
    $<$<CXX_COMPILER_ID:GNU>:-fno-rtti>
    $<$<CXX_COMPILER_ID:Clang>:-fno-rtti>
    $<$<CXX_COMPILER_ID:MSVC>:/GR->

    # Different flags per config
    $<$<CONFIG:Debug>:-O0 -g -fsanitize=address>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
)

target_link_options(mylib PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)
```

### Precompiled Headers

```cmake
target_precompile_headers(mylib PRIVATE
    <vector>
    <string>
    <unordered_map>
    <memory>
    <algorithm>
)
```

---

## 4. Cross-Platform Build

### Platform Detection

```cmake
if(WIN32)
    target_compile_definitions(mylib PRIVATE PLATFORM_WINDOWS)
elseif(APPLE)
    target_compile_definitions(mylib PRIVATE PLATFORM_MACOS)
elseif(UNIX)
    target_compile_definitions(mylib PRIVATE PLATFORM_LINUX)
endif()
```

### Configuring Builds

```bash
# Linux/macOS — Ninja (fastest)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Windows — Visual Studio
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# macOS — Xcode
cmake -B build -G Xcode
cmake --build build --config Release

# Cross-compilation
cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64.cmake
```

### Toolchain File Example (cross-compile)

```cmake
# toolchain-aarch64.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

---

## 5. Package Management

### FetchContent (Download at Configure Time)

```cmake
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.13.0
)

FetchContent_MakeAvailable(googletest fmt spdlog)

target_link_libraries(mylib PRIVATE fmt::fmt spdlog::spdlog)
```

### find_package (System-Installed)

```cmake
# Config mode (preferred — library provides CMake config)
find_package(Eigen3 3.4 REQUIRED CONFIG)
target_link_libraries(mylib PRIVATE Eigen3::Eigen)

# Module mode (uses FindXxx.cmake)
find_package(OpenMP REQUIRED)
target_link_libraries(mylib PRIVATE OpenMP::OpenMP_CXX)

find_package(Threads REQUIRED)
target_link_libraries(mylib PRIVATE Threads::Threads)

# Optional package
find_package(TBB QUIET)
if(TBB_FOUND)
    target_link_libraries(mylib PRIVATE TBB::tbb)
    target_compile_definitions(mylib PRIVATE HAS_TBB)
endif()
```

### Conan Integration (conanfile.txt)

```ini
# conanfile.txt
[requires]
fmt/10.2.1
spdlog/1.13.0
boost/1.84.0

[generators]
CMakeDeps
CMakeToolchain

[layout]
cmake_layout
```

After creating your `conanfile.txt`, run these commands to install the dependencies and build your project using the Conan-generated CMake presets.

```bash
# Build with Conan 2.x
conan install . --output-folder=build --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
```

### vcpkg Integration

```json
// vcpkg.json (manifest mode)
{
    "name": "my-project",
    "version": "1.0.0",
    "dependencies": [
        "fmt",
        "spdlog",
        "gtest",
        {
            "name": "cuda",
            "platform": "linux | windows"
        }
    ]
}
```

To use vcpkg with CMake, point `CMAKE_TOOLCHAIN_FILE` at the vcpkg toolchain script. This lets CMake automatically find all packages declared in your `vcpkg.json` manifest.

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

---

## 6. C++20 Modules Support

### CMake 3.28+ Module Support

```cmake
cmake_minimum_required(VERSION 3.28)
project(modules-example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Experimental module scanning (required as of CMake 3.28)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library(math_module)
target_sources(math_module
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/math.cppm
)

add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE math_module)
```

### Module Source File (math.cppm)

```cpp
// src/math.cppm
export module math;

export namespace math {
    constexpr double pi = 3.14159265358979323846;

    double square(double x) { return x * x; }
    double cube(double x) { return x * x * x; }
}
```

### Consumer (main.cpp)

```cpp
// src/main.cpp
import math;
#include <iostream>

int main() {
    std::cout << math::square(math::pi) << "\n";
}
```

---

## 7. CI/CD: GitHub Actions for C++/CUDA

### Complete Workflow (.github/workflows/ci.yml)

```yaml
name: C++/CUDA CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-cpp:
    runs-on: ubuntu-24.04
    strategy:
      matrix:
        compiler: [gcc-13, clang-17]
        build_type: [Debug, Release]
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y ninja-build cmake

      - name: Configure
        run: |
          cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \
            -DCMAKE_CXX_COMPILER=${{ matrix.compiler == 'gcc-13' && 'g++-13' || 'clang++-17' }} \
            -DBUILD_TESTING=ON

      - name: Build
        run: cmake --build build -j$(nproc)

      - name: Test
        run: ctest --test-dir build --output-on-failure -j$(nproc)

  build-cuda:
    runs-on: ubuntu-24.04
    container:
      image: nvidia/cuda:12.4.1-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v4

      - name: Install tools
        run: |
          apt-get update
          apt-get install -y cmake ninja-build git

      - name: Configure
        run: |
          cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES="80;90"

      - name: Build
        run: cmake --build build -j$(nproc)

  sanitizers:
    runs-on: ubuntu-24.04
    strategy:
      matrix:
        sanitizer: [address, thread, undefined]
    steps:
      - uses: actions/checkout@v4
      - name: Build with sanitizer
        run: |
          cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_CXX_FLAGS="-fsanitize=${{ matrix.sanitizer }}" \
            -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=${{ matrix.sanitizer }}"
          cmake --build build
      - name: Test with sanitizer
        run: ctest --test-dir build --output-on-failure

  clang-tidy:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Run clang-tidy
        run: |
          cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
          cmake --build build
          run-clang-tidy -p build src/
```

---

## 8. Dockerfile: CUDA Development Environment

### Full Development Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    gdb \
    valgrind \
    clang-17 \
    clang-tidy-17 \
    clang-format-17 \
    python3-pip \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install newer CMake (if needed)
ARG CMAKE_VERSION=3.29.3
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz \
    && tar xzf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -C /opt \
    && ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin/ \
    && rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# Install C++ dependencies
RUN pip3 install conan==2.* cmake-format

WORKDIR /workspace

# Builder stage
FROM base AS builder
COPY . /workspace/project
WORKDIR /workspace/project
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    && cmake --build build -j$(nproc)

# Runtime stage (minimal image)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
COPY --from=builder /workspace/project/build/my-app /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/my-app"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: "3.8"
services:
  dev:
    build:
      context: .
      target: base
    volumes:
      - .:/workspace/project
    working_dir: /workspace/project
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: sleep infinity
```

### Usage

```bash
# Build and run
docker compose up -d dev
docker compose exec dev bash

# Inside container
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build
```

---

## 9. Common Issues and Fixes

### Issue: CUDA Toolkit Not Found

```
CMake Error: No CMAKE_CUDA_COMPILER could be found.
```

**Fix:**
```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or specify explicitly
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Issue: Incompatible Host Compiler

```
nvcc fatal: The version ('13.x') of the host compiler ('gcc') is not supported
```

**Fix:**
```bash
# Use a supported GCC version
cmake -B build -DCMAKE_CUDA_HOST_COMPILER=g++-12
# Or use Clang
cmake -B build -DCMAKE_CUDA_HOST_COMPILER=clang++-16
```

### Issue: Linker Errors with CUDA Separable Compilation

```
undefined reference to __cudaRegisterLinkedBinary
```

**Fix:**
```cmake
set_target_properties(my_target PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
```

### Issue: ABI Compatibility Across Compilers

```
undefined symbol: _ZN3foo3barEv (mangled name)
```

**Fix:**
```cmake
# Ensure consistent compiler and standard library
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
# Or use extern "C" for cross-language interfaces
```

### Issue: CMake Not Finding Installed Packages

```cmake
# Set search paths
list(APPEND CMAKE_PREFIX_PATH "/opt/custom/lib/cmake")

# Or set per-package
set(Eigen3_DIR "/opt/eigen/share/eigen3/cmake")

# Debug find_package
cmake -B build -DCMAKE_FIND_DEBUG_MODE=ON
```

### Issue: CUDA + C++ Standard Mismatch

```cmake
# Ensure both compilers use the same standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)  # CUDA 12+ supports C++20
```

### Issue: Slow Builds

```cmake
# Use Ninja (faster than Make)
cmake -B build -G Ninja

# Enable ccache
find_program(CCACHE ccache)
if(CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
endif()

# Unity builds (combine source files)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)

# Precompiled headers
target_precompile_headers(mylib PRIVATE <vector> <string> <memory>)
```

### Issue: link-time optimization (LTO)

```cmake
# Enable LTO for Release
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# Check if LTO is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set_target_properties(mylib PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
    )
endif()
```

---

## 10. CMake Presets (CMakePresets.json)

```json
{
    "version": 6,
    "cmakeMinimumRequired": { "major": 3, "minor": 25, "patch": 0 },
    "configurePresets": [
        {
            "name": "default",
            "hidden": true,
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build/${presetName}",
            "cacheVariables": {
                "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
            }
        },
        {
            "name": "debug",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "CMAKE_CXX_FLAGS": "-fsanitize=address,undefined"
            }
        },
        {
            "name": "release",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_INTERPROCEDURAL_OPTIMIZATION": "ON"
            }
        },
        {
            "name": "cuda-release",
            "inherits": "release",
            "cacheVariables": {
                "CMAKE_CUDA_ARCHITECTURES": "80;90"
            }
        }
    ],
    "buildPresets": [
        { "name": "debug", "configurePreset": "debug" },
        { "name": "release", "configurePreset": "release" },
        { "name": "cuda-release", "configurePreset": "cuda-release" }
    ],
    "testPresets": [
        {
            "name": "debug",
            "configurePreset": "debug",
            "output": { "outputOnFailure": true }
        }
    ]
}
```

Once your `CMakePresets.json` is in place, you can configure, build, and test using named presets instead of manually passing flags each time.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset debug
```

---

*Appendix C — CMake & Build System Cookbook — Part of the CPP-CUDA-Mastery series*
