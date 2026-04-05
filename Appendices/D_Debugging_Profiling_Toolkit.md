# Appendix D — Debugging & Profiling Toolkit Reference

> Complete reference for debugging and profiling C++ and CUDA applications.
> Covers GDB, Valgrind, sanitizers, CUDA tools, Nsight, perf, and common bug patterns.

---

## 1. GDB Cheat Sheet

### Starting GDB

These commands show different ways to launch GDB: running a program directly, passing arguments, attaching to a running process, analyzing a crash dump, or using the split-view TUI mode.

```bash
# Basic
gdb ./my_program
gdb --args ./my_program arg1 arg2

# Attach to running process
gdb -p <PID>

# Core dump analysis
gdb ./my_program core.12345

# TUI mode (split view)
gdb -tui ./my_program
```

### Essential Commands

| Command                     | Short | Description                              |
|-----------------------------|-------|------------------------------------------|
| `run [args]`                | `r`   | Start execution                          |
| `break main`                | `b`   | Set breakpoint at function               |
| `break file.cpp:42`         | `b`   | Set breakpoint at line                   |
| `break func if x > 10`      |       | Conditional breakpoint                   |
| `continue`                  | `c`   | Continue execution                       |
| `next`                      | `n`   | Step over (don't enter functions)        |
| `step`                      | `s`   | Step into function                       |
| `finish`                    | `fin` | Run until current function returns       |
| `until 50`                  | `u`   | Run until line 50                        |
| `print expr`                | `p`   | Print expression value                   |
| `print/x expr`              |       | Print in hex                             |
| `display expr`              |       | Print expr at every stop                 |
| `watch var`                 |       | Break when var changes (hardware)        |
| `rwatch var`                |       | Break when var is read                   |
| `info breakpoints`          | `i b` | List all breakpoints                     |
| `delete 3`                  | `d 3` | Delete breakpoint #3                     |
| `disable 3`                 |       | Disable breakpoint #3                    |
| `backtrace`                 | `bt`  | Print call stack                         |
| `backtrace full`            |       | Call stack with local variables           |
| `frame 3`                   | `f 3` | Select stack frame #3                    |
| `up` / `down`               |       | Navigate stack frames                    |
| `list`                      | `l`   | Show source around current line          |
| `info locals`               |       | Print all local variables                |
| `info args`                 |       | Print function arguments                 |
| `set var x = 42`            |       | Modify variable                          |
| `call func(args)`           |       | Call a function                          |
| `quit`                      | `q`   | Exit GDB                                |

### Thread Commands

These GDB commands help debug multi-threaded programs. You can list all threads, switch between them, get backtraces for every thread at once, and lock scheduling to prevent other threads from running during stepping.

```gdb
info threads                    # List all threads
thread 3                        # Switch to thread 3
thread apply all bt             # Backtrace for all threads
thread apply all bt full        # Full backtrace, all threads
set scheduler-locking on        # Only run current thread
set scheduler-locking step      # Lock during stepping only
break func thread 2             # Breakpoint only for thread 2
```

### STL Pretty Printers

GDB can display STL containers (vectors, maps, etc.) in a human-readable format using pretty printers. This script ensures they are loaded — most GCC installations enable them automatically.

```bash
# Usually auto-loaded with GCC. Verify:
(gdb) info pretty-printer

# Manual setup (add to ~/.gdbinit)
python
import sys
sys.path.insert(0, '/usr/share/gcc/python')
from libstdcxx.v6.printers import register_libstdcxx_printers
register_libstdcxx_printers(None)
end
```

With pretty printers enabled, GDB shows the logical contents of STL containers instead of raw memory. A `vector` shows its elements, a `map` shows its key-value pairs, and a `unique_ptr` shows the managed pointer address.

```gdb
# With pretty printers enabled:
(gdb) print my_vector
$1 = std::vector of length 3, capacity 4 = {10, 20, 30}

(gdb) print my_map
$2 = std::map with 2 elements = {["hello"] = 1, ["world"] = 2}

(gdb) print my_unique_ptr
$3 = std::unique_ptr<MyClass> = {get() = 0x55555576b2c0}
```

### Advanced GDB

These advanced GDB features include reverse debugging (stepping backwards through execution), catching exceptions and system calls, examining raw memory in different formats, and writing custom GDB scripts for repetitive tasks.

```gdb
# Reverse debugging (record/replay)
record
reverse-continue    # rc
reverse-step        # rs
reverse-next        # rn

# Catchpoints
catch throw                     # Break on any exception throw
catch throw std::bad_alloc      # Break on specific exception
catch syscall write             # Break on system call

# Memory examination
x/16xb ptr          # 16 bytes in hex
x/4xw ptr           # 4 words in hex
x/s ptr             # As string
x/10i $pc           # 10 instructions at PC

# GDB scripts (.gdbinit)
define print_vec
    set $i = 0
    while $i < $arg0.size()
        print $arg0[$i]
        set $i = $i + 1
    end
end
```

---

## 2. Valgrind Suite

### Memcheck — Memory Error Detection

These commands run Valgrind's memory checker to detect leaks, buffer overflows, and use-after-free bugs. The `--track-origins=yes` flag traces uninitialized values back to their source, making bugs easier to find.

```bash
# Basic usage
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
    --track-origins=yes ./my_program

# Generate suppressions
valgrind --gen-suppressions=all ./my_program 2> suppressions.txt

# Use suppressions
valgrind --suppressions=my_suppressions.supp ./my_program
```

**Common Memcheck Findings:**

| Error                        | Meaning                              | Fix                           |
|------------------------------|--------------------------------------|-------------------------------|
| Invalid read of size N       | Use-after-free or buffer overread    | Fix lifetime or bounds        |
| Invalid write of size N      | Buffer overflow or use-after-free    | Fix allocation size or lifetime|
| Conditional jump on uninit   | Using uninitialized variable          | Initialize before use         |
| Definitely lost              | Memory leaked, no pointers remain    | Free or use smart pointers    |
| Indirectly lost              | Leaked via another leaked block      | Fix the root leak             |
| Possibly lost               | Interior pointer to block            | Usually false positive         |
| Still reachable              | Allocated, pointer exists at exit    | Usually intentional            |

### Cachegrind — Cache Profiling

Cachegrind profiles how your program uses the CPU cache hierarchy. It counts cache misses at each level, helping you identify memory access patterns that cause performance problems.

```bash
# Profile cache behavior
valgrind --tool=cachegrind ./my_program

# Annotate source with cache stats
cg_annotate cachegrind.out.<pid>

# Compare two runs
cg_diff cachegrind.out.1 cachegrind.out.2
```

**Key Metrics:**
- `I refs` / `D refs`: instruction/data references
- `I1 miss` / `D1 miss`: L1 cache misses
- `LL miss`: Last-level (L3) cache misses
- `D1 miss rate`: should be <5% for well-optimized code

### Callgrind — Call Graph Profiling

Callgrind records function call counts and costs, generating a detailed call graph. Use KCachegrind to visualize which functions consume the most time and how they relate to each other.

```bash
# Profile with call graph
valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./my_program

# Toggle collection on/off during execution
valgrind --tool=callgrind --instr-atstart=no ./my_program
# In another terminal: callgrind_control -i on/off

# Visualize with KCachegrind
kcachegrind callgrind.out
```

### Helgrind / DRD — Thread Error Detection

These tools detect threading bugs. Helgrind finds data races (concurrent unsynchronized memory access) and potential deadlocks (lock ordering violations). DRD is a lighter-weight alternative.

```bash
# Detect data races
valgrind --tool=helgrind ./my_program

# Alternative: DRD (lighter weight)
valgrind --tool=drd ./my_program

# Common findings:
# - Conflicting accesses without locks
# - Lock ordering violations (potential deadlock)
# - Misuse of pthreads API
```

---

## 3. Compiler Sanitizers

### AddressSanitizer (ASan) — Memory Errors

Compile your program with AddressSanitizer by adding the `-fsanitize=address` flag. ASan instruments memory accesses at compile time to catch buffer overflows, use-after-free, and memory leaks at runtime with about 2× slowdown.

```bash
# Compile
g++ -fsanitize=address -fno-omit-frame-pointer -g -O1 main.cpp -o main

# Or with CMake
cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
              -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
```

**What ASan Detects:**
- Heap buffer overflow / underflow
- Stack buffer overflow
- Use-after-free / use-after-return
- Double-free / invalid-free
- Memory leaks (with `ASAN_OPTIONS=detect_leaks=1`)

**Runtime Options:** You can fine-tune ASan behavior through the `ASAN_OPTIONS` environment variable. For example, enable leak detection, continue running after the first error, or point to a suppressions file for known false positives.

```bash
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:print_stats=1"
export ASAN_OPTIONS="suppressions=asan_supp.txt:fast_unwind_on_malloc=0"
```

**Example ASan Output:**
```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014
READ of size 4 at 0x602000000014 thread T0
    #0 0x4006a7 in main test.cpp:5
    #1 0x7f...  in __libc_start_main

0x602000000014 is located 0 bytes to the right of 8-byte region
    allocated by thread T0 here:
    #0 0x7f...  in operator new[](unsigned long)
    #1 0x40068e in main test.cpp:3
```

### ThreadSanitizer (TSan) — Data Race Detection

ThreadSanitizer requires a separate build (it's incompatible with ASan). It detects data races — situations where multiple threads access the same memory without proper synchronization — at runtime with 5-15× slowdown.

```bash
# Compile (separate build — incompatible with ASan)
g++ -fsanitize=thread -g -O2 main.cpp -o main -lpthread

export TSAN_OPTIONS="history_size=7:second_deadlock_stack=1"
```

**What TSan Detects:**
- Data races (unsynchronized concurrent access)
- Deadlocks (lock ordering violations)
- Thread leaks
- Misuse of synchronization primitives

**Example TSan Output:**
```
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 4 at 0x7f... by thread T1:
    #0 increment() race.cpp:8
  Previous read of size 4 at 0x7f... by main thread:
    #0 main() race.cpp:15
  Thread T1 created at:
    #0 pthread_create
    #1 main() race.cpp:13
```

### UndefinedBehaviorSanitizer (UBSan)

UndefinedBehaviorSanitizer catches C/C++ operations with undefined results, such as integer overflow or null pointer dereference. It has almost no overhead (1.2×) and can be combined with ASan in the same build.

```bash
# Compile
g++ -fsanitize=undefined -g main.cpp -o main

# Can combine with ASan
g++ -fsanitize=address,undefined -g main.cpp -o main
```

**What UBSan Detects:**
- Signed integer overflow
- Division by zero
- Null pointer dereference
- Out-of-bounds array access (static arrays)
- Misaligned pointer access
- Invalid enum values
- Invalid bool values
- Shift by negative or too-large amount

**Runtime Options:** Use the `UBSAN_OPTIONS` environment variable to control UBSan's behavior at runtime. Enabling `print_stacktrace` gives you full call stacks on errors, and `halt_on_error` makes the program abort on the first violation instead of continuing.

```bash
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
```

### MemorySanitizer (MSan) — Uninitialized Memory

MemorySanitizer detects reads of uninitialized memory. It's Clang-only and requires all libraries (including libc++) to be compiled with MSan instrumentation, making it harder to use in practice than the other sanitizers.

```bash
# Clang only (not available in GCC)
clang++ -fsanitize=memory -fno-omit-frame-pointer -g main.cpp -o main

# NOTE: All libraries (including libc++) must be compiled with MSan
# This makes it difficult to use in practice
```

### Sanitizer Comparison

| Sanitizer | Slowdown | Memory | Combines With | Key Strength            |
|-----------|----------|--------|---------------|-------------------------|
| ASan      | 2x       | 3x     | UBSan         | Memory errors           |
| TSan      | 5-15x    | 5-10x  | Nothing       | Data races              |
| UBSan     | 1.2x     | 1x     | ASan          | Undefined behavior      |
| MSan      | 3x       | 2x     | Nothing       | Uninitialized reads     |

---

## 4. CUDA Debugging Tools

### compute-sanitizer (Replaces cuda-memcheck)

The `compute-sanitizer` tool (successor to `cuda-memcheck`) runs your CUDA program with instrumentation to detect GPU memory errors, race conditions, synchronization bugs, and uninitialized device memory.

```bash
# Memory checking (default)
compute-sanitizer --tool memcheck ./cuda_program

# Race condition detection
compute-sanitizer --tool racecheck ./cuda_program

# Synchronization checking
compute-sanitizer --tool synccheck ./cuda_program

# Initialization checking
compute-sanitizer --tool initcheck ./cuda_program

# Common options
compute-sanitizer --tool memcheck \
    --show-backtrace yes \
    --report-api-errors all \
    --log-file sanitizer.log \
    ./cuda_program
```

**Common Findings:**
| Error                           | Meaning                            | Fix                            |
|---------------------------------|------------------------------------|--------------------------------|
| Invalid __global__ read/write   | Out-of-bounds device memory        | Check kernel index calculation |
| Misaligned access               | Address not aligned to type size   | Align allocations              |
| Race condition (WAW/RAW/WAR)    | Unsynchronized shared memory       | Add `__syncthreads()`          |
| Barrier error                   | Not all threads reach barrier      | Ensure uniform control flow    |
| Invalid device ordinal          | GPU index out of range             | Check `cudaGetDeviceCount`     |

### cuda-gdb — GPU Debugger

The `cuda-gdb` debugger extends GDB with commands for inspecting GPU state. You can switch between GPU threads and blocks, set breakpoints in kernels with conditions on thread/block indices, and examine device and shared memory.

```bash
# Launch
cuda-gdb ./cuda_program

# CUDA-specific commands
(cuda-gdb) info cuda threads        # List GPU threads
(cuda-gdb) info cuda kernels        # List active kernels
(cuda-gdb) info cuda blocks         # List active blocks
(cuda-gdb) cuda thread (0,0,0)      # Switch to specific thread
(cuda-gdb) cuda block (1,0,0)       # Switch to specific block
(cuda-gdb) cuda kernel 0            # Switch to kernel
(cuda-gdb) print threadIdx          # Print thread index
(cuda-gdb) print blockIdx           # Print block index

# Set breakpoint in kernel
(cuda-gdb) break my_kernel
(cuda-gdb) break my_kernel.cu:42
(cuda-gdb) break my_kernel if threadIdx.x == 0 && blockIdx.x == 0

# Inspect device memory
(cuda-gdb) print @global d_array[0]@10    # 10 elements
(cuda-gdb) print @shared sdata[0]@32      # Shared memory
```

### Printf Debugging in Kernels

CUDA kernels can use `printf` for quick debugging. Limit output to a single thread (e.g., thread 0) to avoid flooding, and watch for NaN/Inf values that indicate numerical issues.

```cuda
__global__ void myKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only print from one thread to avoid output flood
    if (idx == 0) {
        printf("Kernel launched: N=%d, gridDim=%d, blockDim=%d\n",
               N, gridDim.x, blockDim.x);
    }

    if (idx < N) {
        float val = data[idx];
        // Print suspicious values
        if (isnan(val) || isinf(val)) {
            printf("NaN/Inf at idx=%d, block=%d, thread=%d\n",
                   idx, blockIdx.x, threadIdx.x);
        }
    }
}
```

**Notes on CUDA printf:**
- Output buffer is 1MB by default (increase with `cudaDeviceSetLimit`)
- Output appears after `cudaDeviceSynchronize()` or kernel completion
- Very slow — only use for debugging, never in production

---

## 5. Nsight Compute — Kernel Profiling

### Basic Usage

Nsight Compute (`ncu`) profiles individual CUDA kernels in detail. You can target specific kernels, collect full metric sets for roofline analysis, or query specific metrics like SM throughput and memory bandwidth utilization.

```bash
# Profile all kernels
ncu ./cuda_program

# Profile specific kernel
ncu --kernel-name matmul_kernel ./cuda_program

# Full metrics collection (slow but comprehensive)
ncu --set full -o report ./cuda_program

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
./cuda_program

# Launch count control
ncu --launch-count 5 --launch-skip 2 ./cuda_program

# Roofline analysis
ncu --set roofline -o roofline_report ./cuda_program
```

### Key Metrics Dictionary

| Metric                                          | What It Means                          | Target           |
|-------------------------------------------------|----------------------------------------|------------------|
| `sm__throughput.avg.pct_of_peak_sustained`      | SM compute utilization                 | >60% if compute-bound |
| `dram__throughput.avg.pct_of_peak_sustained`     | HBM bandwidth utilization              | >60% if memory-bound  |
| `l1tex__throughput.avg.pct_of_peak_sustained`    | L1/Tex cache throughput                | —                |
| `sm__warps_active.avg.pct_of_peak_sustained`     | Active warps (occupancy indicator)     | >50%             |
| `smsp__warps_launched.avg`                       | Warps launched per SM                  | —                |
| `sm__sass_thread_inst_executed_op_fadd_pred_on`  | FP32 add instructions executed         | —                |
| `l2_cache_hit_rate`                              | L2 cache effectiveness                 | Higher is better |
| `dram__bytes_read.sum + dram__bytes_write.sum`   | Total HBM traffic                      | Minimize         |
| `smsp__cycles_active.avg`                        | Average cycles SM is active            | —                |
| `sm__pipe_tensor_cycles_active.avg`              | Tensor Core utilization                | >50% for GEMM    |
| `launch__occupancy`                              | Theoretical occupancy                  | Depends on kernel |
| `smsp__inst_executed.sum`                        | Total instructions executed            | —                |

### Interpreting Results

```
Is your kernel memory-bound or compute-bound?

If dram__throughput > sm__throughput → Memory-bound
  → Improve coalescing, use shared memory, reduce data movement

If sm__throughput > dram__throughput → Compute-bound
  → Reduce instruction count, use Tensor Cores, increase ILP

If both are low → Latency-bound
  → Increase occupancy, reduce synchronization, increase parallelism
```

---

## 6. Nsight Systems — System-Wide Profiling

### Basic Usage

Nsight Systems (`nsys`) provides a system-wide timeline view showing the interplay between CPU activity, GPU kernels, memory transfers, and API calls. It's the best tool for finding pipeline bottlenecks.

```bash
# Profile entire application
nsys profile -o report ./cuda_program

# With specific options
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o timeline_report \
    ./cuda_program

# Profile MPI application
nsys profile --trace=cuda,mpi,nvtx mpirun -np 4 ./cuda_mpi_program
```

### NVTX Markers (Annotate Code)

NVTX markers let you annotate your code with named regions that appear in the Nsight Systems timeline. This makes it easy to see which phase (forward pass, backward pass, etc.) each GPU operation belongs to. The RAII wrapper automatically ends the region when the scope exits.

```cpp
#include <nvtx3/nvToolsExt.h>

void trainStep() {
    nvtxRangePushA("Forward Pass");
    forward(model, input);
    nvtxRangePop();

    nvtxRangePushA("Backward Pass");
    backward(model, loss);
    nvtxRangePop();

    nvtxRangePushA("Optimizer Step");
    optimizer.step();
    nvtxRangePop();
}

// C++ RAII wrapper
struct NvtxRange {
    NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

void process() {
    NvtxRange r("process");  // auto-pops on scope exit
    // ...
}
```

### What to Look For in Timeline

| Pattern                    | Problem                              | Fix                             |
|----------------------------|--------------------------------------|---------------------------------|
| Gaps between kernels       | CPU overhead / launch latency        | CUDA Graphs, reduce API calls   |
| Long memcpy, idle GPU      | Transfer-bound                       | Overlap with streams, pin memory|
| Small kernels back-to-back | Launch overhead dominates            | Fuse kernels, CUDA Graphs       |
| One GPU busy, others idle  | Poor load balancing                  | Redistribute work               |
| CPU 100%, GPU idle         | CPU bottleneck                       | Move preprocessing to GPU       |
| Sync calls blocking CPU    | Unnecessary synchronization          | Use async APIs, events          |

---

## 7. perf + Flamegraphs

### Linux perf Basics

Linux `perf` profiles CPU-side performance using hardware counters. It measures cache misses, branch mispredictions, and instruction counts — useful for optimizing the host code that launches and manages GPU work.

```bash
# Record CPU performance
perf record -g -F 99 ./my_program
perf report

# Stat summary (hardware counters)
perf stat -d ./my_program

# Specific events
perf stat -e cache-misses,cache-references,branch-misses,instructions,cycles \
    ./my_program

# Record specific events
perf record -e cache-misses -g ./my_program
```

### Generating Flamegraphs

Flamegraphs visualize `perf` profiling data as an interactive SVG. Wide bars indicate functions that consume the most CPU time. The `inferno` tool is a simpler alternative to the original Perl-based FlameGraph scripts.

```bash
# Record
perf record -F 99 -g --call-graph dwarf ./my_program

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# Or using inferno (Rust tool, simpler)
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

### Reading Flamegraphs

```
- Width = time spent (wider = more time)
- Y-axis = stack depth (top = leaf function)
- Color = random (no meaning by default)
- Look for wide plateaus (hot functions)
- Look for deep stacks (potential optimization via inlining)
- Compare before/after to validate optimization
```

### perf + CUDA

Running `perf` on a CUDA application profiles only the CPU side — it shows CUDA API call overhead and driver activity, but not GPU kernel internals. Use Nsight tools for GPU-side profiling.

```bash
# Profile CUDA application (CPU side)
perf record -g ./cuda_program

# For GPU profiling, use Nsight Systems instead
# perf only sees CPU-side activity (API calls, driver overhead)
```

---

## 8. Common Bug Patterns & Fixes

### Pattern 1: Use-After-Free

This demonstrates a classic use-after-free bug: returning a pointer to a stack-local variable that gets destroyed when the function exits. The fix is to return by value, which triggers copy elision (NRVO) for zero-cost.

```cpp
// BUG
std::string* createString() {
    std::string local = "hello";
    return &local;  // dangling pointer
}

// FIX: return by value or use smart pointer
std::string createString() {
    return "hello";  // NRVO/move
}
```

### Pattern 2: Double-Free

This shows a double-free bug caused by ambiguous ownership — both the caller and the function try to free the same memory. Using `unique_ptr` makes ownership explicit: whoever holds the pointer owns the memory.

```cpp
// BUG
void process(int* data) {
    delete[] data;
}
int* arr = new int[100];
process(arr);
delete[] arr;  // double-free!

// FIX: use unique_ptr for clear ownership
void process(std::unique_ptr<int[]> data) {
    // automatically freed
}
auto arr = std::make_unique<int[]>(100);
process(std::move(arr));
```

### Pattern 3: Data Race

This demonstrates a data race: multiple threads incrementing a shared counter without synchronization. The fix uses `std::atomic` which provides hardware-level guarantees that the read-modify-write operation is indivisible.

```cpp
// BUG
int counter = 0;
void increment() { counter++; }  // not thread-safe

// FIX: use atomic
std::atomic<int> counter{0};
void increment() { counter.fetch_add(1, std::memory_order_relaxed); }
```

### Pattern 4: Deadlock

This shows a deadlock caused by two threads acquiring locks in opposite order. Using `std::scoped_lock` acquires both locks simultaneously using a deadlock-avoidance algorithm.

```cpp
// BUG: inconsistent lock ordering
void thread1() { lock(a); lock(b); }
void thread2() { lock(b); lock(a); }  // deadlock!

// FIX: use scoped_lock for simultaneous locking
void thread1() { std::scoped_lock lk(a, b); }
void thread2() { std::scoped_lock lk(a, b); }
```

### Pattern 5: CUDA Illegal Memory Access

This demonstrates an out-of-bounds GPU memory access caused by missing bounds checking. GPU threads beyond the array size must be guarded with an `if (idx < N)` check to prevent illegal memory access.

```cuda
// BUG: out-of-bounds access
__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 0;  // no bounds check!
}

// FIX: bounds check
__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = 0;
    }
}
```

### Pattern 6: CUDA Race Condition in Shared Memory

This shows a shared memory race condition where threads read data that other threads haven't finished writing yet. The `__syncthreads()` barrier ensures all threads complete their writes before any thread reads.

```cuda
// BUG: missing synchronization
__global__ void kernel() {
    __shared__ float sdata[256];
    sdata[threadIdx.x] = globalData[idx];
    // Missing __syncthreads() here!
    float val = sdata[threadIdx.x ^ 1];  // race!
}

// FIX
__global__ void kernel() {
    __shared__ float sdata[256];
    sdata[threadIdx.x] = globalData[idx];
    __syncthreads();  // all threads must store before any reads
    float val = sdata[threadIdx.x ^ 1];
}
```

### Pattern 7: Iterator Invalidation

This demonstrates iterator invalidation: erasing elements from a vector while iterating over it causes the iterator to become invalid. C++20's `std::erase_if` or the pre-C++20 erase-remove idiom handles this correctly.

```cpp
// BUG: modifying container while iterating
std::vector<int> v = {1, 2, 3, 4, 5};
for (auto it = v.begin(); it != v.end(); ++it) {
    if (*it % 2 == 0) {
        v.erase(it);  // invalidates iterator!
    }
}

// FIX: use erase-remove idiom
std::erase_if(v, [](int x) { return x % 2 == 0; });  // C++20
// Or pre-C++20:
v.erase(std::remove_if(v.begin(), v.end(),
    [](int x) { return x % 2 == 0; }), v.end());
```

### Pattern 8: CUDA Forgetting Error Checks

This shows silent CUDA failures caused by unchecked error codes. The `CUDA_CHECK` macro wraps every CUDA API call and kernel launch, printing the error location and message if anything fails.

```cuda
// BUG: silent failure
cudaMalloc(&d_ptr, size);       // might fail
kernel<<<grid, block>>>(d_ptr); // might fail
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

// FIX: check every call
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

CUDA_CHECK(cudaMalloc(&d_ptr, size));
kernel<<<grid, block>>>(d_ptr);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
```

---

## 9. Debugging Decision Tree

```
What's the symptom?
├── Crash (segfault, abort)
│   ├── Run with ASan → pinpoints memory errors
│   ├── Run under GDB → get backtrace
│   └── Check core dump: gdb ./prog core
│
├── Wrong results
│   ├── Run with UBSan → catch undefined behavior
│   ├── Run with MSan → catch uninitialized reads
│   ├── Use GDB watchpoints → find where value changes
│   └── CUDA: use compute-sanitizer → catch device errors
│
├── Hang / deadlock
│   ├── GDB: thread apply all bt → see where threads are stuck
│   ├── TSan → detect lock ordering issues
│   └── CUDA: check for __syncthreads in divergent code
│
├── Memory leak
│   ├── ASan with detect_leaks=1
│   ├── Valgrind memcheck → detailed leak report
│   └── CUDA: check cudaFree calls match cudaMalloc
│
├── Slow performance
│   ├── perf stat → get hardware counter overview
│   ├── Flamegraph → find CPU hotspots
│   ├── Nsight Systems → GPU timeline
│   ├── Nsight Compute → kernel-level metrics
│   └── Cachegrind → cache behavior analysis
│
└── Data race
    ├── TSan → find unsynchronized accesses
    ├── Helgrind → alternative race detector
    └── CUDA: compute-sanitizer --tool racecheck
```

---

*Appendix D — Debugging & Profiling Toolkit Reference — Part of the CPP-CUDA-Mastery series*
