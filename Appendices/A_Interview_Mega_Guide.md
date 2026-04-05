# Appendix A — C++ & CUDA Interview Mega-Guide

> 100+ questions with model answers, organized by category.
> Difficulty: 🟢 Junior | 🟡 Mid | 🔴 Senior | 🔵 Staff

---

## 1. C++ Core (25 Questions)

### Q1. What is RAII and why is it fundamental to C++? 🟢
**A:** RAII (Resource Acquisition Is Initialization) ties resource lifetime to object lifetime. Resources are acquired in constructors and released in destructors. This guarantees cleanup even when exceptions occur, eliminates resource leaks, and makes ownership explicit. Every STL container, smart pointer, and lock guard uses RAII. Without RAII, C++ error handling would require manual cleanup paths like C, making code fragile and error-prone.

### Q2. Explain the difference between `unique_ptr`, `shared_ptr`, and `weak_ptr`. 🟢
**A:** `unique_ptr` models exclusive ownership — one owner, zero overhead, move-only. `shared_ptr` models shared ownership via reference counting — multiple owners, the object is destroyed when the last `shared_ptr` dies. `weak_ptr` is a non-owning observer of a `shared_ptr`-managed object; it breaks circular references and must be locked (`lock()`) before access, returning a `shared_ptr` or null. Prefer `unique_ptr` by default; use `shared_ptr` only when ownership is genuinely shared.

### Q3. What are move semantics and why were they introduced? 🟡
**A:** Move semantics (C++11) allow transferring resources from one object to another without copying. An rvalue reference (`T&&`) binds to temporaries. `std::move` casts an lvalue to an rvalue reference, enabling the move constructor/assignment to steal the source's internals (e.g., pointer, size) and leave it in a valid-but-empty state. This eliminates expensive deep copies for temporaries and return values, dramatically improving performance for containers, strings, and resource handles.

### Q4. What is the Rule of Five? When does the Rule of Zero apply? 🟡
**A:** Rule of Five: if you define any of destructor, copy constructor, copy assignment, move constructor, or move assignment, you should define all five. Rule of Zero: prefer classes that manage no resources directly, delegating to RAII members (smart pointers, containers). Rule of Zero classes need no special member functions — the compiler-generated defaults are correct. Use Rule of Five only for low-level resource wrappers; everything else should follow Rule of Zero.

### Q5. Explain virtual dispatch. What is the vtable? 🟢
**A:** When a class has virtual functions, the compiler creates a vtable — a table of function pointers for each virtual method. Each object of that class contains a hidden vptr pointing to its class's vtable. Calling a virtual function through a base pointer/reference goes through the vptr → vtable → function pointer, enabling runtime polymorphism. Cost: one indirection per call (~2-5ns), one pointer per object. Mark destructors virtual in base classes; use `final` to enable devirtualization.

### Q6. What is the difference between `const`, `constexpr`, and `consteval`? 🟡
**A:** `const` means the variable cannot be modified after initialization — but it may be initialized at runtime. `constexpr` means the value *can* be computed at compile time (and must be for variables, may be for functions). `consteval` (C++20) means the function *must* be evaluated at compile time — any runtime call is a compile error. Use `const` for runtime immutability, `constexpr` for compile-time-capable computation, `consteval` for forced compile-time-only functions.

### Q7. Explain template argument deduction and CTAD. 🟡
**A:** Template argument deduction lets the compiler infer template parameters from function arguments: `template<typename T> void f(T x);` — calling `f(42)` deduces `T=int`. CTAD (Class Template Argument Deduction, C++17) extends this to class templates: `std::vector v{1,2,3};` deduces `vector<int>`. Deduction guides (explicit or compiler-generated) control the mapping. Pitfalls: brace-enclosed lists can be ambiguous, and reference collapsing rules interact with forwarding references.

### Q8. What is SFINAE? How has it evolved? 🔴
**A:** SFINAE (Substitution Failure Is Not An Error): if substituting template arguments causes an invalid type, the overload is silently removed instead of causing a compile error. Used to conditionally enable functions: `enable_if`, `void_t`, `decltype` expressions. Evolution: C++11 added `enable_if`; C++17 added `if constexpr` for simpler branching; C++20 replaced most SFINAE with Concepts, which give cleaner syntax and better error messages. Modern code should prefer Concepts over raw SFINAE.

### Q9. What happens during stack unwinding? 🟡
**A:** When an exception is thrown, the runtime walks up the call stack, destroying local objects in reverse order of construction (calling destructors). This is stack unwinding. If a destructor throws during unwinding, `std::terminate` is called — so destructors must be `noexcept`. Unwinding ensures RAII cleanup but can be expensive (10-40µs on modern x86). `noexcept` functions skip unwinding tables, enabling better optimization.

### Q10. Explain the difference between `new`/`delete` and `malloc`/`free`. 🟢
**A:** `new` calls `operator new` (allocates memory) then the constructor; `delete` calls the destructor then `operator delete` (frees memory). `malloc`/`free` only allocate/free raw bytes — no construction/destruction. `new` is type-safe and exception-aware (throws `bad_alloc`); `malloc` returns `void*` and returns `NULL` on failure. Never mix them. In modern C++, avoid both — use `make_unique`/`make_shared` or containers.

### Q11. What is an allocator in C++? When would you write a custom one? 🔴
**A:** An allocator controls how containers obtain and release memory. The default `std::allocator` uses `new`/`delete`. Custom allocators are used for: arena/pool allocation (eliminate per-object overhead), NUMA-aware placement, shared memory regions, GPU pinned memory, or deterministic embedded systems. C++17 introduced `pmr` (polymorphic memory resources), simplifying custom allocation without retyping containers. Write one when profiling shows allocation is a bottleneck.

### Q12. Explain the difference between `static_cast`, `dynamic_cast`, `reinterpret_cast`, and `const_cast`. 🟢
**A:** `static_cast`: compile-time checked conversions (numeric, up/downcasting without runtime check). `dynamic_cast`: runtime-checked polymorphic downcasting using RTTI (returns null for pointers, throws for references). `reinterpret_cast`: bit-pattern reinterpretation (pointer to int, unrelated pointer types) — dangerous. `const_cast`: adds/removes `const`/`volatile`. Prefer `static_cast` for safe conversions; use `dynamic_cast` sparingly (it's slow); avoid `reinterpret_cast` except for serialization/hardware.

### Q13. What is the One Definition Rule (ODR)? 🟡
**A:** ODR states each entity (variable, function, class) must have exactly one definition across the entire program (for non-inline entities). Inline functions/variables may have multiple identical definitions across translation units. Violating ODR (e.g., defining a function in a header without `inline`) causes linker errors or — worse — silent undefined behavior if definitions differ. Templates are implicitly inline. Header-only libraries rely on this.

### Q14. How does `std::function` work internally? What is type erasure? 🔴
**A:** `std::function` uses type erasure: it stores any callable (lambda, function pointer, functor) behind a uniform interface. Internally it holds a pointer to a polymorphic wrapper that virtual-dispatches to the stored callable. Small callables may be stored inline (Small Buffer Optimization, typically 16-32 bytes). Costs: heap allocation for large callables, virtual dispatch per call. For hot paths, prefer templates or `auto` parameters to avoid this overhead.

### Q15. What is a lambda? How do captures work? 🟢
**A:** A lambda is an anonymous function object. The compiler generates a class with `operator()`. Captures: `[=]` copies, `[&]` references, `[x]` copies x, `[&x]` references x, `[this]` captures the enclosing object, `[*this]` (C++17) copies it. `mutable` allows modifying copied captures. Init captures (`[x = std::move(y)]`) enable move-capture. Generic lambdas (`auto` params) are templated. Beware dangling reference captures.

### Q16. Explain perfect forwarding and `std::forward`. 🔴
**A:** Perfect forwarding preserves the value category (lvalue/rvalue) of arguments through template functions. A forwarding reference (`T&&` where T is deduced) binds to both lvalues and rvalues. `std::forward<T>(arg)` conditionally casts: if T is an lvalue reference, it returns an lvalue; otherwise an rvalue. This allows factory functions, `emplace_back`, etc. to pass arguments to constructors exactly as the caller provided them, avoiding unnecessary copies.

### Q17. What is the `volatile` keyword? Is it useful for multithreading? 🟡
**A:** `volatile` tells the compiler not to optimize away reads/writes to a variable — every access goes to memory. It's for memory-mapped I/O and signal handlers, NOT for multithreading. `volatile` provides no atomicity, no ordering guarantees, and no memory fences. For threads, use `std::atomic`, which provides all three. Combining `volatile` with atomics is occasionally needed for memory-mapped device registers that are also shared between threads.

### Q18. What is undefined behavior? Give 5 examples. 🟡
**A:** UB means the standard imposes no requirements — the compiler may assume it never happens and optimize accordingly. Examples: (1) signed integer overflow, (2) dereferencing a null pointer, (3) accessing array out of bounds, (4) use-after-free, (5) data race on non-atomic variables. UB can cause anything: crashes, wrong results, or apparent correctness that breaks later. Sanitizers (ASan, UBSan) detect many UB cases at runtime.

### Q19. Explain the compilation model: preprocessing, compilation, assembly, linking. 🟢
**A:** (1) Preprocessor: expands `#include`, macros, conditional compilation → translation unit. (2) Compiler: parses C++, generates object code (`.o`/`.obj`). (3) Assembler: converts assembly to machine code (usually integrated). (4) Linker: resolves symbols across object files and libraries, produces executable/shared library. Static linking copies library code in; dynamic linking records references resolved at load time. Understanding this chain is essential for debugging linker errors and build systems.

### Q20. What is the `noexcept` specifier and why does it matter for performance? 🟡
**A:** `noexcept` declares a function won't throw exceptions. Benefits: (1) the compiler can omit stack-unwinding bookkeeping; (2) move operations: containers like `vector` use the move constructor only if it's `noexcept` — otherwise they fall back to copying for strong exception safety. Always mark move constructors/assignment `noexcept`. Mark destructors `noexcept` (they are by default). `noexcept(expr)` is conditional. A throwing function marked `noexcept` calls `std::terminate`.

### Q21. Explain `std::optional`, `std::variant`, and `std::any`. 🟡
**A:** `std::optional<T>`: a value or nothing (replaces nullable pointers, sentinel values). `std::variant<Ts...>`: type-safe tagged union holding one of Ts at a time — use `std::visit` for pattern matching. `std::any`: type-erased container for any copyable type — runtime checked. Prefer `optional` for nullable values, `variant` for closed type sets, and avoid `any` unless interfacing with truly dynamic systems. All three are stack-allocated (no heap).

### Q22. What are aggregate types and designated initializers? 🟡
**A:** An aggregate is a class/struct with no user-declared constructors, no private/protected members, no virtual functions, and no base classes (relaxed in C++17). Aggregates support aggregate initialization: `Point p{1, 2};`. C++20 added designated initializers: `Point p{.x=1, .y=2};` — must be in declaration order. Useful for config structs and POD types, providing named fields without constructor boilerplate.

### Q23. How do virtual base classes solve the diamond problem? 🟡
**A:** In diamond inheritance (B and C inherit A, D inherits both B and C), D gets two copies of A. Virtual inheritance (`class B : virtual public A`) ensures only one shared A subobject. The most-derived class (D) is responsible for constructing the virtual base. Cost: accessing virtual base members requires an indirection through a vptr. Generally, prefer composition over deep inheritance hierarchies.

### Q24. What is `std::string_view` and when should you use it? 🟢
**A:** `string_view` (C++17) is a non-owning view into a contiguous character sequence (pointer + length). Use it as a function parameter to accept `std::string`, `const char*`, or substrings without copying. Never store a `string_view` past the lifetime of the underlying string. It replaces `const std::string&` parameters for read-only access, avoiding the `const char*` → `string` temporary construction. Zero-cost abstraction: no allocation, no copying.

### Q25. Explain the `pImpl` (pointer to implementation) idiom. 🔴
**A:** pImpl hides a class's private members behind a forward-declared implementation class, accessed through a `unique_ptr`. Benefits: (1) compile firewall — changing private members doesn't recompile dependents; (2) ABI stability — the class size stays fixed (one pointer). Cost: heap allocation, one indirection per access. Commonly used in library APIs. Combine with `unique_ptr` and explicit destructor (defined in `.cpp` where `Impl` is complete).

---

## 2. Modern C++ (15 Questions)

### Q26. What are C++20 Concepts? How do they improve templates? 🟡
**A:** Concepts are named constraints on template parameters: `template<std::integral T>` replaces `enable_if` SFINAE. They provide: (1) cleaner syntax, (2) dramatically better error messages (the compiler says "T doesn't satisfy Sortable" instead of pages of template instantiation errors), (3) overload resolution based on constraints. Define custom concepts with `requires` expressions. Concepts subsumption allows partial ordering of overloads.

### Q27. Explain C++20 Ranges and Views. 🟡
**A:** Ranges generalize iterators: a range is anything with `begin()`/`end()`. Views are lazy, composable range adaptors: `views::filter`, `views::transform`, `views::take`. They use the pipe operator: `vec | filter(even) | transform(square) | take(5)`. Views don't own data, don't eagerly compute — they form a pipeline evaluated on iteration. This enables functional-style programming without temporary containers or performance loss.

### Q28. What are C++20 Coroutines? 🔴
**A:** Coroutines are functions that can suspend (`co_await`) and resume execution. They enable lazy generators (`co_yield`), async I/O, and cooperative multitasking without callback hell. The compiler transforms the function into a state machine with a coroutine frame (heap-allocated by default). Key customization points: `promise_type`, `awaiter` interface (`await_ready`, `await_suspend`, `await_resume`). Libraries like `cppcoro` provide task, generator, and async types built on this machinery.

### Q29. Explain `std::span` and how it replaces pointer+size. 🟢
**A:** `std::span<T>` (C++20) is a non-owning view over a contiguous sequence — essentially a pointer + size pair with bounds checking and range-for support. It replaces `(T* ptr, size_t n)` function parameters, accepting arrays, vectors, or other contiguous containers. `span<T, N>` is a fixed-extent span known at compile time. Use it to write functions that work with any contiguous container without templates or overloads.

### Q30. What are `constinit` and `consteval`? 🟡
**A:** `constinit` (C++20) ensures a variable is initialized at compile time but doesn't imply `const` — the variable can be modified at runtime. It prevents the "static initialization order fiasco." `consteval` creates "immediate functions" that *must* produce compile-time constants — any runtime evaluation is a compile error. Together with `constexpr`, C++20 provides fine-grained control: `constexpr` (may be compile-time), `consteval` (must be), `constinit` (init must be, value needn't stay).

### Q31. Explain structured bindings (C++17). 🟢
**A:** Structured bindings decompose aggregates, tuples, and pairs: `auto [x, y, z] = getPoint();`. They work with arrays, structs with all public members, and types providing `tuple_size`/`get<>`. Bindings are references to the decomposed elements (or copies, depending on `auto`/`auto&`/`auto&&`). They simplify range-based loops: `for (auto& [key, val] : map)`. A readability improvement — no more `.first`/`.second`.

### Q32. What is `if constexpr` and how does it differ from regular `if`? 🟡
**A:** `if constexpr` (C++17) evaluates the condition at compile time, discarding the false branch entirely. Unlike regular `if`, the discarded branch doesn't need to be valid code for the current template instantiation. This replaces many SFINAE patterns: `if constexpr (std::is_integral_v<T>) { ... } else { ... }`. Each branch can use operations valid only for its type. The condition must be a constant expression.

### Q33. Explain `std::format` (C++20) and its advantages. 🟢
**A:** `std::format` provides Python-like string formatting: `std::format("Hello {}, you are {} years old", name, age)`. Advantages over `printf`: type-safe, extensible (custom formatters via `std::formatter` specialization), no format-string vulnerabilities. Advantages over `iostream`: faster, positional arguments, compile-time format string checking (C++23 `print`). Supports fill/align, precision, type specifiers, and chrono formatting.

### Q34. What are C++20 Modules and how do they change compilation? 🔴
**A:** Modules replace `#include` with `import`: `import std;` instead of `#include <vector>`. Benefits: (1) no header re-parsing — modules are precompiled, 5-10x faster builds; (2) no macro leakage between translation units; (3) explicit export control. `export module foo;` declares a module, `export` marks public API. Implementation is `module foo; // not exported`. As of 2024, compiler support is improving but not yet universal; CMake 3.28+ adds module support.

### Q35. Explain `std::expected` (C++23) and error handling evolution. 🟡
**A:** `std::expected<T, E>` holds either a value (T) or an error (E) — a vocabulary type for fallible functions. Unlike exceptions (expensive on error path) or error codes (easy to ignore), `expected` forces callers to handle errors and has zero overhead on success. Supports monadic operations: `and_then`, `transform`, `or_else` for chaining. Evolution: C++98 exceptions → error codes → `optional` (C++17) → `expected` (C++23). Inspired by Rust's `Result<T, E>`.

### Q36. What are `std::jthread` and `std::stop_token`? 🟡
**A:** `std::jthread` (C++20) is a joining thread — its destructor automatically requests a stop and joins, preventing detached/leaked threads. `stop_token` enables cooperative cancellation: the thread periodically checks `token.stop_requested()`. `stop_callback` registers a callback when stop is requested. Together they replace the manual "set a flag + join in destructor" pattern. Always prefer `jthread` over `thread` in modern code.

### Q37. What is the spaceship operator `<=>`? 🟡
**A:** The three-way comparison operator (`<=>`, C++20) returns a comparison category: `strong_ordering`, `weak_ordering`, or `partial_ordering`. Defaulting `operator<=>` generates all six comparison operators. `auto operator<=>(const T&) const = default;` enables `==`, `!=`, `<`, `<=`, `>`, `>=` with one declaration. For custom types, implement `<=>` returning the appropriate category. Prefer defaulted comparisons for aggregate types.

### Q38. Explain `std::mdspan` (C++23). 🔴
**A:** `std::mdspan` is a multidimensional non-owning view over contiguous memory. It generalizes `span` to N dimensions with customizable layout (row-major, column-major, strided) and accessor policies. Perfect for matrix operations, image processing, and scientific computing — provides `mdspan<float, extents<3, 4>>` for a 3×4 matrix view. No data ownership, no allocation. Critical for interoperability between C++ and GPU/BLAS libraries.

### Q39. What are `std::generator` and lazy ranges (C++23)? 🔴
**A:** `std::generator<T>` (C++23) is a coroutine-based lazy range. A function using `co_yield` produces elements on-demand: `std::generator<int> fib() { int a=0, b=1; while(true) { co_yield a; tie(a,b) = {b, a+b}; } }`. Elements are computed one at a time — perfect for infinite sequences, file parsing, or tree traversal. Composable with views: `fib() | take(10) | transform(square)`.

### Q40. What are C++23's `std::print` and `std::println`? 🟢
**A:** `std::print` and `std::println` (C++23) bring `std::format` to output streams: `std::println("Hello {}", name);`. They replace `cout << "Hello " << name << "\n"`. Benefits: compile-time format string checking, no `endl` flushing confusion, locale-independent by default, faster than iostreams. They write to `stdout` by default but accept any `FILE*` or `ostream`. A small but welcome quality-of-life improvement.

---

## 3. Concurrency (15 Questions)

### Q41. What is a data race vs. a race condition? 🟡
**A:** Data race: two threads access the same memory location, at least one writes, no synchronization. This is undefined behavior in C++. Race condition: program correctness depends on scheduling order — a logic bug but not necessarily UB. Example: two threads increment a counter without synchronization is a data race; two threads checking-then-acting on a shared flag (TOCTOU) is a race condition. Mutexes prevent data races; careful design prevents race conditions.

### Q42. Explain `std::mutex`, `std::lock_guard`, and `std::scoped_lock`. 🟢
**A:** `std::mutex` provides mutual exclusion — only one thread can hold the lock. `lock_guard` is an RAII wrapper: acquires the lock in constructor, releases in destructor. `scoped_lock` (C++17) can lock multiple mutexes simultaneously using deadlock avoidance algorithms: `std::scoped_lock lk(mtx1, mtx2);`. Always use RAII wrappers, never raw `lock()`/`unlock()`. Prefer `scoped_lock` for multiple mutexes, `unique_lock` when you need to defer, try, or condition-wait.

### Q43. What is `std::atomic` and what memory orderings exist? 🔴
**A:** `std::atomic<T>` provides lock-free (for small T) atomic operations with specified memory orderings: `memory_order_relaxed` (no ordering), `acquire` (reads after this see prior writes), `release` (writes before this are visible to acquirers), `acq_rel` (both), `seq_cst` (total order, default — safest but slowest). On x86, acquire/release are free (strong memory model); on ARM, they emit barriers. Use `relaxed` for counters, `acquire/release` for producer-consumer, `seq_cst` when in doubt.

### Q44. How do you prevent deadlocks? 🟡
**A:** Four strategies: (1) Lock ordering — always acquire mutexes in a consistent global order; (2) `std::scoped_lock` — uses deadlock avoidance (try-and-back-off) for multiple mutexes; (3) `try_lock` with timeout — don't block forever; (4) Lock-free algorithms — eliminate locks entirely. Deadlock requires four conditions (mutual exclusion, hold-and-wait, no preemption, circular wait). Breaking any one prevents it. In practice, consistent lock ordering plus RAII wrappers prevent most deadlocks.

### Q45. Explain `std::condition_variable`. What is the spurious wakeup problem? 🟡
**A:** `condition_variable` allows a thread to sleep until a condition is met, atomically releasing a mutex while waiting. `cv.wait(lock, predicate)` loops until the predicate is true. Spurious wakeups: the OS may wake a thread without `notify` — so you *must* use the predicate form, not raw `wait()`. The pattern: lock mutex → check condition → if false, wait (atomically unlocks) → on wake, re-lock and re-check. Use `notify_one` for single consumer, `notify_all` for broadcast.

### Q46. What is a lock-free data structure? Give an example. 🔴
**A:** A lock-free data structure guarantees system-wide progress: at least one thread completes its operation in finite steps, even if other threads are suspended. Example: lock-free stack using `compare_exchange_weak`: push reads head, creates new node pointing to head, then CAS replaces head. If CAS fails (another thread modified head), retry. Challenges: ABA problem (use tagged pointers or hazard pointers), memory reclamation, complex correctness proofs. Use existing libraries (Folly, Boost.Lockfree) unless you're an expert.

### Q47. Explain `std::async`, `std::future`, and `std::promise`. 🟡
**A:** `std::promise<T>` + `std::future<T>` form a one-shot communication channel: one thread sets the value via `promise.set_value()`, another retrieves it via `future.get()` (blocking). `std::async` launches a task and returns a future: `auto f = std::async(std::launch::async, compute, args);`. The `launch::async` policy guarantees a new thread; `launch::deferred` runs lazily on `get()`. Caution: `std::async`'s returned future blocks in its destructor, making fire-and-forget impossible.

### Q48. What is `std::counting_semaphore` (C++20)? 🟡
**A:** `counting_semaphore<N>` is a lightweight synchronization primitive that maintains a count. `acquire()` decrements the count (blocking if zero); `release()` increments it. Use cases: limiting concurrency (connection pools, thread pools), producer-consumer with bounded buffers. `binary_semaphore` is `counting_semaphore<1>` — like a mutex but without ownership (any thread can release). Semaphores are more flexible than mutexes for certain patterns.

### Q49. Explain `std::latch` and `std::barrier` (C++20). 🟡
**A:** `latch` is a single-use countdown: threads call `count_down()`, one or more wait with `wait()` until the count reaches zero. Use for one-time initialization synchronization. `barrier` is reusable: `N` threads call `arrive_and_wait()`, all block until all N arrive, then a completion function runs and the barrier resets. Perfect for iterative parallel algorithms (e.g., simulation steps). Both are more efficient than implementing the same with mutexes + condition variables.

### Q50. How does `std::atomic_ref` (C++20) work? 🔴
**A:** `atomic_ref<T>` provides atomic operations on a non-atomic variable: `int x = 0; std::atomic_ref<int> ref(x);`. Useful when the same data is sometimes accessed atomically (in parallel sections) and sometimes not (in serial sections). Requirements: the referenced object must be properly aligned and must not be accessed non-atomically while any `atomic_ref` to it exists. Enables atomic operations without the overhead of `std::atomic` wrapper in the data structure.

### Q51. What is the memory model in C++? What does "happens-before" mean? 🔴
**A:** The C++ memory model defines how threads interact through memory. "Happens-before" is a partial order: if A happens-before B, B sees A's effects. Established by: sequencing (within a thread), synchronization (mutex lock/unlock, atomic acquire/release), thread creation/join. Without a happens-before relationship, accesses to the same location from different threads constitute a data race (UB). The model abstracts hardware differences (x86 TSO vs ARM weak ordering).

### Q52. What is false sharing and how do you prevent it? 🔴
**A:** False sharing occurs when threads modify different variables that share the same cache line (typically 64 bytes). Each write invalidates the entire line for other cores, causing excessive cache coherence traffic. Fix: pad or align data to cache-line boundaries. C++17 provides `std::hardware_destructive_interference_size` (typically 64). Use `alignas(64)` on per-thread data. This can cause 10-100x slowdowns in parallel counters and is a common pitfall in concurrent programming.

### Q53. Explain thread pools and `std::execution` (C++26). 🔵
**A:** A thread pool manages a fixed set of worker threads that pick tasks from a queue, amortizing thread creation cost. Implement with a task queue, condition variable, and worker loop. `std::execution` (P2300, expected C++26) provides a standardized sender/receiver model: senders represent async work, receivers consume results, and schedulers place work on execution resources (thread pools, GPU, etc.). This replaces ad-hoc async patterns with a composable, type-safe framework.

### Q54. What is a coroutine thread pool pattern? 🔵
**A:** Combine C++20 coroutines with a thread pool: `co_await pool.schedule()` suspends the coroutine and resumes it on a pool thread. The awaiter's `await_suspend` enqueues the coroutine handle into the pool's task queue. This gives lightweight cooperative multitasking (millions of coroutines) on a small thread pool (one thread per core). Libraries like `libunifex` and `exec` implement this pattern, enabling async I/O without callback spaghetti.

### Q55. How do you implement a lock-free MPSC queue? 🔵
**A:** Multi-Producer Single-Consumer queue: producers use `compare_exchange` to atomically append nodes to a shared tail pointer (linked list or ring buffer). The single consumer reads from head without contention. Key optimizations: cache-line separation between head and tail, avoiding false sharing. Dmitry Vyukov's MPSC queue uses an intrusive linked list with a sentinel node — widely used in actor systems. For MPMC (multi-consumer), the design is significantly more complex.

---

## 4. Systems (10 Questions)

### Q56. Explain the CPU memory hierarchy and cache behavior. 🟡
**A:** Modern CPUs: registers (~0.5ns) → L1 cache (32-64KB, ~1ns) → L2 (256KB-1MB, ~4ns) → L3 (shared, 8-64MB, ~12ns) → DRAM (~100ns). Caches use 64-byte lines, LRU replacement, and write-back policies. Spatial locality: accessing sequential memory prefetches adjacent lines. Temporal locality: recently accessed data stays cached. Cache misses dominate performance for data-intensive code. Optimizations: SoA over AoS, loop tiling, prefetch hints, avoiding pointer chasing.

### Q57. What is SIMD and how do you use it in C++? 🔴
**A:** SIMD (Single Instruction, Multiple Data) processes multiple values per instruction: SSE (128-bit, 4 floats), AVX2 (256-bit, 8 floats), AVX-512 (512-bit, 16 floats). Usage: (1) compiler auto-vectorization (use `-O3 -march=native`), (2) intrinsics (`_mm256_add_ps`), (3) `std::experimental::simd` (C++26 expected). Write SIMD-friendly code: contiguous aligned memory, no branches in hot loops, SoA layout. A 4-8x speedup is common for numerical code.

### Q58. What is the difference between static and dynamic linking? 🟢
**A:** Static linking copies library code into the executable at build time — larger binary, no runtime dependencies, potential for LTO. Dynamic linking (`.so`/`.dll`) loads libraries at runtime — smaller binary, shared across processes, updatable without rebuilding. Trade-offs: static is simpler to deploy (single binary); dynamic reduces memory via shared pages but risks version conflicts ("DLL hell"). Use static for deployment simplicity, dynamic for system libraries and plugin architectures.

### Q59. Explain virtual memory and page faults. 🟡
**A:** Virtual memory maps each process's address space to physical RAM via page tables (4KB pages, managed by the MMU/TLB). A page fault occurs when accessing a page not in physical RAM: the OS loads it from disk (major fault) or allocates a fresh page (minor fault). `mmap` maps files to virtual memory. Huge pages (2MB/1GB) reduce TLB pressure for large allocations. NUMA (Non-Uniform Memory Access) adds locality concerns — accessing remote node memory is 1.5-2x slower.

### Q60. What is an ABI? Why does it matter for C++ libraries? 🔴
**A:** ABI (Application Binary Interface) defines how compiled code interacts: calling conventions, name mangling, vtable layout, struct padding, exception handling. C++ has no standard ABI — each compiler (GCC, Clang, MSVC) and even compiler versions may differ. Breaking ABI changes require recompilation of all dependents. `extern "C"` uses the stable C ABI. Libraries must choose: ABI stability (restrict changes) or ABI freedom (require same-compiler builds). This is why pImpl and C interfaces are popular for shared libraries.

### Q61. What is memory-mapped I/O? When is it faster than `read()`/`write()`? 🔴
**A:** `mmap()` maps a file into virtual memory, making file access look like memory access. The OS handles paging transparently. Faster for: random access patterns (avoids `lseek`), read-only shared data (pages shared across processes), and large files. Slower for: sequential reads (kernel `read()` can readahead more efficiently), writes (dirty page tracking overhead). `madvise()` hints optimize access patterns. Watch for: bus errors on truncated files, coherency with `write()`.

### Q62. How does branch prediction work? What is branch misprediction cost? 🟡
**A:** Modern CPUs predict branch outcomes to keep the pipeline full. A correct prediction costs nothing; a misprediction flushes the pipeline, wasting 15-20 cycles. Predictors use history tables (PHT/TAGE) and are ~95% accurate for regular patterns. Costs in hot loops: replace branches with branchless code (`cmov`, arithmetic masks, `std::min`/`max`). Sorting data before conditional processing improves prediction. Profile with `perf stat` (branch-misses counter).

### Q63. Explain cache coherence protocols (MESI/MOESI). 🔴
**A:** MESI tracks cache line states: Modified (dirty, exclusive), Exclusive (clean, exclusive), Shared (clean, multiple), Invalid. When one core writes a Shared line, it sends invalidation to other cores, transitioning them to Invalid. MOESI adds Owned (dirty, shared — avoids writing back to memory). False sharing triggers excessive invalidations. Understanding MESI helps diagnose concurrent performance issues. The coherence traffic is what makes false sharing expensive and lock contention visible in hardware counters.

### Q64. What is NUMA and how does it affect performance? 🔴
**A:** NUMA (Non-Uniform Memory Access): multi-socket systems where each CPU has local memory. Local access: ~100ns; remote access: ~150-200ns (1.5-2x penalty). Applications: (1) use `numactl --localalloc` to allocate on local node; (2) pin threads to cores near their data; (3) first-touch policy — the OS allocates on the node that first accesses the page. NUMA-unaware programs suffer from remote memory access penalties and interconnect contention. Tools: `numastat`, `lstopo`, `hwloc`.

### Q65. What is Transparent Huge Pages (THP) and when does it help/hurt? 🔴
**A:** THP automatically promotes 4KB pages to 2MB huge pages, reducing TLB misses. Helps: large contiguous allocations (databases, scientific computing). Hurts: allocation latency (compaction), memory waste (internal fragmentation), unpredictable latencies (compaction stalls). Redis and some databases disable THP. Alternative: explicit `madvise(MADV_HUGEPAGE)` for targeted use. Monitor with `/proc/vmstat` (thp_collapse, thp_fault). For CUDA, GPU memory uses its own page management.

---

## 5. CUDA Core (20 Questions)

### Q66. Explain the CUDA thread hierarchy: thread, warp, block, grid. 🟢
**A:** Thread: individual execution unit. Warp: 32 threads executing in lockstep (SIMT). Block: up to 1024 threads sharing shared memory, synchronizable via `__syncthreads()`. Grid: all blocks in a kernel launch. Dimensions: `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (each up to 3D). The global thread ID is computed as `blockIdx.x * blockDim.x + threadIdx.x`. Blocks are distributed across SMs; warps are the scheduling unit. Understanding this hierarchy is essential for writing efficient kernels.

### Q67. What is memory coalescing and why does it matter? 🟡
**A:** Coalescing: when threads in a warp access consecutive memory addresses, the GPU combines them into minimal memory transactions (one 128-byte transaction instead of 32 separate ones). Un-coalesced access wastes bandwidth — e.g., strided access with stride 32 can be 10-20x slower. Rules: thread k should access address `base + k * sizeof(element)`. AoS (Array of Structures) breaks coalescing; SoA (Structure of Arrays) enables it. Check with Nsight Compute's memory throughput metrics.

### Q68. Explain CUDA shared memory and its use in tiled algorithms. 🟡
**A:** Shared memory is fast on-chip SRAM (100+ TB/s) shared within a block. Use for tiled algorithms: load a tile from global memory to shared memory, `__syncthreads()`, compute using shared memory, repeat. This reduces global memory accesses by the tile factor. Bank conflicts: shared memory has 32 banks; if multiple threads in a warp access the same bank (different addresses), accesses serialize. Avoid by padding or index remapping. Configurable: 48KB/block (default) or up to 228KB on Hopper.

### Q69. What is warp divergence? How does it affect performance? 🟡
**A:** Warp divergence occurs when threads in a warp take different paths at a branch (`if/else`). Since a warp executes one instruction at a time, both paths are executed serially — threads not on the active path are masked off. Cost: up to 2x slowdown for simple if/else, worse for complex nesting. Minimize by: ensuring threads in a warp take the same path, restructuring data to avoid divergence, or moving divergent work to separate kernels.

### Q70. What is occupancy and how do you maximize it? 🟡
**A:** Occupancy = active warps / maximum warps per SM. Limited by: registers per thread (register pressure), shared memory per block, and block size. Higher occupancy hides memory latency via warp switching. Use `cudaOccupancyMaxPotentialBlockSize()` to find optimal block size. Check with Nsight Compute's occupancy calculator. However, max occupancy ≠ max performance — sometimes fewer threads with more registers or shared memory per thread wins. Profile to find the sweet spot.

### Q71. Explain the CUDA memory model: global, shared, local, constant, texture. 🟢
**A:** Global: device DRAM (HBM), large (80GB+), high latency (400+ cycles), accessible by all threads. Shared: on-chip SRAM per block, fast (20 cycles), limited (up to 228KB). Local: per-thread, actually in global memory (register spill). Constant: 64KB, cached, broadcast to warp. Texture: cached for spatial locality, hardware interpolation. Registers: fastest, per-thread, limited (255 per thread on modern GPUs). Performance hierarchy: registers > shared >> L2 > global.

### Q72. What is unified memory? When should you use it? 🟡
**A:** Unified Memory (`cudaMallocManaged`) provides a single address space accessible from both CPU and GPU. The driver automatically migrates pages between host and device on access (page faults). Simplifies programming but can cause: migration overhead, page fault stalls, thrashing if both CPU and GPU access the same pages. Use for: prototyping, irregular access patterns, oversubscribed GPU memory. For peak performance, use explicit `cudaMemcpy` with prefetching (`cudaMemPrefetchAsync`).

### Q73. How do you launch a CUDA kernel? Explain the `<<<>>>` syntax. 🟢
**A:** `kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);`. `gridDim`: number of blocks (dim3). `blockDim`: threads per block (dim3). `sharedMemBytes`: dynamic shared memory (default 0). `stream`: CUDA stream (default 0 = default stream). Example: `matMul<<<dim3(N/16, N/16), dim3(16, 16)>>>(A, B, C, N);`. After launch, check for errors with `cudaGetLastError()` and `cudaDeviceSynchronize()`. Kernel launches are asynchronous — they return immediately to the CPU.

### Q74. Explain `cudaMemcpy` and the different transfer types. 🟢
**A:** `cudaMemcpy(dst, src, size, kind)` transfers data. Kinds: `cudaMemcpyHostToDevice` (CPU→GPU), `DeviceToHost` (GPU→CPU), `DeviceToDevice` (GPU→GPU), `HostToHost`. Synchronous by default (blocks CPU). For async: use `cudaMemcpyAsync` with pinned memory (`cudaMallocHost`) and a non-default stream. Pinned memory avoids an extra copy through a staging buffer. Bandwidth: PCIe 5.0 ≈ 64 GB/s (vs HBM3 ≈ 3+ TB/s), so minimize transfers.

### Q75. What is warp shuffle and when should you use it? 🔴
**A:** Warp shuffles (`__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`) exchange data between threads within a warp without shared memory. Use for: warp-level reductions, scan, broadcast. Faster than shared memory (no bank conflicts, no `__syncthreads` needed). Example reduction: `val += __shfl_down_sync(0xFFFFFFFF, val, 16); val += __shfl_down_sync(0xFFFFFFFF, val, 8);` etc. The `mask` parameter specifies participating threads. Available since Kepler (compute 3.0+).

### Q76. Explain CUDA error handling best practices. 🟢
**A:** Every CUDA API call returns `cudaError_t` — always check it: `cudaError_t err = cudaMalloc(...); if(err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); }`. Kernel launches don't return errors directly — use `cudaGetLastError()` after launch and `cudaDeviceSynchronize()` to catch async errors. Common macro: `#define CUDA_CHECK(call) do { cudaError_t e = call; if(e) { ... } } while(0)`. In production, log errors and handle gracefully; don't just `exit()`.

### Q77. What is the difference between `__global__`, `__device__`, and `__host__`? 🟢
**A:** `__global__`: kernel function — called from host, runs on GPU, returns void. `__device__`: device function — called from GPU, runs on GPU. `__host__`: host function — called from and runs on CPU (default). `__host__ __device__`: compiles for both — useful for math helpers. Restrictions: `__global__` functions can't be called from other `__global__` functions (except with dynamic parallelism). `__device__` functions can access device memory only.

### Q78. Explain cooperative groups in CUDA. 🔴
**A:** Cooperative Groups (CUDA 9+) generalize thread synchronization beyond `__syncthreads()`. Hierarchy: `thread_block`, `thread_block_tile<32>` (warp), `grid_group` (entire grid), `multi_grid_group` (multi-GPU). Methods: `sync()`, `size()`, `thread_rank()`. Warp-level tiles enable explicit sub-warp programming without `__shfl`. Grid-wide sync (`grid_group::sync()`) requires cooperative launch. Enables algorithms needing global synchronization without kernel relaunches: producer-consumer, persistent kernels.

### Q79. What is the L2 cache residency control in Ampere+? 🔴
**A:** Ampere introduced `cudaAccessPolicyWindow` to control L2 cache residency. You can mark memory regions as persistent (stay in L2) or streaming (evict quickly). `cudaStreamSetAccessPolicy(stream, &policy)`. Use for: keeping frequently accessed data (lookup tables, embeddings) in L2 while allowing bulk data to stream through. Combined with `cudaMemPoolSetAttribute`, fine-grained control over the 40-50MB L2 cache can significantly improve hit rates for working sets that fit.

### Q80. Explain dynamic parallelism in CUDA. 🔴
**A:** Dynamic parallelism (Kepler+, compute 3.5+) allows GPU kernels to launch other kernels. The child kernel runs on the same device; the parent can synchronize with `cudaDeviceSynchronize()` from the device. Use for: recursive algorithms (quicksort, octree traversal), adaptive mesh refinement. Limitations: higher launch overhead than CPU-side launch, limited nesting depth, separate memory visibility rules. Often, restructuring the algorithm to avoid dynamic parallelism gives better performance.

### Q81. How does `__syncthreads()` work and what are the pitfalls? 🟡
**A:** `__syncthreads()` is a block-wide barrier: all threads in the block must reach it before any proceed. Pitfalls: (1) if placed inside a conditional where some threads don't reach it, behavior is undefined (deadlock or corruption); (2) it only synchronizes within a block, not across blocks; (3) it doesn't imply memory visibility for global memory (use `__threadfence()` for that). Since Volta, warps are independently scheduled — warp-level sync requires explicit `__syncwarp()`.

### Q82. What is register pressure and how do you manage it? 🔴
**A:** Each SM has a fixed register file (65536 on most architectures). More registers per thread = fewer concurrent threads = lower occupancy. Register pressure occurs when a kernel uses too many registers. Manage with: (1) `-maxrregcount=N` compiler flag; (2) `__launch_bounds__(maxThreads, minBlocks)` attribute; (3) simplify kernel logic; (4) use shared memory for intermediate values. Spilled registers go to local memory (global memory speed). Profile with Nsight Compute's register allocation report.

### Q83. Explain `cudaStream_t` and how streams enable concurrency. 🟡
**A:** A CUDA stream is a sequence of operations (kernels, memcpy) executed in order. Operations in different streams can execute concurrently. Default stream (0) synchronizes with all other streams (unless `--default-stream per-thread`). Use multiple streams to overlap: kernel execution with memcpy, multiple kernels. Pattern: `cudaMemcpyAsync(d, h, n, H2D, stream1); kernel<<<g,b,0,stream1>>>(d);`. Event-based dependencies between streams: `cudaEventRecord`, `cudaStreamWaitEvent`.

### Q84. What is the PTX intermediate representation? 🔴
**A:** PTX (Parallel Thread Execution) is NVIDIA's virtual ISA — a stable intermediate representation between CUDA C++ and hardware-specific machine code (SASS). `nvcc` compiles CUDA to PTX, then the driver's JIT compiler translates PTX to SASS for the target GPU. Including PTX in fatbinaries enables forward compatibility — code compiled today can run on future GPUs. You can write inline PTX with `asm()` for fine-grained control or inspect PTX with `cuobjdump`.

### Q85. What is the `__restrict__` keyword in CUDA? 🟡
**A:** `__restrict__` tells the compiler that pointers don't alias (don't point to overlapping memory). This enables aggressive optimizations: the compiler can reorder loads/stores, cache values in registers, and vectorize more aggressively. Without it, the compiler must assume any store through one pointer might affect reads through another. In CUDA, always use `__restrict__` on kernel pointer parameters unless they genuinely alias. Can provide 10-30% speedup for memory-bound kernels.

---

## 6. CUDA Advanced (10 Questions)

### Q86. Explain CUDA streams and events for multi-stream pipelines. 🔴
**A:** Build a pipeline: stream1 transfers batch N while stream2 computes batch N-1. Events synchronize across streams: `cudaEventRecord(event, stream1); cudaStreamWaitEvent(stream2, event);`. Use `cudaEventElapsedTime` for profiling. Best practices: (1) interleave operations across streams; (2) use pinned memory for async copies; (3) use `CUDA_LAUNCH_BLOCKING=1` for debugging. Multi-stream pipelines hide PCIe transfer latency and can improve throughput by 2-3x.

### Q87. How do you profile and optimize a CUDA kernel with Nsight Compute? 🔴
**A:** (1) Profile: `ncu --set full -o report kernel.exe`. (2) Check roofline: is the kernel compute-bound or memory-bound? (3) Key metrics: `sm__throughput.avg.pct_of_peak_sustained`, `dram__throughput.avg.pct_of_peak_sustained`. (4) Memory: check coalescing efficiency, L2 hit rate, shared memory bank conflicts. (5) Compute: check warp execution efficiency, instruction mix. (6) Occupancy: actual vs theoretical. Iterate: fix bottleneck → re-profile → fix next bottleneck.

### Q88. Explain multi-GPU programming with NCCL. 🔵
**A:** NCCL (NVIDIA Collective Communications Library) provides optimized multi-GPU collectives: AllReduce, Broadcast, AllGather, ReduceScatter. It uses NVLink/NVSwitch for intra-node and RDMA for inter-node. Pattern: (1) create communicator `ncclCommInitRank`; (2) launch collectives on per-GPU streams; (3) synchronize. NCCL handles topology detection and routing. Used by PyTorch DDP, DeepSpeed, Megatron. For data parallelism: AllReduce gradients; for tensor parallelism: AllGather/ReduceScatter activations.

### Q89. What are Tensor Cores and how do you use them? 🔴
**A:** Tensor Cores perform matrix multiply-accumulate on small matrices (e.g., 16×16×16) in a single operation: D = A×B + C. Supported types: FP16, BF16, TF32, FP8 (Hopper), INT8, INT4. Access via: (1) `wmma` API (warp-level), (2) `mma.sync` PTX instruction, (3) cuBLAS/cuDNN (easiest). Requirements: matrix dimensions must be multiples of 16, specific memory layouts. Performance: H100 delivers 989 TFLOPs FP16 with Tensor Cores vs 67 TFLOPs FP32 without — a 15x difference.

### Q90. Explain mixed precision training and the role of loss scaling. 🔴
**A:** Mixed precision uses FP16/BF16 for compute and FP32 for accumulation and master weights. Benefits: 2x memory savings, Tensor Core utilization (8-16x throughput). Loss scaling: FP16 has a small dynamic range (6×10⁻⁸ to 65504). Small gradients (e.g., 10⁻⁷) underflow to zero. Loss scaling multiplies the loss by a large factor (e.g., 1024) before backward, scaling gradients into FP16 range, then divides after. Dynamic loss scaling adjusts the factor automatically. BF16 has the same range as FP32, so loss scaling is often unnecessary.

### Q91. What is MIG (Multi-Instance GPU)? 🔴
**A:** MIG (A100+) partitions a single GPU into up to 7 isolated instances, each with dedicated SMs, memory, and cache. Each instance acts as an independent GPU with QoS guarantees. Use for: multi-tenant cloud inference, CI/CD, development. Partition profiles: 1g.5gb, 2g.10gb, 3g.20gb, 4g.20gb, 7g.40gb (A100). Trade-off: each instance has proportionally fewer resources. MIG instances can't communicate via shared memory — they're truly isolated. Manage with `nvidia-smi mig`.

### Q92. Explain CUDA Graphs and their benefits. 🔴
**A:** CUDA Graphs capture a sequence of operations (kernels, memcpy) as a graph, then replay it with minimal CPU overhead. Benefits: (1) reduce launch overhead (10-100µs per kernel → near zero); (2) enable driver-level optimization of the entire workflow; (3) critical for inference where the same computation repeats. API: capture with `cudaStreamBeginCapture` / `cudaStreamEndCapture`, execute with `cudaGraphLaunch`. Update parameters with `cudaGraphExecUpdate` to avoid recapture. Ideal for small, repeated kernels.

### Q93. What is FP8 and the Transformer Engine? 🔵
**A:** FP8 (Hopper+) provides two formats: E4M3 (range ±240, 4-bit exponent) for forward pass and E5M2 (range ±57344, 5-bit exponent) for backward pass. The Transformer Engine (TE) automatically manages FP8 quantization with per-tensor scaling factors, choosing FP8 vs FP16 per layer based on numerical sensitivity. Benefits: 2x speedup over FP16, 4x memory reduction. TE integrates with PyTorch/JAX — users just wrap layers with TE modules. Hopper Tensor Cores deliver 1979 TFLOPs in FP8.

### Q94. Explain thread block clusters (Hopper). 🔵
**A:** Thread block clusters (compute 9.0+) group up to 16 blocks into a cluster, enabling distributed shared memory (DSMEM) — shared memory of one block is directly addressable by other blocks in the cluster. This enables larger tile sizes without global memory intermediation. Use `__cluster_dims__` or cooperative launch. Clusters are co-scheduled on the same GPC, ensuring low-latency inter-block communication. Key for large GEMM tiles and attention computations where shared memory per block is insufficient.

### Q95. How do you optimize memory transfers between CPU and GPU? 🟡
**A:** (1) Use pinned (page-locked) memory: `cudaMallocHost` enables DMA, 2-3x faster than pageable. (2) Overlap transfers with computation using streams. (3) Batch small transfers into large ones. (4) Use `cudaMemcpyAsync` with non-default streams. (5) Consider unified memory with `cudaMemPrefetchAsync` for complex access patterns. (6) Use GPUDirect RDMA for network-to-GPU (bypass CPU). (7) Minimize transfer frequency — keep data on GPU between kernel calls. PCIe is the bottleneck: 64 GB/s vs 3+ TB/s HBM.

---

## 7. Design & Architecture (10 Questions)

### Q96. Describe a system design for a real-time image classification service. 🔵
**A:** Architecture: load balancer → API gateway → inference workers (each with a GPU) → model store (S3/GCS). Pipeline: receive image → preprocess (resize, normalize on CPU) → batch requests (dynamic batching, wait up to 5ms) → GPU inference (TensorRT optimized, FP16) → postprocess → return results. Scaling: autoscale workers based on GPU utilization and queue depth. Latency budget: <50ms p99. Key decisions: batch size trade-off (throughput vs latency), model versioning, A/B testing, fallback to CPU, monitoring with Prometheus/Grafana.

### Q97. How would you optimize a CUDA application that is memory-bound? 🔴
**A:** (1) Check coalescing — ensure contiguous access patterns (SoA not AoS). (2) Use shared memory tiling to amplify data reuse. (3) Increase arithmetic intensity — compute more per byte loaded. (4) Use L2 cache residency control (Ampere+). (5) Compress data (FP16/INT8 instead of FP32). (6) Merge kernels to reduce memory round-trips. (7) Use vectorized loads (`float4` instead of `float`). Profile with Nsight Compute roofline — if below the memory roof, focus on bandwidth utilization.

### Q98. Design a producer-consumer system using lock-free queues. 🔴
**A:** Ring buffer with atomic head/tail indices. Producer: compute next tail = (tail+1) % capacity; if next == head (full), backoff; write data[tail]; release-store tail = next. Consumer: if head == tail (empty), backoff; read data[head]; release-store head = (head+1) % capacity. For MPMC: use per-element sequence counters. Memory ordering: release on writes, acquire on reads. Avoid false sharing: pad head and tail to separate cache lines. Test with ThreadSanitizer.

### Q99. Explain the CRTP (Curiously Recurring Template Pattern) and static polymorphism. 🔴
**A:** CRTP: `class Derived : public Base<Derived>`. The base class calls derived methods without virtual dispatch: `static_cast<Derived*>(this)->impl()`. Benefits: zero-overhead polymorphism (resolved at compile time), enables mixin functionality. Used in: Eigen (expression templates), STL (iterator CRTP in ranges). Trade-off: no runtime polymorphism (can't store different types in a container without type erasure). Also enables "static interface" enforcement. Modern alternative: C++20 Concepts.

### Q100. How do you design a high-performance matrix library? 🔵
**A:** Key techniques: (1) Expression templates to eliminate temporaries: `A + B * C` generates a single fused loop. (2) SIMD intrinsics for inner loops. (3) Cache-oblivious or tiled algorithms for GEMM. (4) Memory pools for allocation amortization. (5) BLAS/LAPACK backend dispatch for large matrices. (6) GPU offload via cuBLAS for large problems. (7) Compile-time size specialization for small matrices. (8) Template metaprogramming for type promotion rules. Reference implementations: Eigen, Blaze, Armadillo.

### Q101. When would you choose a B-tree vs a hash table vs a skip list? 🟡
**A:** Hash table: O(1) average lookup, best for point queries, poor cache for iteration. B-tree: O(log N) with high fanout, excellent cache behavior, sorted iteration, range queries, used in databases/filesystems. Skip list: O(log N) probabilistic, lock-free concurrent friendly (no tree rotations), simpler to implement concurrently. Choose hash for pure lookup; B-tree for sorted/range access; skip list for concurrent sorted maps (like ConcurrentSkipListMap in Java, or LevelDB's memtable).

### Q102. How do you design a plugin system in C++? 🔴
**A:** Approach: (1) Define a pure virtual interface (ABC) as the plugin contract. (2) Plugins compile to shared libraries (`.so`/`.dll`). (3) Host uses `dlopen`/`LoadLibrary` to load plugins at runtime. (4) `extern "C"` factory function creates plugin instances (avoids ABI issues). (5) Use `unique_ptr` with custom deleter for lifetime management. (6) Version the interface — check compatibility at load time. Alternatives: COM (Windows), type erasure with `std::function`, or embedding a scripting language (Lua, Python).

### Q103. Explain the Entity-Component-System (ECS) pattern. 🔴
**A:** ECS separates identity (Entity = ID), data (Components = plain structs), and behavior (Systems = functions). Entities are just integers; components are stored in dense arrays by type (SoA layout); systems iterate over components matching a query. Benefits: cache-friendly iteration (components are contiguous), composition over inheritance, easy serialization. Used in game engines (Unity DOTS, Flecs, EnTT). The SoA layout is identical to GPU-friendly data layout, making ECS natural for GPU-accelerated simulations.

### Q104. Design a task scheduler for heterogeneous CPU+GPU workloads. 🔵
**A:** Architecture: task graph DAG with CPU and GPU task nodes. Scheduler: topological order, ready queue per device. (1) Annotate each task with device affinity (CPU/GPU/either). (2) Estimate cost to choose device when ambiguous. (3) CPU tasks → thread pool; GPU tasks → CUDA stream pool. (4) Dependencies trigger downstream tasks. (5) Memory manager handles CPU↔GPU transfers as implicit tasks. (6) Overlap CPU and GPU tasks when independent. Reference: CUDA Task Graphs, Intel TBB Flow Graph, StarPU. Key challenge: minimizing data movement.

### Q105. How would you reduce inference latency for an LLM serving system? 🔵
**A:** (1) KV-cache optimization: paged attention (vLLM), chunked prefill. (2) Model parallelism: tensor parallel across GPUs for large models. (3) Quantization: GPTQ/AWQ to INT4, or FP8 on Hopper. (4) Continuous batching: add new requests to running batch without waiting. (5) Speculative decoding: draft small model → verify with large model. (6) FlashAttention: fused, IO-aware attention. (7) CUDA Graphs for decode phase. (8) Hardware: NVLink for tensor parallel communication. Target: <100ms TTFT, >100 tokens/sec/user.

---

## Quick Reference: Difficulty Distribution

| Level  | Count | Topics                                        |
|--------|-------|-----------------------------------------------|
| 🟢 Junior | 20  | Basics, syntax, simple memory, kernel launch  |
| 🟡 Mid    | 35  | Design trade-offs, concurrency, optimization  |
| 🔴 Senior | 35  | Lock-free, ABI, advanced CUDA, architecture   |
| 🔵 Staff  | 15  | System design, distributed, cutting-edge      |

---

*Appendix A — C++ & CUDA Interview Mega-Guide — Part of the CPP-CUDA-Mastery series*
