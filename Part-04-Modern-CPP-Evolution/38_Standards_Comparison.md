# Chapter 38 — C++ Standards Comparison & Migration Guide

```yaml
tags:
  - cpp-standards
  - migration
  - modernization
  - cpp11-to-cpp26
  - compiler-support
  - best-practices
```

---

## Theory

C++ evolves on a three-year release cadence. Each standard introduces language features, library additions, and concurrency primitives that make code safer, more expressive, and faster. However, upgrading a production codebase is never a simple compiler-flag change. ABI compatibility, deprecated features, subtle behavior shifts, and toolchain readiness all demand a disciplined migration strategy.

This chapter provides a single reference for **what changed**, **when it landed**, **which compilers support it**, and **how to migrate** an existing codebase step by step.

---

## What — Key Concepts

- **Standard revision**: A numbered ISO release (C++11, C++14, C++17, C++20, C++23, C++26).
- **Feature test macro**: Preprocessor symbols (`__cpp_concepts`, `__cpp_lib_ranges`) that let code detect feature availability at compile time.
- **ABI break**: A change that makes object files compiled under different standards binary-incompatible.
- **Deprecation → Removal cycle**: Features deprecated in one standard are removed one or two revisions later (e.g., `auto_ptr` deprecated C++11, removed C++17).

## Why — Motivation

| Pain Point | How Standards Address It |
|---|---|
| Manual memory bugs | Smart pointers (C++11), `make_unique` (C++14) |
| Boilerplate templates | Concepts (C++20), abbreviated templates |
| Callback hell | Coroutines (C++20), `std::expected` (C++23) |
| Unsafe concurrency | `std::jthread`, latches, barriers (C++20) |
| Verbose transforms | Ranges and views (C++20/C++23) |

## How — Migration Workflow

1. **Audit** — inventory compiler versions, third-party library requirements, and CI matrix.
2. **Enable** — set `-std=c++20` and fix compilation errors.
3. **Modernize** — apply clang-tidy modernize checks module by module.
4. **Test** — full test suite; watch for behavior changes, not just compilation.
5. **Stabilize** — freeze the standard flag and update coding guidelines.

---

## Feature Comparison Table

| Category | C++11 | C++14 | C++17 | C++20 | C++23 | C++26 |
|---|---|---|---|---|---|---|
| **Language Core** | `auto`, lambdas, rvalue refs, `nullptr`, `constexpr` | Generic lambdas, relaxed `constexpr`, variable templates | Structured bindings, `if constexpr`, fold expressions | Concepts, coroutines, modules, `<=>` | `if consteval`, deducing `this` | Reflection, contracts |
| **Library** | `<thread>`, `<chrono>`, smart ptrs | `make_unique`, `exchange` | `optional`, `variant`, `filesystem`, `string_view` | `<ranges>`, `<format>`, `<span>` | `expected`, `flat_map`, `print` | `execution`, `inplace_vector` |
| **Concurrency** | `thread`, `mutex`, `atomic`, `future` | Minor fixes | `shared_mutex`, parallel algorithms | `jthread`, `latch`, `barrier`, semaphore | Minor fixes | Hazard pointers, RCU |
| **Templates** | Variadic templates, `decltype` | Variable templates, generic lambdas | CTAD, `template<auto>` | Concepts, `requires`, abbreviated templates | Multidimensional `operator[]` | Universal template params |

---

## Compiler Support Matrix

| Standard | GCC | Clang | MSVC |
|---|---|---|---|
| C++11 | 4.8+ | 3.3+ | 2015 Update 3 |
| C++14 | 5.0+ | 3.4+ | 2017 (15.0) |
| C++17 | 7.0+ | 5.0+ | 2017 (15.7) |
| C++20 | 10+ (full ~12) | 10+ (full ~16) | 2019 (16.10+) |
| C++23 | 13+ (partial) | 17+ (partial) | 2022 (17.6+) |
| C++26 | 15+ (experimental) | 19+ (experimental) | 2022 (preview) |

> **Tip:** `__cplusplus` values: 201103L, 201402L, 201703L, 202002L, 202302L.

---

## Mermaid Diagram — C++ Evolution Timeline

```mermaid
gantt
    title C++ Standards Evolution Timeline
    dateFormat YYYY
    axisFormat %Y
    section Language Core
    auto, lambdas, rvalue refs           :done, 2011, 2014
    Generic lambdas, relaxed constexpr   :done, 2014, 2017
    Structured bindings, if constexpr    :done, 2017, 2020
    Concepts, coroutines, modules        :done, 2020, 2023
    Deducing this, if consteval          :done, 2023, 2026
    Reflection, contracts                :active, 2026, 2029
    section Library
    thread, chrono, smart ptrs           :done, 2011, 2014
    make_unique, exchange                :done, 2014, 2017
    optional, variant, filesystem        :done, 2017, 2020
    ranges, format, span                 :done, 2020, 2023
    expected, flat_map, print            :done, 2023, 2026
    section Concurrency
    std::thread, mutex, atomic           :done, 2011, 2014
    shared_mutex, parallel algos         :done, 2017, 2020
    jthread, latch, barrier              :done, 2020, 2023
    Hazard pointers, RCU                 :active, 2026, 2029
```

---

## Recommended Modern C++ Subset

### Always Use
- `auto` for local variables where type is obvious
- Smart pointers — never raw `new`/`delete`
- Range-based `for`, structured bindings
- `std::string_view` for read-only string parameters
- `constexpr` wherever possible
- `std::optional` for nullable values; `std::format` over `printf`/iostream formatting
- Concepts to constrain templates

### Use With Caution
- **Coroutines** — powerful but complex; ensure team familiarity
- **Modules** — build-system support still maturing
- **`std::regex`** — known perf issues; consider RE2 or CTRE
- **`std::any`** — prefer `std::variant` when types are known

### Avoid / Replace

| Legacy Pattern | Modern Replacement |
|---|---|
| `auto_ptr` | `unique_ptr` |
| Raw `new`/`delete` | `make_unique` / `make_shared` |
| C-style casts | `static_cast` / `dynamic_cast` |
| `#define` constants | `constexpr` / `inline constexpr` |
| `void*` generics | Templates or `std::variant` |
| Output parameters | `std::optional` or `std::expected` |
| `NULL` / `typedef` | `nullptr` / `using` alias |

---

## Code Examples — Before / After Modernization

### Example 1: Resource Management (C++03 → C++17)

This is the old C++03 approach to resource management using raw `new`/`delete` and manual `fclose` calls. It is fragile because every early return or exception path requires an explicit `delete`, making memory leaks easy to introduce and hard to spot in larger functions.

```cpp
// BEFORE — C++03: manual memory, leak-prone
class FileHandle {
    FILE* f_;
public:
    FileHandle(const char* path) : f_(fopen(path, "r")) {}
    ~FileHandle() { if (f_) fclose(f_); }
    bool is_open() const { return f_ != NULL; }
};
void process() {
    FileHandle* fh = new FileHandle("data.txt");
    if (!fh->is_open()) { delete fh; return; }
    delete fh;  // easy to leak on early return
}
```

The modern C++17 version replaces raw pointers with `std::unique_ptr` and a custom deleter for automatic cleanup via RAII. The factory function returns `std::optional<FileHandle>`, eliminating the need for manual `delete` — resources are freed automatically when they go out of scope, making leaks impossible regardless of control flow.

```cpp
// AFTER — C++17: RAII, optional, no leaks
#include <memory>
#include <optional>
#include <cstdio>
class FileHandle {
    struct Deleter { void operator()(FILE* f) const { if (f) fclose(f); } };
    std::unique_ptr<FILE, Deleter> f_;
public:
    explicit FileHandle(const char* path) : f_(fopen(path, "r")) {}
    bool is_open() const { return f_ != nullptr; }
    FILE* get() const { return f_.get(); }
};
std::optional<FileHandle> open_file(const char* path) {
    FileHandle fh(path);
    if (!fh.is_open()) return std::nullopt;
    return fh;
}
void process() {
    auto fh = open_file("data.txt");
    if (!fh) return;  // no manual delete, no leaks
}
```

### Example 2: Generic Algorithm (C++11 SFINAE → C++20 Concepts)

The old C++11 approach uses SFINAE (`std::enable_if_t`) to constrain templates. While functional, SFINAE produces notoriously unreadable error messages and clutters the function signature with a dummy template parameter that serves no purpose except type checking.

```cpp
// BEFORE — C++11 SFINAE
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
T double_value(T x) { return x * 2; }
```

C++20 Concepts replace the SFINAE boilerplate with a clean, readable constraint (`std::integral`) placed directly on the template parameter. Error messages now point to the unsatisfied concept by name, making debugging dramatically easier.

```cpp
// AFTER — C++20 Concepts
#include <concepts>
template <std::integral T>
T double_value(T x) { return x * 2; }
```

### Example 3: Error Handling (C++11 → C++23)

The old C++11 pattern uses output parameters and a `bool` return value to signal success or failure. This approach is error-prone because callers can easily forget to check the return value, and the function signature doesn't clearly communicate that `result` and `err` are outputs.

```cpp
// BEFORE — output parameters
bool safe_divide(double a, double b, double& result, std::string& err) {
    if (b == 0.0) { err = "division by zero"; return false; }
    result = a / b;
    return true;
}
```

C++23's `std::expected<T, E>` replaces the awkward output-parameter pattern with a single return type that carries either the success value or an error. This makes the function signature self-documenting and enables monadic chaining with `.transform()` — no need to check booleans or inspect separate error variables.

```cpp
// AFTER — std::expected (C++23)
#include <expected>
#include <string>
std::expected<double, std::string> safe_divide(double a, double b) {
    if (b == 0.0) return std::unexpected("division by zero");
    return a / b;
}
// Usage: auto r = safe_divide(10, 3).transform([](double v){ return v * 2; });
```

### Example 4: String Formatting (C++11 → C++23)

The old approach uses C's `snprintf` with a fixed-size buffer, format specifiers (`%d`, `%.2f`, `%s`), and no type safety — passing the wrong type silently produces garbage output or undefined behavior.

```cpp
// BEFORE
char buf[256];
snprintf(buf, sizeof(buf), "ID=%d Score=%.2f Name=%s", id, score, name);
```

C++23's `std::format` and `std::println` provide type-safe, Python-style formatting with `{}` placeholders. Unlike `snprintf`, they catch type mismatches at compile time, handle `std::string` natively, and eliminate buffer-overflow risks.

```cpp
// AFTER — std::format / std::print (C++23)
#include <format>
#include <print>
auto s = std::format("ID={} Score={:.2f} Name={}", id, score, name);
std::println("ID={} Score={:.2f} Name={}", id, score, name);
```

---

## Migration Strategy — C++11 to C++20 Step by Step

### Phase 1: Compiler & Build System (Week 1–2)
1. Upgrade to GCC 12+ / Clang 16+ / MSVC 2022.
2. Set `CMAKE_CXX_STANDARD 20`. Fix new compilation errors.
3. Enable `-Wall -Wextra -Wpedantic`.

### Phase 2: Automated Modernization (Week 3–4)

This command runs clang-tidy with all `modernize-*` checks enabled and the `-fix` flag to automatically rewrite source files in place. It handles bulk conversions like `NULL` → `nullptr`, adding `override`, converting loops to range-for, and replacing raw `new` with `make_unique` — covering 60–70% of a typical migration.

```bash
clang-tidy -checks='modernize-*' -fix src/*.cpp -- -std=c++20
```
Key checks: `modernize-use-auto`, `modernize-use-nullptr`, `modernize-use-override`, `modernize-use-using`, `modernize-loop-convert`, `modernize-make-unique`.

### Phase 3: Manual Refactoring (Week 5–8)

| Area | Action |
|---|---|
| SFINAE templates | Replace with Concepts |
| `std::bind` | Replace with lambdas |
| Output parameters | Return `std::optional` or structured types |
| `boost::optional/variant` | Use `std::optional` / `std::variant` |
| String formatting | Migrate to `std::format` |

### Phase 4: Validation (Week 9–10)
1. Run CI with sanitizers (ASan, UBSan, TSan).
2. Benchmark hot paths for regressions.
3. Update coding guidelines; reject older standard flags in CI.

---

## Codebase Modernization Checklist

- [ ] Replace `NULL` with `nullptr`
- [ ] Replace `typedef` with `using`
- [ ] Add `override` to all virtual overrides
- [ ] Convert C-style casts to named casts
- [ ] Replace raw `new`/`delete` with smart pointers
- [ ] Convert `#define` constants to `constexpr`
- [ ] Use structured bindings for map iteration / tuple returns
- [ ] Replace `std::bind` with lambdas
- [ ] Add `[[nodiscard]]` where ignoring return value is a bug
- [ ] Use `std::string_view` for non-owning string parameters
- [ ] Mark single-argument constructors `explicit`
- [ ] Use `enum class` instead of unscoped `enum`

---

## Common Mistakes

1. **Mixing object files compiled with different `-std=` flags** — causes ABI mismatches that crash at runtime, not compile time. Rebuild all dependencies consistently.
2. **Over-using `auto`** — hurts readability in public APIs. Use explicit types for function signatures.
3. **Ignoring deprecation warnings** — deprecated features *will* be removed. Fix proactively.
4. **Adopting modules without build-system support** — CMake support is still experimental as of 2025.
5. **Assuming `constexpr` means compile-time-only** — `constexpr` functions can run at runtime too.
6. **Forgetting `char8_t` breaking change (C++20)** — `u8""` returns `const char8_t*`, breaking `std::string` assignments.
7. **Adding `std::move` to simple returns** — can *prevent* NRVO. Only use when the compiler can't elide.
8. **`std::filesystem` default exceptions** — use `std::error_code` overloads in perf-critical or exception-free code.
9. **Removed features breaking builds** — `auto_ptr`, `random_shuffle`, `unary_function` removed in C++17.

---

## Exercises

### 🟢 Exercise 1 — Modernize a Loop
Rewrite using C++17 (range-for, structured bindings):

This is the old C++03-style loop using an explicit `const_iterator` type, manual `begin()`/`end()` calls, and `->first`/`->second` to access map entries. Your task is to replace it with C++17 structured bindings and a range-based for loop.

```cpp
std::map<std::string, int> scores;
for (std::map<std::string, int>::const_iterator it = scores.begin();
     it != scores.end(); ++it) {
    if (it->second > 90)
        std::cout << it->first << " passed with " << it->second << "\n";
}
```

### 🟡 Exercise 2 — Replace SFINAE with Concepts
Convert to C++20 Concepts:

This code uses the C++11 SFINAE pattern with `std::enable_if_t` to restrict the template to integral types. Your task is to replace this verbose constraint with a clean C++20 `std::integral` concept.

```cpp
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
T double_value(T x) { return x * 2; }
```

### 🟡 Exercise 3 — Migrate Error Handling
Refactor to return `std::expected<double, std::string>` (C++23):

This function uses the old output-parameter pattern: a `bool` return for success/failure with `result` and `error` passed by reference. Your task is to refactor it to return `std::expected`, eliminating the output parameters entirely.

```cpp
bool safe_divide(double a, double b, double& result, std::string& error) {
    if (b == 0.0) { error = "division by zero"; return false; }
    result = a / b; return true;
}
```

### 🔴 Exercise 4 — Concept-Constrained Container
Write a C++20 `SortedBuffer<T, N>` that uses `std::array<T, N>` internally, constrains `T` with `std::totally_ordered`, provides `insert()` maintaining sorted order, and `contains()` using `std::ranges::binary_search`.

---

## Solutions

### Solution 1

The modernized loop uses C++17 structured bindings (`[name, score]`) to unpack each map entry directly, and a range-based `for` loop to eliminate the verbose iterator boilerplate. This is shorter, more readable, and less error-prone than the explicit iterator version.

```cpp
for (const auto& [name, score] : scores) {
    if (score > 90)
        std::cout << name << " passed with " << score << "\n";
}
```

### Solution 2

The C++20 version replaces the `std::enable_if_t` SFINAE hack with `std::integral` as a direct template constraint. The result is identical behavior with a cleaner signature and far better compiler error messages when the constraint is not satisfied.

```cpp
#include <concepts>
template <std::integral T>
T double_value(T x) { return x * 2; }
```

### Solution 3

The C++23 refactored version returns `std::expected<double, std::string>`, which carries either the computed value or an error string. `std::unexpected` wraps the error case, and callers can chain operations with `.transform()` — no output parameters or boolean checks needed.

```cpp
#include <expected>
#include <string>
std::expected<double, std::string> safe_divide(double a, double b) {
    if (b == 0.0) return std::unexpected("division by zero");
    return a / b;
}
```

### Solution 4

This `SortedBuffer` uses a C++20 `std::totally_ordered` concept to constrain `T`, ensuring only types with valid comparison operators can be stored. Internally it maintains sorted order using `std::ranges::upper_bound` for insertion and `std::ranges::binary_search` for lookup — combining concepts, ranges, and fixed-capacity storage in one clean abstraction.

```cpp
#include <array>
#include <algorithm>
#include <concepts>
#include <ranges>
#include <cstddef>
#include <stdexcept>

template <std::totally_ordered T, std::size_t N>
class SortedBuffer {
    std::array<T, N> data_{};
    std::size_t size_ = 0;
public:
    void insert(const T& value) {
        if (size_ >= N) throw std::overflow_error("buffer full");
        auto pos = std::ranges::upper_bound(data_.begin(), data_.begin() + size_, value);
        std::move_backward(pos, data_.begin() + size_, data_.begin() + size_ + 1);
        *pos = value;
        ++size_;
    }
    bool contains(const T& value) const {
        return std::ranges::binary_search(data_.begin(), data_.begin() + size_, value);
    }
    std::size_t size() const { return size_; }
};
```

---

## Quiz

**Q1.** Which C++ standard introduced structured bindings?
A) C++11 · B) C++14 · C) C++17 ✅ · D) C++20

**Q2.** Minimum GCC version for full C++20 support?
A) GCC 8 · B) GCC 10 · C) GCC 12 ✅ · D) GCC 14

**Q3.** Which feature replaces SFINAE for constraining templates in C++20?
A) `if constexpr` · B) Concepts ✅ · C) Modules · D) Coroutines

**Q4.** `std::auto_ptr` was removed in which standard?
A) C++11 · B) C++14 · C) C++17 ✅ · D) C++20

**Q5.** `__cplusplus == 202002L` indicates?
A) C++17 · B) C++20 ✅ · C) C++23 · D) C++14

**Q6.** Which C++23 type provides monadic error handling without exceptions?
A) `std::optional` · B) `std::variant` · C) `std::expected` ✅ · D) `std::any`

**Q7.** What breaking change did C++20 introduce for `u8""` literals?
A) They became `std::string` · B) They return `const char8_t*` ✅ · C) They were deprecated · D) They require `import`

---

## Key Takeaways

- **Each standard builds incrementally** — adopt features gradually, not all at once.
- **Concepts (C++20) are the biggest template ergonomics leap** since variadic templates.
- **`std::expected` (C++23) bridges exceptions and error codes** with zero-overhead monadic chaining.
- **Compiler support matters** — verify your minimum toolchain before adopting any feature.
- **Automated tools handle 60–70% of migration** — use clang-tidy before manual refactoring.
- **ABI compatibility is the hidden cost** — rebuild all dependencies under one standard.
- **Feature-test macros enable incremental adoption** — wrap new features in `#if` guards.
- **Migration is a process, not an event** — plan it in phases with validation at each step.

---

## Chapter Summary

Migrating a C++ codebase across standards is one of the highest-leverage investments a team can make. Each revision — from C++11 through C++26 — delivers features that reduce boilerplate, eliminate bug categories, and improve expressiveness. The key to success is discipline: audit your toolchain, automate with clang-tidy, refactor in focused phases, and validate with sanitizers and benchmarks. The comparison tables and compiler matrix in this chapter give you a planning reference. The before/after examples show what modernized code looks like. Treat migration as continuous improvement — even a few modern idioms per sprint compounds into a dramatically better codebase.

---

## Real-World Insight

Large-scale C++ codebases (Google, Meta, Bloomberg) maintain a **minimum standard floor** — e.g., C++17 as baseline — while allowing opt-in to newer features via feature-test macros. Google applies clang-tidy modernize checks across billions of lines automatically, proving that automated migration works at scale. Bloomberg wraps standard library types behind aliases, letting one codebase compile under C++14 through C++23. The lesson: design abstractions so upgrading the standard behind them is a localized change, not a codebase-wide rewrite.

---

## Interview Questions

### Q1: What are the three most impactful C++20 features and why?

**Answer:** (1) **Concepts** replace SFINAE with readable constraints — error messages shrink from pages to one line. (2) **Ranges** compose algorithms via pipe, eliminate iterator-pair boilerplate, and enable lazy evaluation. (3) **Coroutines** enable stackless async without callback inversion, making generators and async I/O expressible in linear flow.

### Q2: How would you migrate a 500 KLOC codebase from C++14 to C++20?

**Answer:** Audit dependencies for C++20 support. Upgrade compilers across all platforms. Enable `-std=c++20`, fix errors (narrowing conversions, `char8_t`, removed features). Run clang-tidy modernize in per-module PRs. Adopt Concepts and Ranges in new code first, then retrofit hot paths. Validate with sanitizers and benchmarks. Budget 2–4 months for a 5–10 person team.

### Q3: What is an ABI break and how does it affect migrations?

**Answer:** An ABI break occurs when binary layout, name mangling, or calling conventions change between versions. Object files compiled with different standards can't link safely — causing segfaults, not compile errors. The most common source is `std::string` layout changes (GCC's dual ABI since GCC 5). Mitigation: use identical compiler version and `-std=` flag across the entire dependency tree, or use stable C interfaces at library boundaries.

### Q4: Explain feature-test macros and their role in incremental adoption.

**Answer:** Feature-test macros (`__cpp_concepts`, `__cpp_lib_expected`) are predefined preprocessor symbols whose values correspond to the paper revision that introduced a feature. They enable writing portable code:
```cpp
#if __cpp_lib_expected >= 202202L
    return std::expected<int, Error>{value};
#else
    return fallback_result(value);
#endif
```
This allows gradual adoption — use new features where available, fall back elsewhere.
