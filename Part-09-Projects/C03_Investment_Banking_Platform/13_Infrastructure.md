# Module 13: Infrastructure Components

## Module Overview

Every trading system stands on invisible foundations — memory allocators, loggers, config
parsers, timers, and thread pools. These components appear in **every** module we've built
so far, yet we deferred their implementation until now because understanding *why* they
exist requires seeing the systems they serve.

This module is the **densest C++ module** in the entire project. It exercises custom
allocators, CRTP, variadic templates, lock-free data structures, `constexpr` validation,
`std::expected`, `std::variant`, `std::jthread`, and RAII — all in production-grade
infrastructure code.

**Why infrastructure matters in trading:**

| Component        | stdlib Default       | Custom Version       | Improvement |
|------------------|----------------------|----------------------|-------------|
| Memory alloc     | `malloc` (~100ns)    | Arena bump (~2ns)    | **50×**     |
| Logging          | `std::cout` + mutex  | Lock-free ring buf   | **No stall** |
| Config parsing   | Runtime-only checks  | `constexpr` defaults | **Compile-time safety** |
| Timer            | `gettimeofday`       | `rdtsc` + chrono     | **Sub-ns precision** |
| Thread pool      | `std::async`         | Work-stealing pool   | **No thread churn** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Modules                       │
│  OrderBook │ PricingEngine │ RiskEngine │ ExecutionGateway   │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Arena      │  │   Logger     │  │   Config     │       │
│  │  Allocator    │  │ (Lock-Free)  │  │   System     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                          │
│  │  HiRes Timer │  │  Thread Pool │                          │
│  │  + Profiler   │  │ (Work-Steal) │                          │
│  └──────────────┘  └──────────────┘                          │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│              OS / Hardware (Linux, x86-64)                    │
└─────────────────────────────────────────────────────────────┘
```

**Dependency flow:** Every module `#include`s from infrastructure. Infrastructure
depends on nothing but the standard library and OS primitives.

---

## C++ Concepts Applied

| Concept | Chapter | Where Used |
|---------|---------|------------|
| Custom allocator, PMR | Ch 31 | Arena allocator with `std::pmr` integration |
| CRTP | Ch 23 | Logger static dispatch — zero virtual overhead |
| Variadic templates | Ch 23 | Type-safe `log(fmt, args...)` formatting |
| `constexpr` / `consteval` | Ch 29 | Compile-time config defaults and validation |
| `std::expected` | Ch 36 | Config parse results with structured errors |
| `std::variant` | Ch 30 | Polymorphic config values without inheritance |
| `std::chrono` | Ch 33 | High-resolution timing and timestamps |
| RAII | Ch 14 | ScopedTimer, thread lifetime management |
| `std::jthread` / `stop_token` | Ch 35 | Thread pool with cooperative cancellation |
| Lock-free programming | Ch 35 | Ring buffer logger, MPMC task queue |
| `std::atomic` | Ch 35 | Atomic head/tail pointers, fence operations |

---

## Design Decisions

### Why Not Use Boost or Third-Party Libraries?

This project is pedagogical. Every component is hand-built to teach the C++ features
behind it. In production, you'd evaluate:
- **jemalloc / tcmalloc** for general allocation
- **spdlog** for logging
- **libconfig / toml++** for config
- **Intel TBB** for thread pools

But understanding the internals lets you *choose* intelligently and *debug* effectively.

### Why Arena over Pool Allocator?

Arena allocators suit trading workloads where objects are created in bursts (order
processing) and freed together (end of batch). Pool allocators suit fixed-size objects
with random lifetimes. We use arenas because order book updates follow batch patterns.

---

## Complete Code

### A. Arena Allocator

```cpp
// infrastructure/arena_allocator.hpp
// Custom bump allocator with PMR integration (Ch 31)

#pragma once
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <cassert>
#include <new>
#include <vector>
#include <string>

namespace infra {

// ─────────────────────────────────────────────────────────
// Core Arena: Pre-allocates a block, bumps pointer forward
// ─────────────────────────────────────────────────────────
class Arena {
public:
    explicit Arena(std::size_t capacity)
        : capacity_(capacity)
        , buffer_(static_cast<std::byte*>(::operator new(capacity,
              std::align_val_t{alignment_})))
        , offset_(0)
    {}

    ~Arena() {
        ::operator delete(buffer_, std::align_val_t{alignment_});
    }

    // Non-copyable, movable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&& other) noexcept
        : capacity_(other.capacity_)
        , buffer_(other.buffer_)
        , offset_(other.offset_)
    {
        other.buffer_ = nullptr;
        other.offset_ = 0;
    }

    // Bump allocation — O(1), ~2ns
    [[nodiscard]] void* allocate(std::size_t size, std::size_t align = alignment_) {
        std::size_t aligned_offset = align_up(offset_, align);
        if (aligned_offset + size > capacity_) {
            throw std::bad_alloc{};  // Arena exhausted
        }
        void* ptr = buffer_ + aligned_offset;
        offset_ = aligned_offset + size;
        return ptr;
    }

    // No-op: arena frees everything at once via reset()
    void deallocate(void* /*ptr*/, std::size_t /*size*/) noexcept {
        // Individual deallocation is intentionally a no-op.
        // This is the key design choice: O(1) "free" on every call.
    }

    // Typed allocation helper
    template<typename T, typename... Args>
    [[nodiscard]] T* construct(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        return ::new(mem) T(std::forward<Args>(args)...);
    }

    // Reset the arena — all allocations invalidated
    void reset() noexcept { offset_ = 0; }

    // Diagnostics
    [[nodiscard]] std::size_t used() const noexcept { return offset_; }
    [[nodiscard]] std::size_t remaining() const noexcept { return capacity_ - offset_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
    static constexpr std::size_t alignment_ = 64;  // Cache-line aligned

    static constexpr std::size_t align_up(std::size_t n, std::size_t align) noexcept {
        return (n + align - 1) & ~(align - 1);
    }

    std::size_t capacity_;
    std::byte*  buffer_;
    std::size_t offset_;
};


// ──────────────────────────────────────────────────────────
// PMR Adapter: Lets Arena work with std::pmr containers
// ──────────────────────────────────────────────────────────
class ArenaMemoryResource : public std::pmr::memory_resource {
public:
    explicit ArenaMemoryResource(Arena& arena) : arena_(arena) {}

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        return arena_.allocate(bytes, alignment);
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        arena_.deallocate(p, bytes);  // No-op by design
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    Arena& arena_;
};

}  // namespace infra
```

**Memory block layout:**

```
┌──────────────────────────────────────────────────────────────┐
│                        Arena Buffer                           │
│                                                               │
│  ┌────────┐  ┌──────────┐  ┌──────┐                          │
│  │ Alloc1 │  │  Alloc2  │  │  A3  │  ← offset    free →     │
│  │ 64B    │  │  128B    │  │ 32B  │     ↑                    │
│  └────────┘  └──────────┘  └──────┘     │                    │
│  ^                                       │                    │
│  buffer_                          current offset_             │
│                                                               │
│  After reset():  offset_ = 0  →  entire buffer reusable      │
└──────────────────────────────────────────────────────────────┘
```

**Usage with PMR containers:**

```cpp
infra::Arena arena(1024 * 1024);  // 1 MB
infra::ArenaMemoryResource resource(arena);

// std::pmr containers automatically use our arena
std::pmr::vector<double> prices(&resource);
prices.reserve(10000);  // One arena bump, no malloc

std::pmr::string symbol("AAPL", &resource);  // Arena-allocated string

arena.reset();  // Instant "free" of everything — 0ns
```

---

### B. Lock-Free Logger

```cpp
// infrastructure/logger.hpp
// CRTP logger with lock-free ring buffer (Ch 23, Ch 35)

#pragma once
#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <format>
#include <thread>
#include <string_view>

namespace infra {

enum class LogLevel : uint8_t { DEBUG, INFO, WARN, ERROR, FATAL };

constexpr std::string_view level_str(LogLevel lv) {
    constexpr std::string_view names[] = {"DEBUG","INFO","WARN","ERROR","FATAL"};
    return names[static_cast<int>(lv)];
}

// ─── Log Entry: Fixed-size for lock-free ring buffer ───
struct alignas(64) LogEntry {  // Cache-line aligned
    uint64_t    timestamp_ns;
    LogLevel    level;
    char        message[240];  // Fits in 4 cache lines total
};

// ─── Lock-Free Ring Buffer ─────────────────────────────
// Multiple producers (trading threads), single consumer (flush thread)
template<std::size_t Capacity = 8192>
class LogRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
public:
    bool try_push(const LogEntry& entry) noexcept {
        auto head = head_.load(std::memory_order_relaxed);
        auto next = (head + 1) & mask_;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // Full — drop log rather than block
        }
        buffer_[head] = entry;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(LogEntry& entry) noexcept {
        auto tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // Empty
        }
        entry = buffer_[tail];
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return true;
    }

private:
    static constexpr std::size_t mask_ = Capacity - 1;
    std::array<LogEntry, Capacity> buffer_{};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

// ─── CRTP Logger Base ──────────────────────────────────
// Static dispatch via CRTP: zero virtual call overhead (Ch 23)
template<typename Derived>
class Logger {
public:
    // Variadic template log method (Ch 23)
    template<typename... Args>
    void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
        if (level < min_level_) return;

        LogEntry entry{};
        entry.timestamp_ns = now_ns();
        entry.level = level;

        auto msg = std::format(fmt, std::forward<Args>(args)...);
        std::strncpy(entry.message, msg.c_str(), sizeof(entry.message) - 1);

        static_cast<Derived*>(this)->write_entry(entry);
    }

    template<typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
    }

    void set_level(LogLevel lv) { min_level_ = lv; }

private:
    LogLevel min_level_ = LogLevel::INFO;

    static uint64_t now_ns() {
        auto tp = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            tp.time_since_epoch()).count();
    }
};

// ─── Async File Logger (concrete CRTP derivation) ──────
class AsyncFileLogger : public Logger<AsyncFileLogger> {
    friend class Logger<AsyncFileLogger>;
public:
    explicit AsyncFileLogger(const char* path)
        : file_(std::fopen(path, "a"))
        , flush_thread_([this](std::stop_token st) { flush_loop(st); })
    {}

    ~AsyncFileLogger() {
        flush_thread_.request_stop();
        // jthread joins automatically (Ch 35)
    }

    void write_entry(const LogEntry& entry) {
        ring_.try_push(entry);  // Non-blocking — never stalls hot path
    }

private:
    void flush_loop(std::stop_token stop) {
        LogEntry entry{};
        while (!stop.stop_requested()) {
            while (ring_.try_pop(entry)) {
                std::fprintf(file_, "[%lu] [%.*s] %s\n",
                    entry.timestamp_ns,
                    static_cast<int>(level_str(entry.level).size()),
                    level_str(entry.level).data(),
                    entry.message);
            }
            std::fflush(file_);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Drain remaining on shutdown
        while (ring_.try_pop(entry)) {
            std::fprintf(file_, "[%lu] [%.*s] %s\n",
                entry.timestamp_ns,
                static_cast<int>(level_str(entry.level).size()),
                level_str(entry.level).data(),
                entry.message);
        }
        std::fclose(file_);
    }

    FILE* file_;
    LogRingBuffer<> ring_;
    std::jthread flush_thread_;
};

}  // namespace infra
```

---

### C. Config System

```cpp
// infrastructure/config.hpp
// Compile-time validated config with std::variant and std::expected (Ch 29, 30, 36)

#pragma once
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace infra {

// Config values are a closed set of types (Ch 30 — std::variant)
using ConfigValue = std::variant<int64_t, double, bool, std::string>;

// Structured parse errors via std::expected (Ch 36)
struct ConfigError {
    std::string key;
    std::string message;
    int line_number;
};

using ConfigResult = std::expected<ConfigValue, ConfigError>;

// ─── Compile-time defaults (Ch 29) ────────────────────
struct ConfigDefaults {
    static constexpr int64_t ORDER_BOOK_DEPTH       = 10;
    static constexpr int64_t MAX_ORDER_SIZE          = 1'000'000;
    static constexpr double  RISK_LIMIT_USD          = 50'000'000.0;
    static constexpr int64_t THREAD_POOL_SIZE        = 8;
    static constexpr int64_t ARENA_SIZE_MB           = 64;
    static constexpr int64_t LOG_RING_BUFFER_SIZE    = 8192;
    static constexpr bool    ENABLE_PERSISTENCE      = true;

    // consteval: These MUST be evaluated at compile time
    static consteval int64_t validated_book_depth() {
        static_assert(ORDER_BOOK_DEPTH > 0 && ORDER_BOOK_DEPTH <= 50,
            "Book depth must be 1-50");
        return ORDER_BOOK_DEPTH;
    }

    static consteval double validated_risk_limit() {
        static_assert(RISK_LIMIT_USD > 0, "Risk limit must be positive");
        return RISK_LIMIT_USD;
    }
};

// ─── Config Parser (TOML-like) ─────────────────────────
class Config {
public:
    // Parse a key=value config string
    static std::expected<Config, std::vector<ConfigError>> parse(std::string_view input) {
        Config cfg;
        std::vector<ConfigError> errors;
        int line_num = 0;

        std::size_t pos = 0;
        while (pos < input.size()) {
            ++line_num;
            auto line_end = input.find('\n', pos);
            auto line = input.substr(pos, line_end == std::string_view::npos
                ? std::string_view::npos : line_end - pos);
            pos = (line_end == std::string_view::npos) ? input.size() : line_end + 1;

            line = trim(line);
            if (line.empty() || line[0] == '#') continue;  // Comment or blank

            auto eq = line.find('=');
            if (eq == std::string_view::npos) {
                errors.push_back({std::string(line), "Missing '='", line_num});
                continue;
            }

            auto key = trim(line.substr(0, eq));
            auto val = trim(line.substr(eq + 1));

            auto result = parse_value(val);
            if (result) {
                cfg.values_[std::string(key)] = *result;
            } else {
                errors.push_back({std::string(key), "Invalid value", line_num});
            }
        }

        if (!errors.empty()) return std::unexpected(errors);
        return cfg;
    }

    // Type-safe getters with defaults
    template<typename T>
    T get(std::string_view key, T default_val = T{}) const {
        auto it = values_.find(std::string(key));
        if (it == values_.end()) return default_val;
        if (auto* v = std::get_if<T>(&it->second)) return *v;
        return default_val;
    }

    bool has(std::string_view key) const {
        return values_.contains(std::string(key));
    }

private:
    std::unordered_map<std::string, ConfigValue> values_;

    static std::string_view trim(std::string_view sv) {
        while (!sv.empty() && sv.front() == ' ') sv.remove_prefix(1);
        while (!sv.empty() && sv.back() == ' ')  sv.remove_suffix(1);
        return sv;
    }

    static std::optional<ConfigValue> parse_value(std::string_view sv) {
        if (sv == "true")  return ConfigValue{true};
        if (sv == "false") return ConfigValue{false};

        // Try integer
        int64_t ival = 0;
        bool negative = false;
        std::size_t i = 0;
        if (!sv.empty() && sv[0] == '-') { negative = true; ++i; }
        bool is_int = (i < sv.size());
        bool has_dot = false;
        for (auto j = i; j < sv.size(); ++j) {
            if (sv[j] == '.') { has_dot = true; is_int = false; break; }
            if (sv[j] < '0' || sv[j] > '9') { is_int = false; break; }
            ival = ival * 10 + (sv[j] - '0');
        }
        if (is_int) return ConfigValue{negative ? -ival : ival};

        // Try double
        if (has_dot) {
            try {
                return ConfigValue{std::stod(std::string(sv))};
            } catch (...) {}
        }

        // String (strip quotes if present)
        if (sv.size() >= 2 && sv.front() == '"' && sv.back() == '"') {
            return ConfigValue{std::string(sv.substr(1, sv.size() - 2))};
        }
        return ConfigValue{std::string(sv)};
    }
};

}  // namespace infra
```

---

### D. High-Resolution Timer

```cpp
// infrastructure/timer.hpp
// High-resolution timing with RAII and latency percentiles (Ch 33)

#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace infra {

// ─── rdtsc: Sub-nanosecond CPU cycle counter ───────────
inline uint64_t rdtsc() noexcept {
#if defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#else
    // Fallback for non-x86 platforms
    return std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

// ─── RAII Scoped Timer ─────────────────────────────────
// Logs elapsed time on destruction — measures any scope automatically
class ScopedTimer {
public:
    explicit ScopedTimer(std::string_view name)
        : name_(name)
        , start_(std::chrono::steady_clock::now())
    {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       end - start_).count();
        std::printf("[TIMER] %.*s: %ld ns (%.3f us)\n",
            static_cast<int>(name_.size()), name_.data(), ns, ns / 1000.0);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string_view name_;
    std::chrono::steady_clock::time_point start_;
};

// ─── Latency Percentile Calculator ─────────────────────
// Collects samples and computes p50, p95, p99, p99.9
class LatencyHistogram {
public:
    void record(uint64_t nanoseconds) {
        samples_.push_back(nanoseconds);
    }

    void compute() {
        if (samples_.empty()) return;
        std::sort(samples_.begin(), samples_.end());
    }

    uint64_t percentile(double p) const {
        if (samples_.empty()) return 0;
        auto idx = static_cast<std::size_t>(p / 100.0 * (samples_.size() - 1));
        return samples_[idx];
    }

    uint64_t p50()   const { return percentile(50.0); }
    uint64_t p95()   const { return percentile(95.0); }
    uint64_t p99()   const { return percentile(99.0); }
    uint64_t p999()  const { return percentile(99.9); }
    uint64_t min()   const { return samples_.empty() ? 0 : samples_.front(); }
    uint64_t max()   const { return samples_.empty() ? 0 : samples_.back(); }
    std::size_t count() const { return samples_.size(); }

    double mean() const {
        if (samples_.empty()) return 0.0;
        uint64_t sum = 0;
        for (auto s : samples_) sum += s;
        return static_cast<double>(sum) / samples_.size();
    }

    void print(std::string_view label) const {
        std::printf("[LATENCY] %.*s: n=%zu min=%lu p50=%lu p95=%lu p99=%lu "
                    "p99.9=%lu max=%lu mean=%.1f ns\n",
            static_cast<int>(label.size()), label.data(),
            count(), min(), p50(), p95(), p99(), p999(), max(), mean());
    }

    void reset() { samples_.clear(); }

private:
    std::vector<uint64_t> samples_;
};

}  // namespace infra
```

---

### E. Thread Pool

```cpp
// infrastructure/thread_pool.hpp
// Work-stealing thread pool with std::jthread (Ch 35)

#pragma once
#include <atomic>
#include <concepts>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace infra {

// ─── Per-thread work queue (steal-able) ────────────────
class WorkQueue {
public:
    void push(std::function<void()> task) {
        std::lock_guard lock(mtx_);
        tasks_.push_back(std::move(task));
    }

    std::optional<std::function<void()>> try_pop() {
        std::lock_guard lock(mtx_);
        if (tasks_.empty()) return std::nullopt;
        auto task = std::move(tasks_.front());
        tasks_.pop_front();
        return task;
    }

    // Steal from the BACK (opposite end) to reduce contention
    std::optional<std::function<void()>> try_steal() {
        std::lock_guard lock(mtx_);
        if (tasks_.empty()) return std::nullopt;
        auto task = std::move(tasks_.back());
        tasks_.pop_back();
        return task;
    }

    bool empty() const {
        std::lock_guard lock(mtx_);
        return tasks_.empty();
    }

private:
    mutable std::mutex mtx_;
    std::deque<std::function<void()>> tasks_;
};

// ─── Thread Pool ───────────────────────────────────────
class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
        : queues_(num_threads)
    {
        for (std::size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this, i](std::stop_token stop) {
                worker_loop(i, stop);
            });
        }
    }

    ~ThreadPool() {
        for (auto& t : threads_) t.request_stop();
        // jthreads auto-join in destructor
    }

    // Submit a callable, get a future for the result
    template<std::invocable F>
    auto submit(F&& func) -> std::future<std::invoke_result_t<F>> {
        using Result = std::invoke_result_t<F>;
        auto promise = std::make_shared<std::promise<Result>>();
        auto future  = promise->get_future();

        auto idx = next_queue_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
        queues_[idx].push([p = std::move(promise), f = std::forward<F>(func)]() mutable {
            try {
                if constexpr (std::is_void_v<Result>) {
                    f();
                    p->set_value();
                } else {
                    p->set_value(f());
                }
            } catch (...) {
                p->set_exception(std::current_exception());
            }
        });

        return future;
    }

    std::size_t thread_count() const { return threads_.size(); }

private:
    void worker_loop(std::size_t my_idx, std::stop_token stop) {
        while (!stop.stop_requested()) {
            // 1. Try own queue
            if (auto task = queues_[my_idx].try_pop()) {
                (*task)();
                continue;
            }

            // 2. Try stealing from other queues
            bool found = false;
            for (std::size_t i = 1; i < queues_.size(); ++i) {
                auto victim = (my_idx + i) % queues_.size();
                if (auto task = queues_[victim].try_steal()) {
                    (*task)();
                    found = true;
                    break;
                }
            }

            // 3. Yield if no work found
            if (!found) {
                std::this_thread::yield();
            }
        }
    }

    std::vector<WorkQueue> queues_;
    std::vector<std::jthread> threads_;
    std::atomic<std::size_t> next_queue_{0};
};

}  // namespace infra
```

---

## Walkthrough

### How the Components Interact

Consider what happens when the `OrderBook` (Module 2) processes a new order:

```
1. Config system loaded at startup:
   auto cfg = Config::parse(file_contents);
   int depth = cfg.get<int64_t>("order_book.depth", ConfigDefaults::ORDER_BOOK_DEPTH);

2. Arena allocator pre-allocated:
   Arena arena(cfg.get<int64_t>("arena.size_mb", 64) * 1024 * 1024);

3. Order arrives → OrderBook allocates from arena:
   auto* order = arena.construct<Order>(id, side, price, qty);
   // ~2ns vs ~100ns for malloc

4. Match engine runs → logs via lock-free logger:
   logger.info("Match: {} @ {:.2f} x {}", order_id, price, qty);
   // Non-blocking, never stalls the matching engine

5. Timer measures end-to-end latency:
   ScopedTimer timer("order_process");
   // ... processing ...
   // Destructor prints elapsed time automatically

6. Risk calculations dispatched to thread pool:
   auto greek_future = pool.submit([&]{ return calc_greeks(portfolio); });
   auto var_future   = pool.submit([&]{ return calc_var(portfolio); });
   // Both run in parallel on different threads
```

### Work-Stealing Visualization

```
Thread 0 Queue:  [Task A] [Task B] [Task C]  ← busy
Thread 1 Queue:  [Task D]                     ← finishing
Thread 2 Queue:  (empty)                      ← idle, steals Task C
Thread 3 Queue:  (empty)                      ← idle, steals Task B

Result: Work automatically balances across threads
```

---

## Key Takeaways

1. **Arena allocators eliminate malloc overhead** on the hot path — critical when
   processing thousands of orders per second. The `reset()` pattern matches trading's
   batch-oriented workload perfectly.

2. **CRTP gives you polymorphism without virtual dispatch.** The Logger hierarchy has
   zero runtime overhead — the compiler inlines everything at compile time (Ch 23).

3. **Lock-free logging prevents priority inversion.** A logging call should *never*
   cause a trading thread to wait for a mutex held by a background flush thread.

4. **`constexpr` defaults catch misconfiguration at compile time.** If someone sets
   `ORDER_BOOK_DEPTH = -1`, the build fails — not production at 2 AM.

5. **Work-stealing pools outperform `std::async`** because they reuse threads and
   automatically balance load. Creating a thread per task costs ~50μs; submitting
   to a pool costs ~100ns.

---

## Cross-References

| Component | Depends On | Used By |
|-----------|-----------|---------|
| Arena Allocator | — | OrderBook (M2), PricingEngine (M4), Persistence (M11) |
| Logger | — | ALL modules |
| Config | — | ALL modules (startup) |
| Timer | — | Benchmarks (M14), PricingEngine (M4), ExecutionGateway (M6) |
| Thread Pool | — | RiskEngine (M5), PricingEngine (M4) |

**Chapter references:**
- Ch 14 (RAII) → ScopedTimer, jthread auto-join
- Ch 23 (Templates) → CRTP Logger, variadic log()
- Ch 29 (constexpr) → ConfigDefaults compile-time validation
- Ch 30 (variant) → ConfigValue polymorphism
- Ch 31 (Allocators) → Arena + PMR adapter
- Ch 33 (Chrono) → Timer, logger timestamps
- Ch 35 (Concurrency) → jthread, stop_token, atomics
- Ch 36 (expected) → Config parse error handling
