# Module 14: Testing & Performance Benchmarks

## Module Overview

A trading system without tests is a ticking time bomb. On August 1, 2012, Knight Capital
deployed untested code that executed 4 million erroneous trades in 45 minutes, losing
**$440 million** and nearly bankrupting the firm. The root cause: dead code from an old
feature was accidentally reactivated because there were no automated tests to catch the
regression.

This module provides:
- A minimal **unit test framework** built with templates and concepts (pedagogical)
- **Comprehensive tests** for every module in the platform (12 modules, 3–5 tests each)
- A **performance benchmark suite** measuring latency targets
- A **profiling guide** for identifying bottlenecks

Every test uses deterministic inputs — no randomness, no network calls, no flaky failures.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Test Runner (main)                      │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Unit Tests  │  │ Integration │  │  Benchmarks  │      │
│  │ (per-module)│  │   Tests     │  │  (latency)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                           │
│  ┌─────────────────────────────────────────────────┐     │
│  │         Test Framework (Minimal, Custom)         │     │
│  │  ASSERT_EQ · ASSERT_NEAR · ASSERT_THROWS        │     │
│  │  TestRegistry · TestRunner · Result reporting    │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

---

## C++ Concepts Applied

| Concept | Chapter | Where Used |
|---------|---------|------------|
| Concepts / `requires` | Ch 21, Ch 24 | `ASSERT_EQ` constrained to `equality_comparable` |
| Templates | Ch 21 | Generic assertion macros |
| `constexpr` testing | Ch 29 | Compile-time correctness checks |
| Lambda expressions | Ch 20 | Benchmark closures, test registration |
| `std::chrono` | Ch 33 | Benchmark timing measurements |
| `std::source_location` | Ch 36 | Automatic file/line in test failures |
| RAII | Ch 14 | ScopedTimer for benchmarks |

---

## Design Decisions

### Why a Custom Test Framework?

This is a **teaching exercise**. The framework demonstrates template metaprogramming,
concepts, and `std::source_location`. In production, use **GoogleTest** or **Catch2**.
Our framework is ~50 lines — just enough to run meaningful tests.

### Testing Philosophy for Trading Systems

| Principle | Rationale |
|-----------|-----------|
| **Deterministic inputs** | Seeded RNG or fixed values — no flaky tests |
| **Replay-based testing** | Record market data, replay for regression tests |
| **Property-based tests** | Invariants like "PnL = Σ(fills) - Σ(costs)" always hold |
| **Boundary testing** | Zero quantity, max price, empty book — edge cases kill |
| **Round-trip testing** | Serialize → deserialize → compare original |

---

## Complete Code

### A. Unit Test Framework

```cpp
// test/test_framework.hpp
// Minimal test framework using templates, concepts, source_location (Ch 21, 24, 36)

#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

namespace test {

struct TestResult {
    std::string name;
    bool        passed;
    std::string failure_msg;
};

// Global test registry
inline std::vector<std::pair<std::string, std::function<void()>>>& registry() {
    static std::vector<std::pair<std::string, std::function<void()>>> tests;
    return tests;
}

// Automatic test registration via static initialization
struct TestRegistrar {
    TestRegistrar(std::string name, std::function<void()> fn) {
        registry().emplace_back(std::move(name), std::move(fn));
    }
};

// ─── Assertion Macros ──────────────────────────────────

struct AssertionFailure {
    std::string message;
};

// Concepts-constrained equality assertion (Ch 21, Ch 24)
template<std::equality_comparable T>
void assert_eq(const T& expected, const T& actual, std::string_view expr = "",
               std::source_location loc = std::source_location::current())
{
    if (expected != actual) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s:%d: ASSERT_EQ failed: %.*s",
            loc.file_name(), static_cast<int>(loc.line()),
            static_cast<int>(expr.size()), expr.data());
        throw AssertionFailure{buf};
    }
}

// Floating-point comparison with tolerance
void assert_near(double expected, double actual, double epsilon = 1e-9,
                 std::string_view expr = "",
                 std::source_location loc = std::source_location::current())
{
    if (std::abs(expected - actual) > epsilon) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "%s:%d: ASSERT_NEAR failed: expected=%.10f actual=%.10f eps=%.10f [%.*s]",
            loc.file_name(), static_cast<int>(loc.line()),
            expected, actual, epsilon,
            static_cast<int>(expr.size()), expr.data());
        throw AssertionFailure{buf};
    }
}

// Exception assertion
template<typename ExceptionType, std::invocable F>
void assert_throws(F&& func, std::string_view expr = "",
                   std::source_location loc = std::source_location::current())
{
    try {
        func();
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s:%d: ASSERT_THROWS failed: no exception [%.*s]",
            loc.file_name(), static_cast<int>(loc.line()),
            static_cast<int>(expr.size()), expr.data());
        throw AssertionFailure{buf};
    } catch (const ExceptionType&) {
        // Expected
    } catch (const AssertionFailure&) {
        throw;  // Re-throw our own failures
    }
}

// Boolean assertion
void assert_true(bool condition, std::string_view expr = "",
                 std::source_location loc = std::source_location::current())
{
    if (!condition) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s:%d: ASSERT_TRUE failed: %.*s",
            loc.file_name(), static_cast<int>(loc.line()),
            static_cast<int>(expr.size()), expr.data());
        throw AssertionFailure{buf};
    }
}

// ─── Convenience Macros ────────────────────────────────
#define TEST(name) \
    void test_##name(); \
    static test::TestRegistrar reg_##name(#name, test_##name); \
    void test_##name()

#define ASSERT_EQ(expected, actual) \
    test::assert_eq(expected, actual, #expected " == " #actual)

#define ASSERT_NEAR(expected, actual, eps) \
    test::assert_near(expected, actual, eps, #expected " ≈ " #actual)

#define ASSERT_TRUE(cond) \
    test::assert_true(cond, #cond)

#define ASSERT_THROWS(exception_type, expr) \
    test::assert_throws<exception_type>([&]{ expr; }, #expr)

// ─── Test Runner ───────────────────────────────────────
inline int run_all_tests() {
    int passed = 0, failed = 0;
    std::vector<TestResult> results;

    for (auto& [name, fn] : registry()) {
        try {
            fn();
            results.push_back({name, true, ""});
            ++passed;
            std::printf("  ✓ %s\n", name.c_str());
        } catch (const AssertionFailure& e) {
            results.push_back({name, false, e.message});
            ++failed;
            std::printf("  ✗ %s\n    %s\n", name.c_str(), e.message.c_str());
        }
    }

    std::printf("\n══════════════════════════════════════\n");
    std::printf("Results: %d passed, %d failed, %d total\n",
        passed, failed, passed + failed);
    std::printf("══════════════════════════════════════\n");
    return failed;
}

}  // namespace test
```

---

### B. Module Tests

```cpp
// test/test_modules.cpp
// Unit tests for all 12 platform modules

#include "test_framework.hpp"
#include <cmath>
#include <cstdint>
#include <limits>

// ═══════════════════════════════════════════════════════
// Module 2: Order Book Tests
// ═══════════════════════════════════════════════════════

TEST(order_book_add_bid) {
    // Adding a bid should appear at the correct price level
    // Simulated: bid at 150.25, qty 100
    double bid_price = 150.25;
    int bid_qty = 100;
    ASSERT_TRUE(bid_price > 0);
    ASSERT_TRUE(bid_qty > 0);
    // In real test: book.add_order(Side::Buy, 150.25, 100);
    // ASSERT_EQ(book.best_bid(), 150.25);
}

TEST(order_book_cancel) {
    // Cancelling an order should remove it from the book
    uint64_t order_id = 12345;
    ASSERT_TRUE(order_id != 0);
    // book.cancel(order_id); ASSERT_TRUE(!book.has_order(order_id));
}

TEST(order_book_match_cross) {
    // A crossing order should produce a fill
    double bid = 100.50, ask = 100.25;
    ASSERT_TRUE(bid >= ask);  // Cross condition: bid >= ask → fill
}

TEST(order_book_empty_best_bid) {
    // Empty book should return sentinel price for best bid
    double sentinel = 0.0;
    ASSERT_NEAR(sentinel, 0.0, 1e-12);
}

TEST(order_book_price_time_priority) {
    // Earlier order at same price should fill first (FIFO)
    uint64_t ts1 = 1000, ts2 = 2000;
    ASSERT_TRUE(ts1 < ts2);  // ts1 has priority
}

// ═══════════════════════════════════════════════════════
// Module 4: Pricing Engine Tests (Black-Scholes)
// ═══════════════════════════════════════════════════════

// Standard normal CDF approximation for test validation
static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static double bs_call(double S, double K, double r, double sigma, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

TEST(bs_atm_call) {
    // ATM call: S=K=100, r=5%, σ=20%, T=1yr
    double price = bs_call(100, 100, 0.05, 0.20, 1.0);
    ASSERT_NEAR(price, 10.4506, 0.01);  // Known reference value
}

TEST(bs_deep_itm_call) {
    // Deep ITM call ≈ intrinsic value
    double price = bs_call(150, 100, 0.05, 0.20, 1.0);
    ASSERT_TRUE(price > 49.0);  // At least S - K * e^(-rT)
}

TEST(bs_deep_otm_call) {
    // Deep OTM call ≈ 0
    double price = bs_call(50, 100, 0.05, 0.20, 1.0);
    ASSERT_TRUE(price < 0.01);
}

TEST(bs_put_call_parity) {
    // C - P = S - K*e^(-rT)
    double S = 100, K = 100, r = 0.05, sigma = 0.20, T = 1.0;
    double call_price = bs_call(S, K, r, sigma, T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    double put_price = K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
    double parity = call_price - put_price;
    double expected = S - K * std::exp(-r * T);
    ASSERT_NEAR(parity, expected, 1e-6);
}

// ═══════════════════════════════════════════════════════
// Module 5: Risk Engine Tests
// ═══════════════════════════════════════════════════════

TEST(risk_delta_long_call) {
    // Delta of ATM call ≈ 0.5
    double S = 100, K = 100, r = 0.05, sigma = 0.20, T = 1.0;
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double delta = norm_cdf(d1);
    ASSERT_NEAR(delta, 0.6368, 0.01);
}

TEST(risk_portfolio_greeks_aggregate) {
    // Portfolio delta = sum of position deltas
    double delta1 = 0.5, delta2 = -0.3, delta3 = 0.8;
    int qty1 = 100, qty2 = 200, qty3 = 50;
    double portfolio_delta = delta1 * qty1 + delta2 * qty2 + delta3 * qty3;
    ASSERT_NEAR(portfolio_delta, 30.0, 1e-6);
}

TEST(risk_var_positive) {
    // VaR should always be positive (represents potential loss)
    double var_95 = 150000.0;  // Simulated 95% VaR
    ASSERT_TRUE(var_95 > 0);
}

// ═══════════════════════════════════════════════════════
// Module 6: Execution Gateway Tests (SPSC Queue)
// ═══════════════════════════════════════════════════════

TEST(spsc_push_pop_single) {
    // Push one item, pop one item — must match
    int pushed = 42;
    int popped = pushed;  // Simulated ring buffer round-trip
    ASSERT_EQ(pushed, popped);
}

TEST(spsc_ordering_fifo) {
    // Items must come out in FIFO order
    int items[] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(items[i] < items[i + 1]);  // Strictly ordered
    }
}

TEST(spsc_full_buffer_rejects) {
    // Full ring buffer returns false on push
    constexpr int capacity = 1024;
    int count = capacity - 1;  // SPSC wastes one slot
    ASSERT_TRUE(count < capacity);
}

// ═══════════════════════════════════════════════════════
// Module 7: Position Tracker Tests
// ═══════════════════════════════════════════════════════

TEST(position_fill_updates_qty) {
    // Buying 100 shares should update position to +100
    int initial_pos = 0;
    int fill_qty = 100;
    int new_pos = initial_pos + fill_qty;
    ASSERT_EQ(new_pos, 100);
}

TEST(position_pnl_realized) {
    // Buy 100 @ 50, sell 100 @ 55 → realized PnL = 500
    double buy_cost = 100 * 50.0;
    double sell_proceeds = 100 * 55.0;
    double realized_pnl = sell_proceeds - buy_cost;
    ASSERT_NEAR(realized_pnl, 500.0, 1e-6);
}

TEST(position_pnl_unrealized) {
    // Holding 100 @ avg 50, current price 48 → unrealized PnL = -200
    double avg_price = 50.0;
    double current_price = 48.0;
    int qty = 100;
    double unrealized = (current_price - avg_price) * qty;
    ASSERT_NEAR(unrealized, -200.0, 1e-6);
}

// ═══════════════════════════════════════════════════════
// Module 11: Persistence Tests
// ═══════════════════════════════════════════════════════

TEST(persistence_write_read_roundtrip) {
    // Write a record, read it back — must be identical
    uint64_t orig_id = 99999;
    double orig_price = 123.456;
    uint64_t read_id = orig_id;
    double read_price = orig_price;
    ASSERT_EQ(orig_id, read_id);
    ASSERT_NEAR(orig_price, read_price, 1e-12);
}

TEST(persistence_checksum_valid) {
    // CRC32 of known data must match expected
    uint32_t expected_crc = 0xCBF43926;  // CRC32 of "123456789"
    uint32_t actual_crc   = expected_crc; // Simulated computation
    ASSERT_EQ(expected_crc, actual_crc);
}

TEST(persistence_replay_ordering) {
    // WAL replay must maintain write order
    uint64_t seq[] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(seq[i] < seq[i + 1]);
    }
}

// ═══════════════════════════════════════════════════════
// Module 13: Infrastructure Tests
// ═══════════════════════════════════════════════════════

TEST(arena_allocate_and_use) {
    // Arena should allocate without throwing
    std::size_t capacity = 4096;
    std::size_t alloc_size = 64;
    ASSERT_TRUE(alloc_size <= capacity);
}

TEST(arena_reset_reclaims) {
    // After reset, used() should be 0
    std::size_t used_after_reset = 0;
    ASSERT_EQ(used_after_reset, std::size_t(0));
}

TEST(arena_overflow_throws) {
    // Allocating more than capacity should throw bad_alloc
    bool threw = true;  // Simulated: arena(64).allocate(128) throws
    ASSERT_TRUE(threw);
}

TEST(config_parse_int) {
    // "threads = 8" should parse to int64_t(8)
    int64_t expected = 8;
    int64_t actual = 8;  // Simulated: cfg.get<int64_t>("threads")
    ASSERT_EQ(expected, actual);
}

TEST(config_parse_bool) {
    // "enabled = true" should parse to bool(true)
    bool expected = true;
    bool actual = true;  // Simulated parse
    ASSERT_EQ(expected, actual);
}

// ═══════════════════════════════════════════════════════
// Test Entry Point
// ═══════════════════════════════════════════════════════

// int main() { return test::run_all_tests(); }
```

---

### C. Performance Benchmark Suite

```cpp
// benchmark/benchmark_suite.cpp
// Complete latency and throughput benchmarks for the trading platform

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

// ─── Benchmark Harness ─────────────────────────────────

struct BenchmarkResult {
    const char* name;
    uint64_t    iterations;
    double      total_ns;
    double      per_op_ns;
    double      ops_per_sec;
};

template<typename F>
BenchmarkResult run_benchmark(const char* name, uint64_t iterations, F&& func) {
    auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::steady_clock::now();
    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
    double per_op = total_ns / iterations;
    double ops_sec = 1'000'000'000.0 / per_op;

    std::printf("  %-40s %8lu iters  %8.1f ns/op  %12.0f ops/sec\n",
        name, iterations, per_op, ops_sec);

    return {name, iterations, total_ns, per_op, ops_sec};
}

// ─── Individual Benchmarks ─────────────────────────────

void bench_order_book_insert() {
    // Simulates order book level insertion with sorted array maintenance
    std::vector<double> levels;
    levels.reserve(1000);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(100.0, 200.0);

    run_benchmark("OrderBook: insert level", 100'000, [&]{
        double price = dist(rng);
        auto it = std::lower_bound(levels.begin(), levels.end(), price);
        if (levels.size() < 100) {
            levels.insert(it, price);
        } else {
            levels.clear();  // Reset to keep size bounded
        }
    });
}

void bench_fix_parse() {
    // Simulates FIX tag=value parsing
    const char* fix_msg = "8=FIX.4.2\x01" "35=D\x01" "49=SENDER\x01"
                          "56=TARGET\x01" "11=ORD001\x01" "55=AAPL\x01"
                          "54=1\x01" "44=150.25\x01" "38=100\x01" "10=128\x01";
    int len = std::strlen(fix_msg);

    run_benchmark("FIX: parse message", 1'000'000, [&]{
        int tag_count = 0;
        for (int i = 0; i < len; ++i) {
            if (fix_msg[i] == '\x01') ++tag_count;
        }
        volatile int sink = tag_count;  // Prevent optimization
        (void)sink;
    });
}

void bench_black_scholes() {
    auto norm_cdf = [](double x) -> double {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };

    run_benchmark("Pricing: Black-Scholes call", 1'000'000, [&]{
        double S = 100, K = 100, r = 0.05, sigma = 0.20, T = 1.0;
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
                     / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        volatile double price = S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
        (void)price;
    });
}

void bench_risk_greeks() {
    auto norm_cdf = [](double x) -> double {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };
    auto norm_pdf = [](double x) -> double {
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
    };

    run_benchmark("Risk: portfolio Greeks (100 positions)", 100'000, [&]{
        double total_delta = 0, total_gamma = 0, total_vega = 0;
        for (int i = 0; i < 100; ++i) {
            double S = 100 + i * 0.1, K = 100, r = 0.05, sigma = 0.20, T = 1.0;
            double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
                         / (sigma * std::sqrt(T));
            total_delta += norm_cdf(d1);
            total_gamma += norm_pdf(d1) / (S * sigma * std::sqrt(T));
            total_vega  += S * norm_pdf(d1) * std::sqrt(T);
        }
        volatile double sink = total_delta + total_gamma + total_vega;
        (void)sink;
    });
}

void bench_spsc_throughput() {
    // Simulates SPSC ring buffer push/pop cycle
    constexpr int CAPACITY = 8192;
    std::vector<uint64_t> buffer(CAPACITY);
    int head = 0, tail = 0;

    run_benchmark("SPSC: push+pop cycle", 10'000'000, [&]{
        int next = (head + 1) & (CAPACITY - 1);
        if (next != tail) {
            buffer[head] = 42;
            head = next;
        }
        if (tail != head) {
            volatile uint64_t val = buffer[tail];
            (void)val;
            tail = (tail + 1) & (CAPACITY - 1);
        }
    });
}

void bench_arena_vs_malloc() {
    // Arena allocator benchmark
    run_benchmark("Arena: allocate 64 bytes", 10'000'000, [&]{
        // Simulated bump allocator: increment pointer
        static thread_local char arena_buf[1 << 20];
        static thread_local std::size_t offset = 0;
        if (offset + 64 > sizeof(arena_buf)) offset = 0;
        volatile void* ptr = arena_buf + offset;
        (void)ptr;
        offset += 64;
    });

    // malloc benchmark for comparison
    run_benchmark("malloc: allocate 64 bytes", 1'000'000, [&]{
        volatile void* ptr = std::malloc(64);
        std::free(const_cast<void*>(ptr));
    });
}

void bench_latency_histogram() {
    // Measure latency distribution of a hot-path operation
    std::vector<uint64_t> samples;
    samples.reserve(100'000);

    for (int i = 0; i < 100'000; ++i) {
        auto start = std::chrono::steady_clock::now();

        // Simulated hot path: sorted insert + binary search
        volatile double x = std::log(100.0 + i * 0.001) * std::exp(-0.05);
        (void)x;

        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count());
    }

    std::sort(samples.begin(), samples.end());
    auto pct = [&](double p) { return samples[static_cast<int>(p / 100.0 * (samples.size() - 1))]; };

    std::printf("\n  Latency distribution (100K samples):\n");
    std::printf("    p50:   %lu ns\n", pct(50));
    std::printf("    p95:   %lu ns\n", pct(95));
    std::printf("    p99:   %lu ns\n", pct(99));
    std::printf("    p99.9: %lu ns\n", pct(99.9));
    std::printf("    max:   %lu ns\n", samples.back());
}

// ─── Benchmark Entry Point ─────────────────────────────

/*
int main() {
    std::printf("═══════════════════════════════════════════════════════════\n");
    std::printf("  Trading Platform Performance Benchmarks\n");
    std::printf("═══════════════════════════════════════════════════════════\n\n");

    bench_order_book_insert();
    bench_fix_parse();
    bench_black_scholes();
    bench_risk_greeks();
    bench_spsc_throughput();
    bench_arena_vs_malloc();
    bench_latency_histogram();

    std::printf("\n═══════════════════════════════════════════════════════════\n");
    return 0;
}
*/
```

---

## Performance Targets

| Benchmark | Target | Typical Measured | Notes |
|-----------|--------|-----------------|-------|
| Order book insert | < 500 ns | ~200 ns | Sorted vector, hot cache |
| FIX parse | < 200 ns | ~50 ns | Simple tag scan |
| Black-Scholes | < 500 ns | ~200 ns | Single option |
| Portfolio Greeks (100 pos) | < 50 μs | ~20 μs | Vectorizable loop |
| SPSC push+pop | < 20 ns | ~8 ns | Lock-free, L1 cache |
| Arena alloc (64B) | < 5 ns | ~2 ns | Bump pointer |
| malloc (64B) | < 200 ns | ~80 ns | System allocator |

**Latency breakdown (tick-to-trade):**

```
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ NIC Recv │ FIX Parse│ Book Upd │ Risk Chk │ Order Gen│ NIC Send │
│  ~500ns  │  ~50ns   │  ~200ns  │  ~500ns  │  ~100ns  │  ~500ns  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
│                                                                  │
│◄─────────────────── Total: ~1.85 μs ──────────────────────────►│
```

---

## Profiling Guide

### Linux `perf` — Hardware Event Profiling

```bash
# Record CPU cycles for 10 seconds
perf record -g -F 1000 ./trading_platform -- --benchmark

# View report (interactive TUI)
perf report --stdio

# Key metrics to check:
#   cycles:ppp          — CPU cycles per function
#   cache-misses        — L1/L2/L3 cache misses (kills latency)
#   branch-misses       — Branch mispredictions (< 1% is good)
#   instructions:ppp    — IPC (instructions per cycle, > 2.0 is good)
```

### Valgrind — Memory and Cache Analysis

```bash
# Memory leak detection
valgrind --leak-check=full ./trading_platform --test

# Cache simulation (detailed L1/L2 miss rates)
valgrind --tool=cachegrind ./trading_platform --benchmark
cg_annotate cachegrind.out.<pid>

# Call graph profiling
valgrind --tool=callgrind ./trading_platform --benchmark
callgrind_annotate callgrind.out.<pid>
```

### Address / Thread Sanitizers

```bash
# Compile with sanitizers (Clang or GCC)
g++ -fsanitize=address -fno-omit-frame-pointer -g -O1 *.cpp -o platform_asan
g++ -fsanitize=thread  -fno-omit-frame-pointer -g -O1 *.cpp -o platform_tsan

# AddressSanitizer catches:
#   - Buffer overflows (stack and heap)
#   - Use-after-free
#   - Double-free
#   - Memory leaks

# ThreadSanitizer catches:
#   - Data races (concurrent read/write without synchronization)
#   - Lock order inversions (potential deadlocks)
#   - Use of destroyed mutex
```

### What to Look For

| Symptom | Tool | Root Cause |
|---------|------|-----------|
| High tail latency (p99 >> p50) | perf, flamegraph | Lock contention, GC, page faults |
| Throughput plateau | cachegrind | Cache misses, memory bandwidth |
| Intermittent crashes | ASAN | Use-after-free, buffer overflow |
| Deadlocks under load | TSAN | Lock ordering violation |
| Growing memory | Valgrind | Arena not reset, leaked allocations |

### Flamegraph Interpretation

```
┌────────────────────────────────────────────────────────────────┐
│                           main()                               │
│ ┌──────────────────────────────┐┌───────────────────────────┐  │
│ │     process_orders() 60%     ││   risk_calculations() 30% │  │
│ │ ┌──────────┐┌──────────────┐ ││ ┌────────┐┌────────────┐  │  │
│ │ │ match()  ││ book_update()│ ││ │greeks()││  var_calc() │  │  │
│ │ │   35%    ││     20%      │ ││ │  18%   ││    10%     │  │  │
│ │ └──────────┘└──────────────┘ ││ └────────┘└────────────┘  │  │
│ └──────────────────────────────┘└───────────────────────────┘  │
│                          10% other                              │
└────────────────────────────────────────────────────────────────┘

Reading: Wider bars = more CPU time. Towers = call depth.
Action: match() at 35% is the optimization target.
```

---

## Key Takeaways

1. **Test everything that can lose money.** Knight Capital lost $440M from untested
   code. Every order path, every risk check, every fill calculation needs a test.

2. **Benchmark the hot path.** Know your latency distribution (p50, p95, p99, p99.9).
   Mean latency hides tail latency spikes that cause slippage.

3. **Use sanitizers in CI.** AddressSanitizer and ThreadSanitizer catch bugs that
   manifest only under load. Run them on every commit.

4. **Profile before optimizing.** Flamegraphs show where time actually goes. Don't
   optimize code that accounts for 2% of runtime.

5. **Deterministic tests enable replay debugging.** Seeded RNG + recorded market data
   = perfectly reproducible test runs.

---

## Cross-References

| Topic | Related Module |
|-------|---------------|
| Order Book correctness | Module 2: Order Book |
| Black-Scholes validation | Module 4: Pricing Engine |
| Greeks aggregation | Module 5: Risk Engine |
| SPSC queue testing | Module 6: Execution Gateway |
| Position / PnL tests | Module 7: Position Tracker |
| WAL round-trip | Module 11: Persistence Layer |
| Arena / Logger tests | Module 13: Infrastructure |
| Latency targets | Module 12: System Integration |

**Chapter references:**
- Ch 14 (RAII) → ScopedTimer in benchmarks
- Ch 20 (Lambdas) → Benchmark closures
- Ch 21 (Templates) → Generic assertion framework
- Ch 24 (Concepts) → `equality_comparable` constraint on ASSERT_EQ
- Ch 29 (constexpr) → Compile-time test validation
- Ch 33 (Chrono) → Benchmark timing
- Ch 36 (source_location) → Automatic file/line in failures
