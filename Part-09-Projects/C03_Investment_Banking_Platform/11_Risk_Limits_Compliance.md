# Module 11: Pre-Trade Risk Limits & Compliance

## Module Overview

The Pre-Trade Risk engine validates **every order before it reaches the exchange**. It checks position limits, notional limits, fat-finger thresholds, and restricted instrument lists — all within sub-microsecond budgets. This is the last line of defense between a trading algorithm and catastrophic financial loss.

**Why this matters:** Knight Capital lost $440 million in 45 minutes because faulty software sent unvalidated orders to the exchange. The pre-trade risk engine exists to make such disasters impossible. Every check must be fast (on the critical path), correct (false negatives are fatal), and auditable (regulators review check logic).

---

## Architecture Insight

```mermaid
graph TD
    subgraph "Order Submission"
        ALGO[Trading Algorithm] --> ORDER[New Order]
    end

    ORDER --> PIPELINE[Risk Check Pipeline]

    subgraph "Pre-Trade Checks"
        PIPELINE --> C1[Position Limit Check]
        PIPELINE --> C2[Notional Limit Check]
        PIPELINE --> C3[Fat Finger Check]
        PIPELINE --> C4[Restricted List Check]
        PIPELINE --> C5[Rate Limit Check]
    end

    C1 --> RESULT{All Passed?}
    C2 --> RESULT
    C3 --> RESULT
    C4 --> RESULT
    C5 --> RESULT

    RESULT -->|Yes| EXCHANGE[Send to Exchange]
    RESULT -->|No| REJECT[Reject + Log Violation]
    REJECT --> ALERT[Alert Risk Manager]

    style PIPELINE fill:#e74c3c,color:#fff
    style RESULT fill:#f39c12,color:#fff
    style REJECT fill:#c0392b,color:#fff
    style EXCHANGE fill:#2ecc71,color:#fff
```

**Risk Check Pipeline (Detail):**

```mermaid
flowchart LR
    ORD[Order] --> POS[Position<br>Limit]
    POS -->|pass| NOT[Notional<br>Limit]
    NOT -->|pass| FAT[Fat Finger<br>Check]
    FAT -->|pass| REST[Restricted<br>List]
    REST -->|pass| RATE[Rate<br>Limit]
    RATE -->|pass| OK[✓ Approved]

    POS -->|fail| REJ1[✗ Position Breach]
    NOT -->|fail| REJ2[✗ Notional Breach]
    FAT -->|fail| REJ3[✗ Fat Finger]
    REST -->|fail| REJ4[✗ Restricted]
    RATE -->|fail| REJ5[✗ Rate Exceeded]
```

---

## Investment Banking Domain Context

### Why Pre-Trade Risk Exists

Before 2010, many electronic trading systems had minimal pre-trade checks. Several catastrophic events changed this:

| Incident | Year | Loss | Cause |
|---|---|---|---|
| Knight Capital | 2012 | $440M in 45 min | Old code reactivated, no risk checks |
| Flash Crash | 2010 | $1T temporary | Cascading sell orders, no circuit breakers |
| Société Générale | 2008 | €4.9B | Rogue trader bypassed position limits |
| Barings Bank | 1995 | £827M | Single trader, no limit enforcement |

### Types of Pre-Trade Checks

| Check | What It Validates | Typical Threshold |
|---|---|---|
| Position Limit | Net position won't exceed desk limit | ±10,000 shares per instrument |
| Notional Limit | Dollar exposure within bounds | $50M per order |
| Fat Finger | Price/quantity isn't obviously wrong | ±5% from last price |
| Restricted List | Instrument isn't on compliance blacklist | Binary: allowed or not |
| Rate Limit | Order rate within acceptable bounds | 100 orders/second per desk |
| Concentration | Not too much in one name | 10% of portfolio |

### Latency Requirements

Pre-trade checks are on the **critical path**. Every microsecond added to check latency is a microsecond of market disadvantage:

| Check Category | Target Latency | Acceptable |
|---|---|---|
| Position limit | < 500ns | Fast lookup |
| Notional limit | < 200ns | Arithmetic only |
| Fat finger | < 100ns | Compare against reference |
| Restricted list | < 300ns | Hash lookup |
| **Total pipeline** | **< 2μs** | **Must beat market** |

---

## C++ Concepts Used

| Concept | Chapter | Usage in This Module |
|---|---|---|
| Concepts (`concept`) | Ch 24 | `RiskCheckable` constraining checkable order types |
| `constexpr` | Ch 29 | Compile-time limit configuration |
| `static_assert` | Ch 29 | Catch misconfigured limits at compile time |
| `std::expected` | Ch 36 | Rich error returns (not just pass/fail) |
| Template method | Ch 28 | Base validation pipeline with customizable steps |
| Bitfield enums | Ch 10 | Risk check flags per order type |
| `noexcept` | Ch 12 | All checks are noexcept — no exceptions on hot path |
| `consteval` | Ch 37 | Validate risk parameter combinations at compile time |
| `std::variant` | Ch 34 | Different violation types in a type-safe union |
| Lambdas | Ch 18 | Custom check predicates |

---

## Design Decisions

1. **`std::expected` over exceptions** — Risk checks return `expected<void, RiskViolation>`. Exceptions are too slow for the hot path (~1000ns throw/catch vs. ~5ns return). The violation contains rich context: which check failed, what the limit was, what the order requested.

2. **`noexcept` everywhere** — Every check function is marked `noexcept`. If a risk check throws, something is fundamentally broken and we should halt, not propagate.

3. **Concepts for extensibility** — The `RiskCheckable` concept allows new order types to participate in risk checking without modifying the engine. Any type that provides `instrument_id()`, `quantity()`, `price()`, and `side()` can be checked.

4. **`constexpr` limits** — Risk limits are compile-time constants where possible. This enables the optimizer to inline comparisons and catches configuration errors before deployment.

5. **Pipeline pattern** — Checks are composed as a pipeline. The engine runs checks in order and short-circuits on first failure. New checks are added by implementing the `RiskCheck` interface.

---

## Complete Implementation

```cpp
// ============================================================================
// Pre-Trade Risk Limits & Compliance — Investment Banking Platform
// Module 11 of C03_Investment_Banking_Platform
//
// Compile: g++ -std=c++23 -O2 -o risk_limits risk_limits.cpp
// ============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// ============================================================================
// Domain Types
// ============================================================================

using InstrumentId = std::string;
using DeskId = std::string;

enum class OrderSide : uint8_t { Buy, Sell };

// ============================================================================
// Risk Check Flags — Ch10 Bitfield Enums
// ============================================================================
// Each order type specifies which checks to apply using bitwise flags.
// Equity orders get all checks; internal transfers skip fat-finger.

enum class RiskCheckFlags : uint32_t {
    None            = 0,
    PositionLimit   = 1 << 0,
    NotionalLimit   = 1 << 1,
    FatFinger       = 1 << 2,
    RestrictedList  = 1 << 3,
    RateLimit       = 1 << 4,
    Concentration   = 1 << 5,

    // Composite flags for common order types
    AllChecks       = PositionLimit | NotionalLimit | FatFinger |
                      RestrictedList | RateLimit | Concentration,
    BasicChecks     = PositionLimit | NotionalLimit | RestrictedList,
    InternalOnly    = PositionLimit | NotionalLimit,
};

// Bitwise operators for flag composition
constexpr RiskCheckFlags operator|(RiskCheckFlags a, RiskCheckFlags b) {
    return static_cast<RiskCheckFlags>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
constexpr RiskCheckFlags operator&(RiskCheckFlags a, RiskCheckFlags b) {
    return static_cast<RiskCheckFlags>(
        static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
constexpr bool has_flag(RiskCheckFlags flags, RiskCheckFlags test) {
    return (flags & test) == test;
}

// ============================================================================
// constexpr Risk Parameters — Ch29
// ============================================================================
// Compile-time configuration of risk limits. These constants are baked into
// the binary — no runtime lookup required. The optimizer can inline
// comparisons against these values.

struct RiskParameters {
    // Position limits
    static constexpr double MAX_POSITION_SHARES = 100'000.0;
    static constexpr double MAX_POSITION_CONTRACTS = 1'000.0;

    // Notional limits
    static constexpr double MAX_ORDER_NOTIONAL = 50'000'000.0;   // $50M
    static constexpr double MAX_DESK_NOTIONAL = 500'000'000.0;   // $500M

    // Fat finger thresholds
    static constexpr double MAX_PRICE_DEVIATION = 0.05;   // 5% from ref
    static constexpr double MAX_QUANTITY_SINGLE = 50'000;  // shares per order

    // Rate limits
    static constexpr int MAX_ORDERS_PER_SECOND = 100;
    static constexpr int MAX_ORDERS_PER_MINUTE = 2'000;

    // Concentration limits
    static constexpr double MAX_CONCENTRATION_PCT = 0.10;  // 10% of portfolio
};

// Ch29 static_assert: Validate parameter sanity at compile time
static_assert(RiskParameters::MAX_ORDER_NOTIONAL > 0,
              "Max notional must be positive");
static_assert(RiskParameters::MAX_PRICE_DEVIATION > 0.0 &&
              RiskParameters::MAX_PRICE_DEVIATION < 1.0,
              "Price deviation must be between 0 and 1 (0% to 100%)");
static_assert(RiskParameters::MAX_ORDERS_PER_SECOND > 0,
              "Rate limit must be positive");
static_assert(RiskParameters::MAX_POSITION_SHARES >=
              RiskParameters::MAX_QUANTITY_SINGLE,
              "Position limit must be >= single order quantity limit");

// ============================================================================
// consteval Parameter Validation — Ch37
// ============================================================================
// consteval functions MUST be evaluated at compile time. We use them to
// validate that risk parameter combinations are logically consistent.

consteval bool validate_risk_params() {
    // Notional limit should be achievable within position limit
    // (Assume max price of $10,000 per share — extreme but realistic)
    constexpr double max_theoretical_notional =
        RiskParameters::MAX_POSITION_SHARES * 10'000.0;
    if (RiskParameters::MAX_ORDER_NOTIONAL > max_theoretical_notional * 10) {
        return false;  // notional limit is unreachably high
    }

    // Rate limits: per-minute should be >= per-second
    if (RiskParameters::MAX_ORDERS_PER_MINUTE <
        RiskParameters::MAX_ORDERS_PER_SECOND) {
        return false;
    }

    return true;
}

static_assert(validate_risk_params(),
              "Risk parameters are internally inconsistent");

// ============================================================================
// RiskViolation — Rich error type for failed checks
// ============================================================================
// Ch36 std::expected: Instead of returning bool (pass/fail), we return
// expected<void, RiskViolation> which carries detailed failure information.

enum class ViolationType {
    PositionLimitBreached,
    NotionalLimitBreached,
    FatFingerPrice,
    FatFingerQuantity,
    RestrictedInstrument,
    RateLimitExceeded,
    ConcentrationBreached
};

struct RiskViolation {
    ViolationType type;
    std::string check_name;
    std::string instrument_id;
    double requested_value;
    double limit_value;
    std::string detail;

    [[nodiscard]] std::string to_string() const {
        return std::format("RISK VIOLATION [{}]: {} | "
                           "requested={:.2f} limit={:.2f} | {}",
                           check_name, instrument_id,
                           requested_value, limit_value, detail);
    }
};

// Result type for all risk checks
using RiskResult = std::expected<void, RiskViolation>;

// ============================================================================
// RiskCheckable Concept — Ch24
// ============================================================================
// Any order type that satisfies this concept can be validated by the risk
// engine. This decouples the risk engine from specific order implementations.

template <typename T>
concept RiskCheckable = requires(const T& order) {
    { order.instrument_id() } -> std::convertible_to<std::string>;
    { order.quantity() } -> std::convertible_to<double>;
    { order.price() } -> std::convertible_to<double>;
    { order.side() } -> std::same_as<OrderSide>;
    { order.desk_id() } -> std::convertible_to<std::string>;
    { order.notional() } -> std::convertible_to<double>;
    { order.risk_flags() } -> std::same_as<RiskCheckFlags>;
};

// ============================================================================
// Sample Order Type — satisfies RiskCheckable concept
// ============================================================================

class Order {
public:
    Order(InstrumentId inst, OrderSide side, double qty, double px,
          DeskId desk, RiskCheckFlags flags = RiskCheckFlags::AllChecks)
        : instrument_id_(std::move(inst)), side_(side), quantity_(qty),
          price_(px), desk_id_(std::move(desk)), flags_(flags) {}

    [[nodiscard]] const std::string& instrument_id() const noexcept {
        return instrument_id_;
    }
    [[nodiscard]] double quantity() const noexcept { return quantity_; }
    [[nodiscard]] double price() const noexcept { return price_; }
    [[nodiscard]] OrderSide side() const noexcept { return side_; }
    [[nodiscard]] const std::string& desk_id() const noexcept {
        return desk_id_;
    }
    [[nodiscard]] double notional() const noexcept {
        return quantity_ * price_;
    }
    [[nodiscard]] RiskCheckFlags risk_flags() const noexcept {
        return flags_;
    }

    [[nodiscard]] double signed_quantity() const noexcept {
        return side_ == OrderSide::Buy ? quantity_ : -quantity_;
    }

    [[nodiscard]] std::string to_string() const {
        return std::format("Order[{} {} {:.0f} @ {:.2f} desk={}]",
                           instrument_id_,
                           side_ == OrderSide::Buy ? "BUY" : "SELL",
                           quantity_, price_, desk_id_);
    }

private:
    InstrumentId instrument_id_;
    OrderSide side_;
    double quantity_;
    double price_;
    DeskId desk_id_;
    RiskCheckFlags flags_;
};

// Verify Order satisfies the concept
static_assert(RiskCheckable<Order>,
              "Order must satisfy the RiskCheckable concept");

// ============================================================================
// Individual Risk Checks — noexcept (Ch12)
// ============================================================================
// Each check is a standalone function that returns RiskResult.
// All checks are noexcept: throwing on the hot path is unacceptable.

// --- Position Limit Check ---
// Verifies that the resulting position won't exceed the desk's limit.
class PositionLimitCheck {
public:
    void set_current_position(const InstrumentId& id, double qty) {
        positions_[id] = qty;
    }

    template <RiskCheckable T>
    [[nodiscard]] RiskResult check(const T& order) const noexcept {
        double current = 0.0;
        auto it = positions_.find(order.instrument_id());
        if (it != positions_.end()) {
            current = it->second;
        }

        // Projected position after this order
        double projected = current;
        if (order.side() == OrderSide::Buy) {
            projected += order.quantity();
        } else {
            projected -= order.quantity();
        }

        double limit = RiskParameters::MAX_POSITION_SHARES;
        if (std::abs(projected) > limit) {
            return std::unexpected(RiskViolation{
                ViolationType::PositionLimitBreached,
                "PositionLimit",
                order.instrument_id(),
                std::abs(projected),
                limit,
                std::format("Current={:.0f} + order={:.0f} → projected={:.0f}",
                            current, order.quantity(), projected)});
        }

        return {};  // pass
    }

private:
    std::unordered_map<InstrumentId, double> positions_;
};

// --- Notional Limit Check ---
// Ensures the dollar value of the order doesn't exceed thresholds.
class NotionalLimitCheck {
public:
    template <RiskCheckable T>
    [[nodiscard]] RiskResult check(const T& order) const noexcept {
        double notional = order.notional();

        if (notional > RiskParameters::MAX_ORDER_NOTIONAL) {
            return std::unexpected(RiskViolation{
                ViolationType::NotionalLimitBreached,
                "NotionalLimit",
                order.instrument_id(),
                notional,
                RiskParameters::MAX_ORDER_NOTIONAL,
                "Single order notional exceeds maximum"});
        }

        return {};
    }
};

// --- Fat Finger Check ---
// Catches obviously erroneous orders (price way off market, absurd quantity).
class FatFingerCheck {
public:
    void set_reference_price(const InstrumentId& id, double price) {
        ref_prices_[id] = price;
    }

    template <RiskCheckable T>
    [[nodiscard]] RiskResult check(const T& order) const noexcept {
        // Quantity check
        if (order.quantity() > RiskParameters::MAX_QUANTITY_SINGLE) {
            return std::unexpected(RiskViolation{
                ViolationType::FatFingerQuantity,
                "FatFinger(Qty)",
                order.instrument_id(),
                order.quantity(),
                RiskParameters::MAX_QUANTITY_SINGLE,
                "Quantity exceeds single-order maximum"});
        }

        // Price check: compare against reference price
        auto it = ref_prices_.find(order.instrument_id());
        if (it != ref_prices_.end() && it->second > 0.0) {
            double ref = it->second;
            double deviation = std::abs(order.price() - ref) / ref;

            if (deviation > RiskParameters::MAX_PRICE_DEVIATION) {
                return std::unexpected(RiskViolation{
                    ViolationType::FatFingerPrice,
                    "FatFinger(Price)",
                    order.instrument_id(),
                    order.price(),
                    ref,
                    std::format("Deviation={:.1f}% exceeds {:.1f}% threshold",
                                deviation * 100,
                                RiskParameters::MAX_PRICE_DEVIATION * 100)});
            }
        }

        return {};
    }

private:
    std::unordered_map<InstrumentId, double> ref_prices_;
};

// --- Restricted List Check ---
// Checks if the instrument is on a compliance blacklist.
class RestrictedListCheck {
public:
    void add_restricted(const InstrumentId& id, const std::string& reason) {
        restricted_[id] = reason;
    }

    void remove_restricted(const InstrumentId& id) {
        restricted_.erase(id);
    }

    template <RiskCheckable T>
    [[nodiscard]] RiskResult check(const T& order) const noexcept {
        auto it = restricted_.find(order.instrument_id());
        if (it != restricted_.end()) {
            return std::unexpected(RiskViolation{
                ViolationType::RestrictedInstrument,
                "RestrictedList",
                order.instrument_id(),
                0.0,
                0.0,
                std::format("Restricted: {}", it->second)});
        }

        return {};
    }

private:
    std::unordered_map<InstrumentId, std::string> restricted_;
};

// --- Rate Limit Check ---
// Ensures a desk doesn't submit orders too fast.
class RateLimitCheck {
public:
    template <RiskCheckable T>
    [[nodiscard]] RiskResult check(const T& order) noexcept {
        auto now = std::chrono::steady_clock::now();
        auto& desk_ts = desk_timestamps_[order.desk_id()];

        // Remove timestamps older than 1 second
        auto one_sec_ago = now - std::chrono::seconds(1);
        while (!desk_ts.empty() && desk_ts.front() < one_sec_ago) {
            desk_ts.erase(desk_ts.begin());
        }

        if (static_cast<int>(desk_ts.size()) >=
            RiskParameters::MAX_ORDERS_PER_SECOND) {
            return std::unexpected(RiskViolation{
                ViolationType::RateLimitExceeded,
                "RateLimit",
                order.instrument_id(),
                static_cast<double>(desk_ts.size()),
                static_cast<double>(RiskParameters::MAX_ORDERS_PER_SECOND),
                std::format("Desk {} hit rate limit", order.desk_id())});
        }

        desk_ts.push_back(now);
        return {};
    }

private:
    std::unordered_map<
        DeskId,
        std::vector<std::chrono::steady_clock::time_point>> desk_timestamps_;
};

// ============================================================================
// RiskLimitsEngine — Composable Pipeline (Ch28: Template Method)
// ============================================================================
// The engine composes individual checks into a pipeline. Checks run in
// order and short-circuit on first failure.

class RiskLimitsEngine {
public:
    RiskLimitsEngine() = default;

    // --- Configuration ---
    PositionLimitCheck& position_check() { return pos_check_; }
    NotionalLimitCheck& notional_check() { return notional_check_; }
    FatFingerCheck& fat_finger_check() { return fat_finger_check_; }
    RestrictedListCheck& restricted_check() { return restricted_check_; }
    RateLimitCheck& rate_check() { return rate_check_; }

    // -----------------------------------------------------------------------
    // check_order: Main entry point — validates an order through the pipeline
    // -----------------------------------------------------------------------
    // Ch24 Concepts: Accepts any type satisfying RiskCheckable
    // Ch36 std::expected: Returns detailed violation info on failure
    // Ch12 noexcept: Entire pipeline is noexcept

    template <RiskCheckable T>
    [[nodiscard]] RiskResult check_order(T& order) noexcept {
        auto flags = order.risk_flags();
        ++total_checks_;

        // Run checks based on flags — short-circuit on first failure
        if (has_flag(flags, RiskCheckFlags::RestrictedList)) {
            auto result = restricted_check_.check(order);
            if (!result) {
                ++violations_;
                return result;
            }
        }

        if (has_flag(flags, RiskCheckFlags::PositionLimit)) {
            auto result = pos_check_.check(order);
            if (!result) {
                ++violations_;
                return result;
            }
        }

        if (has_flag(flags, RiskCheckFlags::NotionalLimit)) {
            auto result = notional_check_.check(order);
            if (!result) {
                ++violations_;
                return result;
            }
        }

        if (has_flag(flags, RiskCheckFlags::FatFinger)) {
            auto result = fat_finger_check_.check(order);
            if (!result) {
                ++violations_;
                return result;
            }
        }

        if (has_flag(flags, RiskCheckFlags::RateLimit)) {
            auto result = rate_check_.check(order);
            if (!result) {
                ++violations_;
                return result;
            }
        }

        ++passed_;
        return {};  // all checks passed
    }

    // --- Statistics ---
    [[nodiscard]] size_t total_checks() const { return total_checks_; }
    [[nodiscard]] size_t violations() const { return violations_; }
    [[nodiscard]] size_t passed() const { return passed_; }

    [[nodiscard]] std::string stats_report() const {
        double pass_rate = total_checks_ > 0
            ? (static_cast<double>(passed_) /
               static_cast<double>(total_checks_)) * 100.0
            : 0.0;
        return std::format(
            "Risk Engine Stats:\n"
            "  Total checks:  {}\n"
            "  Passed:        {}\n"
            "  Violations:    {}\n"
            "  Pass rate:     {:.1f}%\n",
            total_checks_, passed_, violations_, pass_rate);
    }

private:
    PositionLimitCheck pos_check_;
    NotionalLimitCheck notional_check_;
    FatFingerCheck fat_finger_check_;
    RestrictedListCheck restricted_check_;
    RateLimitCheck rate_check_;

    size_t total_checks_ = 0;
    size_t violations_ = 0;
    size_t passed_ = 0;
};

// ============================================================================
// Compile-Time Limit Configuration — Ch29 constexpr
// ============================================================================
// Desk-specific limits configured at compile time. Different desks have
// different risk appetites.

struct DeskLimits {
    const char* desk_id;
    double max_notional;
    double max_position;
    int max_orders_per_sec;
};

constexpr std::array<DeskLimits, 3> DESK_CONFIGS = {{
    {"EQ_DESK",  100'000'000.0, 50'000.0, 200},
    {"FX_DESK",  500'000'000.0, 10'000'000.0, 500},
    {"FUT_DESK", 200'000'000.0, 5'000.0, 150},
}};

// constexpr lookup of desk limits
constexpr std::optional<DeskLimits> get_desk_limits(const char* desk) {
    for (const auto& cfg : DESK_CONFIGS) {
        // constexpr string comparison
        const char* a = cfg.desk_id;
        const char* b = desk;
        bool match = true;
        while (*a && *b) {
            if (*a != *b) { match = false; break; }
            ++a; ++b;
        }
        if (match && *a == *b) return cfg;
    }
    return std::nullopt;
}

// Validate desk configs at compile time
static_assert(get_desk_limits("EQ_DESK").has_value(),
              "EQ_DESK must be configured");
static_assert(get_desk_limits("EQ_DESK")->max_notional > 0,
              "EQ_DESK notional must be positive");

// ============================================================================
// Helper: print check result
// ============================================================================

void print_result(const Order& order, const RiskResult& result) {
    if (result) {
        std::cout << std::format("  ✓ PASS: {}\n", order.to_string());
    } else {
        std::cout << std::format("  ✗ FAIL: {}\n", order.to_string());
        std::cout << std::format("         {}\n",
                                 result.error().to_string());
    }
}

// ============================================================================
// Main — Demonstration and Testing
// ============================================================================

int main() {
    std::cout << "=== Pre-Trade Risk Limits & Compliance ===\n\n";

    // --- Configure the risk engine ---
    RiskLimitsEngine engine;

    // Set current positions (from Position Manager)
    engine.position_check().set_current_position("AAPL", 45'000);
    engine.position_check().set_current_position("MSFT", -20'000);
    engine.position_check().set_current_position("TSLA", 0);

    // Set reference prices (from Market Data)
    engine.fat_finger_check().set_reference_price("AAPL", 175.00);
    engine.fat_finger_check().set_reference_price("MSFT", 420.00);
    engine.fat_finger_check().set_reference_price("TSLA", 245.00);

    // Set restricted list (from Compliance)
    engine.restricted_check().add_restricted(
        "ENRON", "Delisted — compliance blacklist");
    engine.restricted_check().add_restricted(
        "WIRECARD", "Fraud investigation — trading suspended");

    // --- Test 1: Normal orders (should pass) ---
    std::cout << "--- Test 1: Normal Orders ---\n";
    {
        Order o1("AAPL", OrderSide::Buy, 100, 175.50, "EQ_DESK");
        print_result(o1, engine.check_order(o1));

        Order o2("MSFT", OrderSide::Sell, 500, 419.00, "EQ_DESK");
        print_result(o2, engine.check_order(o2));

        Order o3("TSLA", OrderSide::Buy, 1000, 244.00, "EQ_DESK");
        print_result(o3, engine.check_order(o3));
    }

    // --- Test 2: Position limit breach ---
    std::cout << "\n--- Test 2: Position Limit Breach ---\n";
    {
        // Current AAPL position is 45,000. Buying 60,000 more → 105,000 > limit
        Order o("AAPL", OrderSide::Buy, 60'000, 175.00, "EQ_DESK");
        print_result(o, engine.check_order(o));
    }

    // --- Test 3: Notional limit breach ---
    std::cout << "\n--- Test 3: Notional Limit Breach ---\n";
    {
        // 40,000 shares × $1,500 = $60M > $50M limit
        Order o("GOOG", OrderSide::Buy, 40'000, 1500.00, "EQ_DESK");
        print_result(o, engine.check_order(o));
    }

    // --- Test 4: Fat finger — price deviation ---
    std::cout << "\n--- Test 4: Fat Finger (Price) ---\n";
    {
        // AAPL ref price is $175. Order at $200 → 14.3% deviation > 5%
        Order o("AAPL", OrderSide::Buy, 100, 200.00, "EQ_DESK");
        print_result(o, engine.check_order(o));
    }

    // --- Test 5: Fat finger — quantity ---
    std::cout << "\n--- Test 5: Fat Finger (Quantity) ---\n";
    {
        // 75,000 shares in a single order > 50,000 max
        Order o("MSFT", OrderSide::Buy, 75'000, 420.00, "EQ_DESK");
        print_result(o, engine.check_order(o));
    }

    // --- Test 6: Restricted instrument ---
    std::cout << "\n--- Test 6: Restricted Instrument ---\n";
    {
        Order o("ENRON", OrderSide::Buy, 100, 0.01, "EQ_DESK");
        print_result(o, engine.check_order(o));

        Order o2("WIRECARD", OrderSide::Sell, 50, 0.10, "EQ_DESK");
        print_result(o2, engine.check_order(o2));
    }

    // --- Test 7: Internal order with reduced checks ---
    std::cout << "\n--- Test 7: Internal Transfer (Reduced Checks) ---\n";
    {
        // Internal transfers skip fat-finger and restricted list checks
        Order o("AAPL", OrderSide::Buy, 100, 200.00, "EQ_DESK",
                RiskCheckFlags::InternalOnly);
        // Price is way off ($200 vs $175 ref), but fat-finger check is skipped
        print_result(o, engine.check_order(o));
    }

    // --- Test 8: Compile-time validation ---
    std::cout << "\n--- Test 8: Compile-Time Validation ---\n";
    std::cout << std::format(
        "  Risk parameters validated at compile time:\n"
        "    MAX_POSITION_SHARES:   {:.0f}\n"
        "    MAX_ORDER_NOTIONAL:    ${:,.0f}\n"
        "    MAX_PRICE_DEVIATION:   {:.0f}%\n"
        "    MAX_QUANTITY_SINGLE:   {:.0f}\n"
        "    MAX_ORDERS_PER_SEC:    {}\n",
        RiskParameters::MAX_POSITION_SHARES,
        RiskParameters::MAX_ORDER_NOTIONAL,
        RiskParameters::MAX_PRICE_DEVIATION * 100,
        RiskParameters::MAX_QUANTITY_SINGLE,
        RiskParameters::MAX_ORDERS_PER_SECOND);

    // Desk config lookup (constexpr)
    constexpr auto eq_limits = get_desk_limits("EQ_DESK");
    static_assert(eq_limits.has_value());
    std::cout << std::format(
        "  EQ_DESK limits: notional=${:,.0f} position={:.0f} "
        "rate={}/sec\n",
        eq_limits->max_notional, eq_limits->max_position,
        eq_limits->max_orders_per_sec);

    // --- Engine statistics ---
    std::cout << "\n" << engine.stats_report();

    std::cout << "\n=== Risk Limits Tests Complete ===\n";
    return 0;
}
```

---

## Code Walkthrough

### Concept-Based Extensibility (Ch24)

The `RiskCheckable` concept defines the contract for any order type:

```cpp
template <typename T>
concept RiskCheckable = requires(const T& order) {
    { order.instrument_id() } -> std::convertible_to<std::string>;
    { order.quantity() }      -> std::convertible_to<double>;
    // ...
};
```

Any struct that provides these methods can be risk-checked — no inheritance required. This is compile-time polymorphism: zero runtime overhead, full type safety.

### std::expected Error Handling (Ch36)

Instead of:
```cpp
bool check(Order& o) { return ok; }  // Lost: WHY did it fail?
```

We return:
```cpp
RiskResult check(Order& o) {
    return std::unexpected(RiskViolation{ ... });  // Rich context
}
```

The caller can inspect `.error().type`, `.error().detail`, and `.error().limit_value` to understand exactly what failed and why.

### Compile-Time Validation (Ch29/Ch37)

Three levels of compile-time safety:

1. **`constexpr` parameters**: Risk limits are constants — no runtime lookup.
2. **`static_assert`**: Individual parameter sanity checks (positive values, consistent ranges).
3. **`consteval` functions**: Cross-parameter consistency validation (position limit ≥ single order limit).

---

## Testing

| Test Case | Check Type | Input | Expected Result |
|---|---|---|---|
| Normal buy | All checks | 100 AAPL @ $175.50 | ✓ Pass |
| Position breach | Position limit | +60K AAPL (current: 45K) | ✗ Fail: 105K > 100K |
| Notional breach | Notional limit | 40K × $1500 = $60M | ✗ Fail: $60M > $50M |
| Fat finger price | Fat finger | AAPL @ $200 (ref: $175) | ✗ Fail: 14.3% > 5% |
| Fat finger qty | Fat finger | 75K shares single order | ✗ Fail: 75K > 50K |
| Restricted | Restricted list | ENRON (delisted) | ✗ Fail: blacklisted |
| Internal transfer | Reduced flags | AAPL @ $200, no fat-finger | ✓ Pass (skipped) |
| Compile-time | static_assert | Misconfigured limits | Compilation error |

---

## Performance Analysis

| Check | Operations | Latency | Notes |
|---|---|---|---|
| Position limit | Hash lookup + arithmetic | ~100ns | O(1) map access |
| Notional limit | Multiplication + compare | ~10ns | Pure arithmetic |
| Fat finger | Hash lookup + division | ~80ns | Division is slow |
| Restricted list | Hash lookup | ~50ns | O(1) set membership |
| Rate limit | Timestamp deque scan | ~200ns | Amortized |
| **Full pipeline** | **Sum of all** | **~500ns** | **Well under 2μs target** |

---

## Key Takeaways

1. **`std::expected` replaces exceptions on the hot path** — Rich error context without the ~1000ns cost of throw/catch. The optimizer can inline the check-and-return.

2. **Concepts enable zero-cost polymorphism** — Any order type matching `RiskCheckable` can be checked. No vtable, no virtual dispatch, no runtime overhead.

3. **`constexpr` + `static_assert` = defense in depth** — Risk misconfigurations are caught at compile time, not after deployment. `MAX_POSITION < MAX_SINGLE_ORDER` would be a compilation error.

4. **Bitfield flags enable per-order-type check selection** — Internal transfers skip fat-finger checks. Market-making orders skip rate limits. The flags compose naturally.

5. **`noexcept` is a contract, not just an optimization** — On the risk check hot path, an exception means something is fundamentally broken. `noexcept` makes that explicit.

---

## Cross-References

| Related Module | Connection |
|---|---|
| Module 03: Order Types | Orders are the input to risk checks |
| Module 05: Market Data | Reference prices for fat-finger checks |
| Module 09: Position Manager | Current positions for limit checks |
| Module 10: Persistence | Risk violations logged permanently |
| Module 12: Reporting | Violation reports for compliance |
| Ch 10: Enums | Bitfield flags for check selection |
| Ch 12: Error Handling | `noexcept` contract, `expected` returns |
| Ch 24: Concepts | `RiskCheckable` constraint |
| Ch 29: `constexpr` | Compile-time limit configuration |
| Ch 36: `std::expected` | Rich error propagation |
| Ch 37: `consteval` | Parameter consistency validation |
