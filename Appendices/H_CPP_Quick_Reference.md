# Appendix H — Modern C++ Quick Reference (C++17 / 20 / 23)

> **Print-friendly cheat sheet.** Syntax and key facts — not a tutorial.

---

## 1. Types & Literals

### Primitive Types (typical x86-64)

| Type | Bytes | Range / Notes |
|------|-------|---------------|
| `bool` | 1 | `true` / `false` |
| `char` | 1 | −128 … 127 (signed on most platforms) |
| `char8_t` (C++20) | 1 | UTF-8 code unit |
| `short` | 2 | −32 768 … 32 767 |
| `int` | 4 | −2³¹ … 2³¹−1 |
| `long` | 8 (LP64) / 4 (Win) | Platform-dependent |
| `long long` | 8 | −2⁶³ … 2⁶³−1 |
| `float` | 4 | ~7 digits, IEEE 754 |
| `double` | 8 | ~15 digits, IEEE 754 |
| `std::size_t` | 8 (64-bit) | Unsigned, for sizes/indices |
| `std::int32_t` | 4 | Fixed-width from `<cstdint>` |

### Type Deduction & Aliases

```cpp
auto x = 42;                     // int
decltype(x) w = x;               // same type as x
decltype(auto) r = func();       // preserves ref/cv qualifiers
using Vec = std::vector<int>;    // alias (preferred over typedef)
template <typename T>
using Matrix = std::vector<std::vector<T>>;  // alias template
```

### Literal Suffixes

| Suffix | Type | Example |
|--------|------|---------|
| `f` | `float` | `3.14f` |
| `L` | `long` / `long double` | `42L`, `3.14L` |
| `ULL` | `unsigned long long` | `42ULL` |
| `uz` (C++23) | `std::size_t` | `42uz` |
| `s` | `std::string` | `"hello"s` (needs `using namespace std::literals;`) |
| `sv` | `std::string_view` | `"hello"sv` |
| `ms/us/ns` | `std::chrono::duration` | `100ms`, `2s` |

```cpp
auto mask = 0b1111'0000;   // binary with digit separator
auto addr = 0xFF'AA'00'11; // hex with digit separator
```

---

## 2. Smart Pointers

```cpp
auto p  = std::make_unique<T>(args);  // exclusive ownership
auto sp = std::make_shared<T>(args);  // shared (ref-counted) ownership
std::weak_ptr<T> wp = sp;            // non-owning observer
if (auto locked = wp.lock()) locked->method();  // promote to shared
auto p2 = std::move(p);              // transfer ownership; p → nullptr
auto fd = std::unique_ptr<FILE, decltype(&fclose)>(fopen("f","r"), fclose);
```

| Situation | Use |
|-----------|-----|
| Single owner | `unique_ptr` |
| Multiple owners, shared lifetime | `shared_ptr` |
| Break reference cycles | `weak_ptr` |
| C API / custom cleanup | `unique_ptr` + custom deleter |
| Non-owning reference | Raw pointer or reference |

---

## 3. STL Containers at a Glance

| Container | Structure | Insert | Find | Ordered? | When to Use |
|-----------|-----------|--------|------|----------|-------------|
| `vector<T>` | Dynamic array | O(1) amort back | O(n) | By index | **Default choice** |
| `array<T,N>` | Fixed array | — | O(n) | By index | Compile-time size |
| `deque<T>` | Block array | O(1) front/back | O(n) | By index | Push both ends |
| `list<T>` | Doubly-linked | O(1) at pos | O(n) | No | Frequent splice |
| `map<K,V>` | Red-black tree | O(log n) | O(log n) | Yes | Sorted keys |
| `unordered_map<K,V>` | Hash table | O(1) avg | O(1) avg | No | **Fast lookup** |
| `set<T>` | Red-black tree | O(log n) | O(log n) | Yes | Sorted unique |
| `unordered_set<T>` | Hash table | O(1) avg | O(1) avg | No | Membership test |
| `stack<T>` | Adapter | O(1) push | — | LIFO | LIFO semantics |
| `queue<T>` | Adapter | O(1) push | — | FIFO | FIFO semantics |
| `priority_queue<T>` | Binary heap | O(log n) | — | Priority | Top-k, scheduling |
| `span<T>` (C++20) | Non-owning view | — | O(n) | By index | View into contiguous memory |

**Tip:** `vector` beats `list` almost always due to cache locality.

---

## 4. STL Algorithms Cheat Sheet

```cpp
#include <algorithm>
#include <numeric>   // accumulate, reduce, iota

std::vector<int> v = {5, 3, 1, 4, 2};

// ── Sorting ──
std::sort(v.begin(), v.end());                        // ascending
std::sort(v.begin(), v.end(), std::greater<>{});      // descending
std::ranges::sort(v);                                 // C++20
std::nth_element(v.begin(), v.begin()+2, v.end());    // partition around median

// ── Searching ──
auto it = std::find(v.begin(), v.end(), 42);
auto it2 = std::find_if(v.begin(), v.end(), [](int x){ return x > 10; });
bool found = std::binary_search(v.begin(), v.end(), 42);  // sorted range!

// ── Counting & Aggregation ──
int n = std::count_if(v.begin(), v.end(), [](int x){ return x > 0; });
int sum = std::accumulate(v.begin(), v.end(), 0);

// ── Transformation ──
std::transform(v.begin(), v.end(), v.begin(), [](int x){ return x * 2; });
std::for_each(v.begin(), v.end(), [](int& x){ x += 1; });

// ── Erase-remove idiom ──
v.erase(std::remove_if(v.begin(), v.end(), [](int x){ return x<0; }), v.end());
std::vector<int> dst;
std::copy_if(v.begin(), v.end(), std::back_inserter(dst), [](int x){ return x>0; });

// ── Min/Max & Predicates ──
auto [lo, hi] = std::minmax_element(v.begin(), v.end());
bool any  = std::any_of(v.begin(), v.end(),  [](int x){ return x > 10; });
bool all  = std::all_of(v.begin(), v.end(),  [](int x){ return x > 0; });
bool none = std::none_of(v.begin(), v.end(), [](int x){ return x < 0; });

// ── Filling ──
std::iota(v.begin(), v.end(), 1);   // 1,2,3,4,5
```

---

## 5. Lambda Syntax

```cpp
auto f = [captures](params) mutable noexcept -> RetType { body };

// ── Capture modes ──
[=]        // copy all        [&]        // ref all
[x]        // copy x          [&x]       // ref x
[=, &x]    // copy all, ref x [&, x]     // ref all, copy x
[this]     // capture this    [*this]    // copy *this (C++17)
[p = std::move(ptr)]          // init-capture / move into lambda (C++14)

// ── Generic lambdas (C++14) ──
auto add = [](auto a, auto b) { return a + b; };

// ── Template lambda (C++20) ──
auto sized = []<typename T>(std::vector<T> const& v) { return v.size(); };

// ── Immediately-invoked ──
const auto val = [&]{ /* complex init */ return result; }();

// ── Recursive (C++23 deducing this) ──
auto fib = [](this auto self, int n) -> int {
    return n <= 1 ? n : self(n-1) + self(n-2);
};
```

---

## 6. Move Semantics Quick Reference

```cpp
std::string a = "hello";
std::string b = std::move(a);  // a is now valid-but-unspecified

// Perfect forwarding
template <typename... Args>
auto make(Args&&... args) { return T(std::forward<Args>(args)...); }
```

### Rule of 0 / 3 / 5

| Rule | Provide | When |
|------|---------|------|
| **0** | None of the 5 special members | Only RAII members (smart ptrs, containers) |
| **3** | Destructor + copy ctor + copy assign | Manages raw resource |
| **5** | Rule of 3 + move ctor + move assign | Same + want move efficiency |

### When to `std::move`

| ✅ Do | ❌ Don't |
|-------|----------|
| Sink param you won't reuse | Return local (NRVO handles it) |
| `vec.push_back(std::move(s))` | From objects you'll use later |
| Move ctor/assign implementation | From a `const` object (silently copies!) |

---

## 7. Template Syntax

```cpp
// Function template
template <typename T> T square(T x) { return x * x; }

// Class template
template <typename T, int N = 10>
class FixedVec { std::array<T, N> data_; };

// Variable template (C++14)
template <typename T> constexpr T pi = T(3.14159265358979L);

// Concepts (C++20)
template <typename T> concept Numeric = std::is_arithmetic_v<T>;
template <Numeric T> T add(T a, T b) { return a + b; }
auto add2(Numeric auto a, Numeric auto b) { return a + b; }  // shorthand

// requires clause
template <typename T> requires std::is_integral_v<T> && (sizeof(T) >= 4)
T bitwise_op(T a, T b) { return a ^ b; }

// requires expression
template <typename T>
concept Printable = requires(T t, std::ostream& os) {
    { os << t } -> std::same_as<std::ostream&>;
};

// CTAD (C++17)
std::vector v = {1, 2, 3};     // vector<int>
std::pair p = {1, 3.14};       // pair<int,double>

// if constexpr (C++17)
if constexpr (std::is_same_v<T, std::string>) return val;
else return std::to_string(val);

// Fold expressions (C++17)
template <typename... Args> auto sum(Args... a) { return (a + ...); }
```

---

## 8. Modern C++ Feature Quick-Look

| Feature | Std | One-line Example |
|---------|-----|------------------|
| Structured bindings | 17 | `auto [key, val] = *map.begin();` |
| `if constexpr` | 17 | `if constexpr (std::is_integral_v<T>) { ... }` |
| `std::optional<T>` | 17 | `auto o = std::optional{42}; if (o) use(*o);` |
| `std::variant<Ts...>` | 17 | `std::variant<int,string> v = "hi"; std::get<string>(v);` |
| `std::any` | 17 | `std::any a = 42; int x = std::any_cast<int>(a);` |
| `std::string_view` | 17 | `void f(std::string_view sv); // non-owning, no alloc` |
| `std::filesystem` | 17 | `for (auto& e : fs::directory_iterator(".")) ...` |
| Init-stmt in `if` | 17 | `if (auto it = m.find(k); it != m.end()) { ... }` |
| Fold expressions | 17 | `template<class...T> auto sum(T...v){return (v+...);}` |
| Concepts | 20 | `template <std::integral T> T add(T a, T b);` |
| Ranges | 20 | `auto evens = v \| views::filter([](int x){return x%2==0;});` |
| `std::span<T>` | 20 | `void process(std::span<const int> data);` |
| `<=>` spaceship | 20 | `auto operator<=>(const T&) const = default;` |
| `consteval` | 20 | `consteval int sqr(int n) { return n*n; }` |
| `std::format` | 20 | `std::format("x={}, y={:.2f}", 42, 3.14);` |
| Coroutines | 20 | `generator<int> iota(int n) { for(int i=0;i<n;++i) co_yield i; }` |
| Modules | 20 | `export module math; export int add(int a, int b);` |
| `contains()` | 20 | `if (my_map.contains(key)) { ... }` |
| `starts_with/ends_with` | 20 | `"hello"sv.starts_with("he"); // true` |
| `std::jthread` | 20 | `std::jthread t(func); // auto-joins on destruction` |
| Deducing `this` | 23 | `void f(this auto&& self) { ... }` |
| `std::expected<T,E>` | 23 | `std::expected<int,Err> r = parse(s); if (r) use(*r);` |
| `std::print/println` | 23 | `std::println("x = {}", 42);` |
| `std::mdspan` | 23 | `std::mdspan m(ptr, 3, 4); m[1,2] = 5.0;` |
| `std::generator<T>` | 23 | `std::generator<int> fib() { co_yield a; ... }` |
| `std::flat_map` | 23 | `std::flat_map<int,string> fm; // cache-friendly` |
| `string::contains` | 23 | `"hello world"s.contains("world"); // true` |

---

## 9. Concurrency Quick Reference

```cpp
#include <thread>
#include <mutex>
#include <atomic>
#include <future>

// ── Threads ──
std::thread t(func, arg1, arg2);  t.join();
std::jthread jt([](std::stop_token st) {      // C++20: auto-joins + stop_token
    while (!st.stop_requested()) { /* work */ }
});

// ── Mutexes ──
std::mutex mtx;
{ std::lock_guard lk(mtx); /* critical section */ }          // RAII lock
{ std::scoped_lock lk(mtx1, mtx2); /* deadlock-free */ }     // C++17
{ std::unique_lock lk(mtx); /* for cond_var or deferred */ } // flexible

// ── Atomics ──
std::atomic<int> counter{0};
counter.fetch_add(1, std::memory_order_relaxed);
counter.store(42);  int v = counter.load();

// ── Condition Variables ──
std::condition_variable cv;  std::mutex cv_m;  bool ready = false;
// Producer: { std::lock_guard lk(cv_m); ready = true; } cv.notify_one();
// Consumer: std::unique_lock lk(cv_m); cv.wait(lk, [&]{ return ready; });

// ── Async / Future ──
auto fut = std::async(std::launch::async, [](int x){ return x*x; }, 42);
int result = fut.get();   // blocks until ready

// ── Promise ──
std::promise<int> prom;
std::future<int> f = prom.get_future();
std::thread([&]{ prom.set_value(42); }).detach();
int r = f.get();

// ── C++20 synchronization primitives ──
std::latch done(3);           // one-shot countdown: count_down() + wait()
std::barrier sync(n, on_complete);  // reusable: arrive_and_wait()
std::counting_semaphore<4> sem(4);  // acquire() / release()
```

---

## 10. Common Type Traits

```cpp
#include <type_traits>  // use _v suffix for values, _t for types

// ── Type checks ──
std::is_integral_v<T>             std::is_floating_point_v<T>
std::is_arithmetic_v<T>           std::is_pointer_v<T>
std::is_same_v<T, U>              std::is_base_of_v<Base, Derived>
std::is_convertible_v<From, To>   std::is_trivially_copyable_v<T>
std::is_const_v<T>                std::is_default_constructible_v<T>

// ── Type transformations ──
std::remove_reference_t<int&>          // → int
std::remove_const_t<const int>         // → int
std::decay_t<int(&)[3]>               // → int*
std::conditional_t<true, int, float>   // → int
std::common_type_t<int, double>        // → double
std::add_pointer_t<int>               // → int*
std::make_unsigned_t<int>             // → unsigned int

// ── SFINAE (pre-C++20; prefer concepts) ──
template <typename T>
std::enable_if_t<std::is_integral_v<T>, T> checked_add(T a, T b);
```

---

## 11. String Operations

```cpp
std::string s = "Hello, World!";

// ── Search ──
auto pos = s.find("World");       // index or std::string::npos
auto pos2 = s.rfind("l");        // reverse find

// ── Modify ──
s.substr(7, 5);                   // "World"
s.replace(7, 5, "C++");          // "Hello, C++!"
s.insert(5, "!");    s.erase(5, 1);    s += " again";

// ── C++20/23 ──
s.starts_with("Hello");   s.ends_with("!");   // C++20
s.contains("World");                           // C++23

// ── Conversions ──
int n = std::stoi("42");   auto ns = std::to_string(42);

// ── string_view (non-owning, zero-copy) ──
std::string_view sv = s;
sv.remove_prefix(7);  sv.remove_suffix(1);  // modifies view, not string

// ── std::format (C++20) ──
std::format("name={}, val={:.2f}", "x", 3.14);  // "name=x, val=3.14"
std::format("{:#x}", 255);    // "0xff"
std::format("{:>10}", "hi");  // "        hi"
// Spec: {[index]:[fill][align][width][.prec][type]}  align: < > ^

// ── std::println (C++23) ──
std::println("x = {}", 42);
```

---

## 12. Error Handling Patterns

```cpp
// ── Exceptions ──
try { throw std::runtime_error("oops"); }
catch (const std::exception& e) { std::cerr << e.what(); }
catch (...) { /* catch-all */ }

// ── noexcept ──
void safe() noexcept;   // move ctors should be noexcept for vector efficiency

// ── std::optional (C++17) — value or nothing ──
std::optional<int> find(int id) { return found ? val : std::nullopt; }
int v = find(42).value_or(-1);

// ── std::expected (C++23) — value or typed error ──
std::expected<int, Error> parse(std::string_view s) {
    if (s.empty()) return std::unexpected(Error::empty);
    return 42;
}
auto r = parse("123").transform([](int x){ return x*2; });

// ── std::error_code (filesystem, system APIs) ──
std::error_code ec;
auto sz = std::filesystem::file_size("f.txt", ec);
if (ec) std::cerr << ec.message();

// ── When to use what ──
// Hot loops / CUDA kernels → error codes / std::expected
// Application logic        → exceptions
// "May be absent"          → std::optional
// "Value or specific error"→ std::expected
```

---

## 13. Compiler Flags Quick Reference

### GCC / Clang

| Flag | Purpose |
|------|---------|
| `-std=c++20` | Language standard (17, 20, 23, 2c) |
| `-Wall -Wextra -Wpedantic` | Comprehensive warnings |
| `-Werror` | Warnings → errors |
| `-O0 / -O2 / -O3 / -Ofast` | Optimization levels |
| `-g` | Debug symbols |
| `-DNDEBUG` | Disable `assert()` |
| `-march=native` | Optimize for current CPU |
| `-flto` | Link-time optimization |
| `-fsanitize=address` | Buffer overflow, use-after-free |
| `-fsanitize=thread` | Data race detection |
| `-fsanitize=undefined` | Undefined behavior |
| `-fno-exceptions` | Disable exceptions |
| `-fno-rtti` | Disable RTTI |

```bash
# Dev:     g++ -std=c++20 -Wall -Wextra -Wpedantic -Werror -g -O0 -fsanitize=address,undefined
# Release: g++ -std=c++20 -O3 -DNDEBUG -march=native -flto
```

### Attributes

```cpp
[[nodiscard]]            // warn if return value ignored
[[nodiscard("reason")]]  // C++20: with message
[[maybe_unused]]         // suppress unused warnings
[[deprecated("use v2")]] // mark deprecated
[[likely]] / [[unlikely]] // C++20: branch hints
[[fallthrough]]          // intentional switch fallthrough
[[no_unique_address]]    // C++20: empty member optimization
[[assume(expr)]]         // C++23: optimizer hint
```

---

*Appendix H — Modern C++ Quick Reference • CPP-CUDA-Mastery*
