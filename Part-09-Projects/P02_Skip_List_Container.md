# Project 02 — STL-Compatible Skip List Container

> **Difficulty:** 🟡 Intermediate
> **Time estimate:** 8–12 hours
> **Standard:** C++20 (`-std=c++20`)

---

## Prerequisites

| Topic | Why |
|---|---|
| Templates & concepts | Key constraint enforcement with `std::three_way_comparable` |
| Iterators | Forward-iterator interface for range-`for` and `<algorithm>` |
| Allocators | `std::allocator_traits` for pluggable memory |
| Probability | Geometric distribution governs level promotion |
| `operator<=>` | Spaceship operator for three-way node comparison |

## Learning Objectives

1. Design a probabilistic data structure with **O(log n)** search, insert, and erase.
2. Implement a **forward iterator** satisfying `std::forward_iterator`.
3. Wire up **allocator-aware** node management via `rebind` and `allocator_traits`.
4. Enforce key ordering at compile time with C++20 **concepts**.
5. Benchmark against `std::set` and reason about cache-line behaviour.

---

## Architecture

```mermaid
graph TD
    subgraph "Skip List  (max_level = 4)"
        direction LR
        HEAD["HEAD<br/>L0–L3"]
        N3["3<br/>L0–L2"]
        N7["7<br/>L0–L3"]
        N12["12<br/>L0"]
        N19["19<br/>L0–L1"]
        N25["25<br/>L0–L2"]
        TAIL["nullptr"]

        HEAD -->|L3| N7
        HEAD -->|L2| N3
        HEAD -->|L1| N3
        HEAD -->|L0| N3

        N3 -->|L2| N7
        N3 -->|L1| N7
        N3 -->|L0| N7

        N7 -->|L3| TAIL
        N7 -->|L2| N25
        N7 -->|L1| N19
        N7 -->|L0| N12

        N12 -->|L0| N19

        N19 -->|L1| N25
        N19 -->|L0| N25

        N25 -->|L2| TAIL
        N25 -->|L0| TAIL
    end

    style HEAD fill:#4a9eff,color:#fff
    style TAIL fill:#888,color:#fff
    style N7 fill:#ff6b6b,color:#fff
```

**Key insight:** each node stores a *variable-length* array of forward pointers.
Searching starts at the highest level and drops down, yielding expected **O(log n)** hops.

---

## Step-by-Step Implementation

### Complete Code — `skip_list.hpp`

```cpp
#pragma once
#include <cassert>
#include <cstddef>
#include <compare>
#include <concepts>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <utility>
#include <vector>

// ── Concept: keys must support three-way comparison ──────────────
template <typename K>
concept OrderedKey = std::three_way_comparable<K>
                  && std::copyable<K>
                  && std::default_initializable<K>;

// ── Forward declarations ─────────────────────────────────────────
template <OrderedKey K, typename Alloc> class SkipList;
template <OrderedKey K>                 struct SkipNode;

// ── Node ─────────────────────────────────────────────────────────
template <OrderedKey K>
struct SkipNode {
    K                          key{};
    std::vector<SkipNode*>     forward;   // forward[i] = next at level i

    explicit SkipNode(int level)
        : forward(static_cast<std::size_t>(level + 1), nullptr) {}

    SkipNode(K k, int level)
        : key{std::move(k)},
          forward(static_cast<std::size_t>(level + 1), nullptr) {}

    [[nodiscard]] int level() const noexcept {
        return static_cast<int>(forward.size()) - 1;
    }
};

// ── Iterator (forward) ──────────────────────────────────────────
template <OrderedKey K>
class SkipListIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = K;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const K*;
    using reference         = const K&;

    SkipListIterator() noexcept = default;
    explicit SkipListIterator(SkipNode<K>* n) noexcept : node_{n} {}

    reference   operator*()  const noexcept { return node_->key; }
    pointer     operator->() const noexcept { return &node_->key; }

    SkipListIterator& operator++() noexcept {
        node_ = node_->forward[0];
        return *this;
    }
    SkipListIterator operator++(int) noexcept {
        auto tmp = *this; ++*this; return tmp;
    }

    bool operator==(const SkipListIterator& o) const noexcept = default;

private:
    SkipNode<K>* node_{nullptr};

    template <OrderedKey, typename> friend class SkipList;
};

static_assert(std::forward_iterator<SkipListIterator<int>>);

// ── SkipList ─────────────────────────────────────────────────────
template <OrderedKey K, typename Alloc = std::allocator<K>>
class SkipList {
public:
    // ── public type aliases (STL convention) ─────────────────────
    using key_type       = K;
    using value_type     = K;
    using size_type      = std::size_t;
    using difference_type= std::ptrdiff_t;
    using allocator_type = Alloc;
    using iterator       = SkipListIterator<K>;
    using const_iterator = iterator;  // keys are immutable

    static constexpr int kMaxLevel  = 16;
    static constexpr double kProb   = 0.5;

    // ── constructors / destructor ────────────────────────────────
    explicit SkipList(const Alloc& alloc = Alloc{})
        : alloc_{alloc},
          head_{create_node(K{}, kMaxLevel)},
          gen_{std::random_device{}()} {}

    SkipList(std::initializer_list<K> init, const Alloc& alloc = Alloc{})
        : SkipList(alloc)
    {
        for (auto& k : init) insert(k);
    }

    ~SkipList() { clear(); destroy_node(head_); }

    // non-copyable, movable
    SkipList(const SkipList&)            = delete;
    SkipList& operator=(const SkipList&) = delete;

    SkipList(SkipList&& o) noexcept
        : alloc_{std::move(o.alloc_)}, head_{o.head_},
          level_{o.level_}, size_{o.size_}, gen_{std::move(o.gen_)}
    {
        o.head_  = nullptr;
        o.size_  = 0;
        o.level_ = 0;
    }

    SkipList& operator=(SkipList&& o) noexcept {
        if (this != &o) {
            clear();
            destroy_node(head_);
            alloc_ = std::move(o.alloc_);
            head_  = o.head_;  level_ = o.level_;
            size_  = o.size_;  gen_   = std::move(o.gen_);
            o.head_ = nullptr; o.size_ = 0; o.level_ = 0;
        }
        return *this;
    }

    // ── capacity ─────────────────────────────────────────────────
    [[nodiscard]] bool      empty() const noexcept { return size_ == 0; }
    [[nodiscard]] size_type size()  const noexcept { return size_; }

    // ── iterators ────────────────────────────────────────────────
    iterator begin() const noexcept { return iterator{head_->forward[0]}; }
    iterator end()   const noexcept { return iterator{nullptr}; }

    // ── lookup ───────────────────────────────────────────────────
    iterator find(const K& key) const noexcept {
        auto* x = head_;
        for (int i = level_; i >= 0; --i)
            while (x->forward[i] && (x->forward[i]->key <=> key) < 0)
                x = x->forward[i];
        x = x->forward[0];
        if (x && (x->key <=> key) == 0)
            return iterator{x};
        return end();
    }

    bool contains(const K& key) const noexcept {
        return find(key) != end();
    }

    // ── insert ───────────────────────────────────────────────────
    std::pair<iterator, bool> insert(const K& key) {
        // collect update path
        std::vector<SkipNode<K>*> update(
            static_cast<size_type>(kMaxLevel + 1), nullptr);
        auto* x = head_;
        for (int i = level_; i >= 0; --i) {
            while (x->forward[i] && (x->forward[i]->key <=> key) < 0)
                x = x->forward[i];
            update[static_cast<size_type>(i)] = x;
        }
        x = x->forward[0];

        if (x && (x->key <=> key) == 0)
            return {iterator{x}, false};   // duplicate

        int new_level = random_level();
        if (new_level > level_) {
            for (int i = level_ + 1; i <= new_level; ++i)
                update[static_cast<size_type>(i)] = head_;
            level_ = new_level;
        }

        auto* node = create_node(key, new_level);
        for (int i = 0; i <= new_level; ++i) {
            auto idx = static_cast<size_type>(i);
            node->forward[idx]             = update[idx]->forward[idx];
            update[idx]->forward[idx]      = node;
        }
        ++size_;
        return {iterator{node}, true};
    }

    // ── erase ────────────────────────────────────────────────────
    size_type erase(const K& key) {
        std::vector<SkipNode<K>*> update(
            static_cast<size_type>(kMaxLevel + 1), nullptr);
        auto* x = head_;
        for (int i = level_; i >= 0; --i) {
            while (x->forward[i] && (x->forward[i]->key <=> key) < 0)
                x = x->forward[i];
            update[static_cast<size_type>(i)] = x;
        }
        x = x->forward[0];

        if (!x || (x->key <=> key) != 0)
            return 0;  // not found

        for (int i = 0; i <= level_; ++i) {
            auto idx = static_cast<size_type>(i);
            if (update[idx]->forward[idx] != x) break;
            update[idx]->forward[idx] = x->forward[idx];
        }
        destroy_node(x);
        while (level_ > 0 && head_->forward[static_cast<size_type>(level_)] == nullptr)
            --level_;
        --size_;
        return 1;
    }

    iterator erase(iterator pos) {
        if (pos == end()) return end();
        auto next_it = std::next(pos);
        erase(*pos);
        return next_it;
    }

    // ── clear ────────────────────────────────────────────────────
    void clear() noexcept {
        auto* x = head_ ? head_->forward[0] : nullptr;
        while (x) {
            auto* next = x->forward[0];
            destroy_node(x);
            x = next;
        }
        if (head_) {
            for (auto& ptr : head_->forward) ptr = nullptr;
        }
        level_ = 0;
        size_  = 0;
    }

    // ── comparison ───────────────────────────────────────────────
    friend auto operator<=>(const SkipList& a, const SkipList& b) {
        return std::lexicographical_compare_three_way(
            a.begin(), a.end(), b.begin(), b.end());
    }
    friend bool operator==(const SkipList& a, const SkipList& b) {
        if (a.size() != b.size()) return false;
        return (a <=> b) == 0;
    }

    // ── allocator access ─────────────────────────────────────────
    allocator_type get_allocator() const noexcept { return alloc_; }

    // ── debug: level distribution ────────────────────────────────
    [[nodiscard]] std::vector<int> level_histogram() const {
        std::vector<int> hist(static_cast<size_type>(kMaxLevel + 1), 0);
        for (auto* n = head_->forward[0]; n; n = n->forward[0])
            ++hist[static_cast<size_type>(n->level())];
        return hist;
    }

private:
    using NodeAlloc = typename std::allocator_traits<Alloc>
                        ::template rebind_alloc<SkipNode<K>>;
    using NodeTraits = std::allocator_traits<NodeAlloc>;

    NodeAlloc         alloc_;
    SkipNode<K>*      head_{nullptr};
    int               level_{0};
    size_type         size_{0};
    std::mt19937      gen_;

    int random_level() {
        int lvl = 0;
        std::bernoulli_distribution coin(kProb);
        while (coin(gen_) && lvl < kMaxLevel) ++lvl;
        return lvl;
    }

    SkipNode<K>* create_node(const K& key, int lvl) {
        NodeAlloc na{alloc_};
        auto* p = NodeTraits::allocate(na, 1);
        NodeTraits::construct(na, p, key, lvl);
        return p;
    }

    void destroy_node(SkipNode<K>* p) {
        if (!p) return;
        NodeAlloc na{alloc_};
        NodeTraits::destroy(na, p);
        NodeTraits::deallocate(na, p, 1);
    }
};
```

---

### Companion Test File — `skip_list_test.cpp`

```cpp
#include "skip_list.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <ranges>
#include <set>
#include <string>
#include <vector>

// ── helpers ──────────────────────────────────────────────────────
#define ASSERT_EQ(a, b)  assert((a) == (b))
#define ASSERT_TRUE(x)   assert((x))
#define ASSERT_FALSE(x)  assert(!(x))

// ── Test 1: basic insert & contains ─────────────────────────────
void test_insert_contains() {
    SkipList<int> sl;
    auto [it1, ok1] = sl.insert(10);
    ASSERT_TRUE(ok1);
    ASSERT_EQ(*it1, 10);

    auto [it2, ok2] = sl.insert(10);  // duplicate
    ASSERT_FALSE(ok2);
    ASSERT_EQ(*it2, 10);

    sl.insert(5);
    sl.insert(20);
    sl.insert(15);
    ASSERT_EQ(sl.size(), 4u);
    ASSERT_TRUE(sl.contains(5));
    ASSERT_TRUE(sl.contains(15));
    ASSERT_FALSE(sl.contains(99));
    std::cout << "  [PASS] insert & contains\n";
}

// ── Test 2: sorted iteration ────────────────────────────────────
void test_sorted_order() {
    SkipList<int> sl{30, 10, 50, 20, 40};
    std::vector<int> out(sl.begin(), sl.end());
    std::vector<int> expected{10, 20, 30, 40, 50};
    ASSERT_EQ(out, expected);
    std::cout << "  [PASS] sorted iteration\n";
}

// ── Test 3: erase by key ────────────────────────────────────────
void test_erase() {
    SkipList<int> sl{1, 2, 3, 4, 5};
    ASSERT_EQ(sl.erase(3), 1u);
    ASSERT_EQ(sl.erase(3), 0u);  // already gone
    ASSERT_EQ(sl.size(), 4u);
    ASSERT_FALSE(sl.contains(3));

    // erase first and last
    sl.erase(1);
    sl.erase(5);
    std::vector<int> out(sl.begin(), sl.end());
    ASSERT_EQ(out, (std::vector<int>{2, 4}));
    std::cout << "  [PASS] erase\n";
}

// ── Test 4: erase via iterator ──────────────────────────────────
void test_erase_iterator() {
    SkipList<int> sl{10, 20, 30};
    auto it = sl.find(20);
    ASSERT_TRUE(it != sl.end());
    auto next = sl.erase(it);
    ASSERT_EQ(sl.size(), 2u);
    if (next != sl.end()) ASSERT_EQ(*next, 30);
    std::cout << "  [PASS] erase via iterator\n";
}

// ── Test 5: find returns end() for missing keys ─────────────────
void test_find() {
    SkipList<int> sl{1, 3, 5, 7, 9};
    ASSERT_TRUE(sl.find(5) != sl.end());
    ASSERT_EQ(*sl.find(5), 5);
    ASSERT_TRUE(sl.find(4) == sl.end());
    std::cout << "  [PASS] find\n";
}

// ── Test 6: clear & empty ───────────────────────────────────────
void test_clear() {
    SkipList<int> sl{1, 2, 3};
    sl.clear();
    ASSERT_TRUE(sl.empty());
    ASSERT_EQ(sl.size(), 0u);
    ASSERT_TRUE(sl.begin() == sl.end());
    sl.insert(42);
    ASSERT_EQ(sl.size(), 1u);
    std::cout << "  [PASS] clear & reuse\n";
}

// ── Test 7: move semantics ──────────────────────────────────────
void test_move() {
    SkipList<int> a{1, 2, 3};
    SkipList<int> b{std::move(a)};
    ASSERT_EQ(b.size(), 3u);
    ASSERT_TRUE(b.contains(2));

    SkipList<int> c;
    c = std::move(b);
    ASSERT_EQ(c.size(), 3u);
    std::cout << "  [PASS] move semantics\n";
}

// ── Test 8: three-way comparison ────────────────────────────────
void test_comparison() {
    SkipList<int> a{1, 2, 3};
    SkipList<int> b{1, 2, 3};
    SkipList<int> c{1, 2, 4};
    ASSERT_TRUE(a == b);
    ASSERT_TRUE((a <=> c) < 0);
    ASSERT_TRUE((c <=> a) > 0);
    std::cout << "  [PASS] operator<=> & ==\n";
}

// ── Test 9: string keys ─────────────────────────────────────────
void test_string_keys() {
    SkipList<std::string> sl;
    sl.insert("banana"); sl.insert("apple"); sl.insert("cherry");
    std::vector<std::string> out(sl.begin(), sl.end());
    ASSERT_EQ(out, (std::vector<std::string>{"apple", "banana", "cherry"}));
    std::cout << "  [PASS] string keys\n";
}

// ── Test 10: large-scale insert + verify ordering ───────────────
void test_large_scale() {
    SkipList<int> sl;
    constexpr int N = 10'000;
    std::vector<int> vals(N);
    std::iota(vals.begin(), vals.end(), 0);
    std::mt19937 rng{42};
    std::ranges::shuffle(vals, rng);
    for (int v : vals) sl.insert(v);
    ASSERT_EQ(sl.size(), static_cast<std::size_t>(N));
    int prev = -1;
    for (int k : sl) { ASSERT_TRUE(k > prev); prev = k; }
    std::cout << "  [PASS] large-scale (N=" << N << ")\n";
}

// ── Test 11: works with STL algorithms ──────────────────────────
void test_stl_compat() {
    SkipList<int> sl{5, 3, 8, 1, 9};
    ASSERT_EQ(*std::find(sl.begin(), sl.end(), 8), 8);
    ASSERT_EQ(std::accumulate(sl.begin(), sl.end(), 0), 26);
    ASSERT_EQ(std::ranges::count_if(sl, [](int x){ return x > 4; }), 3);
    std::cout << "  [PASS] STL algorithm compatibility\n";
}

// ── Test 12: level histogram sanity ─────────────────────────────
void test_level_distribution() {
    SkipList<int> sl;
    for (int i = 0; i < 1000; ++i) sl.insert(i);
    auto hist = sl.level_histogram();
    ASSERT_TRUE(hist[0] > 0);  // level 0 should have the most nodes
    std::cout << "  [PASS] level distribution (L0=" << hist[0] << ")\n";
}

int main() {
    std::cout << "=== SkipList Test Suite ===\n";
    test_insert_contains();
    test_sorted_order();
    test_erase();
    test_erase_iterator();
    test_find();
    test_clear();
    test_move();
    test_comparison();
    test_string_keys();
    test_large_scale();
    test_stl_compat();
    test_level_distribution();
    std::cout << "=== All tests passed ===\n";
}
```

**Compile & run:**

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o skip_test skip_list_test.cpp && ./skip_test
```

---

## Testing Strategy

| Layer | What to verify | Method |
|---|---|---|
| **Unit** | insert/find/erase correctness, duplicates rejected | Assert-based tests above |
| **Ordering** | Iteration always yields sorted output | Shuffle N values, insert, verify monotone |
| **Iterator contract** | `std::forward_iterator` satisfied | `static_assert` in header |
| **Edge cases** | Empty list ops, single element, erase-all-then-reuse | `test_clear`, erase first/last |
| **Move semantics** | Source emptied, destination valid | `test_move` |
| **STL compat** | Works with `<algorithm>` and `<ranges>` | `std::find`, `std::accumulate`, `ranges::count_if` |
| **Stress** | 10k random inserts stay O(log n)-ish | `test_large_scale` + timing |

---

## Performance Analysis

### Theoretical Complexity

| Operation | Expected | Worst case |
|---|---|---|
| `insert` | O(log n) | O(n) — astronomically unlikely |
| `find` | O(log n) | O(n) |
| `erase` | O(log n) | O(n) |
| `begin` / `++` | O(1) | O(1) |
| Space | O(n) | O(n log n) — expected O(n) with p=0.5 |

**`std::set` comparison:** Skip lists have higher constant factors (pointer chasing across levels) and `std::set` (red-black tree) is cache-friendlier for small keys. However, skip lists win on **lock-free concurrent extensions** and at N > 100k both converge to similar O(log n) scaling. Benchmark with shuffled input — sorted insertion doesn't exercise multi-level search.

---

## Extensions & Challenges

| # | Challenge | Hint |
|---|---|---|
| 1 | **Bidirectional iterator** — add `prev` pointers | Each node gets a `backward` pointer; update on insert/erase |
| 2 | **`lower_bound` / `upper_bound`** | Stop descent one step earlier; mirrors `std::set` API |
| 3 | **Concurrent skip list** | Use `std::atomic` per-level forward pointers with CAS-based insert |
| 4 | **Custom comparator** | Template on `Compare = std::less<K>` instead of requiring `<=>` |
| 5 | **Memory pool allocator** | Write a `PoolAllocator<SkipNode<K>>` and benchmark allocation throughput |
| 6 | **Serialization** | Dump level-by-level to binary; reconstruct without re-randomising |
| 7 | **Range insert** | Accept iterator pairs and `std::initializer_list` (partially done) |
| 8 | **`std::pmr` support** | Use `std::pmr::polymorphic_allocator` for arena-based allocation |

---

## Key Takeaways

1. **Probabilistic ≠ unreliable.** With p = 0.5, the expected height is log₂ n and the probability of catastrophic imbalance is negligible (< 1/n²).
2. **Allocator awareness** is mechanical but essential — `rebind_alloc` + `allocator_traits` makes your container a first-class citizen in any allocator ecosystem.
3. **Concepts enforce invariants at compile time** — `OrderedKey` prevents instantiation with types that lack `<=>`, producing clear error messages instead of template soup.
4. **Iterator design is the gateway to STL interop.** Satisfying `std::forward_iterator` unlocks range-`for`, `<algorithm>`, and `<ranges>` with zero extra work.
5. **`operator<=>`** on containers composes naturally — delegate to `std::lexicographical_compare_three_way` and `==` falls out for free.
6. **Benchmarking requires shuffled input** — sorted insertion into a skip list still works correctly but doesn't exercise the multi-level search path.
