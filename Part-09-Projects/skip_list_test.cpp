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
