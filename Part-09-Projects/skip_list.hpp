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
