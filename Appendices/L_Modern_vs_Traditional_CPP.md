# Appendix L — Modern C++ vs Traditional C++: A Comprehensive Side-by-Side Comparison

> **Every major C++ concept shown two ways: the old way that worked, and the modern way that's safer, shorter, and faster.**

This appendix covers 33 side-by-side comparisons across memory management, type safety, error handling, iteration, classes, concurrency, data structures, and modern idioms — plus a full before/after program and a modernization checklist.

**Standard coverage:** C++11 · C++14 · C++17 · C++20 · C++23

---

## Table of Contents

| # | Topic | Category |
|---|-------|----------|
| 1 | [Raw new/delete → unique_ptr/shared_ptr](#1-raw-newdelete--unique_ptrshared_ptr) | Memory |
| 2 | [C-style arrays → std::array/std::vector](#2-c-style-arrays--stdarraystdvector) | Memory |
| 3 | [C-strings → std::string/string_view](#3-c-strings-char--stdstringstring_view) | Memory |
| 4 | [Manual memory tracking → RAII](#4-manual-memory-tracking--raii) | Memory |
| 5 | [malloc/free → new/delete → make_unique](#5-mallocfree--newdelete--make_unique) | Memory |
| 6 | [Manual resource cleanup → RAII destructors](#6-manual-resource-cleanup--raii-destructors) | Memory |
| 7 | [void* casting → std::variant/std::any](#7-void-casting--stdvariantstdany) | Type Safety |
| 8 | [C-style casts → static_cast/dynamic_cast](#8-c-style-casts--static_castdynamic_cast) | Type Safety |
| 9 | [#define constants → constexpr](#9-define-constants--constexpr) | Type Safety |
| 10 | [NULL → nullptr](#10-null--nullptr) | Type Safety |
| 11 | [Error codes → exceptions](#11-error-codes--errno--exceptions) | Errors |
| 12 | [Sentinel returns → std::optional](#12-returning--1--sentinel--stdoptional) | Errors |
| 13 | [Exceptions for expected errors → std::expected](#13-exceptions-for-expected-errors--stdexpected) | Errors |
| 14 | [C-style for → range-based for](#14-c-style-for-loops--range-based-for) | Iteration |
| 15 | [Manual loops → STL algorithms](#15-manual-loops--stl-algorithms) | Iteration |
| 16 | [Function pointers → lambdas](#16-function-pointers--lambdas) | Iteration |
| 17 | [Hand-written comparators → lambda + auto](#17-hand-written-sort-comparators--lambda--auto) | Iteration |
| 18 | [Manual copy constructor → Rule of Zero](#18-manual-copy-constructor--rule-of-zero) | Classes |
| 19 | [Virtual + raw pointers → CRTP](#19-virtual--raw-pointers--crtp-for-static-polymorphism) | Classes |
| 20 | [printf → std::format/std::print](#20-printf--stdformatstdprint) | Classes |
| 21 | [Macro generics → templates + concepts](#21-macro-based-generics--templates--concepts) | Classes |
| 22 | [pthreads → std::thread/jthread](#22-pthreads--stdthreadjthread) | Concurrency |
| 23 | [pthread_mutex → std::mutex/scoped_lock](#23-pthread_mutex--stdmutexlock_guardscoped_lock) | Concurrency |
| 24 | [Manual threads → std::async/future](#24-manual-thread-management--stdasyncfuture) | Concurrency |
| 25 | [Linked list → std::vector](#25-linked-list--stdvector-almost-always) | Data |
| 26 | [Manual hash map → std::unordered_map](#26-manual-hash-map--stdunordered_map) | Data |
| 27 | [.first/.second → structured bindings](#27-pair-with-firstsecond--structured-bindings) | Data |
| 28 | [Output parameters → return by value](#28-output-parameters--return-by-value-move-semantics) | Modern |
| 29 | [SFINAE → concepts](#29-type-traits--sfinae--concepts) | Modern |
| 30 | [#include → import modules](#30-include-headers--import-modules) | Modern |
| 31 | [Callback hell → coroutines](#31-callback-hell--coroutines) | Modern |
| 32 | [sprintf → std::format](#32-sprintf--stdformat) | Modern |
| 33 | [Full program: Before & After](#33-the-complete-beforeafter) | Bonus |

---

# Memory Management

---

## 1. Raw new/delete → unique_ptr/shared_ptr

### ❌ Traditional C++ (Pre-C++11)

```cpp
class Engine {
    Sensor* sensor;
    Logger* logger;
public:
    Engine() {
        sensor = new Sensor();
        logger = new Logger();       // if this throws, sensor leaks!
    }
    ~Engine() {
        delete sensor;
        delete logger;
    }
    // Must also write copy ctor + operator= to avoid double-free
};

void process() {
    Widget* w = new Widget();
    if (w->validate()) {
        delete w;                    // easy to forget on early return
        return;
    }
    w->run();
    delete w;                        // duplicated cleanup
}
```

**Problems:** Every `new` must be paired with `delete`. If an exception is thrown between allocation and deallocation, memory leaks. Copy semantics lead to double-free. Every early-return path must remember cleanup. This is the #1 source of bugs in legacy C++.

### ✅ Modern C++ (C++11/14)

```cpp
class Engine {
    std::unique_ptr<Sensor> sensor;
    std::unique_ptr<Logger> logger;
public:
    Engine()
        : sensor(std::make_unique<Sensor>())   // C++14
        , logger(std::make_unique<Logger>())    // exception-safe
    {}
    // No destructor, no copy ctor, no operator= needed!
};

void process() {
    auto w = std::make_unique<Widget>();
    if (w->validate()) return;       // automatic cleanup on any exit
    w->run();
}                                    // automatic cleanup here too
```

**Why Better:** Ownership is explicit in the type system. No leaks possible — cleanup is automatic even with exceptions or early returns. No need to write destructors, copy constructors, or assignment operators.

### 💡 What the Code Conveys

`unique_ptr` says "I own this exclusively" and `shared_ptr` says "we share ownership." The type itself documents lifetime management — no comments needed.

### 🔗 Cross-References

- See: [Chapter 9 — Dynamic Memory](../Part-01-CPP-Foundations/09_Dynamic_Memory.md) for allocation fundamentals
- See: [Chapter 16 — Smart Pointers](../Part-02-CPP-Intermediate/16_Smart_Pointers.md) for full coverage
- Used in: [Project P03 — Thread Pool](../Part-09-Projects/P03_Thread_Pool.md)

---

## 2. C-style Arrays → std::array/std::vector

### ❌ Traditional C++ (Pre-C++11)

```cpp
void process_data() {
    int scores[100];                 // stack array: no bounds checking
    int* dynamic = new int[n];       // heap array: must track size manually

    for (int i = 0; i <= 100; i++)   // off-by-one bug: silent corruption
        scores[i] = i * 10;

    sort(dynamic, dynamic + n);      // must pass raw pointers + size
    delete[] dynamic;                // forget this = leak; forget [] = UB
}

void print_array(int* arr, int size) {  // size travels separately — easy to mismatch
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
}
```

**Problems:** No bounds checking. Size is disconnected from the data. Off-by-one errors cause silent memory corruption. Must remember `delete[]` (not `delete`). Cannot return arrays from functions. Decays to pointer, losing size information.

### ✅ Modern C++ (C++11/17)

```cpp
void process_data() {
    std::array<int, 100> scores{};    // fixed-size, stack, bounds-checked with .at()
    std::vector<int> dynamic(n);      // heap, auto-resizing, RAII

    for (auto& s : scores)            // range-based: no off-by-one possible
        s = (&s - scores.data()) * 10;

    std::ranges::sort(dynamic);       // knows its own size (C++20)
}   // vector freed automatically

void print_array(std::span<const int> arr) {   // C++20: non-owning view
    for (int x : arr)
        std::print("{} ", x);
}
```

**Why Better:** Size is part of the type (`array`) or object (`vector`). `.at()` provides bounds-checked access. RAII handles deallocation. `std::span` replaces pointer+size pairs.

### 💡 What the Code Conveys

Modern containers carry their size with them. You can't have a size mismatch, and you can't forget to free them.

### 🔗 Cross-References

- See: [Chapter 6 — Arrays & Strings](../Part-01-CPP-Foundations/06_Arrays_Strings.md) for fundamentals
- See: [Chapter 17 — STL Containers](../Part-02-CPP-Intermediate/17_STL_Containers.md) for full coverage
- Used in: [Lab 09 — Reduction Optimization](../CUDA-Labs/Lab09_Reduction_Optimization.md)

---

## 3. C-strings (char*) → std::string/string_view

### ❌ Traditional C++ (Pre-C++11)

```cpp
char* build_greeting(const char* name) {
    char* buf = (char*)malloc(strlen("Hello, ") + strlen(name) + 2);
    if (!buf) return NULL;
    strcpy(buf, "Hello, ");
    strcat(buf, name);               // buffer overflow if we miscalculated
    strcat(buf, "!");
    return buf;                      // caller must remember to free()
}

void use() {
    char* msg = build_greeting("Alice");
    printf("%s\n", msg);
    free(msg);                       // forget this = leak
}
```

**Problems:** Manual buffer management. Must calculate sizes exactly or risk buffer overflow (a top security vulnerability). Caller owns the memory but nothing in the type says so. `NULL` confusion between "empty string" and "no string."

### ✅ Modern C++ (C++11/17)

```cpp
std::string build_greeting(std::string_view name) {   // C++17: no copy
    return std::format("Hello, {}!", name);            // C++20
}

void use() {
    auto msg = build_greeting("Alice");
    std::println("{}", msg);          // C++23
}   // msg freed automatically
```

**Why Better:** No buffer calculations, no overflow risk, no manual free. `string_view` avoids copies for read-only access. `std::format` is type-safe. Return by value is efficient thanks to move semantics and RVO.

### 💡 What the Code Conveys

Strings are values, not pointers. Ownership and lifetime are self-evident. Security vulnerabilities from buffer overflows are eliminated by design.

### 🔗 Cross-References

- See: [Chapter 6 — Arrays & Strings](../Part-01-CPP-Foundations/06_Arrays_Strings.md) for string fundamentals
- See: [Chapter 34 — C++17 Enhancements](../Part-04-Modern-CPP-Evolution/34_CPP17_Enhancements.md) for string_view
- Used in: [Project P01 — JSON Parser](../Part-09-Projects/P01_JSON_Parser.md)

---

## 4. Manual Memory Tracking → RAII

### ❌ Traditional C++ (Pre-C++11)

```cpp
bool process_file(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return false;

    char* buffer = (char*)malloc(4096);
    if (!buffer) {
        fclose(f);                   // must clean up f before returning
        return false;
    }

    DatabaseConn* db = db_connect("localhost");
    if (!db) {
        free(buffer);                // must clean up buffer AND f
        fclose(f);
        return false;
    }

    // ... actual work ...

    db_disconnect(db);               // cleanup in reverse order
    free(buffer);
    fclose(f);
    return true;
}
```

**Problems:** Each new resource adds cleanup to every previous error path. N resources → O(N²) cleanup code. One missed cleanup path = leak. Exception-unsafe — if any line throws, everything leaks.

### ✅ Modern C++ (C++11)

```cpp
bool process_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) return false;          // auto-closed on scope exit

    std::vector<char> buffer(4096);   // auto-freed on scope exit

    auto db = DatabaseConnection("localhost");  // RAII wrapper
    if (!db.connected()) return false;          // everything auto-cleaned

    // ... actual work ...
    return true;
}   // all resources released automatically, in reverse order
```

**Why Better:** Every resource is tied to a stack object. Cleanup happens automatically in reverse order of construction. Exception-safe by construction. Zero cleanup code. Adding a new resource doesn't touch any existing error paths.

### 💡 What the Code Conveys

RAII is the most important idiom in C++. If every resource is owned by a stack object, resource leaks become structurally impossible.

### 🔗 Cross-References

- See: [Chapter 9 — Dynamic Memory](../Part-01-CPP-Foundations/09_Dynamic_Memory.md) for RAII motivation
- See: [Chapter 16 — Smart Pointers](../Part-02-CPP-Intermediate/16_Smart_Pointers.md) for smart pointer RAII
- Used in: [Chapter 13 — OOP Classes](../Part-02-CPP-Intermediate/13_OOP_Classes.md)

---

## 5. malloc/free → new/delete → make_unique

### ❌ Traditional C++ (Pre-C++11)

```cpp
// C era: malloc/free
Widget* w1 = (Widget*)malloc(sizeof(Widget));  // no constructor called!
free(w1);                                       // no destructor called!

// C++98: new/delete
Widget* w2 = new Widget(42);
Widget* w3 = new Widget(99);
process(w2, w3);         // if new w3 throws, w2 leaks
delete w3;
delete w2;

// Array version: must match new[] with delete[]
int* arr = new int[100];
delete arr;              // BUG: should be delete[] — undefined behavior
```

**Problems:** `malloc` doesn't call constructors. `new` can leak on exceptions. `delete` vs `delete[]` mismatch is undefined behavior. Raw `new` in application code is a code smell.

### ✅ Modern C++ (C++14)

```cpp
// Single ownership
auto w1 = std::make_unique<Widget>(42);     // exception-safe, no leak
auto w2 = std::make_unique<Widget>(99);
process(*w1, *w2);      // if make_unique throws, previous ones auto-freed

// Shared ownership
auto shared = std::make_shared<Widget>(42); // single allocation for object+refcount

// Array version (C++20)
auto arr = std::make_unique_for_overwrite<int[]>(100); // uninitialized, fast
```

**Why Better:** `make_unique` is exception-safe, prevents `new`/`delete` mismatch, and eliminates `delete[]` confusion. The type encodes ownership semantics. Zero chance of using `free` on a `new`ed object.

### 💡 What the Code Conveys

In modern C++, raw `new` and `delete` should never appear in application code. Use `make_unique` for exclusive ownership, `make_shared` for shared ownership.

### 🔗 Cross-References

- See: [Chapter 9 — Dynamic Memory](../Part-01-CPP-Foundations/09_Dynamic_Memory.md) for allocation basics
- See: [Chapter 16 — Smart Pointers](../Part-02-CPP-Intermediate/16_Smart_Pointers.md) for make_unique/make_shared
- Used in: [Project P06 — CUDA Vector Ops](../Part-09-Projects/P06_CUDA_Vector_Ops.md)

---

## 6. Manual Resource Cleanup → RAII Destructors

### ❌ Traditional C++ (Pre-C++11)

```cpp
class NetworkSession {
    int socket_fd;
    SSL* ssl_ctx;
    char* recv_buffer;
public:
    NetworkSession(const char* host) {
        socket_fd = connect_to(host);
        ssl_ctx = ssl_init(socket_fd);
        recv_buffer = new char[8192];
    }

    // Must implement all of these correctly:
    ~NetworkSession() {
        delete[] recv_buffer;
        ssl_free(ssl_ctx);
        close(socket_fd);
    }
    NetworkSession(const NetworkSession&);             // deep copy? disable?
    NetworkSession& operator=(const NetworkSession&);  // same problem
};
```

**Problems:** Rule of Three: if you write a destructor, you must also write copy constructor and assignment operator. Forgetting any of them causes double-free or resource leaks on copy.

### ✅ Modern C++ (C++11)

```cpp
class NetworkSession {
    UniqueSocket socket;              // RAII wrapper for socket fd
    UniqueSSL ssl_ctx;                // RAII wrapper for SSL context
    std::vector<char> recv_buffer;    // RAII container
public:
    NetworkSession(std::string_view host)
        : socket(connect_to(host))
        , ssl_ctx(ssl_init(socket.get()))
        , recv_buffer(8192)
    {}
    // Rule of Zero: no destructor, no copy, no assignment needed
    // Implicitly move-only (because UniqueSocket is move-only)
};
```

**Why Better:** Each sub-resource manages itself. The class becomes trivially correct — no destructor, no copy operations to get wrong. Move semantics are auto-generated. The compiler does all the work.

### 💡 What the Code Conveys

The Rule of Zero: if every member manages its own lifetime, the class needs no special member functions. Correctness is compositional.

### 🔗 Cross-References

- See: [Chapter 13 — OOP Classes](../Part-02-CPP-Intermediate/13_OOP_Classes.md) for class design
- See: [Chapter 20 — Move Semantics](../Part-02-CPP-Intermediate/20_Move_Semantics.md) for Rule of Zero/Five
- Used in: [Project P04 — HTTP Server](../Part-09-Projects/P04_HTTP_Server.md)

---

# Type Safety

---

## 7. void* Casting → std::variant/std::any

### ❌ Traditional C++ (Pre-C++11)

```cpp
struct Config {
    void* value;
    int type;    // 0=int, 1=double, 2=string — must keep in sync
};

double get_double(Config* c) {
    if (c->type != 1)                // easy to get wrong
        return 0.0;                  // silent fallback — hides bugs
    return *(double*)c->value;       // undefined behavior if type is wrong
}

// Union alternative — same problems
union Value { int i; double d; char* s; };
struct TaggedValue { int tag; Value val; };
```

**Problems:** Type information is disconnected from the value. Wrong cast = undefined behavior. No compiler help to ensure all types are handled. `void*` erases all type information.

### ✅ Modern C++ (C++17)

```cpp
using ConfigValue = std::variant<int, double, std::string>;

double get_double(const ConfigValue& v) {
    return std::visit(overloaded{
        [](double d) { return d; },
        [](int i)    { return static_cast<double>(i); },
        [](const std::string&) -> double {
            throw std::invalid_argument("not numeric");
        }
    }, v);
}

// Compiler ERROR if you forget to handle a type!
```

**Why Better:** Type-safe: impossible to access the wrong type without a compile-time or runtime error. `std::visit` forces you to handle every alternative. The variant knows its own active type.

### 💡 What the Code Conveys

`variant` is a type-safe union — the compiler enforces exhaustive handling. `any` is the modern replacement for `void*` when the set of types is open-ended.

### 🔗 Cross-References

- See: [Chapter 30 — Type Erasure](../Part-03-CPP-Advanced/30_Type_Erasure.md) for variant/any/visit
- See: [Chapter 34 — C++17 Enhancements](../Part-04-Modern-CPP-Evolution/34_CPP17_Enhancements.md) for vocabulary types

---

## 8. C-style Casts → static_cast/dynamic_cast

### ❌ Traditional C++ (Pre-C++11)

```cpp
void process(void* data) {
    int* p = (int*)data;              // C cast: no checking at all
    double d = (double)(*p);          // could be reinterpret or static — who knows?

    Base* b = get_object();
    Derived* d = (Derived*)b;         // no runtime check — UB if wrong type
    d->derived_method();              // may crash or corrupt memory silently
}
```

**Problems:** C-style casts silently combine `static_cast`, `reinterpret_cast`, and `const_cast`. You can't search for them reliably. No runtime checking for polymorphic downcasts. If the type is wrong, you get silent undefined behavior.

### ✅ Modern C++ (C++11)

```cpp
void process(void* data) {
    auto* p = static_cast<int*>(data);         // intent is clear: type conversion
    auto d = static_cast<double>(*p);          // explicit, searchable

    Base* b = get_object();
    if (auto* d = dynamic_cast<Derived*>(b)) { // runtime-checked
        d->derived_method();                    // safe — type verified
    } else {
        handle_wrong_type();
    }
}
```

**Why Better:** Named casts are searchable, express intent, and catch errors. `static_cast` = safe conversion. `dynamic_cast` = runtime-checked downcast. `reinterpret_cast` = dangerous (rare and visible). `const_cast` = removing const (code smell flag).

### 💡 What the Code Conveys

Named casts document what kind of conversion is happening. Code review can spot `reinterpret_cast` instantly — C casts hide the danger.

### 🔗 Cross-References

- See: [Chapter 3 — Operators & Expressions](../Part-01-CPP-Foundations/03_Operators_Expressions.md) for cast fundamentals
- See: [Chapter 14 — Inheritance & Polymorphism](../Part-02-CPP-Intermediate/14_Inheritance_Polymorphism.md) for dynamic_cast

---

## 9. #define Constants → constexpr

### ❌ Traditional C++ (Pre-C++11)

```cpp
#define MAX_BUFFER_SIZE 4096
#define PI 3.14159265358979
#define SQUARE(x) ((x) * (x))       // macro: no type checking

void example() {
    int x = 5;
    int y = SQUARE(x++);             // BUG: x incremented twice! Expands to ((x++) * (x++))
    char buf[MAX_BUFFER_SIZE];       // no type, no scope, pollutes everything
}
```

**Problems:** Macros are text substitution — no type checking, no scope, no debugger support. Macro "functions" evaluate arguments multiple times, causing subtle bugs. `#define` pollutes the global namespace and can clash with any identifier.

### ✅ Modern C++ (C++11/14/20)

```cpp
constexpr std::size_t max_buffer_size = 4096;   // typed, scoped, debuggable
constexpr double pi = 3.14159265358979;

constexpr auto square(auto x) {                  // C++20: type-safe, evaluated once
    return x * x;
}

void example() {
    int x = 5;
    int y = square(x++);             // x incremented exactly once
    std::array<char, max_buffer_size> buf{};
}

// Compile-time computation
consteval int factorial(int n) {     // C++20: MUST be evaluated at compile time
    return n <= 1 ? 1 : n * factorial(n - 1);
}
static_assert(factorial(5) == 120);
```

**Why Better:** Type-safe. Scoped (respects namespaces). Debuggable. No multiple-evaluation bugs. `constexpr` enables compile-time computation. `consteval` guarantees it.

### 💡 What the Code Conveys

Constants should be typed values, not text substitution. `constexpr` replaces `#define` for constants and simple functions. There is almost never a reason to use `#define` for a constant.

### 🔗 Cross-References

- See: [Chapter 29 — Compile-Time Programming](../Part-03-CPP-Advanced/29_Compile_Time_Programming.md) for constexpr/consteval
- See: [Chapter 35 — C++20 Big Four](../Part-04-Modern-CPP-Evolution/35_CPP20_Big_Four.md) for consteval

---

## 10. NULL → nullptr

### ❌ Traditional C++ (Pre-C++11)

```cpp
#define NULL 0     // or ((void*)0) in C

void process(int x)    { std::cout << "int\n"; }
void process(char* p)  { std::cout << "pointer\n"; }

void example() {
    int* p = NULL;
    process(NULL);       // calls process(int)! NULL is just 0
    process(0);          // same ambiguity

    if (p == 0)          // comparing pointer to integer — confusing
        std::cout << "null\n";
}
```

**Problems:** `NULL` is the integer `0`, not a pointer type. This causes overload ambiguity. Comparing pointers with `0` is semantically confusing. Implicit conversions between `0`, `NULL`, and pointers cause subtle bugs.

### ✅ Modern C++ (C++11)

```cpp
void process(int x)       { std::cout << "int\n"; }
void process(char* p)     { std::cout << "pointer\n"; }

void example() {
    int* p = nullptr;
    process(nullptr);      // calls process(char*) — unambiguous!

    if (p == nullptr)      // clear intent: checking for null pointer
        std::cout << "null\n";

    // int x = nullptr;    // COMPILE ERROR: nullptr is not an integer
}
```

**Why Better:** `nullptr` has its own type (`std::nullptr_t`). It resolves overload ambiguity correctly. It cannot be implicitly converted to an integer. The code reads more clearly.

### 💡 What the Code Conveys

`nullptr` means "no object" — not "zero." The type system enforces the distinction. Every use of `NULL` or `0` as a null pointer should be replaced with `nullptr`.

### 🔗 Cross-References

- See: [Chapter 7 — Pointers Deep Dive](../Part-01-CPP-Foundations/07_Pointers_Deep_Dive.md) for pointer fundamentals
- See: [Chapter 33 — C++11/14 Revolution](../Part-04-Modern-CPP-Evolution/33_CPP11_14_Revolution.md) for nullptr

---

# Error Handling

---

## 11. Error Codes / errno → Exceptions

### ❌ Traditional C++ (Pre-C++11)

```cpp
int parse_config(const char* path, Config* out) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;                         // error code: what went wrong?

    char buf[1024];
    if (fread(buf, 1, 1024, f) == 0) {
        fclose(f);
        return -2;                             // different code — must document
    }

    if (parse_json(buf, out) != 0) {
        fclose(f);
        return -3;                             // caller must check EVERY return
    }

    fclose(f);
    return 0;                                  // 0 = success (by convention)
}

// Caller:
Config cfg;
int err = parse_config("app.json", &cfg);      // easy to ignore return value
if (err != 0) { /* handle... somehow */ }
```

**Problems:** Return values are easily ignored. Error codes are meaningless without documentation. Error-checking code drowns out the logic. Resources must be manually cleaned up on each error path.

### ✅ Modern C++ (C++11)

```cpp
Config parse_config(const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open: " + path);

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    return parse_json(content);   // throws on parse error with meaningful message
}

// Caller:
try {
    auto cfg = parse_config("app.json");
    // use cfg — we know it's valid
} catch (const std::exception& e) {
    std::cerr << "Config error: " << e.what() << '\n';
}
```

**Why Better:** Exceptions cannot be silently ignored. Error messages are descriptive. The happy path is clean and readable. RAII ensures resources are cleaned up automatically when exceptions propagate.

### 💡 What the Code Conveys

Exceptions separate error handling from normal logic. Combined with RAII, they make error handling both correct and invisible in the happy path.

### 🔗 Cross-References

- See: [Chapter 12 — Error Handling](../Part-01-CPP-Foundations/12_Error_Handling.md) for exception fundamentals
- See: [Chapter 22 — File I/O](../Part-02-CPP-Intermediate/22_File_IO.md) for file error handling

---

## 12. Returning -1 / Sentinel → std::optional

### ❌ Traditional C++ (Pre-C++11)

```cpp
int find_user_age(const char* name) {
    // returns -1 if not found
    for (int i = 0; i < num_users; i++)
        if (strcmp(users[i].name, name) == 0)
            return users[i].age;
    return -1;                        // sentinel: but what if age IS -1?
}

double* find_max(double* arr, int n) {
    if (n == 0) return NULL;          // caller must check for NULL
    double* max = &arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > *max) max = &arr[i];
    return max;                       // dangling pointer if arr is freed
}
```

**Problems:** Sentinel values are in-band — they overlap with valid values. `NULL` returns create null pointer dereference risks. The type doesn't communicate "might not have a value." Every sentinel convention must be documented and remembered.

### ✅ Modern C++ (C++17)

```cpp
std::optional<int> find_user_age(std::string_view name) {
    for (const auto& user : users)
        if (user.name == name)
            return user.age;          // has value
    return std::nullopt;              // explicitly: no value
}

// Caller — must handle the empty case:
if (auto age = find_user_age("Alice")) {
    std::cout << "Age: " << *age << '\n';
} else {
    std::cout << "User not found\n";
}

// Or with value_or:
int age = find_user_age("Bob").value_or(0);
```

**Why Better:** The type itself communicates "might be empty." No sentinel values needed. The compiler and the reader both know this function might not return a result. `value_or` provides safe defaults.

### 💡 What the Code Conveys

`std::optional<T>` means "a T or nothing." It replaces sentinel values, output parameters, and null pointers for representing absent values.

### 🔗 Cross-References

- See: [Chapter 30 — Type Erasure](../Part-03-CPP-Advanced/30_Type_Erasure.md) for optional usage
- See: [Chapter 34 — C++17 Enhancements](../Part-04-Modern-CPP-Evolution/34_CPP17_Enhancements.md) for vocabulary types

---

## 13. Exceptions for Expected Errors → std::expected

### ❌ Traditional C++ (Pre-C++11 / C++11 with exceptions)

```cpp
// Problem: using exceptions for expected failure modes
User parse_user(const std::string& json) {
    try {
        auto doc = parse_json(json);
        return User{doc["name"], doc["age"]};    // might throw on bad input
    } catch (const JsonError& e) {
        // Is this exceptional? Bad JSON is expected in production.
        // Exceptions are expensive when they happen frequently.
        throw;
    }
}

// Alternative: output parameter + bool
bool parse_user(const std::string& json, User& out, std::string& error) {
    // Three parameters for one logical operation — messy API
}
```

**Problems:** Exceptions are expensive for non-exceptional errors. Error codes lose type information. Output parameters are awkward. No standard way to return "value or error" cleanly before C++23.

### ✅ Modern C++ (C++23)

```cpp
std::expected<User, ParseError> parse_user(std::string_view json) {
    auto doc = parse_json(json);
    if (!doc)
        return std::unexpected(ParseError::invalid_json);

    auto name = doc->get_string("name");
    auto age = doc->get_int("age");
    if (!name || !age)
        return std::unexpected(ParseError::missing_field);

    return User{*name, *age};
}

// Caller:
auto result = parse_user(input);
if (result) {
    use_user(*result);
} else {
    log_error(result.error());   // typed error — not just a string
}
```

**Why Better:** Returns either a value or a typed error — no exceptions needed for expected failures. Zero overhead on the happy path. Error type is part of the function signature. Composes naturally with pattern matching.

### 💡 What the Code Conveys

`std::expected` is for operations that can fail in expected ways. Use exceptions for truly exceptional situations. Use `expected` for parse failures, network errors, and validation — things that happen routinely.

### 🔗 Cross-References

- See: [Chapter 36 — C++23 Refinements](../Part-04-Modern-CPP-Evolution/36_CPP23_Refinements.md) for std::expected
- See: [Chapter 12 — Error Handling](../Part-01-CPP-Foundations/12_Error_Handling.md) for exception philosophy

---

# Iteration & Algorithms

---

## 14. C-style for Loops → Range-based for

### ❌ Traditional C++ (Pre-C++11)

```cpp
std::vector<std::string> names;
// ... fill names ...

for (std::vector<std::string>::iterator it = names.begin();
     it != names.end(); ++it) {
    std::cout << *it << std::endl;          // verbose, noisy
}

// Or with indices — off-by-one risk
for (int i = 0; i < names.size(); i++) {    // signed/unsigned mismatch warning
    std::cout << names[i] << std::endl;
}
```

**Problems:** Iterator boilerplate is enormous. Index-based loops risk signed/unsigned mismatch and off-by-one errors. The intent ("process each element") is buried in mechanics. `std::endl` flushes unnecessarily.

### ✅ Modern C++ (C++11/20)

```cpp
std::vector<std::string> names;

for (const auto& name : names) {            // range-based for (C++11)
    std::cout << name << '\n';
}

// With index if needed (C++20 views)
for (auto [i, name] : names | std::views::enumerate) {   // C++23
    std::cout << i << ": " << name << '\n';
}

// Filter + transform in one pass
for (auto& name : names | std::views::filter([](auto& n) { return n.size() > 3; })
                        | std::views::transform([](auto& n) { return to_upper(n); })) {
    std::cout << name << '\n';
}
```

**Why Better:** Range-based for is shorter, safer (no off-by-one), and expresses intent. Views compose lazily — no intermediate allocations. `auto` eliminates type noise.

### 💡 What the Code Conveys

"For each element" is the most common loop pattern. Range-based for makes this pattern a one-liner. Views (C++20) enable composable, lazy pipelines.

### 🔗 Cross-References

- See: [Chapter 4 — Control Flow](../Part-01-CPP-Foundations/04_Control_Flow.md) for loop fundamentals
- See: [Chapter 35 — C++20 Big Four](../Part-04-Modern-CPP-Evolution/35_CPP20_Big_Four.md) for ranges/views

---

## 15. Manual Loops → STL Algorithms

### ❌ Traditional C++ (Pre-C++11)

```cpp
// Count elements matching a condition
int count = 0;
for (int i = 0; i < vec.size(); i++)
    if (vec[i] > threshold)
        count++;

// Find maximum
int max_val = vec[0];
for (int i = 1; i < vec.size(); i++)
    if (vec[i] > max_val)
        max_val = vec[i];

// Copy matching elements
std::vector<int> filtered;
for (int i = 0; i < vec.size(); i++)
    if (vec[i] % 2 == 0)
        filtered.push_back(vec[i]);
```

**Problems:** Every loop re-implements a well-known algorithm. Bugs hide in boilerplate. Intent is obscured by mechanics. Optimizer has less information to work with compared to named algorithms.

### ✅ Modern C++ (C++11/20)

```cpp
// Count elements matching a condition
auto count = std::ranges::count_if(vec, [](int x) { return x > threshold; });

// Find maximum
auto max_val = std::ranges::max(vec);

// Copy matching elements
auto filtered = vec | std::views::filter([](int x) { return x % 2 == 0; })
                    | std::ranges::to<std::vector>();   // C++23
```

**Why Better:** Algorithm names communicate intent instantly. Battle-tested implementations — fewer bugs. Ranges (C++20) compose naturally. The optimizer can apply SIMD and other optimizations to known algorithms.

### 💡 What the Code Conveys

If you're writing a loop, ask: "Is there an algorithm for this?" The answer is almost always yes. Named algorithms are faster to read, faster to review, and harder to get wrong.

### 🔗 Cross-References

- See: [Chapter 18 — STL Algorithms](../Part-02-CPP-Intermediate/18_STL_Algorithms.md) for full algorithm catalog
- See: [Chapter 35 — C++20 Big Four](../Part-04-Modern-CPP-Evolution/35_CPP20_Big_Four.md) for ranges

---

## 16. Function Pointers → Lambdas

### ❌ Traditional C++ (Pre-C++11)

```cpp
// Must define a named function elsewhere
bool is_even(int x) { return x % 2 == 0; }
bool greater_than_10(int x) { return x > 10; }  // need a new function for each threshold!

void process(std::vector<int>& v) {
    // Function pointer: no state capture
    std::remove_if(v.begin(), v.end(), is_even);

    // For stateful predicates: need a functor class (!)
    struct GreaterThan {
        int threshold;
        GreaterThan(int t) : threshold(t) {}
        bool operator()(int x) const { return x > threshold; }
    };
    std::remove_if(v.begin(), v.end(), GreaterThan(42));
}
```

**Problems:** Function pointers can't capture local state. Functor classes require 5+ lines of boilerplate for a one-line predicate. The logic is far from the call site. Each variation needs a new named entity.

### ✅ Modern C++ (C++11/14/20)

```cpp
void process(std::vector<int>& v) {
    // Inline, stateless
    std::erase_if(v, [](int x) { return x % 2 == 0; });     // C++20

    // Capture local state
    int threshold = 42;
    std::erase_if(v, [threshold](int x) { return x > threshold; });

    // Generic lambda (C++14)
    auto print = [](const auto& x) { std::cout << x << '\n'; };
    std::ranges::for_each(v, print);

    // Immediately invoked (IIFE) for complex initialization
    const auto config = [&] {
        Config c;
        c.load("settings.json");
        return c;
    }();
}
```

**Why Better:** Lambdas are inline, can capture state, and are the natural unit of customization for algorithms. Generic lambdas (C++14) work with any type. They keep logic at the point of use.

### 💡 What the Code Conveys

Lambdas are anonymous functions defined right where they're needed. They replace function pointers, functor classes, and `std::bind` — all in one concise syntax.

### 🔗 Cross-References

- See: [Chapter 19 — Lambdas & Functional](../Part-02-CPP-Intermediate/19_Lambdas_Functional.md) for full lambda coverage
- See: [Chapter 18 — STL Algorithms](../Part-02-CPP-Intermediate/18_STL_Algorithms.md) for algorithm + lambda patterns

---

## 17. Hand-written Sort Comparators → Lambda + auto

### ❌ Traditional C++ (Pre-C++11)

```cpp
struct Employee {
    std::string name;
    int salary;
    int department;
};

// Must define comparator as a separate struct
struct CompareBySalaryDesc {
    bool operator()(const Employee& a, const Employee& b) const {
        return a.salary > b.salary;
    }
};

struct CompareByNameAsc {
    bool operator()(const Employee& a, const Employee& b) const {
        return a.name < b.name;
    }
};

// Usage — comparator type is far from sort call
std::sort(employees.begin(), employees.end(), CompareBySalaryDesc());
```

**Problems:** Each sort order requires a separate named struct. The comparison logic is far from the sort call. Changing sort criteria means adding more boilerplate classes. Hard to compose multi-key sorts.

### ✅ Modern C++ (C++11/20)

```cpp
struct Employee {
    std::string name;
    int salary;
    int department;
};

// Inline comparator — logic at point of use
std::ranges::sort(employees, [](const auto& a, const auto& b) {
    return a.salary > b.salary;
});

// Even better with projections (C++20)
std::ranges::sort(employees, std::greater{}, &Employee::salary);

// Multi-key sort with spaceship operator (C++20)
std::ranges::sort(employees, [](const auto& a, const auto& b) {
    if (auto cmp = a.department <=> b.department; cmp != 0) return cmp < 0;
    return a.salary > b.salary;
});
```

**Why Better:** Lambda comparators are inline and readable. Projections (C++20) separate "what to compare" from "how to compare." The spaceship operator (`<=>`) simplifies multi-key sorting.

### 💡 What the Code Conveys

Sorting with projections reads like English: "sort employees by salary, descending." This is the highest level of abstraction for a comparison — no boilerplate classes needed.

### 🔗 Cross-References

- See: [Chapter 18 — STL Algorithms](../Part-02-CPP-Intermediate/18_STL_Algorithms.md) for sort + algorithms
- See: [Chapter 19 — Lambdas & Functional](../Part-02-CPP-Intermediate/19_Lambdas_Functional.md) for lambda patterns

---

# Classes & Polymorphism

---

## 18. Manual Copy Constructor → Rule of Zero

### ❌ Traditional C++ (Pre-C++11)

```cpp
class Buffer {
    char* data;
    size_t size;
public:
    Buffer(size_t n) : data(new char[n]), size(n) {}

    // Rule of Three: must implement all three
    ~Buffer() { delete[] data; }

    Buffer(const Buffer& other) : data(new char[other.size]), size(other.size) {
        std::memcpy(data, other.data, size);    // deep copy
    }

    Buffer& operator=(const Buffer& other) {
        if (this != &other) {                   // self-assignment check
            delete[] data;                       // free old data
            data = new char[other.size];          // allocate new
            size = other.size;
            std::memcpy(data, other.data, size); // copy
        }
        return *this;
    }
    // 30+ lines just for resource management!
};
```

**Problems:** Rule of Three requires writing destructor, copy constructor, and copy assignment. Each is error-prone. Self-assignment check is easy to forget. No move semantics — copies are always deep and expensive. Enormous boilerplate for a simple buffer.

### ✅ Modern C++ (C++11)

```cpp
class Buffer {
    std::vector<char> data;       // vector handles everything
public:
    Buffer(size_t n) : data(n) {}
    // Rule of Zero: no destructor, no copy, no move — all auto-generated!
    // - Copy: deep copy via vector's copy constructor
    // - Move: efficient pointer swap via vector's move constructor
    // - Destroy: automatic via vector's destructor
};

// If you need custom behavior, use Rule of Five (C++11):
class UniqueBuffer {
    std::unique_ptr<char[]> data;
    size_t size;
public:
    UniqueBuffer(size_t n) : data(std::make_unique<char[]>(n)), size(n) {}
    // Move-only: unique_ptr makes copy deleted automatically
};
```

**Why Better:** The Rule of Zero: if all members manage their own resources, the class needs zero special member functions. 30 lines become 3. Move semantics are free. Correctness is guaranteed by composition.

### 💡 What the Code Conveys

Prefer composition over manual resource management. Use `vector`, `string`, `unique_ptr` as members, and the compiler writes your copy/move/destroy for you.

### 🔗 Cross-References

- See: [Chapter 20 — Move Semantics](../Part-02-CPP-Intermediate/20_Move_Semantics.md) for Rule of Zero/Five
- See: [Chapter 13 — OOP Classes](../Part-02-CPP-Intermediate/13_OOP_Classes.md) for class design patterns

---

## 19. Virtual + Raw Pointers → CRTP for Static Polymorphism

### ❌ Traditional C++ (Pre-C++11)

```cpp
class Shape {
public:
    virtual double area() const = 0;
    virtual void draw() const = 0;
    virtual ~Shape() {}                        // virtual destructor required
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
    void draw() const override { /* ... */ }
};

// Runtime cost: vtable lookup on every call
void process(Shape* shapes[], int n) {
    for (int i = 0; i < n; i++)
        total += shapes[i]->area();            // indirect call — no inlining
}
```

**Problems:** Virtual dispatch has runtime cost (vtable lookup, indirect branch, no inlining). Requires heap allocation and pointer indirection. Virtual destructor adds overhead. Performance-critical code (inner loops, CUDA host setup) pays this cost millions of times.

### ✅ Modern C++ (C++11/20)

```cpp
// CRTP: Compile-time polymorphism — zero overhead
template <typename Derived>
class Shape {
public:
    double area() const {
        return static_cast<const Derived*>(this)->area_impl();
    }
};

class Circle : public Shape<Circle> {
    double radius;
public:
    Circle(double r) : radius(r) {}
    double area_impl() const { return 3.14159 * radius * radius; }
};

// C++20: Concepts instead of CRTP
template <typename T>
concept ShapeLike = requires(const T& s) {
    { s.area() } -> std::convertible_to<double>;
};

template <ShapeLike S>
double total_area(std::span<const S> shapes) {
    return std::ranges::fold_left(shapes, 0.0,
        [](double sum, const auto& s) { return sum + s.area(); });
}
```

**Why Better:** CRTP eliminates vtable overhead — calls are resolved at compile time and can be inlined. Concepts (C++20) provide cleaner syntax for static polymorphism. Use virtual when you need runtime polymorphism; use CRTP/concepts when types are known at compile time.

### 💡 What the Code Conveys

Not all polymorphism needs to be runtime. When types are known at compile time, static polymorphism gives you the abstraction without the performance cost.

### 🔗 Cross-References

- See: [Chapter 23 — Template Metaprogramming](../Part-03-CPP-Advanced/23_Template_Metaprogramming.md) for CRTP
- See: [Chapter 24 — Concepts & Constraints](../Part-03-CPP-Advanced/24_Concepts_Constraints.md) for concept-based polymorphism

---

## 20. printf → std::format/std::print

### ❌ Traditional C++ (Pre-C++11)

```cpp
// printf: type-unsafe, format string must match arguments
printf("Name: %s, Age: %d, Score: %.2f\n",
       name, age, score);          // wrong format specifier = UB

// %s with std::string = crash (must use .c_str())
std::string user = "Alice";
printf("User: %s\n", user);       // UB! should be user.c_str()

// iostream: type-safe but ugly
std::cout << "Name: " << name << ", Age: " << age
          << ", Score: " << std::fixed << std::setprecision(2)
          << score << std::endl;   // verbose, stateful formatting
```

**Problems:** `printf` format strings aren't type-checked — wrong specifier = undefined behavior. `%s` with `std::string` is a common crash. `iostream` is type-safe but extremely verbose and uses stateful formatting manipulators.

### ✅ Modern C++ (C++20/23)

```cpp
// std::format (C++20): type-safe, Python-like
auto msg = std::format("Name: {}, Age: {}, Score: {:.2f}", name, age, score);

// std::print (C++23): format + output in one step
std::println("Name: {}, Age: {}, Score: {:.2f}", name, age, score);

// Positional arguments
std::println("{1} is {0} years old", age, name);

// Custom type formatting
template <>
struct std::formatter<Point> {
    constexpr auto parse(auto& ctx) { return ctx.begin(); }
    auto format(const Point& p, auto& ctx) const {
        return std::format_to(ctx.out(), "({}, {})", p.x, p.y);
    }
};
std::println("Location: {}", my_point);   // "Location: (3, 4)"
```

**Why Better:** Type-safe at compile time. Clean Python/Rust-like syntax. No `c_str()` needed. Custom formatters for user types. `std::print` combines formatting and output.

### 💡 What the Code Conveys

`std::format` combines the safety of `iostream` with the readability of `printf`. It's the "best of both worlds" and should be the default for all new code.

### 🔗 Cross-References

- See: [Chapter 36 — C++23 Refinements](../Part-04-Modern-CPP-Evolution/36_CPP23_Refinements.md) for std::print/format
- See: [Chapter 1 — Hello C++](../Part-01-CPP-Foundations/01_Hello_CPP.md) for output basics

---

## 21. Macro-based Generics → Templates + Concepts

### ❌ Traditional C++ (Pre-C++11)

```cpp
// Macro "generics": textual substitution, no type safety
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SWAP(type, a, b) { type temp = a; a = b; b = temp; }

int result = MAX(x++, y++);   // BUG: one of x,y incremented twice

// C++98 templates: better but error messages are terrible
template <typename T>
T clamp(T value, T low, T high) {
    // What if T doesn't support operator< ?
    // Error message: 200 lines of template instantiation backtrace
    return value < low ? low : (value > high ? high : value);
}

clamp(std::string("hello"), 1, 10);  // nonsensical but compiles (maybe)
```

**Problems:** Macros have no type safety, cause double-evaluation bugs, and can't be debugged. C++98 templates are type-safe but produce incomprehensible error messages when misused. No way to constrain what types are valid.

### ✅ Modern C++ (C++20)

```cpp
// Concepts: explicit type requirements, clear errors
template <std::totally_ordered T>
T clamp(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}

// Custom concept
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
T safe_divide(T a, T b) {
    if (b == T{0}) throw std::domain_error("division by zero");
    return a / b;
}

// Compile error with clear message:
// safe_divide(std::string("a"), std::string("b"));
// error: constraint 'Numeric<std::string>' not satisfied
```

**Why Better:** Concepts constrain templates with readable requirements. Error messages say what's wrong, not 200 lines of template noise. Code is self-documenting: `Numeric` tells you exactly what types are valid.

### 💡 What the Code Conveys

Concepts are "type requirements as code." They replace SFINAE, `enable_if`, and documentation comments with compiler-checked constraints.

### 🔗 Cross-References

- See: [Chapter 21 — Templates](../Part-02-CPP-Intermediate/21_Templates.md) for template fundamentals
- See: [Chapter 24 — Concepts & Constraints](../Part-03-CPP-Advanced/24_Concepts_Constraints.md) for concepts

---

# Concurrency

---

## 22. pthreads → std::thread/jthread

### ❌ Traditional C++ (Pre-C++11)

```cpp
#include <pthread.h>

struct ThreadData {
    int id;
    double* results;
};

void* worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;   // void* cast — unsafe
    data->results[data->id] = compute(data->id);
    return NULL;
}

void parallel_compute() {
    pthread_t threads[4];
    ThreadData data[4];
    double results[4];

    for (int i = 0; i < 4; i++) {
        data[i] = {i, results};
        int err = pthread_create(&threads[i], NULL, worker, &data[i]);
        if (err != 0) { /* handle error... cleanup previous threads? */ }
    }

    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);    // must join every thread
}
```

**Problems:** `void*` parameter passing is unsafe. Manual thread lifecycle management. Not portable (POSIX-only). Must join every thread or face undefined behavior. Error handling is error-code based. Data sharing requires manual struct packing.

### ✅ Modern C++ (C++11/20)

```cpp
#include <thread>

void parallel_compute() {
    std::array<double, 4> results{};

    // C++20 jthread: auto-joins on destruction
    std::vector<std::jthread> threads;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back([&results, i] {   // type-safe lambda capture
            results[i] = compute(i);
        });
    }
}   // all jthreads auto-join here — no manual cleanup

// With stop token (C++20)
std::jthread worker([](std::stop_token st) {
    while (!st.stop_requested()) {
        do_work();
    }
});
worker.request_stop();   // cooperative cancellation
```

**Why Better:** Portable across all platforms. Type-safe parameter passing via lambdas. `jthread` auto-joins — no resource leak possible. Stop tokens enable cooperative cancellation. No `void*` casts.

### 💡 What the Code Conveys

`std::jthread` is RAII for threads — it joins automatically on destruction, just like `unique_ptr` frees memory. Combined with lambdas, thread creation is a one-liner.

### 🔗 Cross-References

- See: [Chapter 25 — Concurrency](../Part-03-CPP-Advanced/25_Concurrency.md) for thread fundamentals
- See: [Chapter 35 — C++20 Big Four](../Part-04-Modern-CPP-Evolution/35_CPP20_Big_Four.md) for jthread
- Used in: [Project P03 — Thread Pool](../Part-09-Projects/P03_Thread_Pool.md)

---

## 23. pthread_mutex → std::mutex/lock_guard/scoped_lock

### ❌ Traditional C++ (Pre-C++11)

```cpp
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void transfer(Account& from, Account& to, double amount) {
    pthread_mutex_lock(&mutex);
    pthread_mutex_lock(&mutex2);         // DEADLOCK RISK: lock ordering

    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance += amount;
    }

    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex);        // must unlock in reverse order

    // What if an exception is thrown between lock and unlock?
    // Mutex stays locked forever → deadlock
}
```

**Problems:** Manual lock/unlock is error-prone. Forgetting to unlock (especially on exception or early return) causes deadlocks. Two mutexes must be locked in consistent order to avoid deadlocks. No RAII — not exception-safe.

### ✅ Modern C++ (C++11/17)

```cpp
std::mutex mtx_from, mtx_to;

void transfer(Account& from, Account& to, double amount) {
    // C++17 scoped_lock: locks both mutexes atomically, no deadlock
    std::scoped_lock lock(mtx_from, mtx_to);

    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance += amount;
    }
}   // both mutexes automatically unlocked — even on exception

// Read-heavy workload: shared_mutex (C++17)
std::shared_mutex rw_mutex;

int read_data() {
    std::shared_lock lock(rw_mutex);     // multiple readers allowed
    return cached_value;
}

void write_data(int val) {
    std::unique_lock lock(rw_mutex);     // exclusive write access
    cached_value = val;
}
```

**Why Better:** `scoped_lock` locks multiple mutexes deadlock-free and unlocks automatically. Exception-safe — mutex is always released. `shared_mutex` enables multiple readers / single writer. No manual lock/unlock calls.

### 💡 What the Code Conveys

Mutexes follow RAII: lock in constructor, unlock in destructor. `scoped_lock` handles multiple mutexes atomically. You should never call `.lock()` and `.unlock()` directly.

### 🔗 Cross-References

- See: [Chapter 25 — Concurrency](../Part-03-CPP-Advanced/25_Concurrency.md) for mutex fundamentals
- See: [Chapter 27 — Memory Model & Lock-Free](../Part-03-CPP-Advanced/27_Memory_Model_Lock_Free.md) for advanced patterns

---

## 24. Manual Thread Management → std::async/future

### ❌ Traditional C++ (Pre-C++11)

```cpp
struct Result { double value; bool ready; };

Result global_result;
pthread_mutex_t result_mutex;

void* compute_async(void* arg) {
    double val = expensive_computation(*(int*)arg);

    pthread_mutex_lock(&result_mutex);
    global_result.value = val;
    global_result.ready = true;
    pthread_mutex_unlock(&result_mutex);
    return NULL;
}

double get_result() {
    // Busy-wait or use condition variables (more boilerplate)
    while (true) {
        pthread_mutex_lock(&result_mutex);
        if (global_result.ready) {
            double val = global_result.value;
            pthread_mutex_unlock(&result_mutex);
            return val;
        }
        pthread_mutex_unlock(&result_mutex);
        usleep(1000);   // polling — wastes CPU or adds latency
    }
}
```

**Problems:** Global mutable state. Manual synchronization with mutexes. Busy-waiting or complex condition variable setup. No standard way to propagate exceptions from worker thread. Thread lifecycle management is manual.

### ✅ Modern C++ (C++11)

```cpp
// std::async: fire-and-forget parallelism
auto future = std::async(std::launch::async, [](int x) {
    return expensive_computation(x);
}, 42);

// Do other work while computation runs...

double result = future.get();   // blocks until ready, propagates exceptions

// Multiple parallel tasks
std::vector<std::future<double>> futures;
for (int i = 0; i < 8; i++) {
    futures.push_back(std::async(std::launch::async, compute, i));
}

double total = 0;
for (auto& f : futures)
    total += f.get();   // collect results, exceptions auto-propagated
```

**Why Better:** No global state. No manual synchronization. Exceptions propagate naturally through `future.get()`. The runtime manages thread lifecycle. Clean separation of "start computation" and "get result."

### 💡 What the Code Conveys

`std::async` + `std::future` is the simplest correct way to run work in parallel. The future is a handle to a value that will be available later — a promise of a result.

### 🔗 Cross-References

- See: [Chapter 26 — Async & Parallel](../Part-03-CPP-Advanced/26_Async_Parallel.md) for async/future patterns
- Used in: [Project P03 — Thread Pool](../Part-09-Projects/P03_Thread_Pool.md)

---

# Data Structures

---

## 25. Linked List → std::vector (Almost Always)

### ❌ Traditional C++ (Pre-C++11)

```cpp
struct Node {
    int data;
    Node* next;
};

class LinkedList {
    Node* head;
public:
    LinkedList() : head(NULL) {}

    void push_front(int val) {
        Node* n = new Node;
        n->data = val;
        n->next = head;
        head = n;
    }

    ~LinkedList() {
        while (head) {
            Node* tmp = head;
            head = head->next;
            delete tmp;            // manual cleanup, node by node
        }
    }
    // Must also implement copy ctor, operator=...
};
// Each node: separate heap allocation, cache-hostile pointer chasing
```

**Problems:** Each node is a separate heap allocation — terrible cache locality. Pointer chasing kills prefetcher. O(n) access by index. Manual memory management for each node. Must implement Rule of Three.

### ✅ Modern C++ (C++11)

```cpp
// std::vector is almost always the right choice
std::vector<int> data;
data.push_back(42);                        // amortized O(1)
data.insert(data.begin(), 99);             // O(n) but cache-friendly
data.erase(data.begin() + 3);             // O(n) but still fast

// Even for "middle insertion" workloads, vector often wins due to cache:
// Linked list: 1M nodes, each a cache miss → slow
// Vector: 1M elements, contiguous memory → fast even with shifts

// If you really need a list (rare): use std::list
std::list<int> linked;                     // RAII, no manual memory management
linked.push_front(42);
linked.splice(linked.begin(), other_list); // O(1) splice — list's one advantage
```

**Why Better:** `std::vector` is contiguous in memory — the CPU prefetcher loves it. Even O(n) operations on vector are often faster than O(1) list operations due to cache effects. Bjarne Stroustrup showed this in benchmarks: vector beats list for almost all workloads.

### 💡 What the Code Conveys

Default to `std::vector`. Use `std::list` only when you need iterator stability during insertion/deletion and have profiling data showing list is faster. This is rare.

### 🔗 Cross-References

- See: [Chapter 17 — STL Containers](../Part-02-CPP-Intermediate/17_STL_Containers.md) for container selection
- See: [Appendix B — Performance Cheat Sheet](B_Performance_Cheat_Sheet.md) for cache effects

---

## 26. Manual Hash Map → std::unordered_map

### ❌ Traditional C++ (Pre-C++11)

```cpp
#define TABLE_SIZE 1024

struct Entry {
    char key[64];
    int value;
    Entry* next;        // chaining for collisions
};

Entry* table[TABLE_SIZE];

unsigned hash(const char* key) {
    unsigned h = 0;
    while (*key) h = h * 31 + *key++;
    return h % TABLE_SIZE;
}

void insert(const char* key, int value) {
    unsigned idx = hash(key);
    Entry* e = (Entry*)malloc(sizeof(Entry));
    strcpy(e->key, key);               // buffer overflow if key > 63 chars
    e->value = value;
    e->next = table[idx];
    table[idx] = e;
}

int lookup(const char* key) {
    unsigned idx = hash(key);
    for (Entry* e = table[idx]; e; e = e->next)
        if (strcmp(e->key, key) == 0)
            return e->value;
    return -1;                          // sentinel — what if -1 is a valid value?
}
```

**Problems:** Manual hash function. Buffer overflow risk. No automatic resizing. Sentinel return values. Memory leaks if entries aren't freed. Not generic — hardcoded for `char*` → `int`.

### ✅ Modern C++ (C++11)

```cpp
std::unordered_map<std::string, int> table;

table["Alice"] = 42;
table.emplace("Bob", 99);          // in-place construction

// Safe lookup with optional-like pattern
if (auto it = table.find("Alice"); it != table.end()) {   // C++17 init-stmt
    std::cout << it->second << '\n';
}

// Or with structured bindings (C++17)
for (const auto& [name, value] : table) {
    std::println("{}: {}", name, value);
}

// C++20: contains()
if (table.contains("Alice")) { /* ... */ }
```

**Why Better:** Auto-resizing hash table with good default hash. Type-safe. No buffer overflows. No sentinel values. RAII handles all memory. Generic — works with any hashable key type.

### 💡 What the Code Conveys

`std::unordered_map` is the standard hash table. Use it instead of writing your own. For ordered keys, use `std::map` (red-black tree).

### 🔗 Cross-References

- See: [Chapter 17 — STL Containers](../Part-02-CPP-Intermediate/17_STL_Containers.md) for container comparison
- Used in: [Project P01 — JSON Parser](../Part-09-Projects/P01_JSON_Parser.md)

---

## 27. Pair with .first/.second → Structured Bindings

### ❌ Traditional C++ (Pre-C++11)

```cpp
std::map<std::string, int> scores;

// Insert and check success — .first and .second are cryptic
std::pair<std::map<std::string, int>::iterator, bool> result =
    scores.insert(std::make_pair("Alice", 95));

if (result.second) {
    std::cout << "Inserted: " << result.first->second << std::endl;
    // result.first->second: the value of the pair pointed to by the iterator
    // ... which member of which pair? Unreadable.
}

// Iterating:
for (std::map<std::string, int>::const_iterator it = scores.begin();
     it != scores.end(); ++it) {
    std::cout << it->first << ": " << it->second << std::endl;
}
```

**Problems:** `.first` and `.second` carry no semantic meaning. Nested pairs are incomprehensible (`result.first->second`). Full type names are enormous. Code is write-only — unreadable after a week.

### ✅ Modern C++ (C++17)

```cpp
std::map<std::string, int> scores;

// Structured bindings: meaningful names
auto [iter, inserted] = scores.emplace("Alice", 95);   // C++17

if (inserted) {
    auto& [name, score] = *iter;
    std::println("Inserted: {} = {}", name, score);
}

// Iterating with structured bindings
for (const auto& [name, score] : scores) {
    std::println("{}: {}", name, score);
}

// Works with arrays, tuples, and any aggregate
auto [x, y, z] = get_coordinates();
auto [min_val, max_val] = std::minmax({3, 1, 4, 1, 5});
```

**Why Better:** Names replace `.first`/`.second` — code reads like prose. Works with pairs, tuples, arrays, and structs. Combined with `auto`, eliminates type noise entirely.

### 💡 What the Code Conveys

Structured bindings give names to the unnamed. They turn `result.first->second` into `score` — the single biggest readability improvement in C++17.

### 🔗 Cross-References

- See: [Chapter 34 — C++17 Enhancements](../Part-04-Modern-CPP-Evolution/34_CPP17_Enhancements.md) for structured bindings
- Used in: [Chapter 17 — STL Containers](../Part-02-CPP-Intermediate/17_STL_Containers.md)

---

# Modern Idioms

---

## 28. Output Parameters → Return by Value (Move Semantics)

### ❌ Traditional C++ (Pre-C++11)

```cpp
// Output parameter: caller must pre-allocate, function fills in
void get_large_data(std::vector<double>& out) {
    out.clear();
    out.reserve(1000000);
    for (int i = 0; i < 1000000; i++)
        out.push_back(compute(i));
}

// Pointer to output: even worse
bool read_file(const char* path, std::string* content, std::string* error) {
    // Three pointer parameters for one logical operation
    // Which are inputs? Which are outputs? Unclear.
}

// Why? Because returning by value copies the entire vector.
// Copying 1M doubles = 8MB memcpy. Too expensive.
```

**Problems:** Output parameters invert the natural data flow. Callers must pre-allocate containers. It's unclear which parameters are inputs vs outputs. The only reason for this pattern was the cost of copying, which is now solved by move semantics and RVO.

### ✅ Modern C++ (C++11)

```cpp
// Return by value: natural data flow, zero-cost thanks to move/RVO
std::vector<double> get_large_data() {
    std::vector<double> result;
    result.reserve(1000000);
    for (int i = 0; i < 1000000; i++)
        result.push_back(compute(i));
    return result;   // NRVO: no copy, no move — constructed in place
}

// Multiple return values: use structured bindings
auto read_file(const std::string& path) -> std::expected<std::string, Error> {
    // Single return value: the content or the error
}

// Usage: clean, natural
auto data = get_large_data();              // zero-cost return
auto content = read_file("data.txt");      // expected<string, Error>
```

**Why Better:** Natural data flow: functions produce values, not fill buffers. Return Value Optimization (RVO/NRVO) eliminates copies entirely — the object is constructed directly in the caller's variable. Move semantics provide a fallback: moving a million-element vector is just 3 pointer swaps.

### 💡 What the Code Conveys

In modern C++, returning objects by value is free. Prefer returning values over output parameters. The compiler's RVO and move semantics make this the fastest option.

### 🔗 Cross-References

- See: [Chapter 20 — Move Semantics](../Part-02-CPP-Intermediate/20_Move_Semantics.md) for move/RVO details
- See: [Chapter 8 — References & Value Categories](../Part-01-CPP-Foundations/08_References_Value_Categories.md) for value categories

---

## 29. Type Traits + SFINAE → Concepts

### ❌ Traditional C++ (C++11/14)

```cpp
// SFINAE: "Substitution Failure Is Not An Error" — clever but cryptic
template <typename T,
          typename = typename std::enable_if<
              std::is_arithmetic<T>::value &&
              !std::is_same<T, bool>::value
          >::type>
T safe_divide(T a, T b) {
    if (b == T{0}) throw std::domain_error("division by zero");
    return a / b;
}

// Error message when misused:
// error: no matching function for call to 'safe_divide(std::string, std::string)'
// note: candidate template ignored: substitution failure [with T = std::string]:
//       no type named 'type' in 'struct std::enable_if<false, void>'
// ... what?
```

**Problems:** SFINAE is an accidental language feature exploited for metaprogramming. The syntax is impenetrable. Error messages reference template internals, not the actual constraint. Composing multiple constraints is a bracket-counting nightmare.

### ✅ Modern C++ (C++20)

```cpp
// Concepts: constraints as readable, composable declarations
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T> && !std::same_as<T, bool>;

template <Arithmetic T>
T safe_divide(T a, T b) {
    if (b == T{0}) throw std::domain_error("division by zero");
    return a / b;
}

// Error message when misused:
// error: constraint 'Arithmetic<std::string>' not satisfied
// note: because 'std::is_arithmetic_v<std::string>' evaluated to false
// Beautiful.

// Composable constraints:
template <typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template <typename T>
concept Loggable = Arithmetic<T> && Printable<T>;
```

**Why Better:** Concepts are readable, composable, and produce clear error messages. They replace `enable_if`, `void_t`, and tag dispatch with a first-class language feature. Constraints are self-documenting.

### 💡 What the Code Conveys

SFINAE was a hack. Concepts are the real thing. They let you write "this function works with arithmetic types" directly in the language.

### 🔗 Cross-References

- See: [Chapter 24 — Concepts & Constraints](../Part-03-CPP-Advanced/24_Concepts_Constraints.md) for full concepts coverage
- See: [Chapter 23 — Template Metaprogramming](../Part-03-CPP-Advanced/23_Template_Metaprogramming.md) for SFINAE history

---

## 30. #include Headers → import Modules

### ❌ Traditional C++ (Pre-C++20)

```cpp
// header.h
#ifndef MY_HEADER_H          // include guard boilerplate
#define MY_HEADER_H

#include <vector>             // transitively includes <memory>, <algorithm>, ...
#include <string>             // order matters! macros from previous includes affect this

struct Widget {
    std::string name;
    void process();           // declaration only
};

#endif

// translation unit 1
#include "header.h"           // entire header textually pasted — parsed again
// translation unit 2
#include "header.h"           // same header parsed AGAIN — O(N*M) compile time

// Problems: macros from one header break another, include order matters,
// compile times scale with #include depth × translation units
```

**Problems:** Headers are textually included — parsed repeatedly across translation units. Include order can cause different behavior. Macros leak across headers. No encapsulation. Compile times are proportional to total included lines × files.

### ✅ Modern C++ (C++20)

```cpp
// widget.cppm — module interface unit
export module widget;

import <string>;           // only imports the interface, not macros
import <vector>;

export struct Widget {
    std::string name;
    void process();        // can also define inline
};

// Only exported symbols are visible — true encapsulation
// Non-exported helpers are hidden from importers

// consumer.cpp
import widget;             // binary interface — parsed once, reused everywhere
// No macros leak, no include order issues, faster compilation
```

**Why Better:** Modules are compiled once and imported as binary interfaces — dramatically faster build times. No macro leakage. True encapsulation (non-exported symbols are hidden). No include guards needed.

### 💡 What the Code Conveys

Modules are C++'s answer to the 40-year-old `#include` problem. They give C++ the module system that Python, Rust, and Go have had all along.

### 🔗 Cross-References

- See: [Chapter 35 — C++20 Big Four](../Part-04-Modern-CPP-Evolution/35_CPP20_Big_Four.md) for modules
- See: [Chapter 11 — Namespaces & Headers](../Part-01-CPP-Foundations/11_Namespaces_Headers.md) for header fundamentals

---

## 31. Callback Hell → Coroutines

### ❌ Traditional C++ (Pre-C++20)

```cpp
// Nested callbacks for async operations
void fetch_user_data(int user_id, std::function<void(User)> on_success,
                     std::function<void(Error)> on_error) {
    http_get("/api/users/" + std::to_string(user_id),
        [=](Response resp) {
            if (resp.status != 200) {
                on_error(Error{"HTTP " + std::to_string(resp.status)});
                return;
            }
            parse_json_async(resp.body,
                [=](User user) {
                    fetch_permissions(user.id,
                        [=](Perms perms) {          // 4 levels deep!
                            user.perms = perms;
                            on_success(user);
                        },
                        on_error);
                },
                on_error);
        },
        on_error);
}
```

**Problems:** Each async operation nests another callback. Error handling is duplicated at every level. Logic flows right and down instead of top-to-bottom. Debugging is nearly impossible — stack traces show callback dispatch, not the logical flow.

### ✅ Modern C++ (C++20)

```cpp
// Coroutines: async code that reads like sync code
Task<User> fetch_user_data(int user_id) {
    auto resp = co_await http_get("/api/users/" + std::to_string(user_id));

    if (resp.status != 200)
        throw Error{"HTTP " + std::to_string(resp.status)};

    auto user = co_await parse_json<User>(resp.body);
    user.perms = co_await fetch_permissions(user.id);

    co_return user;
}

// Generator coroutine — lazy infinite sequence
Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        std::tie(a, b) = std::pair{b, a + b};
    }
}

for (int x : fibonacci() | std::views::take(10)) {
    std::println("{}", x);
}
```

**Why Better:** Coroutines turn nested callbacks into sequential code. `co_await` suspends and resumes without blocking a thread. `co_yield` creates lazy generators. The logical flow reads top-to-bottom.

### 💡 What the Code Conveys

Coroutines let you write asynchronous code that reads like synchronous code. They're the foundation for async I/O, generators, and state machines in modern C++.

### 🔗 Cross-References

- See: [Chapter 35 — C++20 Big Four](../Part-04-Modern-CPP-Evolution/35_CPP20_Big_Four.md) for coroutines
- See: [Chapter 26 — Async & Parallel](../Part-03-CPP-Advanced/26_Async_Parallel.md) for async patterns

---

## 32. sprintf → std::format

### ❌ Traditional C++ (Pre-C++20)

```cpp
char buffer[256];

// sprintf: no bounds checking — buffer overflow
sprintf(buffer, "User %s (age %d) scored %.1f%%",
        name, age, score);

// snprintf: bounds-checked but truncation is silent
int written = snprintf(buffer, sizeof(buffer),
    "User %s (age %d) scored %.1f%%",
    name, age, score);
if (written >= sizeof(buffer)) {
    // truncated — now what?
}

// Wrong format specifier = undefined behavior:
sprintf(buffer, "%d", 3.14);    // UB! %d expects int, got double
sprintf(buffer, "%s", 42);      // UB! %s expects char*, got int — crash
```

**Problems:** `sprintf` has no bounds checking — the #1 security vulnerability in C/C++ code. `snprintf` truncates silently. Format specifiers must match argument types exactly — mismatch is undefined behavior. Not type-safe.

### ✅ Modern C++ (C++20/23)

```cpp
// std::format: type-safe, no buffer overflow, Python-like syntax
auto msg = std::format("User {} (age {}) scored {:.1f}%", name, age, score);

// Compile-time format string validation (C++20)
// std::format("{:d}", "hello");   // COMPILE ERROR: invalid format for string

// Dynamic width/precision
auto table_row = std::format("{:<20} {:>8.2f}", product_name, price);

// std::format_to: write to any output iterator
std::string result;
std::format_to(std::back_inserter(result),
    "({}, {})", point.x, point.y);

// std::print: direct output (C++23) — no intermediate string
std::println("User {} (age {}) scored {:.1f}%", name, age, score);
```

**Why Better:** Compile-time format string validation prevents type mismatches. No buffer overflow — returns `std::string`. Python/Rust-style syntax is more readable. `std::print` (C++23) is the complete replacement for both `printf` and `iostream`.

### 💡 What the Code Conveys

`std::format` is the end of the `printf` vs `iostream` debate. It combines type safety with readability. Every new C++ project should use it.

### 🔗 Cross-References

- See: [Chapter 36 — C++23 Refinements](../Part-04-Modern-CPP-Evolution/36_CPP23_Refinements.md) for std::format/print
- See: [Chapter 33 — C++11/14 Revolution](../Part-04-Modern-CPP-Evolution/33_CPP11_14_Revolution.md) for to_string

---

# Bonus: The Complete Before/After

---

## 33. The Complete Before/After

A realistic program: read a CSV file of employee records, filter by department, compute average salary, and output a report.

### ❌ Traditional C++ — The Old Way (~90 lines)

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Employee {
    char name[64];
    char department[32];
    double salary;
};

// Parse CSV line into Employee, return 0 on success, -1 on error
int parse_employee(const char* line, Employee* out) {
    char* copy = strdup(line);
    if (!copy) return -1;

    char* token = strtok(copy, ",");
    if (!token) { free(copy); return -1; }
    strncpy(out->name, token, 63);
    out->name[63] = '\0';

    token = strtok(NULL, ",");
    if (!token) { free(copy); return -1; }
    strncpy(out->department, token, 31);
    out->department[31] = '\0';

    token = strtok(NULL, ",");
    if (!token) { free(copy); return -1; }
    out->salary = atof(token);

    free(copy);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <file> <department>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }

    Employee* employees = NULL;
    int count = 0;
    int capacity = 0;
    char line[256];

    // Skip header
    fgets(line, sizeof(line), f);

    while (fgets(line, sizeof(line), f)) {
        // Remove newline
        line[strcspn(line, "\n")] = 0;

        if (count >= capacity) {
            capacity = capacity == 0 ? 16 : capacity * 2;
            Employee* tmp = (Employee*)realloc(employees,
                                                capacity * sizeof(Employee));
            if (!tmp) {
                free(employees);
                fclose(f);
                fprintf(stderr, "Out of memory\n");
                return 1;
            }
            employees = tmp;
        }

        if (parse_employee(line, &employees[count]) == 0) {
            count++;
        }
    }
    fclose(f);

    // Filter by department and compute average
    double total_salary = 0;
    int dept_count = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(employees[i].department, argv[2]) == 0) {
            total_salary += employees[i].salary;
            dept_count++;
            printf("  %s: $%.2f\n", employees[i].name, employees[i].salary);
        }
    }

    if (dept_count > 0) {
        printf("\nDepartment: %s\n", argv[2]);
        printf("Employees: %d\n", dept_count);
        printf("Average Salary: $%.2f\n", total_salary / dept_count);
    } else {
        printf("No employees in department '%s'\n", argv[2]);
    }

    free(employees);
    return 0;
}
```

**Problems with this code:**
- Manual dynamic array with `realloc` — easy to corrupt
- `strdup` / `strtok` / `strncpy` — buffer overflow risks everywhere
- Multiple `free`/`fclose` cleanup paths — each must be correct
- Error handling clutters the logic
- `atof` silently returns 0 on bad input
- Not exception-safe (though no exceptions are used)
- 90 lines of mostly mechanical code

---

### ✅ Modern C++ — The Modern Way (~45 lines)

```cpp
#include <algorithm>
#include <fstream>
#include <numeric>
#include <print>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

struct Employee {
    std::string name;
    std::string department;
    double salary{};
};

Employee parse_employee(std::string_view line) {
    std::istringstream stream{std::string(line)};
    Employee emp;
    std::getline(stream, emp.name, ',');
    std::getline(stream, emp.department, ',');
    stream >> emp.salary;
    return emp;                             // RVO — zero-cost return
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::println(stderr, "Usage: {} <file> <department>", argv[0]);
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::println(stderr, "Cannot open {}", argv[1]);
        return 1;
    }

    std::string header;
    std::getline(file, header);

    std::vector<Employee> employees;
    for (std::string line; std::getline(file, line);)
        employees.push_back(parse_employee(line));
    // file auto-closed here

    auto dept = std::string_view(argv[2]);
    auto in_dept = employees | std::views::filter(
        [&](const auto& e) { return e.department == dept; });

    for (const auto& e : in_dept)
        std::println("  {}: ${:.2f}", e.name, e.salary);

    auto salaries = in_dept | std::views::transform(&Employee::salary);
    auto total = std::ranges::fold_left(salaries, 0.0, std::plus{});
    auto count = std::ranges::distance(in_dept);

    if (count > 0) {
        std::println("\nDepartment: {}", dept);
        std::println("Employees: {}", count);
        std::println("Average Salary: ${:.2f}", total / count);
    } else {
        std::println("No employees in department '{}'", dept);
    }
}
```

**What changed and why:**

| Aspect | Old | Modern | Improvement |
|--------|-----|--------|-------------|
| Strings | `char[64]`, `strncpy` | `std::string` | No buffer overflow |
| Dynamic array | `realloc` manual | `std::vector` | Automatic, safe |
| File I/O | `FILE*`, `fclose` | `std::ifstream` | RAII, auto-close |
| Parsing | `strtok`, `atof` | `getline`, `>>` | Type-safe, no mutation |
| Filtering | Manual loop | `views::filter` | Composable, lazy |
| Aggregation | Manual loop | `fold_left` | Named algorithm |
| Output | `printf` | `std::println` | Type-safe formatting |
| Memory | `malloc`/`free` | Automatic | No leaks possible |
| Lines | ~90 | ~45 | 50% reduction |

### 💡 What the Code Conveys

Modern C++ isn't just "newer syntax" — it's a fundamentally different approach. Resources manage themselves. Algorithms have names. Types carry meaning. The code reads like a description of what it does, not how it does it.

---

# Modernization Checklist

> **"Find and replace" patterns for upgrading a legacy C++ codebase.**

Use this checklist when modernizing existing code. Ordered by impact — start from the top.

### 🔴 Critical — Fix First (Safety & Correctness)

| # | Find (Old Pattern) | Replace With | Standard | Risk if Ignored |
|---|-------------------|--------------|----------|-----------------|
| 1 | `new T(...)` / `delete p` | `std::make_unique<T>(...)` | C++14 | Memory leak, double-free |
| 2 | `new T[n]` / `delete[] p` | `std::vector<T>(n)` or `make_unique<T[]>(n)` | C++11 | Array/non-array delete mismatch |
| 3 | `NULL` or `0` for pointers | `nullptr` | C++11 | Overload ambiguity |
| 4 | C-style cast `(Type*)expr` | `static_cast<Type*>(expr)` | C++11 | Silent reinterpret |
| 5 | `sprintf` / `strcpy` / `strcat` | `std::format` / `std::string` | C++20 | Buffer overflow (CVE-class) |
| 6 | `malloc`/`free` | `new`/smart pointers | C++11 | No ctor/dtor called |

### 🟡 Important — Fix Next (Reliability & Readability)

| # | Find (Old Pattern) | Replace With | Standard | Benefit |
|---|-------------------|--------------|----------|---------|
| 7 | `#define CONST 42` | `constexpr int CONST = 42;` | C++11 | Type safety, scoping |
| 8 | `#define MACRO(x) ...` | `constexpr` function or template | C++11 | No double-eval bugs |
| 9 | `typedef old new` | `using new = old;` | C++11 | Template alias support |
| 10 | `for (iter = begin; ...)` | `for (auto& x : container)` | C++11 | No off-by-one |
| 11 | `.first` / `.second` | Structured bindings `auto [a, b]` | C++17 | Readability |
| 12 | Output parameters `void f(T& out)` | Return by value `T f()` | C++11 | Natural data flow |
| 13 | `push_back(T(...))` | `emplace_back(...)` | C++11 | Avoids temporary |

### 🟢 Recommended — Modernize (Clarity & Performance)

| # | Find (Old Pattern) | Replace With | Standard | Benefit |
|---|-------------------|--------------|----------|---------|
| 14 | Function pointer | Lambda | C++11 | Capture, inline |
| 15 | Manual `for` loop (search/count/transform) | STL algorithm | C++11 | Named intent |
| 16 | `std::bind` | Lambda | C++14 | Clearer, faster |
| 17 | `enable_if` / SFINAE | `concept` / `requires` | C++20 | Clear errors |
| 18 | `pthread_t` | `std::jthread` | C++20 | RAII, portable |
| 19 | `#include` (new modules) | `import` | C++20 | Build speed |
| 20 | `printf` / `cout <<` | `std::print` / `std::println` | C++23 | Type-safe, clean |

### 🔧 Automated Tools

| Tool | Purpose |
|------|---------|
| `clang-tidy` | Automated modernization checks (`modernize-*` checks) |
| `clang-tidy --checks='modernize-*' --fix` | Auto-fix many patterns above |
| `cppcheck` | Find legacy patterns, memory issues |
| `include-what-you-use` | Clean up `#include` dependencies |
| `clang-format` | Consistent code style |

### Key `clang-tidy` modernize checks:

```bash
# Run all modernize checks with auto-fix
clang-tidy --checks='modernize-*' --fix source.cpp -- -std=c++20

# Most impactful individual checks:
#   modernize-use-nullptr            → NULL to nullptr
#   modernize-use-auto               → explicit types to auto
#   modernize-loop-convert           → C loops to range-based for
#   modernize-make-unique            → new to make_unique
#   modernize-make-shared            → new to make_shared
#   modernize-use-override           → add override keyword
#   modernize-use-using              → typedef to using
#   modernize-pass-by-value          → const& to value + move
#   modernize-return-braced-init-list → return T{} to return {}
```

---

## Quick Decision Tree

```
Need to allocate a single object?
  └─ Use std::make_unique<T>(args...)

Need a dynamic array?
  └─ Use std::vector<T>

Need a fixed-size array?
  └─ Use std::array<T, N>

Need a string?
  └─ Own it: std::string
  └─ View it: std::string_view

Need a nullable value?
  └─ Use std::optional<T>

Need a value-or-error?
  └─ Use std::expected<T, E>   (C++23)
  └─ Use exceptions             (unexpected errors)

Need to iterate?
  └─ Use range-based for
  └─ Or ranges::views for filtering/transforming

Need a callback?
  └─ Use a lambda

Need a hash map?
  └─ Use std::unordered_map<K, V>

Need a thread?
  └─ Use std::jthread (C++20) or std::thread (C++11)

Need to lock a mutex?
  └─ Use std::scoped_lock (C++17) or std::lock_guard (C++11)

Need formatted output?
  └─ Use std::println (C++23) or std::format (C++20)
```

---

## Summary: The Core Philosophy Shift

| Traditional C++ | Modern C++ |
|----------------|------------|
| Manual resource management | RAII — resources manage themselves |
| Programmer tracks lifetimes | Types encode lifetimes |
| Error-prone boilerplate | Compiler-generated correctness |
| Text substitution (`#define`) | Type-safe abstractions |
| Raw loops | Named algorithms |
| Function pointers | Lambdas |
| Output parameters | Return by value |
| Implicit conventions | Explicit in the type system |
| "Trust the programmer" | "Help the programmer" |

> **The single biggest insight:** Modern C++ doesn't add complexity — it removes it. Every feature listed above makes code shorter, safer, and faster simultaneously. There is no trade-off.

---

*Cross-reference: [Appendix H — C++ Quick Reference](H_CPP_Quick_Reference.md) · [Chapter 33 — C++11/14 Revolution](../Part-04-Modern-CPP-Evolution/33_CPP11_14_Revolution.md) · [Chapter 38 — Standards Comparison](../Part-04-Modern-CPP-Evolution/38_Standards_Comparison.md)*
