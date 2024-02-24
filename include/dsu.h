#pragma once
#include <atomic>
#include <cassert>
#include <chrono>
#include <random>
#include <vector>

#define likely_if(x) if(__builtin_expect((bool)(x), true))
#define unlikely_if(x) if (__builtin_expect((bool)(x), false))

template <class T>
struct DSUMergeRet {
  bool merged;  // true if a merge actually occured
  T root;       // new root
  T child;      // new child of root
};

template <class T>
class DisjointSetUnion {
 private:
  // number of items in the DSU
  T n;

  // parent and priority arrays
  T* parent;
  T* priority;

  // Order based on size and break ties with
  // a simple fixed ordering of the edge
  // if sum even, smaller first
  // if sum odd, larger first
  inline void order_edge(T& a, T& b) {
    unlikely_if(priority[a] == priority[b] && a > b) std::swap(a, b);

    if (priority[a] < priority[b]) std::swap(a, b);
  }

 public:
  DisjointSetUnion(T n) : n(n), parent(new T[n]), priority(new T[n]) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    size_t seed = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    std::mt19937_64 prio_gen(seed);
    for (T i = 0; i < n; i++) {
      parent[i] = i;
      priority[i] = prio_gen();
    }
  }

  ~DisjointSetUnion() {
    delete[] parent;
    delete[] priority;
  }

  // make a copy of the DSU
  DisjointSetUnion(const DisjointSetUnion& oth) : n(oth.n), parent(new T[n]), priority(new T[n]) {
    for (T i = 0; i < n; i++) {
      parent[i] = oth.parent[i];
      priority[i] = oth.priority[i];
    }
  }

  // move the DSU to a new object
  DisjointSetUnion(DisjointSetUnion&& oth) : n(oth.n), parent(oth.parent), priority(oth.priority) {
    oth.n = 0;
    oth.parent = nullptr;
    oth.priority = nullptr;
  }

  DisjointSetUnion operator=(const DisjointSetUnion& oth) = delete;

  inline T find_root(T u) {
    assert(0 <= u && u < n);
    while (parent[parent[u]] != parent[u]) {
      parent[u] = parent[parent[u]];
      u = parent[u];
    }
    return parent[u];
  }

  inline DSUMergeRet<T> merge(T u, T v) {
    T a = find_root(u);
    T b = find_root(v);
    assert(0 <= a && a < n);
    assert(0 <= b && b < n);
    if (a == b) return {false, 0, 0};

    order_edge(a, b);
    parent[b] = a;
    return {true, a, b};
  }

  inline void reset() {
    for (T i = 0; i < n; i++) {
      parent[i] = i;
    }
  }
};

// Disjoint set union that uses atomics to be thread safe
// thus is a little slower for single threaded use cases
template <class T>
class DisjointSetUnion_MT {
 private:
  // number of items in the DSU
  T n;

  // parent and node priority arrays
  std::atomic<T>* parent;
  T* priority;

  // Order based on priority and break ties using node id.
  // Smaller ids have higher priority.
  inline void order_edge(T& a, T& b) {
    unlikely_if(priority[a] == priority[b] && a > b) std::swap(a, b);

    if (priority[a] < priority[b]) std::swap(a, b);
  }

 public:
  DisjointSetUnion_MT(T n) : n(n), parent(new std::atomic<T>[n]), priority(new T[n]) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    size_t seed = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    std::mt19937_64 prio_gen(seed);
    for (T i = 0; i < n; i++) {
      parent[i] = i;
      priority[i] = prio_gen();
    }
  }

  ~DisjointSetUnion_MT() {
    delete[] parent;
    delete[] priority;
  }

  // make a copy of the DSU
  DisjointSetUnion_MT(const DisjointSetUnion_MT& oth)
      : n(oth.n), parent(new std::atomic<T>[n]), priority(new T[n]) {
    for (T i = 0; i < n; i++) {
      parent[i] = oth.parent[i].load();
      priority[i] = oth.priority[i];
    }
  }
  DisjointSetUnion_MT& operator=(const DisjointSetUnion_MT& oth) = default;

  inline T find_root(T u) {
    assert(0 <= u && u < n);
    while (parent[parent[u]] != parent[u]) {
      parent[u] = parent[parent[u]].load();
      u = parent[u];
    }
    return parent[u];
  }

  // use CAS in this function to allow for simultaneous merge calls
  inline DSUMergeRet<T> merge(T a, T b) {
    while ((a = find_root(a)) != (b = find_root(b))) {
      assert(0 <= a && a < n);
      assert(0 <= b && b < n);
      order_edge(a, b);

      // if parent of b has not been modified by another thread -> replace with a
      if (parent[b].compare_exchange_weak(b, a)) {
        return {true, a, b};
      }
    }
    return {false, 0, 0};
  }

  inline void reset() {
    for (T i = 0; i < n; i++) {
      parent[i] = i;
    }
  }
};
