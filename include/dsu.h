#pragma once
#include <vector>
#include <atomic>
#include <cassert>

template<class T>
struct DSUMergeRet {
  bool merged; // true if a merge actually occured
  T root; // new root
  T child; // new child of root
};

template <class T>
class DisjointSetUnion {
  // number of items in the DSU
  T n;

  // parent and size arrays
  T* parent;
  T* size;
public:
  DisjointSetUnion(T n) : n(n), parent(new T[n]), size(new T[n]) {
    for (T i = 0; i < n; i++) {
      parent[i] = i;
      size[i] = 1;
    }
  }

  ~DisjointSetUnion() {
    delete[] parent;
    delete[] size;
  }

  // make a copy of the DSU
  DisjointSetUnion(const DisjointSetUnion &oth) : n(oth.n), parent(new T[n]), size(new T[n]) {
    for (T i = 0; i < n; i++) {
      parent[i] = oth.parent[i];
      size[i] = oth.size[i];
    }
  }

  // move the DSU to a new object
  DisjointSetUnion(DisjointSetUnion &&oth) : n(oth.n), parent(oth.parent), size(oth.size) {
    oth.n = 0;
    oth.parent = nullptr;
    oth.size = nullptr;
  }

  DisjointSetUnion operator=(const DisjointSetUnion &oth) = delete;

  inline T find_root(T u) {
    assert(0 <= u && u < n);
    while(parent[parent[u]] != u) {
      parent[u] = parent[parent[u]];
      u = parent[u];
    }
    return u;
  }

  inline DSUMergeRet<T> merge(T u, T v) {
    T a = find_root(u);
    T b = find_root(v);
    assert(0 <= a && a < n);
    assert(0 <= b && b < n);
    if (a == b) return {false, 0, 0};

    if (size[a] < size[b]) std::swap(a,b);
    parent[b] = a;
    size[a] += size[b];
    return {true, a, b};
  }

  inline void reset() {
    for (T i = 0; i < n; i++) {
      parent[i] = i;
      size[i] = 1;
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

  // parent and size arrays
  std::atomic<T>* parent;
  std::atomic<T>* size;
public:
  DisjointSetUnion_MT(T n) : n(n), parent(new std::atomic<T>[n]), size(new std::atomic<T>[n]) {
    for (T i = 0; i < n; i++) {
      parent[i] = i;
      size[i] = 1;
    }
  }

  ~DisjointSetUnion_MT() {
    delete[] parent;
    delete[] size;
  }

  // make a copy of the DSU
  DisjointSetUnion_MT(const DisjointSetUnion_MT &oth) : n(oth.n), parent(new T[n]), size(new T[n]) {
    for (T i = 0; i < n; i++) {
      parent[i] = oth.parent[i].load();
      size[i] = oth.size[i].load();
    }
  }

  // move the DSU to a new object
  DisjointSetUnion_MT(DisjointSetUnion_MT &&oth) : n(oth.n), parent(oth.parent), size(oth.size) {
    oth.n = 0;
    oth.parent = nullptr;
    oth.size = nullptr;
  }

  inline T find_root(T u) {
    assert(0 <= u && u < n);
    while (parent[parent[u]] != u) {
      parent[u] = parent[parent[u]].load();
      u = parent[u];
    }
    return u;
  }

  // use CAS in this function to allow for simultaneous merge calls
  inline DSUMergeRet<T> merge(T u, T v) {
    while ((u = find_root(u)) != (v = find_root(v))) {
      assert(0 <= u && u < n);
      assert(0 <= v && v < n);
      if (size[u] < size[v])
        std::swap(u, v);

      // if parent of b has not been modified by another thread -> replace with a
      if (std::atomic_compare_exchange_weak(&parent[u], &v, u)) {
        size[u] += size[v];
        return {true, u, v};
      }
    }
    return {false, 0, 0};
  }

  inline void reset() {
    for (T i = 0; i < n; i++) {
      parent[i] = i;
      size[i] = 1;
    }
  }
};
