#pragma once
#include <vector>
#include <atomic>

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
    }
  }

  ~DisjointSetUnion() {
    delete[] parent;
    delete[] size;
  }

  inline T find_root(T u) {
    while(parent[parent[u]] != u) {
      parent[u] = parent[parent[u]];
      u = parent[u];
    }
    return u;
  }

  inline DSUMergeRet<T> merge(T u, T v) {
    T a = find_root(u);
    T b = find_root(v);
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

  inline T find_root(T u) {
    while (parent[parent[u]] != u) {
      parent[u] = parent[parent[u]].load();
      u = parent[u];
    }
    return u;
  }

  // use CAS in this function to allow for simultaneous merge calls
  inline DSUMergeRet<T> merge(T u, T v) {
    while ((u = find_root(u)) != (v = find_root(v))) {
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
