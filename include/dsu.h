#pragma once
#include <vector>

template<class T>
struct DSUMergeRet {
  bool merged; // true if a merge actually occured
  T root; // new root
  T child; // new child of root
};

template <class T>
class DisjointSetUnion {
  std::vector<T> parent;
  std::vector<T> size;
public:
  DisjointSetUnion(T n) : parent(n), size(n, 1) {
    for (T i = 0; i < n; i++) {
      parent[i] = i;
    }
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
    for (T i = 0; i < parent.size(); i++) {
      parent[i] = i;
      size[i] = 1;
    }
  }
};
