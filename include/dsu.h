#pragma once
#include <vector>

template <class T>
class DisjointSetUnion {
  std::vector<T> parent;
  std::vector<T> size;
public:
  DisjointSetUnion();
  DisjointSetUnion(T n);

  // return the parent of the resultant
  T link(std::vector<T>& ts);
  T find_set(T i);
  void union_sets(std::vector<T>& ts);
};

template <class T>
DisjointSetUnion<T>::DisjointSetUnion() {}

template <class T>
DisjointSetUnion<T>::DisjointSetUnion(T n) : parent(n), size(n, 1) {
  for (T i = 0; i < n; i++) {
    parent[i] = i;
  }
}

template<class T>
T DisjointSetUnion<T>::link(std::vector<T>& ts) {
  auto n = ts.size();
  // find largest
  decltype(n) max_idx;
  T max_size = 0;

  for (int i = 0; i < n; ++i) {
    if (size[ts[i]] > max_size) {
      max_size = size[ts[i]];
      max_idx = i;
    }
  }
  for (int i = 0; i < n; ++i) {
    if (i == max_idx) continue;
    parent[ts[i]] = max_idx;
    size[ts[max_idx]] += size[ts[i]];
  }
  return ts[max_idx];
}

template <class T>
T DisjointSetUnion<T>::find_set(T i) {
  if (parent[i] == i) return i;
  return parent[i] = find_set(parent[i]);
}

template <class T>
void DisjointSetUnion<T>::union_sets(std::vector<T>& ts) {
  std::vector<T> ts_parent;
  for (auto& t : ts) {
    ts_parent.push_back(find_set(t));
  }
  link(ts_parent);
}
