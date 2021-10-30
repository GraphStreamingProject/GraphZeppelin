#pragma once
#include <vector>

template <class T>
class DisjointSetUnion {
  std::vector<T> parent;
  std::vector<T> size;
public:
  DisjointSetUnion();
  DisjointSetUnion(T n);

  void link(T i, T j);
  T find_set(T i);
  void union_set(T i, T j);
};

template <class T>
DisjointSetUnion<T>::DisjointSetUnion() {}

template <class T>
DisjointSetUnion<T>::DisjointSetUnion(T n) : parent(n), size(n, 1) {
  for (T i = 0; i < n; i++) {
    parent[i] = i;
  }
}

template <class T>
void DisjointSetUnion<T>::link(T i, T j) {
  if (size[i] < size[j]) std::swap(i,j);
  parent[j] = i;
  size[i] += size[j];
}

template <class T>
T DisjointSetUnion<T>::find_set(T i) {
  if (parent[i] == i) return i;
  return parent[i] = find_set(parent[i]);
}

template <class T>
void DisjointSetUnion<T>::union_set(T i, T j) {
  link(find_set(i), find_set(j));
}
