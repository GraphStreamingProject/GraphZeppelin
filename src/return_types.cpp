#include "return_types.h"

#include <map>
#include <algorithm>

ConnectedComponents::ConnectedComponents(node_id_t num_vertices,
                                         DisjointSetUnion_MT<node_id_t> &dsu)
    : parent_arr(new node_id_t[num_vertices]), num_vertices(num_vertices) {
  size_t temp_cc = 0;
#pragma omp parallel for
  for (node_id_t i = 0; i < num_vertices; i++) {
    parent_arr[i] = dsu.find_root(i);
    if (parent_arr[i] == i) {
#pragma omp atomic update
      temp_cc += 1;
    }
  }

  num_cc = temp_cc;
}

ConnectedComponents::~ConnectedComponents() { delete[] parent_arr; }

std::vector<std::set<node_id_t>> ConnectedComponents::get_component_sets() {
  std::map<node_id_t, std::set<node_id_t>> temp;
  for (node_id_t i = 0; i < num_vertices; ++i) temp[parent_arr[i]].insert(i);
  std::vector<std::set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto &it : temp) retval.push_back(it.second);
  return retval;
}

SpanningForest::SpanningForest(node_id_t num_vertices,
                               const std::unordered_set<node_id_t> *spanning_forest)
    : num_vertices(num_vertices) {
  edges.reserve(num_vertices);
  for (node_id_t src = 0; src < num_vertices; src++) {
    for (node_id_t dst : spanning_forest[src]) {
      edges.push_back({src, dst});
    }
  }
}

const std::vector<Edge> &SpanningForest::get_sorted_adjacency() {
  if (has_adjacency) return sorted_adjacency;

  size_t num = edges.size();
  sorted_adjacency.resize(edges.size() * 2);

#pragma omp parallel for
  for (size_t i = 0; i < num; i++) {
    sorted_adjacency[i] = edges[i];
    sorted_adjacency[i + num] = sorted_adjacency[i];
    std::swap(sorted_adjacency[i + num].src, sorted_adjacency[i + num].dst);
  }

  // sort the edges
  std::sort(sorted_adjacency.begin(), sorted_adjacency.end());

  has_adjacency = true;
  return sorted_adjacency;
}
