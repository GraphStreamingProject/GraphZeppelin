// This file defines the query return types from the cc algorithm class
#pragma once
#include <cstddef>
#include <iterator>
#include <set>
#include <unordered_set>
#include <vector>

#include "dsu.h"
#include "types.h"

// This class defines the connected components of a graph
class ConnectedComponents {
 private:
  node_id_t *parent_arr;
  node_id_t num_vertices;
  node_id_t num_cc;

 public:
  ConnectedComponents(node_id_t num_vertices, DisjointSetUnion_MT<node_id_t> &dsu);
  ~ConnectedComponents();

  std::vector<std::set<node_id_t>> get_component_sets();
  bool is_connected(node_id_t a, node_id_t b) const { return parent_arr[a] == parent_arr[b]; }
  node_id_t size() const { return num_cc; }
};

// This class defines a spanning forest of a graph
class SpanningForest {
 private:
  node_id_t num_vertices;
  std::vector<Edge> edges;
  std::vector<Edge> sorted_adjacency;
  bool has_adjacency = false;
 public:
  SpanningForest(node_id_t num_vertices, const std::unordered_set<node_id_t> *spanning_forest);

  const std::vector<Edge>& get_edges() const { return edges; }
  const std::vector<Edge>& get_sorted_adjacency();
};
