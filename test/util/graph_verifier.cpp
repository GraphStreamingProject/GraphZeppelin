#include "graph_verifier.h"
#include <ascii_file_stream.h>

#include <map>
#include <iostream>
#include <algorithm>
#include <cassert>

GraphVerifier::GraphVerifier(node_id_t num_vertices)
    : num_vertices(num_vertices), kruskal_dsu(num_vertices) {
  // initialize adjacency matrix
  adj_matrix = std::vector<std::vector<bool>>(num_vertices);
  for (node_id_t i = 0; i < num_vertices; ++i)
    adj_matrix[i] = std::vector<bool>(num_vertices - i);
}

GraphVerifier::GraphVerifier(node_id_t num_vertices, const std::string &cumul_file_name)
    : num_vertices(num_vertices), kruskal_dsu(num_vertices) {
  // initialize adjacency matrix
  adj_matrix = std::vector<std::vector<bool>>(num_vertices);
  for (node_id_t i = 0; i < num_vertices; ++i)
    adj_matrix[i] = std::vector<bool>(num_vertices - i);

  // cumulative files do not have update types
  AsciiFileStream stream(cumul_file_name, false);

  GraphStreamUpdate stream_upd;
  stream.get_update_buffer(&stream_upd, 1);

  node_id_t src = stream_upd.edge.src;
  node_id_t dst = stream_upd.edge.dst;
  UpdateType type = static_cast<UpdateType>(stream_upd.type);

  while (type != BREAKPOINT) {
    if (src > dst)
      std::swap(src, dst);
    dst -= src;
    adj_matrix[src][dst] = !adj_matrix[src][dst];

    stream.get_update_buffer(&stream_upd, 1);
    src = stream_upd.edge.src;
    dst = stream_upd.edge.dst;
    type = static_cast<UpdateType>(stream_upd.type);
  }

  kruskal();
}

void GraphVerifier::edge_update(Edge edge) {
  auto src = edge.src;
  auto dst = edge.dst;

  if (src >= num_vertices || dst >= num_vertices) {
    throw BadEdgeException("Source " + std::to_string(src) + " or Destination " +
                           std::to_string(dst) + " out of bounds!");
  }
  if (src > dst) std::swap(src, dst);

  dst = dst - src;

  // update adj_matrix entry
  adj_matrix[src][dst] = !adj_matrix[src][dst];
  need_query_compute = true;
}

void GraphVerifier::kruskal() {
  if (!need_query_compute)
    return;

  kruskal_ccs = num_vertices;
  kruskal_dsu.reset();
  for (node_id_t i = 0; i < num_vertices; i++) {
    for (node_id_t j = 0; j < adj_matrix[i].size(); j++) {
      if (adj_matrix[i][j] && kruskal_dsu.merge(i, i + j).merged)
        kruskal_ccs -= 1;
    }
  }
  need_query_compute = false;
}

void GraphVerifier::verify_edge(Edge edge) {
  // verify that the edge in question actually exists
  if (edge.src > edge.dst) std::swap(edge.src, edge.dst);
  if (!adj_matrix[edge.src][edge.dst - edge.src]) {
    printf("Got an error on edge (%u, %u): edge is not in adj_matrix\n", edge.src, edge.dst);
    throw BadEdgeException("The edge is not in the cut of the sample!");
  }
}

void GraphVerifier::verify_connected_components(const ConnectedComponents &cc) {
  // compute the connected components for the verifier
  kruskal();

  // first check that the number of components is the same for both
  if (kruskal_ccs != cc.size()) {
    throw IncorrectCCException("Incorrect number of components!");
  }

  // then check that we agree on where all the vertices belong
  for (node_id_t i = 0; i < num_vertices; i++) {
    node_id_t root = kruskal_dsu.find_root(i);
    if (!cc.is_connected(root, i))
      throw IncorrectCCException("Incorrect Connectivity!");
  }
}

void GraphVerifier::verify_spanning_forests(std::vector<SpanningForest> SFs) {
  // backup the adjacency matrix
  std::vector<std::vector<bool>> backup(adj_matrix);

  for (SpanningForest &forest : SFs) {
    kruskal();

    DisjointSetUnion<node_id_t> forest_ccs(num_vertices);
    for (auto edge : forest.get_edges()) {
      // every edge in the spanning forest must encode connectivity info
      if (!forest_ccs.merge(edge.src, edge.dst).merged) {
        adj_matrix = backup;
        throw IncorrectForestException(
            "Found an edge: (" + std::to_string(edge.src) + ", " +
            std::to_string(edge.dst) + ") that is redundant within a single spanning forest!");
      }

      try {
        verify_edge(edge);
      } catch (...) {
        adj_matrix = backup;
        throw;
      }
      edge_update(edge);
    }

    // root map allows us to translate from the kruskal_dsu's roots to the forest_ccs' roots
    std::map<node_id_t, node_id_t> root_map;

    for (node_id_t i = 0; i < num_vertices; i++) {
      node_id_t kruskal_root = kruskal_dsu.find_root(i);

      if (root_map.count(kruskal_root) == 0) {
        root_map[kruskal_root] = forest_ccs.find_root(i);
      }
      else if (root_map[kruskal_root] != forest_ccs.find_root(i)) {
        adj_matrix = backup;
        throw IncorrectForestException("Forest does not match expected component sets!");
      }
    }
  }
  adj_matrix = backup;
}

void GraphVerifier::combine(const GraphVerifier &oth) {
  for (size_t i = 0; i < adj_matrix.size(); i++) {
    for (size_t j = 0; j < adj_matrix[i].size(); j++) {
      adj_matrix[i][j] = adj_matrix[i][j] != oth.adj_matrix[i][j];
    }
  }
}
