#pragma once
#include <set>
#include "../supernode.h"
#include "../dsu.h"
#include "graph_verifier.h"

/**
 * A plugin for the Graph class that runs Boruvka alongside the graph algorithm
 * and verifies the edges and connected components that the graph algorithm
 * generates. Takes a reference graph from a packed in-memory adjacency matrix.
 */
class MatGraphVerifier : public GraphVerifier {
  std::vector<std::set<node_id_t>> kruskal_ref;
  std::vector<std::set<node_id_t>> boruvka_cc;
  std::vector<std::vector<bool>> adj_graph;
  DisjointSetUnion<node_id_t> sets;

  node_id_t n;

  /**
   * Runs Kruskal's (deterministic) CC algo.
   * @param input_file the file to read input from.
   * @return an array of connected components.
   */
  std::vector<std::set<node_id_t>> kruskal();
public:
  MatGraphVerifier(node_id_t n);
  
  void reset_cc_state();       // run this function before using as a verifier in CC
  void edge_update(node_id_t src, node_id_t dst);

  void verify_edge(Edge edge);
  void verify_cc(node_id_t node);
  void verify_soln(std::vector<std::set<node_id_t>> &retval);
};