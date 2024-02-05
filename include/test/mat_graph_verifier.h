#pragma once
#include "graph_verifier.h"

#include <iostream>

#include "dsu.h"

/**
 * A plugin for the Graph class that runs Boruvka alongside the graph algorithm
 * and verifies the edges and connected components that the graph algorithm
 * generates. Takes a reference graph from a packed in-memory adjacency matrix.
 */
class MatGraphVerifier : public GraphVerifier {
  std::vector<std::set<node_id_t>> kruskal_ref;
  node_id_t n;

  /**
   * Runs Kruskal's (deterministic) CC algo.
   * @param input_file the file to read input from.
   * @return an array of connected components.
   */
  std::vector<std::set<node_id_t>> kruskal();
public:
  MatGraphVerifier(node_id_t n);

  // When we want to build a MatGraphVerifier without iterative edge_updates
  MatGraphVerifier(node_id_t n, std::vector<std::vector<bool>> _adj)
   : GraphVerifier(_adj), n(n) { reset_cc_state(); };
  
  void reset_cc_state();       // run this function before using as a verifier in CC
  void edge_update(node_id_t src, node_id_t dst);

  void verify_edge(Edge edge);
  void verify_soln(std::vector<std::set<node_id_t>> &retval);
};
