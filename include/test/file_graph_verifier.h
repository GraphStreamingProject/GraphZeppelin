#pragma once
#include <set>
#include "../supernode.h"
#include "../dsu.h"
#include "graph_verifier.h"

/**
 * A plugin for the Graph class that runs Boruvka alongside the graph algorithm
 * and verifies the edges and connected components that the graph algorithm
 * generates. Takes a reference graph from a file.
 */
class FileGraphVerifier : public GraphVerifier {
  std::vector<std::set<node_id_t>> kruskal_ref;
  std::vector<std::set<node_id_t>> boruvka_cc;
  DisjointSetUnion<node_id_t> sets;

public:
  FileGraphVerifier(node_id_t n, const std::string& input_file);

  void verify_edge(Edge edge);
  void verify_cc(node_id_t node);
  void verify_soln(std::vector<std::set<node_id_t>>& retval);

  /**
   * Runs Kruskal's (deterministic) CC algo.
   * @param input_file the file to read input from.
   * @return an array of connected components.
   */
  static std::vector<std::set<node_id_t>> kruskal(const std::string& input_file = "cumul_sample.txt");
};
