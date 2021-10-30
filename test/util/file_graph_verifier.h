#pragma once
#include <set>
#include "../../include/supernode.h"
#include "../../include/dsu.h"
#include "graph_verifier.h"

/**
 * A plugin for the Graph class that runs Boruvka alongside the graph algorithm
 * and verifies the edges and connected components that the graph algorithm
 * generates. Takes a reference graph from a file.
 */
class FileGraphVerifier : public GraphVerifier {
  std::vector<std::set<node_t>> kruskal_ref;
  std::vector<std::set<node_t>> boruvka_cc;
  std::vector<std::set<node_t>> det_graph;
  DisjointSetUnion<node_t> sets;

public:
  FileGraphVerifier(const string& input_file = "./cumul_sample.txt");

  void verify_edge(Edge edge);
  void verify_cc(node_t node);
  void verify_soln(vector<set<node_t>>& retval);

  /**
   * Runs Kruskal's (deterministic) CC algo.
   * @param input_file the file to read input from.
   * @return an array of connected components.
   */
  static std::vector<std::set<node_t>> kruskal(const string& input_file = "cumul_sample.txt");
};
