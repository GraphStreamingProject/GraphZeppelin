#pragma once
#include <set>
#include <vector>

#include "types.h"
#include "return_types.h"

/**
 * A plugin for the Graph class that runs Boruvka alongside the graph algorithm
 * and verifies the edges and connected components that the graph algorithm
 * generates.
 */
class GraphVerifier {
private:
  const node_id_t num_vertices;
  std::vector<std::vector<bool>> adj_matrix;
  DisjointSetUnion<node_id_t> kruskal_dsu;
  node_id_t kruskal_ccs;
  bool need_query_compute = true;

  /**
   * Runs Kruskal's (deterministic) CC algo to compute the kruskal dsu.
   */
  void kruskal();
public:
  /**
   * Empty Graph Verifier constructor
   */
  GraphVerifier(node_id_t vertices);
  
  /**
   * Construct GraphVerifier from a cumulative stream file
   */
  GraphVerifier(node_id_t num_vertices, const std::string &cumul_file_name);

  /**
   * Copy a GraphVerifier
   */
  GraphVerifier(const GraphVerifier &oth_verifier)
      : num_vertices(oth_verifier.num_vertices),
        adj_matrix(oth_verifier.adj_matrix),
        kruskal_dsu(oth_verifier.kruskal_dsu) {};

  /**
   * Flip an edge in the adjacency list.
   * @param edge   the edge to flip
   */
  void edge_update(Edge edge);

  /**
   * Verifies an edge exists in the graph.
   * @param edge the edge to be tested.
   * @throws BadEdgeException if the edge does not exist in the graph.
   */
  void verify_edge(Edge edge);

  /**
   * Verifies the connected components solution is correct. Compares
   * retval against kruskal_ref.
   * @throws IncorrectCCException if the solution cannot be verified
   */
  void verify_connected_components(const ConnectedComponents &cc);

  /**
   * Verifies that one or more spanning forests are valid
   * Additionally, enforces that spanning forests must be edge disjoint.
   * @param SFs    the spanning forests to check
   * @throws IncorrectForestException if a bad spanning forest is found
   */
  void verify_spanning_forests(std::vector<SpanningForest> SFs);

  /*
   * Merge two GraphVerifiers that have seen two different streams.
   * Yields a GraphVerifier that has seen both streams.
   * @param oth   a GraphVerifier to combine into this one.
   */
  void combine(const GraphVerifier &oth);

  std::vector<std::vector<bool>> extract_adj_matrix() { return adj_matrix; }
  node_id_t get_num_kruskal_ccs() { return kruskal_ccs; }

  bool operator==(const GraphVerifier &oth) { return adj_matrix == oth.adj_matrix; }
  bool operator!=(const GraphVerifier &oth) { return !(*this == oth); }
};

class BadEdgeException : public std::exception {
 private:
  std::string err_msg;
 public:
  BadEdgeException(std::string err) : err_msg(err) {};
  virtual const char* what() const throw() { return err_msg.c_str(); }
};

class IncorrectCCException : public std::exception {
 private:
  std::string err_msg;
 public:
  IncorrectCCException(std::string err) : err_msg(err) {};
  virtual const char* what() const throw() { return err_msg.c_str(); }
};

class IncorrectForestException : public std::exception {
 private:
  std::string err_msg;
 public:
  IncorrectForestException(std::string err) : err_msg(err) {};
  virtual const char* what() const throw() { return err_msg.c_str(); }
};
