#ifndef MAIN_GRAPH_H
#define MAIN_GRAPH_H
#include <cstdlib>
#include <exception>
#include <set>
#include <unordered_map>
#include <list>
#include <fstream>
#include "supernode.h"

#ifdef VERIFY_SAMPLES_F
#include "../test/util/graph_verifier.h"
#endif

using namespace std;

enum UpdateType {
  INSERT = 0,
  DELETE = 1,
};

typedef pair<Edge, UpdateType> GraphUpdate;

/**
 * Undirected graph object with n nodes labelled 0 to n-1, no self-edges,
 * multiple edges, or weights.
 */
class Graph{
  const uint64_t num_nodes;
  bool update_locked = false;
  // a set containing one "representative" from each supernode
  map<Node, size_t>* representatives;
  Supernode** supernodes;
  // DSU representation of supernode relationship
  Node* parent;
  Node get_parent(Node node);
public:
  explicit Graph(uint64_t num_nodes);
  ~Graph();
  Graph(const Graph& g);
  void update(GraphUpdate upd);

  /**
   * Update all the sketches in supernode, given a batch of updates.
   * @param src The supernode where the edges originate
   * @param edges A vector of <destination, delta> pairs
   */
  void batch_update(uint64_t src, const std::vector<uint64_t>& edges);

  /**
   * Main algorithm utilizing Boruvka and L_0 sampling.
   * @return a vector of the connected components in the graph.
   */
  vector<set<Node>> connected_components();

  /** 
   * Implements Boruvka on the graph sketch to find a spanning
   * forest, where a spanning forest of G is defined as a maximal
   * acyclic subgraph of G.  
   *
   * @return a vector of adjacency lists, one for each component,
   * where each adjacency list is implemented as an unordered_map
   * from nodes in the componenent to vectors of their neighbors.
   */
  vector<unordered_map<Node, vector<Node>>> spanning_forest();

  /**
   * Let $G$ be the graph represented by this sketch, and define 
   * $F_i$ for $1 <= i <= k$ as a spanning forest of 
   * $G - \cup_{j = 1}^{i-1} F_j$, where subtraction between graphs
   * excludes nodes.
   *
   * @param k The number of edge-disjoint spanning forests to take
   * the union of.
   * 
   * @return U $\cup_{i = 1}^{k} F_i$
   */ 
  vector<vector<Node>> k_edge_disjoint_span_forests_union (int k);

#ifdef VERIFY_SAMPLES_F
  std::string cum_in = "./cum_sample.txt";

  /**
   * Set the filepath to search for cumulative graph input.
   */
  void set_cum_in(const std::string& input_file) {
    cum_in = input_file;
  }
#endif
};

class UpdateLockedException : public exception {
  virtual const char* what() const throw() {
    return "The graph cannot be updated: Connected components algorithm has "
           "already started";
  }
};

#endif //MAIN_GRAPH_H
