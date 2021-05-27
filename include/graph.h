#ifndef MAIN_GRAPH_H
#define MAIN_GRAPH_H
#include <cstdlib>
#include <exception>
#include <set>
#include <fstream>
#include "supernode.h"
#include "dsu.h"

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
  set<Node>* representatives;
  Supernode** supernodes;
  // DSU representation of supernode relationship
  DSU dsu;
public:
  explicit Graph(uint64_t num_nodes);
  ~Graph();
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
  std::vector<std::pair<std::vector<Node>, std::vector<Edge>>>
      connected_components();
};

class UpdateLockedException : public exception {
  virtual const char* what() const throw() {
    return "The graph cannot be updated: Connected components algorithm has "
           "already started";
  }
};

#endif //MAIN_GRAPH_H
