#ifndef MAIN_GRAPH_H
#define MAIN_GRAPH_H
#include <cstdlib>
#include <exception>
#include <set>
#include <fstream>
#include "supernode.h"

#ifdef VERIFY_SAMPLES_F
#include "../test/util/deterministic.h"
#endif

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

#ifdef VERIFY_SAMPLES_F
  std::string cum_in = "./cum_sample.txt";
#endif

  // DSU representation of supernode relationship
  Node* parent;
  Node get_parent(Node node);
public:
  explicit Graph(uint64_t num_nodes);
  ~Graph();
  void update(GraphUpdate upd);

  /**
   * Main algorithm utilizing Boruvka and L_0 sampling.
   * @return a vector of the connected components in the graph.
   */
  vector<set<Node>> connected_components();

#ifdef VERIFY_SAMPLES_F
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
