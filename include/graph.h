#ifndef MAIN_GRAPH_H
#define MAIN_GRAPH_H
#include <cstdlib>
#include <exception>
#include <memory>
#include <set>
#include <fstream>
#include "supernode.h"
#include <atomic>  // REMOVE LATER

#ifdef VERIFY_SAMPLES_F
#include "../test/util/graph_verifier.h"
#endif

using namespace std;

class BufferTree;
class GraphWorker;

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
  set<Node> representatives;
  std::vector<std::unique_ptr<Supernode>> supernodes;
  // DSU representation of supernode relationship
  std::vector<Node> parent;
  Node get_parent(Node node);

  // BufferTree for buffering inputs
  std::unique_ptr<BufferTree> bf;
public:
  explicit Graph(uint64_t num_nodes);
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
  
#ifdef VERIFY_SAMPLES_F
  std::string cum_in = "./cum_sample.txt";

  /**
   * Set the filepath to search for cumulative graph input.
   */
  void set_cum_in(const std::string& input_file) {
    cum_in = input_file;
  }
#endif

  // temp to verify number of updates -- REMOVE later
  std::atomic<uint64_t> num_updates;
};

class UpdateLockedException : public exception {
  virtual const char* what() const throw() {
    return "The graph cannot be updated: Connected components algorithm has "
           "already started";
  }
};

#endif //MAIN_GRAPH_H
