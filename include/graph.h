#ifndef MAIN_GRAPH_H
#define MAIN_GRAPH_H
#include <cstdlib>
#include <exception>
#include <set>
#include <fstream>
#include "supernode.h"

#ifndef USE_FBT_F
#include "work_queue.h"
#endif

#ifdef VERIFY_SAMPLES_F
#include "../test/util/graph_verifier.h"
#endif

using namespace std;

class BufferTree;
class GraphWorker;

enum class UpdateType : int {
  INSERT = 0,
  DELETE = 1,
};

struct GraphUpdate {
  Edge edge;
  UpdateType type;
};

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
  Node* parent;
  Node get_parent(Node node);

#ifdef USE_FBT_F
  // BufferTree for buffering inputs
  BufferTree *bf;
#else
  // In-memory buffering system
  WorkQueue *wq;
#endif
public:
  explicit Graph(uint64_t num_nodes);
  ~Graph();
  void ingest_graph(std::string path);
  void update(GraphUpdate upd);
  GraphUpdate get_graph_update() { return {}; };
  const std::vector<Sketch> &get_supernode_sketches(uint64_t src) const;
  void apply_supernode_deltas(uint64_t src, const std::vector<Sketch>& deltas);
  static std::vector<vec_t> make_updates(uint64_t src, const std::vector<uint64_t>& edges);

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
  std::vector<std::set<Node>> connected_components();


#ifdef USE_FBT_F
  BufferTree &get_buffer_tree() { return *bf; };
#else
  WorkQueue &get_work_queue() { return *wq; };
#endif

#ifdef VERIFY_SAMPLES_F
  std::string cum_in = "../test/res/multiples_graph_1024.txt";

  /**
   * Set the filepath to search for cumulative graph input.
   */
  void set_cum_in(const std::string input_file) {
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
