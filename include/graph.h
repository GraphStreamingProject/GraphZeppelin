#ifndef MAIN_GRAPH_H
#define MAIN_GRAPH_H
#include <cstdlib>
#include <exception>
#include <set>
#include <fstream>
#include "supernode.h"
#include <atomic>  // REMOVE LATER

#ifndef USE_FBT_F
#include "work_queue.h"
#endif

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
class Graph {
  uint64_t num_nodes;
  long seed;
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
  explicit Graph(std::ifstream& in);
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
  vector<set<Node>> connected_components();

  /**
   * Parallel version of Boruvka.
   * @return a vector of the connected components in the graph.
   */
  vector<set<Node>> parallel_connected_components();
  /*
   * Call this function to indicate to the graph that it should
   * begin accepting updates again. It is important that the sketches
   * be restored to their pre-connected_components state before
   * calling this function
   * Unpauses the graph workers and sets allow update flag
   */
  void post_cc_resume();

  /**
   * Serialize the graph node sketches to an output stream.
   * @param out
   */
  void write_to_stream(std::ofstream& out);

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

  static Supernode* generate_delta_node(uint64_t node_n, long node_seed, uint64_t src,
                                  const vector<uint64_t> &edges);
};

class UpdateLockedException : public exception {
  virtual const char* what() const throw() {
    return "The graph cannot be updated: Connected components algorithm has "
           "already started";
  }
};

#endif //MAIN_GRAPH_H
