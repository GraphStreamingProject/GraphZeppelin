#pragma once
#include <guttering_system.h>

#include <atomic>  // REMOVE LATER
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <unordered_set>

#include "graph_configuration.h"
#include "sketch.h"
#include "worker_thread.h"

#ifdef VERIFY_SAMPLES_F
#include "test/graph_verifier.h"
#endif

// Exceptions the Graph class may throw
class UpdateLockedException : public std::exception {
  virtual const char *what() const throw() {
    return "The graph cannot be updated: Connected components algorithm has "
           "already started";
  }
};

class MultipleGraphsException : public std::exception {
  virtual const char *what() const throw() {
    return "Only one Graph may be open at one time. The other Graph must be deleted.";
  }
};

/**
 * Undirected graph object with n nodes labelled 0 to n-1, no self-edges,
 * multiple edges, or weights.
 */
class Graph {
 protected:
  node_id_t num_nodes;
  uint64_t seed;
  bool update_locked = false;
  bool modified = false;
  // a set containing one "representative" from each supernode
  std::set<node_id_t> *representatives;
  Sketch **sketches;
  // DSU representation of supernode relationship
#ifdef USE_EAGER_DSU
  std::atomic<node_id_t> *parent;
#else
  node_id_t *parent;
#endif
  node_id_t *size;
  node_id_t get_parent(node_id_t node);
  bool dsu_valid = true;

  std::unordered_set<node_id_t> *spanning_forest;
  std::mutex *spanning_forest_mtx;

  // Guttering system for batching updates
  GutteringSystem *gts;

  WorkerThreadGroup *worker_group;

  void backup_to_disk(const std::vector<node_id_t> &ids_to_backup);
  void restore_from_disk(const std::vector<node_id_t> &ids_to_restore);

  /**
   * Update the query array with new samples
   * @param query  an array of sketch sample results
   * @param reps   an array containing node indices for the representative of each supernode
   */
  virtual void sample_supernodes(std::pair<Edge, SampleSketchRet> *query,
                                 std::vector<node_id_t> &reps);

  /**
   * @param copy_sketches  an array to be filled with sketches
   * @param to_merge       an list of lists of supernodes to be merged
   *
   */
  void merge_supernodes(Sketch **copy_sketches, std::vector<node_id_t> &new_reps,
                        std::vector<std::vector<node_id_t>> &to_merge, bool make_copy);

  /**
   * Run the disjoint set union to determine what supernodes
   * Should be merged together.
   * Map from nodes to a vector of nodes to merge with them
   * @param query  an array of sketch sample results
   * @param reps   an array containing node indices for the representative of each supernode
   */
  std::vector<std::vector<node_id_t>> supernodes_to_merge(std::pair<Edge, SampleSketchRet> *query,
                                                          std::vector<node_id_t> &reps);

  /**
   * Main parallel algorithm utilizing Boruvka and L_0 sampling.
   * @return a vector of the connected components in the graph.
   */
  std::vector<std::set<node_id_t>> boruvka_emulation(bool make_copy);

  /**
   * Generates connected components from this graph's dsu
   * @return a vector of the connected components in the graph.
   */
  std::vector<std::set<node_id_t>> cc_from_dsu();

  std::string backup_file;  // where to backup the supernodes

  FRIEND_TEST(GraphTestSuite, TestCorrectnessOfReheating);
  FRIEND_TEST(GraphTest, TestSupernodeRestoreAfterCCFailure);

  GraphConfiguration config;

  static bool open_graph;

 public:
  explicit Graph(node_id_t num_nodes, int num_inserters = 1)
      : Graph(num_nodes, GraphConfiguration(), num_inserters){};
  explicit Graph(const std::string &input_file, int num_inserters = 1)
      : Graph(input_file, GraphConfiguration(), num_inserters){};
  explicit Graph(const std::string &input_file, GraphConfiguration config, int num_inserters = 1);
  explicit Graph(node_id_t num_nodes, GraphConfiguration config, int num_inserters = 1);

  virtual ~Graph();

  inline void update(GraphUpdate upd, int thr_id = 0) {
    if (update_locked) throw UpdateLockedException();
    Edge &edge = upd.edge;

    gts->insert({edge.src, edge.dst}, thr_id);
    std::swap(edge.src, edge.dst);
    gts->insert({edge.src, edge.dst}, thr_id);
#ifdef USE_EAGER_DSU
    if (dsu_valid) {
      auto src = std::min(edge.src, edge.dst);
      auto dst = std::max(edge.src, edge.dst);
      std::lock_guard<std::mutex> sflock(spanning_forest_mtx[src]);
      if (spanning_forest[src].find(dst) != spanning_forest[src].end()) {
        dsu_valid = false;
      } else {
        node_id_t a = src, b = dst;
        while ((a = get_parent(a)) != (b = get_parent(b))) {
          if (size[a] < size[b]) {
            std::swap(a, b);
          }
          if (std::atomic_compare_exchange_weak(&parent[b], &b, a)) {
            size[a] += size[b];
            spanning_forest[src].insert(dst);
            break;
          }
        }
      }
    }
#else
    unlikely_if(dsu_valid) dsu_valid = false;
#endif  // USE_EAGER_DSU
  }

  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param src        The node where the edges originate.
   * @param edges      A vector of destinations.
   * @param delta_loc  Memory location where we should initialize the delta sketch.
   */
  void batch_update(node_id_t src, const std::vector<node_id_t> &edges, Sketch *delta_loc);

  /**
   * Main parallel query algorithm utilizing Boruvka and L_0 sampling.
   * If cont is true, allow for additional updates when done.
   * @param cont
   * @return a vector of the connected components in the graph.
   */
  std::vector<std::set<node_id_t>> connected_components(bool cont = false);

  /**
   * Point query algorithm utilizing Boruvka and L_0 sampling.
   * Allows for additional updates when done.
   * @param a, b
   * @return true if a and b are in the same connected component, false otherwise.
   */
  bool point_query(node_id_t a, node_id_t b);

#ifdef VERIFY_SAMPLES_F
  std::unique_ptr<GraphVerifier> verifier;
  void set_verifier(std::unique_ptr<GraphVerifier> verifier) {
    this->verifier = std::move(verifier);
  }

  // to induce a failure mid-CC
  bool fail_round_2 = false;
  void should_fail_CC() { fail_round_2 = true; }
#endif

  // number of updates
  std::atomic<uint64_t> num_updates;

  /**
   * Generate a delta node for the purposes of updating a node sketch
   * (Sketch).
   * @param src           the src id.
   * @param edges         a list of node ids to which src is connected.
   * @param delta_sketch  the sketch to use as the delta.
   * @returns nothing (Sketch delta is in delta_sketch).
   */
  void apply_batch_updates(node_id_t src, const std::vector<node_id_t> &edges,
                           Sketch &delta_sketch);

  /**
   * Serialize the graph data to a binary file.
   * @param filename the name of the file to (over)write data to.
   */
  void write_binary(const std::string &filename);

  // time hooks for experiments
  std::chrono::steady_clock::time_point flush_start;
  std::chrono::steady_clock::time_point flush_end;
  std::chrono::steady_clock::time_point cc_alg_start;
  std::chrono::steady_clock::time_point cc_alg_end;

  // getters
  inline node_id_t get_num_vertices() { return num_nodes; }
  inline size_t get_seed() { return seed; }
};
