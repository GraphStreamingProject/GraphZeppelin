#pragma once

#include <atomic>  // REMOVE LATER
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <unordered_set>
#include <vector>
#include <memory>

#include "cc_alg_configuration.h"
#include "sketch.h"

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

/**
 * Undirected graph object with n nodes labelled 0 to n-1, no self-edges,
 * multiple edges, or weights.
 */
class CCSketchAlg {
 protected:
  node_id_t num_nodes;
  uint64_t seed;
  bool update_locked = false;
  bool modified = false;
  // a set containing one "representative" from each supernode
  std::set<node_id_t> *representatives;
  Sketch **sketches;
  // DSU representation of supernode relationship
#ifndef NO_EAGER_DSU
  std::atomic<node_id_t> *parent;
#else
  node_id_t *parent;
#endif
  node_id_t *size;
  node_id_t get_parent(node_id_t node);

  // if dsu valid then we have a cached query answer. Additionally, we need to update the DSU in
  // pre_insert()
  bool dsu_valid = true;

  // for accessing if the DSU is valid from threads that do not perform updates
  std::atomic<bool> shared_dsu_valid;

  std::unordered_set<node_id_t> *spanning_forest;
  std::mutex *spanning_forest_mtx;

  // threads use these sketches to apply delta updates to our sketches
  Sketch **delta_sketches = nullptr;
  size_t num_delta_sketches;

  void backup_to_disk(const std::vector<node_id_t> &ids_to_backup);
  void restore_from_disk(const std::vector<node_id_t> &ids_to_restore);

  /**
   * Update the query array with new samples
   * @param query  an array of sketch sample results
   * @param reps   an array containing node indices for the representative of each supernode
   */
  void sample_supernodes(std::pair<Edge, SampleSketchRet> *query, std::vector<node_id_t> &reps);

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
  std::vector<std::set<node_id_t>> boruvka_emulation();

  /**
   * Generates connected components from this graph's dsu
   * @return a vector of the connected components in the graph.
   */
  std::vector<std::set<node_id_t>> cc_from_dsu();

  std::string backup_file;  // where to backup the supernodes

  FRIEND_TEST(GraphTestSuite, TestCorrectnessOfReheating);
  FRIEND_TEST(GraphTest, TestSupernodeRestoreAfterCCFailure);

  CCAlgConfiguration config;

 public:
  CCSketchAlg(node_id_t num_nodes, CCAlgConfiguration config = CCAlgConfiguration());
  CCSketchAlg(const std::string &input_file, CCAlgConfiguration config = CCAlgConfiguration());
  ~CCSketchAlg();

  /**
   * Returns the number of buffered updates we would like to have in the update batches
   */
  size_t get_desired_updates_per_batch() { 
    size_t num = sketches[0]->bucket_array_bytes() / sizeof(node_id_t);
    num *= config._batch_factor;
    return num;
  }

  /**
   * Action to take on an update before inserting it to the guttering system.
   * We use this function to manage the eager dsu.
   */
  void pre_insert(GraphUpdate upd, int thr_id = 0);

  /**
   * Allocate memory for the worker threads to use when updating this algorithm's sketches
   */
  void allocate_worker_memory(size_t num_workers) {
    num_delta_sketches = num_workers;
    delta_sketches = new Sketch *[num_delta_sketches];
    for (size_t i = 0; i < num_delta_sketches; i++) {
      delta_sketches[i] = new Sketch(Sketch::calc_vector_length(num_nodes), seed,
                                     Sketch::calc_cc_samples(num_nodes));
    }
  }

  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  /**
   * 
   */
  void apply_raw_buckets_update(node_id_t src_vertex, Bucket *raw_buckets);

  /**
   * The function performs a direct update to the associated sketch.
   * For performance reasons, do not use this function if possible.
   * 
   * This function is not thread-safe
   */
  void update(GraphUpdate upd);

  /**
   * Return if we have cached an answer to query.
   * This allows the driver to avoid flushing the gutters before calling query functions.
   */
  bool has_cached_query() { return shared_dsu_valid; }

  /**
   * Main parallel query algorithm utilizing Boruvka and L_0 sampling.
   * @return a vector of the connected components in the graph.
   */
  std::vector<std::set<node_id_t>> connected_components();

  /**
   * Point query algorithm utilizing Boruvka and L_0 sampling.
   * Allows for additional updates when done.
   * @param a, b
   * @return true if a and b are in the same connected component, false otherwise.
   */
  bool point_query(node_id_t a, node_id_t b);

  /**
   * Return a spanning forest of the graph utilizing Boruvka and L_0 sampling
   * IMPORTANT: The updates to this algorithm MUST NOT be a function of the output of this query
   * that is, unless you really know what you're doing.
   * @return an adjacency list representation of the spanning forest of the graph
   */
  std::vector<std::pair<node_id_t, std::vector<node_id_t>>> calc_spanning_forest();

#ifdef VERIFY_SAMPLES_F
  std::unique_ptr<GraphVerifier> verifier;
  void set_verifier(std::unique_ptr<GraphVerifier> verifier) {
    this->verifier = std::move(verifier);
  }

  // to induce a failure mid-CC
  bool fail_round_2 = false;
  void should_fail_CC() { fail_round_2 = true; }
#endif

  /**
   * Serialize the graph data to a binary file.
   * @param filename the name of the file to (over)write data to.
   */
  void write_binary(const std::string &filename);

  // time hooks for experiments
  std::chrono::steady_clock::time_point cc_alg_start;
  std::chrono::steady_clock::time_point cc_alg_end;

  // getters
  inline node_id_t get_num_vertices() { return num_nodes; }
  inline size_t get_seed() { return seed; }
};
