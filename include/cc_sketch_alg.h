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
#include <cassert>

#include "cc_alg_configuration.h"
#include "sketch.h"
#include "dsu.h"

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
 * Algorithm for computing connected components on undirected graph streams
 * (no self-edges or multi-edges)
 */
class CCSketchAlg {
 protected:
  node_id_t num_nodes;
  size_t seed;
  bool update_locked = false;
  // a set containing one "representative" from each supernode
  std::set<node_id_t> *representatives;
  Sketch **sketches;
  // DSU representation of supernode relationship
  DisjointSetUnion_MT<node_id_t> dsu;

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

  /**
   * Update the query array with new samples
   * @param query  an array of sketch sample results
   * @param reps   an array containing node indices for the representative of each supernode
   */
  bool sample_supernodes(std::vector<node_id_t> &merge_instr);

  /**
   * @param reps         set containing the roots of each supernode
   * @param merge_instr  a list of lists of supernodes to be merged
   */
  void merge_supernodes(const size_t next_round,
                        const std::vector<node_id_t> &merge_instr);

  /**
   * @param reps         set containing the roots of each supernode
   * @param merge_instr  an array where each vertex indicates its supernode root
   */
  void undo_merge_supernodes(const size_t cur_round,
                             const std::vector<node_id_t> &merge_instr);

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

  FRIEND_TEST(GraphTestSuite, TestCorrectnessOfReheating);

  CCAlgConfiguration config;

  // constructor for use when reading from a serialized file
  CCSketchAlg(node_id_t num_nodes, size_t seed, std::ifstream &binary_stream,
              CCAlgConfiguration config);

 public:
  CCSketchAlg(node_id_t num_nodes, CCAlgConfiguration config = CCAlgConfiguration());
  ~CCSketchAlg();

  // construct a CC algorithm from a serialized file
  static CCSketchAlg * construct_from_serialized_data(
      const std::string &input_file, CCAlgConfiguration config = CCAlgConfiguration());

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
   * Return if we have cached an answer to query.
   * This allows the driver to avoid flushing the gutters before calling query functions.
   */
  bool has_cached_query() { return shared_dsu_valid; }

  /**
   * Print the configuration of the connected components graph sketching.
   */
  void print_configuration() {
    std::cout << config << std::endl;
  }

  /**
   * Apply a batch of updates that have already been processed into a sketch delta.
   * Specifically, the delta is in the form of a pointer to raw bucket data.
   * @param src_vertex   The vertex where the all edges originate.
   * @param raw_buckets  Pointer to the array of buckets from the delta sketch
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
