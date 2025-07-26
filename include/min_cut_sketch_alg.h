#pragma once
#include <iostream>
#include <vector>
#include <memory>

#include "cc_sketch_alg.h"
#include "edge_store.h"
#include "mc_configuration.h"

// Minimum cut sketch algorithm class
class MinCutSketchAlg {
 private:
  struct ThreadData {
    std::vector<std::vector<node_id_t>> cc_buffers;
    std::vector<SubgraphTaggedUpdate> edge_store_buffer;
  };

  const node_id_t num_vertices;
  const size_t seed;
  const size_t subgraph_seed;
  MCAlgConfiguration config;
  const size_t max_subgraphs;
  const size_t k;
  std::atomic<size_t> cur_subgraphs;
  std::mutex advance_subgraph_lock;

  const double sketch_factor;
  const size_t sketch_samples;
  const size_t buffer_elms;

  CCSketchAlg **cc_sketches;
  EdgeStore edge_store;

  Sketch *delta_sketches = nullptr;
  ThreadData *thread_data = nullptr;
  size_t num_delta_sketches = 0;
  size_t num_workers;

#ifdef VERIFY_SAMPLES_F
  std::unique_ptr<GraphVerifier> verifier;
  std::unique_ptr<GraphVerifier> adj_verifier;
#endif

  CCAlgConfiguration cc_config;

  void advance_cur_subgraph(size_t new_cur_subgraphs);

  void create_subgraph_verifiers();
 public:
  /**
   * Construct an instance of the Minimum Cut Sketching Algorithm
   * param _num_vertices  number of graph vertices
   * param _seed          seed to hash functions
   * param _config        Configuration options for minimum cut sketch algorithm
   */
  MinCutSketchAlg(node_id_t _num_vertices, size_t _seed,
                  MCAlgConfiguration _config = MCAlgConfiguration());

  ~MinCutSketchAlg();

  /**
   * Allocate memory for the worker threads to use when updating this algorithm's sketches
   */
  void allocate_worker_memory(size_t num_workers);

  /**
   * Returns the number of buffered updates we would like to have in the update batches
   */
  size_t get_desired_updates_per_batch() {
    return cc_sketches[0]->get_desired_updates_per_batch();
  }

  /**
   * Action to take on an update before inserting it to the guttering system.
   * We use this function to manage the eager dsu.
   */
  void pre_insert(GraphUpdate upd, node_id_t thr_id);


  /**
   * Update all the sketches for a vertex, given a batch of updates.
   * param thr_id         The id of the thread performing the update [0, num_threads)
   * param src_vertex     The vertex where the edges originate.
   * param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(size_t thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  /**
   * Set the verifier this algorithm will use to check its correctness
   * param: _verifier   the verifier to use, should contain all edges processed at this point
   */
#ifdef VERIFY_SAMPLES_F
  void set_verifier(std::unique_ptr<GraphVerifier> _verifier);
#endif

  /**
   * Main query routine of this algorithm.
   * Returns an approximation of the minimum cut of the graph defined by the graph stream
   * seen thus far. This approximation is guaranteed to be within 1 +/- epsilon of the true
   * minimum cut.
   */
  MinCut calc_minimum_cut();

  /**
   * Return if we have cached an answer to query.
   * This allows the driver to avoid flushing the gutters before calling query functions.
   * TODO: Is there something intelligent we can do here for mincut/k-conn
   */
  bool has_cached_query(int query_type) {
    if (query_type != MINIMUMCUT) return cc_sketches[0]->has_cached_query(query_type);
    return false; 
  }

  /**
   * Print the configuration of minimum cut graph sketching algorithm.
   */
  void print_configuration() {
    std::cout << config;
    std::cout << "MCAlg using the following CCAlg config:" << std::endl;
    std::cout << cc_config << std::endl;
  }

  node_id_t get_num_vertices() { return num_vertices; }

  // time hooks for experiments
  std::chrono::duration<double> total_mc_duration;
  std::chrono::duration<double> sf_total_duration;
  std::chrono::duration<double> viecut_duration;
};
