#pragma once
#include <iostream>
#include <vector>
#include <memory>

#include "cc_sketch_alg.h"
#include "edge_store.h"


// Configuration options for the minimum cut sketch algorithm
class MCAlgConfiguration {
 private:
  // How large to make update batches as factor of sketch size
  double _batch_factor = 1;
  
  // Returned min-cut guaranteed to be a +/- epsilon multiplicative approx of the true min cut.
  double _epsilon = 0.5;

  // Number of subgraphs for which we use a delta sketch
  // When applying sketch updates to other subgraphs, apply updates directly to sketch
  size_t _num_subgraphs_use_delta = 2;

  friend class MinCutSketchAlg;
 public:
  // setters
  MCAlgConfiguration& batch_factor(double batch_factor) {
    if (batch_factor <= 0) {
      std::cerr << "WARNING: Batch factor in MCAlgConfiguration must be > 0." << std::endl;
      std::cerr << "         Setting to default value: " << _batch_factor << std::endl;
    } else {
      _batch_factor = batch_factor;
    }
    return *this;
  }
  MCAlgConfiguration& epsilon(double epsilon) {
    if (epsilon <= 0 || epsilon > 1) {
      std::cerr << "WARNING: MCAlgConfiguration epsilon must be in range (0, 1]." << std::endl;
      std::cerr << "         Setting to default value: " << _epsilon << std::endl;
    } else {
      _epsilon = epsilon;
    }
    return *this;
  }
  MCAlgConfiguration& num_subgraphs_use_delta(size_t num_subgraphs) {
    _num_subgraphs_use_delta = num_subgraphs;
    return *this;
  }

  // getters
  double get_batch_factor() { return _batch_factor; }
  double get_epsilon() { return _epsilon; }
  size_t get_num_subgraphs_use_delta() { return _num_subgraphs_use_delta; }

  friend std::ostream& operator<< (std::ostream &out, const MCAlgConfiguration &conf) {
    out << "Minimum Cut Algorithm Configuration:" << std::endl;
    out << "  batch_factor = " << conf._batch_factor << std::endl;
    return out;
  }
};

// Minimum cut sketch algorithm class
class MinCutSketchAlg {
 private:
  const node_id_t num_vertices;
  const size_t seed;
  MCAlgConfiguration config;
  const size_t max_subgraphs;
  size_t cur_subgraphs;

  const double sketch_factor;
  const size_t sketch_samples;

  CCSketchAlg **cc_sketches;
  EdgeStore edge_store;

  Sketch *delta_sketches = nullptr;
  node_id_t **update_buffers = nullptr;
  size_t num_delta_sketches = 0;
  size_t num_upd_buffers = 0;

#ifdef VERIFY_SAMPLES_F
  std::unique_ptr<GraphVerifier> verifier;
#endif

  CCAlgConfiguration cc_config;
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
    return config._batch_factor; // TODO: Fill in correctly
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
   * TODO: What is the right way to use verifier for minimum cut?
   */
#ifdef VERIFY_SAMPLES_F
  void set_verifier(std::unique_ptr<GraphVerifier> verifier) {
    this->verifier = std::move(verifier);
  }
#endif

  /**
   * Main query routine of this algorithm.
   * Returns an approximation of the minimum cut of the graph defined by the graph stream
   * seen thus far. This approximation is guaranteed to be within 1 +/- epsilon of the true
   * minimum cut.
   */
  size_t calc_minimum_cut();

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
    std::cout << config << std::endl;
  }

  node_id_t get_num_vertices() { return num_vertices; }
};
