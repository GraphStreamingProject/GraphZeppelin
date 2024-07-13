#pragma once

#include <atomic>
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
#include "return_types.h"
#include "sketch.h"
#include "dsu.h"

#ifdef VERIFY_SAMPLES_F
#include "test/graph_verifier.h"
#endif

// Exceptions the Connected Components algorithm may throw
class UpdateLockedException : public std::exception {
  virtual const char *what() const throw() {
    return "Cannot update the algorithm: Connected components currently running";
  }
};

struct MergeInstr {
  node_id_t root;
  node_id_t child;

  inline bool operator< (const MergeInstr &oth) const {
    if (root == oth.root)
      return child < oth.child;
    return root < oth.root;
  }
};

struct alignas(64) GlobalMergeData {
  Sketch sketch;
  std::mutex mtx;
  size_t num_merge_needed = -1;
  size_t num_merge_done = 0;

  GlobalMergeData(node_id_t num_vertices, size_t seed, double sketches_factor)
      : sketch(Sketch::calc_vector_length(num_vertices), seed,
               Sketch::calc_cc_samples(num_vertices, sketches_factor)) {}

  GlobalMergeData(const GlobalMergeData&& other)
  : sketch(other.sketch) {
    num_merge_needed = other.num_merge_needed;
    num_merge_done = other.num_merge_done;
  }
};

// What type of query is the user going to perform. Used for has_cached_query()
enum QueryCode {
  CONNECTIVITY,     // connected components and spanning forest of graph
  KSPANNINGFORESTS, // k disjoint spanning forests
  MINIMUMCUT,       // compute the minimum cut of the graph
};

struct MinCut {
  std::set<node_id_t> left_vertices;
  std::set<node_id_t> right_vertices;
  size_t value;
};

/**
 * Algorithm for computing minimum cut on undirected graph streams
 * (no self-edges or multi-edges)
 */
class MCSketchAlg {
 private:
  int max_sketch_graphs;
  int num_sketch_graphs;
  node_id_t num_vertices;
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

  CCAlgConfiguration config;
#ifdef VERIFY_SAMPLES_F
  std::unique_ptr<GraphVerifier> verifier;
#endif

  /**
   * Run the first round of Boruvka. We can do things faster here because we know there will
   * be no merging we have to do.
   */
  bool run_round_zero();

  // run_round_zero for k connectivitiy
  bool run_k_round_zero(int graph_id);

  /**
   * Sample a single supernode represented by a single sketch containing one or more vertices.
   * Updates the dsu and spanning forest with query results if edge contains new connectivity info.
   * @param skt   sketch to sample
   * @return      [bool] true if the query result indicates we should run an additional round.
   */
  bool sample_supernode(Sketch &skt);


  /**
   * Calculate the instructions for what vertices to merge to form each component
   */
  void create_merge_instructions(std::vector<MergeInstr> &merge_instr);

  /**
   * @param reps         set containing the roots of each supernode
   * @param merge_instr  a list of lists of supernodes to be merged
   */
  bool perform_boruvka_round(const size_t cur_round, const std::vector<MergeInstr> &merge_instr,
                             std::vector<GlobalMergeData> &global_merges);

  // Boruvka round for k connectivitiy
  bool perform_k_boruvka_round(const size_t cur_round, const std::vector<MergeInstr> &merge_instr,
                             std::vector<GlobalMergeData> &global_merges, int graph_id);

  /**
   * Main parallel algorithm utilizing Boruvka and L_0 sampling.
   * Ensures that the DSU represents the Connected Components of the stream when called
   */
  void boruvka_emulation();

  // Boruvka algorithm for k connectivitiy
  void k_boruvka_emulation(int graph_id);

  // constructor for use when reading from a serialized file
  MCSketchAlg(node_id_t num_vertices, size_t seed, std::ifstream &binary_stream,
              CCAlgConfiguration config);

 public:
  MCSketchAlg(node_id_t num_vertices, size_t seed, Bucket* first_graph_buckets, int _max_sketch_graphs, CCAlgConfiguration config = CCAlgConfiguration());
  ~MCSketchAlg();

  // construct a MC algorithm from a serialized file
  static MCSketchAlg * construct_from_serialized_data(
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
      delta_sketches[i] =
          new Sketch(Sketch::calc_vector_length(num_vertices), seed,
                     Sketch::calc_cc_samples(num_vertices, config.get_sketches_factor()));
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
   * Update all the sketches for a node, given a batch of updates. (Multi-graphs version, Applies update to specified graph)
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param graph_id       The id of sketch graph
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch_single_graph(int thr_id, int graph_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  /**
   * Return if we have cached an answer to query.
   * This allows the driver to avoid flushing the gutters before calling query functions.
   */
  bool has_cached_query(int query_code) {
    QueryCode code = (QueryCode) query_code;
    if (code == CONNECTIVITY)
      return shared_dsu_valid;
    else
      return false;
  }

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
   * @return  the connected components in the graph.
   */
  ConnectedComponents connected_components();

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
   * @return  the spanning forest of the graph
   */
  SpanningForest calc_spanning_forest();

  SpanningForest get_k_spanning_forest(int graph_id);

  // compute the minimum cut of the graph defined by the input stream
  MinCut calc_minimum_cut(const std::vector<Edge> &edges);

#ifdef VERIFY_SAMPLES_F
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
  size_t last_query_rounds = 0;

  // getters
  inline node_id_t get_num_vertices() { return num_vertices; }
  inline size_t get_seed() { return seed; }
  inline size_t max_rounds() { return sketches[0]->get_num_samples(); }
  inline bool get_update_locked() { return update_locked; }
};
