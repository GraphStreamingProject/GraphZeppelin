#include "mc_sketch_alg.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <omp.h>
#include <unordered_map>

#include <algorithms/global_mincut/algorithms.h>
#include <algorithms/global_mincut/minimum_cut.h>
#include <data_structure/graph_access.h>
#include <data_structure/mutable_graph.h>

MCSketchAlg::MCSketchAlg(node_id_t num_vertices, size_t seed, Bucket* buckets, int _max_sketch_graphs, CCAlgConfiguration config)
    : num_vertices(num_vertices), seed(seed), dsu(num_vertices), config(config) {
  representatives = new std::set<node_id_t>();
  max_sketch_graphs = _max_sketch_graphs;
  vec_t sketch_vec_len = Sketch::calc_vector_length(num_vertices);
  size_t sketch_num_samples = Sketch::calc_cc_samples(num_vertices, config.get_sketches_factor());
  size_t sketch_num_columns = sketch_num_samples * Sketch::default_cols_per_sample;
  size_t sketch_bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_vertices));
  size_t sketch_num_buckets = sketch_num_columns * sketch_bkt_per_col + 1;

  sketches = new Sketch *[num_vertices * max_sketch_graphs];

  for (int graph_id = 0; graph_id < max_sketch_graphs; graph_id++) {
    for (node_id_t i = 0; i < num_vertices; ++i) {
      sketches[(graph_id * num_vertices) + i] = new Sketch(sketch_vec_len, seed, i, &buckets[graph_id * num_vertices * sketch_num_buckets], sketch_num_samples);
    }
  }
  
  for (node_id_t i = 0; i < num_vertices; ++i) {
    representatives->insert(i);
  }
  spanning_forest = new std::unordered_set<node_id_t>[num_vertices];
  spanning_forest_mtx = new std::mutex[num_vertices];

  // Note: Turn these off for k tree graphs
  dsu_valid = false;
  shared_dsu_valid = false;
}

// Note: Not being used currently
MCSketchAlg *MCSketchAlg::construct_from_serialized_data(const std::string &input_file,
                                                        CCAlgConfiguration config) {
  double sketches_factor;
  auto binary_in = std::ifstream(input_file, std::ios::binary);
  size_t seed;
  node_id_t num_vertices;
  binary_in.read((char *)&seed, sizeof(seed));
  binary_in.read((char *)&num_vertices, sizeof(num_vertices));
  binary_in.read((char *)&sketches_factor, sizeof(sketches_factor));

  config.sketches_factor(sketches_factor);

  return new MCSketchAlg(num_vertices, seed, binary_in, config);
}

// Note: Not being used currently
MCSketchAlg::MCSketchAlg(node_id_t num_vertices, size_t seed, std::ifstream &binary_stream,
                         CCAlgConfiguration config)
    : num_vertices(num_vertices), seed(seed), dsu(num_vertices), config(config) {
  representatives = new std::set<node_id_t>();
  sketches = new Sketch *[num_vertices];

  vec_t sketch_vec_len = Sketch::calc_vector_length(num_vertices);
  size_t sketch_num_samples = Sketch::calc_cc_samples(num_vertices, config.get_sketches_factor());

  for (node_id_t i = 0; i < num_vertices; ++i) {
    representatives->insert(i);
    sketches[i] = new Sketch(sketch_vec_len, seed, binary_stream, sketch_num_samples);
  }
  binary_stream.close();

  spanning_forest = new std::unordered_set<node_id_t>[num_vertices];
  spanning_forest_mtx = new std::mutex[num_vertices];
  dsu_valid = false;
  shared_dsu_valid = false;
}

MCSketchAlg::~MCSketchAlg() {
  for (size_t i = 0; i < num_vertices * max_sketch_graphs; ++i) delete sketches[i];
  delete[] sketches;
  if (delta_sketches != nullptr) {
    for (size_t i = 0; i < num_delta_sketches; i++) delete delta_sketches[i];
    delete[] delta_sketches;
  }

  delete representatives;
  delete[] spanning_forest;
  delete[] spanning_forest_mtx;
}

void MCSketchAlg::pre_insert(GraphUpdate upd, int /* thr_id */) {
#ifdef NO_EAGER_DSU
  (void)upd;
  // reason we have an if statement: avoiding cache coherency issues
  unlikely_if(dsu_valid) {
    dsu_valid = false;
    shared_dsu_valid = false;
  }
#else
  if (dsu_valid) {
    Edge edge = upd.edge;
    auto src = std::min(edge.src, edge.dst);
    auto dst = std::max(edge.src, edge.dst);
    std::lock_guard<std::mutex> sflock(spanning_forest_mtx[src]);
    if (dsu.merge(src, dst).merged) {
      // this edge adds new connectivity information so add to spanning forest
      spanning_forest[src].insert(dst);
    }
    else if (spanning_forest[src].find(dst) != spanning_forest[src].end()) {
      // this update deletes one of our spanning forest edges so mark dsu invalid
      dsu_valid = false;
      shared_dsu_valid = false;
    }
  }
#endif  // NO_EAGER_DSU
}

void MCSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (update_locked) throw UpdateLockedException();
  Sketch &delta_sketch = *delta_sketches[thr_id];
  delta_sketch.zero_contents();

  for (const auto &dst : dst_vertices) {
    delta_sketch.update(static_cast<vec_t>(concat_pairing_fn(src_vertex, dst)));
  }

  std::lock_guard<std::mutex> lk(sketches[src_vertex]->mutex);
  sketches[src_vertex]->merge(delta_sketch);
}

void MCSketchAlg::apply_update_batch_single_graph(int thr_id, int graph_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (update_locked) throw UpdateLockedException();
  Sketch &delta_sketch = *delta_sketches[thr_id];
  delta_sketch.zero_contents();

  for (const auto &dst : dst_vertices) {
    delta_sketch.update(static_cast<vec_t>(concat_pairing_fn(src_vertex, dst)));
  }

  std::lock_guard<std::mutex> lk(sketches[(graph_id * num_vertices) + src_vertex]->mutex);
  sketches[(graph_id * num_vertices) + src_vertex]->merge(delta_sketch);                                 
}

void MCSketchAlg::apply_raw_buckets_update(node_id_t src_vertex, Bucket *raw_buckets) {
  std::lock_guard<std::mutex> lk(sketches[src_vertex]->mutex);
  sketches[src_vertex]->merge_raw_bucket_buffer(raw_buckets);
}

// Note: for performance reasons route updates through the driver instead of calling this function
//       whenever possible.
void MCSketchAlg::update(GraphUpdate upd) {
  pre_insert(upd, 0);
  Edge edge = upd.edge;

  sketches[edge.src]->update(static_cast<vec_t>(concat_pairing_fn(edge.src, edge.dst)));
  sketches[edge.dst]->update(static_cast<vec_t>(concat_pairing_fn(edge.src, edge.dst)));
}

// sample from a sketch that represents a supernode of vertices
// that is, 1 or more vertices merged together during Boruvka
inline bool MCSketchAlg::sample_supernode(Sketch &skt) {
  bool modified = false;
  SketchSample sample = skt.sample();

  Edge e = inv_concat_pairing_fn(sample.idx);
  SampleResult result_type = sample.result;

  // std::cout << " " << result_type << " e:" << e.src << " " << e.dst << std::endl;

  if (result_type == FAIL) {
    modified = true;
  } else if (result_type == GOOD) {
    DSUMergeRet<node_id_t> m_ret = dsu.merge(e.src, e.dst);
    if (m_ret.merged) {
#ifdef VERIFY_SAMPLES_F
      verifier->verify_edge(e);
#endif
      modified = true;
      // Update spanning forest
      auto src = std::min(e.src, e.dst);
      auto dst = std::max(e.src, e.dst);
      {
        std::lock_guard<std::mutex> lk(spanning_forest_mtx[src]);
        spanning_forest[src].insert(dst);
      }
    }
  }

  return modified;
}

/*
 * Returns the ith half-open range in the division of [0, length] into divisions segments.
 */
inline std::pair<node_id_t, node_id_t> get_ith_partition(node_id_t length, size_t i,
                                                         size_t divisions) {
  double div_factor = (double)length / divisions;
  return {ceil(div_factor * i), ceil(div_factor * (i + 1))};
}

/*
 * Returns the half-open range idx that contains idx
 * Inverse of get_ith_partition
 */
inline size_t get_partition_idx(node_id_t length, node_id_t idx, size_t divisions) {
  double div_factor = (double)length / divisions;
  return idx / div_factor;
}

inline node_id_t find_last_partition_of_root(const std::vector<MergeInstr> &merge_instr,
                                             const node_id_t root, node_id_t min_hint,
                                             size_t num_threads) {
  node_id_t max = merge_instr.size() - 1;
  node_id_t min = min_hint;
  MergeInstr target = {root, (node_id_t) -1};

  while (min < max) {
    node_id_t mid = min + (max - min) / 2;

    if (merge_instr[mid] < target) {
      min = mid + 1;
    } else {
      max = mid;
    }
  }

  if (merge_instr[min].root != root)
    min = min - 1;

  assert(merge_instr[min].root == root);
  assert(min == merge_instr.size() - 1 || merge_instr[min + 1].root > root);
  return get_partition_idx(merge_instr.size(), min, num_threads);
}

// merge the global and return if it is safe to query now
inline bool merge_global(const size_t cur_round, const Sketch &local_sketch,
                         GlobalMergeData &global) {
  std::lock_guard<std::mutex> lk(global.mtx);
  global.sketch.range_merge(local_sketch, cur_round, 1);
  ++global.num_merge_done;
  assert(global.num_merge_done <= global.num_merge_needed);

  return global.num_merge_done >= global.num_merge_needed;
}

// faster query procedure optimized for when we know there is no merging to do (i.e. round 0)
inline bool MCSketchAlg::run_round_zero() {
  bool modified = false;
  bool except = false;
  std::exception_ptr err;
#pragma omp parallel for
  for (node_id_t i = 0; i < num_vertices; i++) {
    try {
      // num_query += 1;
      if (sample_supernode(*sketches[i]) && !modified) modified = true;
    } catch (...) {
      except = true;
#pragma omp critical
      err = std::current_exception();
    }
  }
  if (except) {
    // if one of our threads produced an exception throw it here
    std::rethrow_exception(err);
  }

  return modified;
}

inline bool MCSketchAlg::run_k_round_zero(int graph_id) {
  bool modified = false;
  bool except = false;
  std::exception_ptr err;
#pragma omp parallel for
  for (node_id_t i = 0; i < num_vertices; i++) {
    try {
      // num_query += 1;
      if (sample_supernode(*sketches[(graph_id * num_vertices) + i]) && !modified) modified = true;
    } catch (...) {
      except = true;
#pragma omp critical
      err = std::current_exception();
    }
  }
  if (except) {
    // if one of our threads produced an exception throw it here
    std::rethrow_exception(err);
  }

  return modified;
}

bool MCSketchAlg::perform_boruvka_round(const size_t cur_round,
                                        const std::vector<MergeInstr> &merge_instr,
                                        std::vector<GlobalMergeData> &global_merges) {
  if (cur_round == 0) {
    return run_round_zero();
  }

  bool modified = false;
  bool except = false;
  std::exception_ptr err;
  for (size_t i = 0; i < global_merges.size(); i++) {
    global_merges[i].sketch.zero_contents();
    global_merges[i].num_merge_needed = -1;
    global_merges[i].num_merge_done = 0;
  }

#pragma omp parallel default(shared)
  {
    // some thread local variables
    Sketch local_sketch(Sketch::calc_vector_length(num_vertices), seed,
                        Sketch::calc_cc_samples(num_vertices, config.get_sketches_factor()));

    size_t thr_id = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    std::pair<node_id_t, node_id_t> partition = get_ith_partition(num_vertices, thr_id, num_threads);
    node_id_t start = partition.first;
    node_id_t end = partition.second;
    assert(start <= end);
    bool local_except = false;
    std::exception_ptr local_err;

    bool root_from_left = false;
    if (start > 0) {
      root_from_left = merge_instr[start - 1].root == merge_instr[start].root;
    }
    bool root_exits_right = false;
    if (end < num_vertices) {
      root_exits_right = merge_instr[end - 1].root == merge_instr[end].root;
    }

    node_id_t cur_root = merge_instr[start].root;

    // std::cout << thr_id << std::endl;
    // std::cout << "  Component " << cur_root << ":";
    for (node_id_t i = start; i < end; i++) {
      node_id_t root = merge_instr[i].root;
      node_id_t child = merge_instr[i].child;

      if (root != cur_root) {
        if (root_from_left) {
          // we hold the global for this merge
          // std::cout << " merge global (we own)" << std::endl;
          bool query_ready = merge_global(cur_round, local_sketch, global_merges[thr_id]);
          if (query_ready) {
            // std::cout << "Performing query!";
            try {
              // num_query += 1;
              if (sample_supernode(global_merges[thr_id].sketch) && !modified) modified = true;
            } catch (...) {
              local_except = true;
              local_err = std::current_exception();
            }
          }

          // set root_from_left to false
          root_from_left = false;
        } else {
          // This is an entirely local computation
          // std::cout << " query local";
          try {
            // num_query += 1;
            if (sample_supernode(local_sketch) && !modified) modified = true;
          } catch (...) {
            local_except = true;
            local_err = std::current_exception();
          }
        }

        cur_root = root;
        // std::cout << "  Component " << cur_root << ":";
        local_sketch.zero_contents();
      }

      // std::cout << " " << child;
      local_sketch.range_merge(*sketches[child], cur_round, 1);
    }

    if (root_exits_right || root_from_left) {
      // global merge where we may or may not own it
      size_t global_id = find_last_partition_of_root(merge_instr, cur_root, start, num_threads);
      // std::cout << " merge global (" << global_id << ")" << std::endl;
      if (!root_from_left) {
        // Resolved root_from_left, so we are the first thread to encounter this root
        // set the number of threads that will merge into this component
        std::lock_guard<std::mutex> lk(global_merges[global_id].mtx);
        global_merges[global_id].num_merge_needed = global_id - thr_id + 1;
      }
      bool query_ready = merge_global(cur_round, local_sketch, global_merges[global_id]);
      if (query_ready) {
        // std::cout << "Performing query!";
        try {
          // num_query += 1;
          if (sample_supernode(global_merges[global_id].sketch) && !modified) modified = true;
        } catch (...) {
          local_except = true;
          local_err = std::current_exception();
        }
      }
    } else {
      // This is an entirely local computation
      // std::cout << " query local";
      try {
        // num_query += 1;
        if (sample_supernode(local_sketch) && !modified) modified = true;
      } catch (...) {
        local_except = true;
        local_err = std::current_exception();
      }
    }
    if (local_except) {
#pragma omp critical
      err = local_err;
      except = true;
    }
  }

  // std::cout << "Number of roots queried = " << num_query << std::endl;

  if (except) {
    // if one of our threads produced an exception throw it here
    std::rethrow_exception(err);
  }

  return modified;
}

bool MCSketchAlg::perform_k_boruvka_round(const size_t cur_round,
                                        const std::vector<MergeInstr> &merge_instr,
                                        std::vector<GlobalMergeData> &global_merges,
                                        int graph_id) {
  if (cur_round == 0) {
    return run_k_round_zero(graph_id);
  }

  bool modified = false;
  bool except = false;
  std::exception_ptr err;
  for (size_t i = 0; i < global_merges.size(); i++) {
    global_merges[i].sketch.zero_contents();
    global_merges[i].num_merge_needed = -1;
    global_merges[i].num_merge_done = 0;
  }

#pragma omp parallel default(shared)
  {
    // some thread local variables
    Sketch local_sketch(Sketch::calc_vector_length(num_vertices), seed,
                        Sketch::calc_cc_samples(num_vertices, config.get_sketches_factor()));

    size_t thr_id = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    std::pair<node_id_t, node_id_t> partition = get_ith_partition(num_vertices, thr_id, num_threads);
    node_id_t start = partition.first;
    node_id_t end = partition.second;
    assert(start <= end);
    bool local_except = false;
    std::exception_ptr local_err;

    bool root_from_left = false;
    if (start > 0) {
      root_from_left = merge_instr[start - 1].root == merge_instr[start].root;
    }
    bool root_exits_right = false;
    if (end < num_vertices) {
      root_exits_right = merge_instr[end - 1].root == merge_instr[end].root;
    }

    node_id_t cur_root = merge_instr[start].root;

    // std::cout << thr_id << std::endl;
    // std::cout << "  Component " << cur_root << ":";
    for (node_id_t i = start; i < end; i++) {
      node_id_t root = merge_instr[i].root;
      node_id_t child = merge_instr[i].child;

      if (root != cur_root) {
        if (root_from_left) {
          // we hold the global for this merge
          // std::cout << " merge global (we own)" << std::endl;
          bool query_ready = merge_global(cur_round, local_sketch, global_merges[thr_id]);
          if (query_ready) {
            // std::cout << "Performing query!";
            try {
              // num_query += 1;
              if (sample_supernode(global_merges[thr_id].sketch) && !modified) modified = true;
            } catch (...) {
              local_except = true;
              local_err = std::current_exception();
            }
          }

          // set root_from_left to false
          root_from_left = false;
        } else {
          // This is an entirely local computation
          // std::cout << " query local";
          try {
            // num_query += 1;
            if (sample_supernode(local_sketch) && !modified) modified = true;
          } catch (...) {
            local_except = true;
            local_err = std::current_exception();
          }
        }

        cur_root = root;
        // std::cout << "  Component " << cur_root << ":";
        local_sketch.zero_contents();
      }

      // std::cout << " " << child;
      local_sketch.range_merge(*sketches[(graph_id * num_vertices) + child], cur_round, 1);
    }

    if (root_exits_right || root_from_left) {
      // global merge where we may or may not own it
      size_t global_id = find_last_partition_of_root(merge_instr, cur_root, start, num_threads);
      // std::cout << " merge global (" << global_id << ")" << std::endl;
      if (!root_from_left) {
        // Resolved root_from_left, so we are the first thread to encounter this root
        // set the number of threads that will merge into this component
        std::lock_guard<std::mutex> lk(global_merges[global_id].mtx);
        global_merges[global_id].num_merge_needed = global_id - thr_id + 1;
      }
      bool query_ready = merge_global(cur_round, local_sketch, global_merges[global_id]);
      if (query_ready) {
        // std::cout << "Performing query!";
        try {
          // num_query += 1;
          if (sample_supernode(global_merges[global_id].sketch) && !modified) modified = true;
        } catch (...) {
          local_except = true;
          local_err = std::current_exception();
        }
      }
    } else {
      // This is an entirely local computation
      // std::cout << " query local";
      try {
        // num_query += 1;
        if (sample_supernode(local_sketch) && !modified) modified = true;
      } catch (...) {
        local_except = true;
        local_err = std::current_exception();
      }
    }
    if (local_except) {
#pragma omp critical
      err = local_err;
      except = true;
    }
  }

  // std::cout << "Number of roots queried = " << num_query << std::endl;

  if (except) {
    // if one of our threads produced an exception throw it here
    std::rethrow_exception(err);
  }

  return modified;
}

inline void MCSketchAlg::create_merge_instructions(std::vector<MergeInstr> &merge_instr) {
  std::vector<node_id_t> cc_prefix(num_vertices, 0);
  node_id_t range_sums[omp_get_max_threads()];

#pragma omp parallel default(shared)
  {
    // thread local variables
    std::unordered_map<node_id_t, std::vector<node_id_t>> local_ccs;
    std::vector<node_id_t> local_cc_idx;

    size_t thr_id = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
    std::pair<node_id_t, node_id_t> partition = get_ith_partition(num_vertices, thr_id, num_threads);
    node_id_t start = partition.first;
    node_id_t end = partition.second;

    for (node_id_t i = start; i < end; i++) {
      node_id_t child = merge_instr[i].child;
      node_id_t root = dsu.find_root(child);
      if (local_ccs.count(root) == 0) {
        local_ccs[root] = {child};
      } else {
        local_ccs[root].push_back(child);
      }
    }

    // each thread loops over its local_ccs and updates cc_prefix
    for (auto const &cc : local_ccs) {
      node_id_t root = cc.first;
      const std::vector<node_id_t> &vertices = cc.second;

      node_id_t idx;
#pragma omp atomic capture
      {idx = cc_prefix[root]; cc_prefix[root] += vertices.size(); }

      local_cc_idx.push_back(idx);
    }
#pragma omp barrier

    // perform a prefix sum over cc_prefix
    for (node_id_t i = start + 1; i < end; i++) {
      cc_prefix[i] += cc_prefix[i-1];
    }
#pragma omp barrier

    // perform single threaded prefix sum of the resulting sums from each thread
#pragma omp single
    {
      range_sums[0] = 0;
      for (int t = 1; t < omp_get_num_threads(); t++) {
        node_id_t cur = get_ith_partition(num_vertices, t - 1, num_threads).second - 1;
        range_sums[t] = cc_prefix[cur] + range_sums[t - 1];
      }
    }

    // in parallel finish the prefix sums
    if (thr_id > 0) {
      for (node_id_t i = start; i < end; i++) {
        cc_prefix[i] += range_sums[thr_id];
      }
    }
#pragma omp barrier

    // Finally, write the local_ccs to the correct portion of the merge_instr array
    node_id_t i = 0;
    for (auto const &cc : local_ccs) {
      node_id_t root = cc.first;
      const std::vector<node_id_t> &vertices = cc.second;
      node_id_t thr_idx = local_cc_idx[i];

      node_id_t placement = thr_idx;
      if (root > 0)
        placement += cc_prefix[root - 1];

      for (size_t j = 0; j < vertices.size(); j++) {
        merge_instr[placement + j] = {root, vertices[j]};
      }
      i++;
    }
  }
}

void MCSketchAlg::boruvka_emulation() {
  // auto start = std::chrono::steady_clock::now();
  update_locked = true;

  cc_alg_start = std::chrono::steady_clock::now();
  std::vector<MergeInstr> merge_instr(num_vertices);

  size_t num_threads = omp_get_max_threads();
  std::vector<GlobalMergeData> global_merges;
  global_merges.reserve(num_threads);
  for (size_t i = 0; i < num_threads; i++) {
    global_merges.emplace_back(num_vertices, seed, config.get_sketches_factor());
  }

  dsu.reset();
  for (node_id_t i = 0; i < num_vertices; ++i) {
    merge_instr[i] = {i, i};
    spanning_forest[i].clear();
  }
  size_t round_num = 0;
  bool modified = true;
  // std::cout << std::endl;
  // std::cout << "  pre boruvka processing = "
  //             << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
  //             << std::endl;

  while (true) {
    // std::cout << "   Round: " << round_num << std::endl;
    // start = std::chrono::steady_clock::now();
    modified = perform_boruvka_round(round_num, merge_instr, global_merges);
    // std::cout << "     perform_boruvka_round = "
    //           << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
    //           << std::endl;

    if (!modified) break;

    // calculate updated merge instructions for next round
    // start = std::chrono::steady_clock::now();
    create_merge_instructions(merge_instr);
    // std::cout << "     create_merge_instructions = "
    //           << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
    //           << std::endl;
    ++round_num;
  }
  last_query_rounds = round_num;

  dsu_valid = true;
  shared_dsu_valid = true;
  update_locked = false;
}

void MCSketchAlg::k_boruvka_emulation(int graph_id) {
  // auto start = std::chrono::steady_clock::now();
  update_locked = true;

  cc_alg_start = std::chrono::steady_clock::now();
  std::vector<MergeInstr> merge_instr(num_vertices);

  size_t num_threads = omp_get_max_threads();
  std::vector<GlobalMergeData> global_merges;
  global_merges.reserve(num_threads);
  for (size_t i = 0; i < num_threads; i++) {
    global_merges.emplace_back(num_vertices, seed, config.get_sketches_factor());
  }

  dsu.reset();
  for (node_id_t i = 0; i < num_vertices; ++i) {
    merge_instr[i] = {i, i};
    spanning_forest[i].clear();
  }
  size_t round_num = 0;
  bool modified = true;

  while (true) {
    modified = perform_k_boruvka_round(round_num, merge_instr, global_merges, graph_id);

    if (!modified) break;
    create_merge_instructions(merge_instr);
    ++round_num;
  }
  last_query_rounds = round_num;

  //dsu_valid = true;
  //shared_dsu_valid = true;
  update_locked = false;
}

ConnectedComponents MCSketchAlg::connected_components() {
  cc_alg_start = std::chrono::steady_clock::now();

  // if the DSU holds the answer, use that
  if (shared_dsu_valid) {
#ifdef VERIFY_SAMPLES_F
    for (node_id_t src = 0; src < num_vertices; ++src) {
      for (const auto &dst : spanning_forest[src]) {
        verifier->verify_edge({src, dst});
      }
    }
#endif
  }
  // The DSU does not hold the answer, make it so
  else {
    bool except = false;
    std::exception_ptr err;
    try {
      // auto start = std::chrono::steady_clock::now();
      boruvka_emulation();
      // std::cout << " boruvka's algorithm = "
      //         << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
      //         << std::endl;
    } catch (...) {
      except = true;
      err = std::current_exception();
    }

    // get ready for ingesting more from the stream by resetting the sketches sample state
    for (node_id_t i = 0; i < num_vertices; i++) {
      sketches[i]->reset_sample_state();
    }

    if (except) std::rethrow_exception(err);
  }

  ConnectedComponents cc(num_vertices, dsu);
#ifdef VERIFY_SAMPLES_F
  verifier->verify_connected_components(cc);
#endif
  cc_alg_end = std::chrono::steady_clock::now();
  return cc;
}

SpanningForest MCSketchAlg::calc_spanning_forest() {
  // TODO: Could probably optimize this a bit by writing new code
  connected_components();

  SpanningForest ret(num_vertices, spanning_forest);
#ifdef VERIFY_SAMPLES_F
  verifier->verify_spanning_forests(std::vector<SpanningForest>{ret});
#endif
  return ret;
}

SpanningForest MCSketchAlg::get_k_spanning_forest(int graph_id) {
  bool except = false;
  std::exception_ptr err;
  try {
    k_boruvka_emulation(graph_id);
  } catch (...) {
    except = true;
    err = std::current_exception();
  }

  ConnectedComponents cc(num_vertices, dsu);

  // Note: Get num_cc for spanning forest
  std::cout << "    round = " << last_query_rounds << " cc size = " << cc.size() << "\n";

  // Note: Turning these off for now for performance, but turn it back on if run into OutOfSamplesException 
  // get ready for ingesting more from the stream by resetting the sketches sample state
  for (node_id_t i = 0; i < num_vertices * max_sketch_graphs; i++) {
    sketches[i]->reset_sample_state();
  }

  if (except) std::rethrow_exception(err);

  return SpanningForest(num_vertices, spanning_forest);
}

bool MCSketchAlg::point_query(node_id_t a, node_id_t b) {
  cc_alg_start = std::chrono::steady_clock::now();

  // if the DSU holds the answer, use that
  if (dsu_valid) {
#ifdef VERIFY_SAMPLES_F
    for (node_id_t src = 0; src < num_vertices; ++src) {
      for (const auto &dst : spanning_forest[src]) {
        verifier->verify_edge({src, dst});
      }
    }
#endif
  } 
  // The DSU does not hold the answer, make it so
  else {
    bool except = false;
    std::exception_ptr err;
    try {
      boruvka_emulation();
    } catch (...) {
      except = true;
      err = std::current_exception();
    }

    // get ready for ingesting more from the stream
    // reset dsu and resume graph workers
    for (node_id_t i = 0; i < num_vertices; i++) {
      sketches[i]->reset_sample_state();
    }

    // check if boruvka errored
    if (except) std::rethrow_exception(err);
  }

#ifdef VERIFY_SAMPLES_F
  ConnectedComponents cc(num_vertices, dsu);
  verifier->verify_connected_components(cc);
#endif

  bool retval = (dsu.find_root(a) == dsu.find_root(b));
  cc_alg_end = std::chrono::steady_clock::now();
  return retval;
}

MinCut MCSketchAlg::calc_minimum_cut(const std::vector<Edge> &edges) {
  typedef VieCut::mutable_graph Graph;
  typedef std::shared_ptr<VieCut::mutable_graph> GraphPtr;

  // Create a VieCut graph
  GraphPtr G = std::make_shared<Graph>();
  G->start_construction(num_vertices, edges.size());

  // Add edges to VieCut graph
  for (auto edge : edges) {
    G->new_edge(edge.src, edge.dst);
  }

  // finish construction and compute degrees
  // TODO: Don't know if degrees are necessary. Its in the VieCut code tho
  G->finish_construction();
  G->computeDegrees();

  // Perform the mincut computation
  VieCut::EdgeWeight cut;
  VieCut::minimum_cut* mc = new VieCut::viecut<GraphPtr>();
  cut = mc->perform_minimum_cut(G);

  // Return answer
  std::set<node_id_t> left;
  std::set<node_id_t> right;

  for (node_id_t i = 0; i < num_vertices; i++) {
    if (G->getNodeInCut(i))
      left.insert(i);
    else
      right.insert(i);
  }

  return {left, right, cut};
}

void MCSketchAlg::write_binary(const std::string &filename) {
  auto binary_out = std::fstream(filename, std::ios::out | std::ios::binary);
  binary_out.write((char *)&seed, sizeof(seed));
  binary_out.write((char *)&num_vertices, sizeof(num_vertices));
  binary_out.write((char *)&config._sketches_factor, sizeof(config._sketches_factor));
  for (node_id_t i = 0; i < num_vertices; ++i) {
    sketches[i]->serialize(binary_out);
  }
  binary_out.close();
}
