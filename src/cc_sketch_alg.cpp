#include "cc_sketch_alg.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <random>

CCSketchAlg::CCSketchAlg(node_id_t num_vertices, size_t seed, CCAlgConfiguration config)
    : num_vertices(num_vertices), seed(seed), dsu(num_vertices), config(config) {
  representatives = new std::set<node_id_t>();
  sketches = new Sketch *[num_vertices];

  vec_t sketch_vec_len = Sketch::calc_vector_length(num_vertices);
  size_t sketch_num_samples = Sketch::calc_cc_samples(num_vertices);
  for (node_id_t i = 0; i < num_vertices; ++i) {
    representatives->insert(i);
    sketches[i] = new Sketch(sketch_vec_len, seed, sketch_num_samples);
  }

  spanning_forest = new std::unordered_set<node_id_t>[num_vertices];
  spanning_forest_mtx = new std::mutex[num_vertices];
  dsu_valid = true;
  shared_dsu_valid = true;
}

CCSketchAlg *CCSketchAlg::construct_from_serialized_data(const std::string &input_file,
                                                        CCAlgConfiguration config) {
  double sketches_factor;
  auto binary_in = std::ifstream(input_file, std::ios::binary);
  size_t seed;
  node_id_t num_vertices;
  binary_in.read((char *)&seed, sizeof(seed));
  binary_in.read((char *)&num_vertices, sizeof(num_vertices));
  binary_in.read((char *)&sketches_factor, sizeof(sketches_factor));

  config.sketches_factor(sketches_factor);

  return new CCSketchAlg(num_vertices, seed, binary_in, config);
}

CCSketchAlg::CCSketchAlg(node_id_t num_vertices, size_t seed, std::ifstream &binary_stream,
                         CCAlgConfiguration config)
    : num_vertices(num_vertices), seed(seed), dsu(num_vertices), config(config) {
  representatives = new std::set<node_id_t>();
  sketches = new Sketch *[num_vertices];

  vec_t sketch_vec_len = Sketch::calc_vector_length(num_vertices);
  size_t sketch_num_samples = Sketch::calc_cc_samples(num_vertices);
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

CCSketchAlg::~CCSketchAlg() {
  for (size_t i = 0; i < num_vertices; ++i) delete sketches[i];
  delete[] sketches;
  if (delta_sketches != nullptr) {
    for (size_t i = 0; i < num_delta_sketches; i++) delete delta_sketches[i];
    delete[] delta_sketches;
  }

  delete representatives;
  delete[] spanning_forest;
  delete[] spanning_forest_mtx;
}

void CCSketchAlg::pre_insert(GraphUpdate upd, int /* thr_id */) {
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
    if (spanning_forest[src].find(dst) != spanning_forest[src].end()) {
      dsu_valid = false;
      shared_dsu_valid = false;
    } else {
      spanning_forest[src].insert(dst);
      dsu.merge(src, dst);
    }
  }
#endif  // NO_EAGER_DSU
}

void CCSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (update_locked) throw UpdateLockedException();
  Sketch &delta_sketch = *delta_sketches[thr_id];
  delta_sketch.zero_contents();

  for (const auto &dst : dst_vertices) {
    delta_sketch.update(static_cast<vec_t>(concat_pairing_fn(src_vertex, dst)));
  }

  std::unique_lock<std::mutex>(sketches[src_vertex]->mutex);
  sketches[src_vertex]->merge(delta_sketch);
}

void CCSketchAlg::apply_raw_buckets_update(node_id_t src_vertex, Bucket *raw_buckets) {
  std::unique_lock<std::mutex>(sketches[src_vertex]->mutex);
  sketches[src_vertex]->merge_raw_bucket_buffer(raw_buckets);
}

// Note: for performance reasons route updates through the driver instead of calling this function
//       whenever possible.
void CCSketchAlg::update(GraphUpdate upd) {
  pre_insert(upd, 0);
  Edge edge = upd.edge;

  sketches[edge.src]->update(static_cast<vec_t>(concat_pairing_fn(edge.src, edge.dst)));
  sketches[edge.dst]->update(static_cast<vec_t>(concat_pairing_fn(edge.src, edge.dst)));
}

bool CCSketchAlg::sample_supernodes(std::vector<node_id_t> &merge_instr) {
  bool except = false;
  bool modified = false;
  std::exception_ptr err;
#pragma omp parallel for default(shared)
    for (node_id_t root = 0; root < num_vertices; root++) {
      if (merge_instr[root] != root) {
        // don't query non-roots
        continue;
      }

      SketchSample sample_result;

      // wrap in a try/catch because exiting through exception is undefined behavior in OMP
      try {
        sample_result = sketches[root]->sample();
      } catch (...) {
        except = true;
        err = std::current_exception();
      }

      Edge e = inv_concat_pairing_fn(sample_result.idx);
      SampleResult result_type = sample_result.result;

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
            std::unique_lock<std::mutex> lk(spanning_forest_mtx[src]);
            spanning_forest[src].insert(dst);
          }
        }
      }
    }
  
  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);
  return modified;
}

void CCSketchAlg::merge_supernodes(const size_t next_round,
                                   const std::vector<node_id_t> &merge_instr) {
#pragma omp parallel default(shared)
  {
    // some thread local variables
    Sketch local_sketch(Sketch::calc_vector_length(num_vertices), seed,
                        Sketch::calc_cc_samples(num_vertices));
    node_id_t cur_root = 0;
    bool first_root = true;
#pragma omp for
    for (node_id_t i = 0; i < num_vertices; i++) {
      if (merge_instr[i] == i) continue;

      node_id_t root = merge_instr[i];
      if (root != cur_root || first_root) {
        if (!first_root) {
          std::unique_lock<std::mutex> lk(sketches[cur_root]->mutex);
          sketches[cur_root]->range_merge(local_sketch, next_round, 1);
        }
        cur_root = root;
        local_sketch.zero_contents();
        first_root = false;
      }

      local_sketch.range_merge(*sketches[i], next_round, 1);
    }

    if (!first_root) {
      std::unique_lock<std::mutex> lk(sketches[cur_root]->mutex);
      sketches[cur_root]->range_merge(local_sketch, next_round, 1);
    }
  }
}

void CCSketchAlg::undo_merge_supernodes(const size_t cur_round,
                                        const std::vector<node_id_t> &merge_instr) {
  if (cur_round > 0) merge_supernodes(cur_round, merge_instr);
}

std::vector<std::set<node_id_t>> CCSketchAlg::boruvka_emulation() {
  update_locked = true;

  cc_alg_start = std::chrono::steady_clock::now();
  std::vector<node_id_t> merge_instr(num_vertices);

  dsu.reset();
  for (node_id_t i = 0; i < num_vertices; ++i) {
    merge_instr[i] = i;
    spanning_forest[i].clear();
  }
  size_t round_num = 0;
  bool modified = true;
  while (true) {
    // auto start = std::chrono::steady_clock::now();
    try {
      modified = sample_supernodes(merge_instr);
    } catch (...) {
      undo_merge_supernodes(round_num, merge_instr);
      std::rethrow_exception(std::current_exception());
    }
    // std::cout << "sample: "
    //           << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
    //           << std::endl;

    // start = std::chrono::steady_clock::now();
    undo_merge_supernodes(round_num, merge_instr);
    // std::cout << "undo merge: "
    //           << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
    //           << std::endl;

    if (!modified) break;

    // calculate updated merge instructions
#pragma omp parallel for
    for (node_id_t i = 0; i < num_vertices; i++)
      merge_instr[i] = dsu.find_root(i);

    // prepare for the next round by merging
    // start = std::chrono::steady_clock::now();
    merge_supernodes(round_num + 1, merge_instr);
    // std::cout << "merge: "
    //           << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
    //           << std::endl;
    ++round_num;
  }
  last_query_rounds = round_num;

  dsu_valid = true;
  shared_dsu_valid = true;

  auto retval = cc_from_dsu();
  cc_alg_end = std::chrono::steady_clock::now();
  update_locked = false;
  return retval;
}

std::vector<std::set<node_id_t>> CCSketchAlg::connected_components() {
  // if the DSU holds the answer, use that
  if (shared_dsu_valid) {
    cc_alg_start = std::chrono::steady_clock::now();
#ifdef VERIFY_SAMPLES_F
    for (node_id_t src = 0; src < num_vertices; ++src) {
      for (const auto &dst : spanning_forest[src]) {
        verifier->verify_edge({src, dst});
      }
    }
#endif
    auto retval = cc_from_dsu();
#ifdef VERIFY_SAMPLES_F
    verifier->verify_soln(retval);
#endif
    cc_alg_end = std::chrono::steady_clock::now();
    return retval;
  }

  std::vector<std::set<node_id_t>> ret;

  bool except = false;
  std::exception_ptr err;
  try {
    ret = boruvka_emulation();
#ifdef VERIFY_SAMPLES_F
    verifier->verify_soln(ret);
#endif
  } catch (...) {
    except = true;
    err = std::current_exception();
  }

  // get ready for ingesting more from the stream
  // reset dsu and resume graph workers
  for (node_id_t i = 0; i < num_vertices; i++) {
    sketches[i]->reset_sample_state();
  }

  // check if boruvka error'd
  if (except) std::rethrow_exception(err);

  return ret;
}

std::vector<std::pair<node_id_t, std::vector<node_id_t>>> CCSketchAlg::calc_spanning_forest() {
  // TODO: Could probably optimize this a bit by writing new code
  connected_components();
  
  std::vector<std::pair<node_id_t, std::vector<node_id_t>>> forest;

  for (node_id_t src = 0; src < num_vertices; src++) {
    if (spanning_forest[src].size() > 0) {
      std::vector<node_id_t> edge_list;
      edge_list.reserve(spanning_forest[src].size());
      for (node_id_t dst : spanning_forest[src]) {
        edge_list.push_back(dst);
      }
      forest.push_back({src, edge_list});
    }
  }
  return forest;
}

bool CCSketchAlg::point_query(node_id_t a, node_id_t b) {
  // DSU check before calling force_flush()
  if (dsu_valid) {
    cc_alg_start = std::chrono::steady_clock::now();
#ifdef VERIFY_SAMPLES_F
    for (node_id_t src = 0; src < num_vertices; ++src) {
      for (const auto &dst : spanning_forest[src]) {
        verifier->verify_edge({src, dst});
      }
    }
#endif
    bool retval = (dsu.find_root(a) == dsu.find_root(b));
    cc_alg_end = std::chrono::steady_clock::now();
    return retval;
  }

  bool except = false;
  std::exception_ptr err;
  bool ret;
  try {
    std::vector<std::set<node_id_t>> ccs = boruvka_emulation();
#ifdef VERIFY_SAMPLES_F
    verifier->verify_soln(ccs);
#endif
    ret = (dsu.find_root(a) == dsu.find_root(b));
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

  return ret;
}

std::vector<std::set<node_id_t>> CCSketchAlg::cc_from_dsu() {
  // calculate connected components using DSU structure
  std::map<node_id_t, std::set<node_id_t>> temp;
  for (node_id_t i = 0; i < num_vertices; ++i) temp[dsu.find_root(i)].insert(i);
  std::vector<std::set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto &it : temp) retval.push_back(it.second);
  return retval;
}

void CCSketchAlg::write_binary(const std::string &filename) {
  auto binary_out = std::fstream(filename, std::ios::out | std::ios::binary);
  binary_out.write((char *)&seed, sizeof(seed));
  binary_out.write((char *)&num_vertices, sizeof(num_vertices));
  binary_out.write((char *)&config._sketches_factor, sizeof(config._sketches_factor));
  for (node_id_t i = 0; i < num_vertices; ++i) {
    sketches[i]->serialize(binary_out);
  }
  binary_out.close();
}
