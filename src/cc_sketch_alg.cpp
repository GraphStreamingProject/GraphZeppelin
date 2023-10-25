#include "cc_sketch_alg.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <random>

CCSketchAlg::CCSketchAlg(node_id_t num_nodes, CCAlgConfiguration config)
    : num_nodes(num_nodes), dsu(num_nodes), config(config) {
  representatives = new std::set<node_id_t>();
  sketches = new Sketch *[num_nodes];
  seed = std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
             .count();
  std::mt19937_64 r(seed);
  seed = r();

  vec_t sketch_vec_len = Sketch::calc_vector_length(num_nodes);
  size_t sketch_num_samples = Sketch::calc_cc_samples(num_nodes);
  for (node_id_t i = 0; i < num_nodes; ++i) {
    representatives->insert(i);
    sketches[i] = new Sketch(sketch_vec_len, seed, sketch_num_samples);
  }

  spanning_forest = new std::unordered_set<node_id_t>[num_nodes];
  spanning_forest_mtx = new std::mutex[num_nodes];
  dsu_valid = true;
  shared_dsu_valid = true;
  std::cout << config << std::endl;  // print the graph configuration
}

CCSketchAlg *CCSketchAlg::construct_from_serialized_data(const std::string &input_file,
                                                        CCAlgConfiguration config) {
  double sketches_factor;
  auto binary_in = std::ifstream(input_file, std::ios::binary);
  size_t seed;
  node_id_t num_nodes;
  binary_in.read((char *)&seed, sizeof(seed));
  binary_in.read((char *)&num_nodes, sizeof(num_nodes));
  binary_in.read((char *)&sketches_factor, sizeof(sketches_factor));

  config.sketches_factor(sketches_factor);

  return new CCSketchAlg(num_nodes, seed, binary_in, config);
}

CCSketchAlg::CCSketchAlg(node_id_t num_nodes, size_t seed, std::ifstream &binary_stream,
                         CCAlgConfiguration config)
    : num_nodes(num_nodes), seed(seed), dsu(num_nodes), config(config) {
  representatives = new std::set<node_id_t>();
  sketches = new Sketch *[num_nodes];

  vec_t sketch_vec_len = Sketch::calc_vector_length(num_nodes);
  size_t sketch_num_samples = Sketch::calc_cc_samples(num_nodes);
  for (node_id_t i = 0; i < num_nodes; ++i) {
    representatives->insert(i);
    sketches[i] = new Sketch(sketch_vec_len, seed, binary_stream, sketch_num_samples);
  }
  binary_stream.close();

  spanning_forest = new std::unordered_set<node_id_t>[num_nodes];
  spanning_forest_mtx = new std::mutex[num_nodes];
  dsu_valid = false;
  shared_dsu_valid = false;
  std::cout << config << std::endl;  // print the graph configuration
}

CCSketchAlg::~CCSketchAlg() {
  for (size_t i = 0; i < num_nodes; ++i) delete sketches[i];
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

// Note: for performance reasons route updates through the driver instead of calling this function
//       whenever possible.
void CCSketchAlg::update(GraphUpdate upd) {
  pre_insert(upd, 0);
  Edge edge = upd.edge;

  sketches[edge.src]->update(static_cast<vec_t>(concat_pairing_fn(edge.src, edge.dst)));
  sketches[edge.dst]->update(static_cast<vec_t>(concat_pairing_fn(edge.src, edge.dst)));
}

void CCSketchAlg::sample_supernodes(std::pair<Edge, SampleSketchRet> *query,
                                    const std::unordered_set<node_id_t> &roots) {
  bool except = false;
  std::exception_ptr err;
#pragma omp parallel for default(none) shared(query, roots, except, err)
  for (node_id_t root = 0; root < num_nodes; root++) {
    if (roots.count(root) == 0) {
      // don't query non-roots
      continue;
    }

    std::pair<vec_t, SampleSketchRet> query_ret;

    // wrap in a try/catch because exiting through exception is undefined behavior in OMP
    try {
      query_ret = sketches[root]->sample();
    } catch (...) {
      except = true;
      err = std::current_exception();
    }

    Edge e = inv_concat_pairing_fn(query_ret.first);
    query[root] = {e, query_ret.second};
  }
  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);
}

void CCSketchAlg::supernodes_to_merge(const std::pair<Edge, SampleSketchRet> *query,
                                      std::unordered_set<node_id_t> &roots,
                                      std::vector<std::vector<node_id_t>> &to_merge) {
  for (auto it = roots.begin(); it != roots.end();) {
    // unpack query result
    node_id_t root = *it;
    Edge edge = query[root].first;
    SampleSketchRet ret_code = query[root].second;

    // try this query again next round as it failed this round
    if (ret_code == FAIL) {
      ++it;
      continue;
    }
    // This root will not grow anymore, so remove it from the set
    else if (ret_code == ZERO) {
      it = roots.erase(it);
      to_merge[root].clear();
      continue;
    }

    // query dsu
    DSUMergeRet<node_id_t> m_ret = dsu.merge(edge.src, edge.dst);
    if (m_ret.merged) {
#ifdef VERIFY_SAMPLES_F
      verifier->verify_edge(edge);
#endif

      if (m_ret.child == root) {
        it = roots.erase(it);
      } else {
        roots.erase(m_ret.child);
      }

      // add b and any of the nodes to merge with it to a's vector
      to_merge[m_ret.root].push_back(m_ret.child);
      to_merge[m_ret.root].insert(to_merge[m_ret.root].end(), to_merge[m_ret.child].begin(), to_merge[m_ret.child].end());
      to_merge[m_ret.child].clear();

      // Update spanning forest
      auto src = std::min(edge.src, edge.dst);
      auto dst = std::max(edge.src, edge.dst);
      spanning_forest[src].insert(dst);
    } else {
      ++it;
    }
  }
}

// TODO: There's probably a better way to do this. Let's try this for now though
void CCSketchAlg::merge_supernodes(const size_t cur_round,
                                   const std::vector<std::vector<node_id_t>> &to_merge) {
#pragma omp parallel for default(shared)
  for (node_id_t root = 0; root < num_nodes; root++) {
    // single thread merge (not worth cost to parallelize)
    for (node_id_t child : to_merge[root]) {
      sketches[root]->range_merge(*sketches[child], cur_round, 1);
      assert(dsu.find_root(child) == root);
    }
  }
}

void CCSketchAlg::undo_merge_supernodes(const size_t prev_round,
                                        const std::vector<std::vector<node_id_t>> &to_merge) {
  if (prev_round > 0) merge_supernodes(prev_round, to_merge);
}

std::vector<std::set<node_id_t>> CCSketchAlg::boruvka_emulation() {
  update_locked = true;

  cc_alg_start = std::chrono::steady_clock::now();
  std::pair<Edge, SampleSketchRet> *query = new std::pair<Edge, SampleSketchRet>[num_nodes];
  std::unordered_set<node_id_t> roots;
  std::vector<std::vector<node_id_t>> to_merge(num_nodes);

  dsu.reset();
  for (node_id_t i = 0; i < num_nodes; ++i) {
    roots.insert(i);
    spanning_forest[i].clear();
  }
  size_t round_num = 0;
  do {
    auto start = std::chrono::steady_clock::now();
    try {
      sample_supernodes(query, roots);
    } catch (...) {
      delete[] query;
      undo_merge_supernodes(round_num, to_merge);
      std::rethrow_exception(std::current_exception());
    }
    std::cout << "sample: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << std::endl;

    start = std::chrono::steady_clock::now();
    undo_merge_supernodes(round_num, to_merge);
    std::cout << "undo merge: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << std::endl;

    start = std::chrono::steady_clock::now();
    supernodes_to_merge(query, roots, to_merge);
    std::cout << "parse samples: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << std::endl;

    // prepare for the next round by merging
    start = std::chrono::steady_clock::now();
    merge_supernodes(round_num + 1, to_merge);
    std::cout << "merge: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << std::endl;
    std::cout << round_num << " remaining nodes = " << roots.size() << std::endl;
    ++round_num;
  } while (roots.size() > 1);

  undo_merge_supernodes(round_num, to_merge);
  delete[] query;
  dsu_valid = true;
  shared_dsu_valid = true;

  std::cout << "Query complete in " << round_num << " rounds." << std::endl;
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
    for (node_id_t src = 0; src < num_nodes; ++src) {
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
  for (node_id_t i = 0; i < num_nodes; i++) {
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

  for (node_id_t src = 0; src < num_nodes; src++) {
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
    for (node_id_t src = 0; src < num_nodes; ++src) {
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
  for (node_id_t i = 0; i < num_nodes; i++) {
    sketches[i]->reset_sample_state();
  }

  // check if boruvka errored
  if (except) std::rethrow_exception(err);

  return ret;
}

std::vector<std::set<node_id_t>> CCSketchAlg::cc_from_dsu() {
  // calculate connected components using DSU structure
  std::map<node_id_t, std::set<node_id_t>> temp;
  for (node_id_t i = 0; i < num_nodes; ++i) temp[dsu.find_root(i)].insert(i);
  std::vector<std::set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto &it : temp) retval.push_back(it.second);
  return retval;
}

void CCSketchAlg::write_binary(const std::string &filename) {
  auto binary_out = std::fstream(filename, std::ios::out | std::ios::binary);
  binary_out.write((char *)&seed, sizeof(seed));
  binary_out.write((char *)&num_nodes, sizeof(num_nodes));
  binary_out.write((char *)&config._sketches_factor, sizeof(config._sketches_factor));
  for (node_id_t i = 0; i < num_nodes; ++i) {
    sketches[i]->serialize(binary_out);
  }
  binary_out.close();
}
