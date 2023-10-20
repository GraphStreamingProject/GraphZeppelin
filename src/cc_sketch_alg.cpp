#include "cc_sketch_alg.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <random>

CCSketchAlg::CCSketchAlg(node_id_t num_nodes, CCAlgConfiguration config)
    : num_nodes(num_nodes), config(config) {
  representatives = new std::set<node_id_t>();
  sketches = new Sketch *[num_nodes];
  parent = new std::remove_reference<decltype(*parent)>::type[num_nodes];
  size = new node_id_t[num_nodes];
  seed = std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
             .count();
  std::mt19937_64 r(seed);
  seed = r();

  std::fill(size, size + num_nodes, 1);
  for (node_id_t i = 0; i < num_nodes; ++i) {
    representatives->insert(i);
    sketches[i] = new Sketch(num_nodes, seed);
    parent[i] = i;
  }
  backup_file = config._disk_dir + "supernode_backup.data";

  spanning_forest = new std::unordered_set<node_id_t>[num_nodes];
  spanning_forest_mtx = new std::mutex[num_nodes];
  dsu_valid = true;
  shared_dsu_valid = true;
  std::cout << config << std::endl;  // print the graph configuration
}

CCSketchAlg::CCSketchAlg(std::string input_file, CCAlgConfiguration config) : config(config) {
  double sketches_factor;
  auto binary_in = std::fstream(input_file, std::ios::in | std::ios::binary);
  binary_in.read((char *)&seed, sizeof(seed));
  binary_in.read((char *)&num_nodes, sizeof(num_nodes));
  binary_in.read((char *)&sketches_factor, sizeof(sketches_factor));

  representatives = new std::set<node_id_t>();
  sketches = new Sketch *[num_nodes];
  parent = new std::remove_reference<decltype(*parent)>::type[num_nodes];
  size = new node_id_t[num_nodes];
  std::fill(size, size + num_nodes, 1);
  for (node_id_t i = 0; i < num_nodes; ++i) {
    representatives->insert(i);
    sketches[i] = new Sketch(num_nodes, seed, binary_in, FULL);
    parent[i] = i;
  }
  binary_in.close();
  backup_file = config._disk_dir + "supernode_backup.data";

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

  delete[] parent;
  delete[] size;
  delete representatives;
  delete[] spanning_forest;
  delete[] spanning_forest_mtx;
}

void CCSketchAlg::pre_insert(GraphUpdate upd, int /* thr_id */) {
#ifndef NO_EAGER_DSU
  if (dsu_valid) {
    Edge edge = upd.edge;
    auto src = std::min(edge.src, edge.dst);
    auto dst = std::max(edge.src, edge.dst);
    std::lock_guard<std::mutex> sflock(spanning_forest_mtx[src]);
    if (spanning_forest[src].find(dst) != spanning_forest[src].end()) {
      dsu_valid = false;
      shared_dsu_valid = false;
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
  (void)upd;
  unlikely_if(dsu_valid) {
    dsu_valid = false;
    shared_dsu_valid = false;
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
                                    std::vector<node_id_t> &reps) {
  bool except = false;
  std::exception_ptr err;
#pragma omp parallel for default(none) shared(query, reps, except, err)
  for (node_id_t i = 0; i < reps.size(); ++i) {  // NOLINT(modernize-loop-convert)
    // wrap in a try/catch because exiting through exception is undefined behavior in OMP
    try {
      std::pair<vec_t, SampleSketchRet> query_ret = sketches[reps[i]]->sample();
      Edge e = inv_concat_pairing_fn(query_ret.first);
      query[reps[i]] = {e, query_ret.second};

    } catch (...) {
      except = true;
      err = std::current_exception();
    }
  }
  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);
}

std::vector<std::vector<node_id_t>> CCSketchAlg::supernodes_to_merge(
    std::pair<Edge, SampleSketchRet> *query, std::vector<node_id_t> &reps) {
  std::vector<std::vector<node_id_t>> to_merge(num_nodes);
  std::vector<node_id_t> new_reps;
  for (auto i : reps) {
    // unpack query result
    Edge edge = query[i].first;
    SampleSketchRet ret_code = query[i].second;

    // try this query again next round as it failed this round
    if (ret_code == FAIL) {
      modified = true;
      new_reps.push_back(i);
      continue;
    }
    if (ret_code == ZERO) continue;

    // query dsu
    node_id_t a = get_parent(edge.src);
    node_id_t b = get_parent(edge.dst);
    if (a == b) continue;

#ifdef VERIFY_SAMPLES_F
    verifier->verify_edge(edge);
#endif

    // make a the parent of b
    if (size[a] < size[b]) std::swap(a, b);
    parent[b] = a;
    size[a] += size[b];

    // add b and any of the nodes to merge with it to a's vector
    to_merge[a].push_back(b);
    to_merge[a].insert(to_merge[a].end(), to_merge[b].begin(), to_merge[b].end());
    to_merge[b].clear();
    modified = true;

    // Update spanning forest
    auto src = std::min(edge.src, edge.dst);
    auto dst = std::max(edge.src, edge.dst);
    spanning_forest[src].insert(dst);
  }

  // remove nodes added to new_reps due to sketch failures that
  // did end up being able to merge after all
  std::vector<node_id_t> temp_vec;
  for (node_id_t a : new_reps)
    if (to_merge[a].empty()) temp_vec.push_back(a);
  std::swap(new_reps, temp_vec);

  // add to new_reps all the nodes we will merge into
  for (node_id_t a = 0; a < num_nodes; a++)
    if (!to_merge[a].empty()) new_reps.push_back(a);

  reps = new_reps;
  return to_merge;
}

void CCSketchAlg::merge_supernodes(Sketch **copy_sketches, std::vector<node_id_t> &new_reps,
                                   std::vector<std::vector<node_id_t>> &to_merge, bool make_copy) {
  bool except = false;
  std::exception_ptr err;
// loop over the to_merge vector and perform supernode merging
#pragma omp parallel for default(shared)
  for (node_id_t i = 0; i < new_reps.size(); i++) {  // NOLINT(modernize-loop-convert)
    // OMP requires a traditional for-loop to work
    node_id_t a = new_reps[i];
    try {
      if (make_copy && config._backup_in_mem) {  // make a copy of a
        copy_sketches[a] = new Sketch(*sketches[a]);
      }

      // perform merging of nodes b into node a
      for (node_id_t b : to_merge[a]) {
        sketches[a]->merge(*sketches[b]);
      }
    } catch (...) {
      except = true;
      err = std::current_exception();
    }
  }

  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);
}

std::vector<std::set<node_id_t>> CCSketchAlg::boruvka_emulation() {
  update_locked = true;

  cc_alg_start = std::chrono::steady_clock::now();
  bool first_round = true;
  Sketch **copy_sketches;
  if (config._backup_in_mem) {
    copy_sketches = new Sketch *[num_nodes];
    for (node_id_t i = 0; i < num_nodes; ++i) copy_sketches[i] = nullptr;
  }
  std::pair<Edge, SampleSketchRet> *query = new std::pair<Edge, SampleSketchRet>[num_nodes];
  std::vector<node_id_t> reps(num_nodes);
  std::vector<node_id_t> backed_up;
  std::fill(size, size + num_nodes, 1);
  for (node_id_t i = 0; i < num_nodes; ++i) reps[i] = i;

  // function to restore sketches after CC or failure
  auto cleanup_copy = [this, &backed_up, &copy_sketches]() {
    if (config._backup_in_mem) {
      // restore original sketches and free memory
      for (node_id_t i : backed_up) {
        if (sketches[i] != nullptr) free(sketches[i]);
        sketches[i] = copy_sketches[i];
      }
      delete[] copy_sketches;
    } else {
      restore_from_disk(backed_up);
    }
  };

  for (node_id_t i = 0; i < num_nodes; ++i) {
    parent[i] = i;
    spanning_forest[i].clear();
  }
  size_t round_num = 1;
  try {
    do {
      modified = false;
      sample_supernodes(query, reps);
      std::vector<std::vector<node_id_t>> to_merge = supernodes_to_merge(query, reps);
      // make a copy if first round
      if (first_round) {
        backed_up = reps;
        if (!config._backup_in_mem) backup_to_disk(backed_up);
      }

      merge_supernodes(copy_sketches, reps, to_merge, first_round);

#ifdef VERIFY_SAMPLES_F
      if (!first_round && fail_round_2) {
        std::cerr << "inducing an error for testing!" << std::endl;
        throw OutOfQueriesException();
      }
#endif
      first_round = false;
      ++round_num;
    } while (modified);
  } catch (...) {
    cleanup_copy();
    delete[] query;
    std::rethrow_exception(std::current_exception());
  }
  cleanup_copy();
  delete[] query;
  dsu_valid = true;
  shared_dsu_valid = true;

  std::cout << "Query complete in " << round_num << " rounds." << std::endl;
  auto retval = cc_from_dsu();
  cc_alg_end = std::chrono::steady_clock::now();
  update_locked = false;
  return retval;
}

void CCSketchAlg::backup_to_disk(const std::vector<node_id_t> &ids_to_backup) {
  // Make a copy on disk
  std::fstream binary_out(backup_file, std::ios::out | std::ios::binary);
  if (!binary_out.is_open()) {
    std::cerr << "Failed to open file for writing backup!" << backup_file << std::endl;
    exit(EXIT_FAILURE);
  }
  for (node_id_t idx : ids_to_backup) {
    sketches[idx]->serialize(binary_out);
  }
  binary_out.close();
}

// given a list of ids restore those sketches from disk
// IMPORTANT: ids_to_restore must be the same as ids_to_backup
void CCSketchAlg::restore_from_disk(const std::vector<node_id_t> &ids_to_restore) {
  // restore from disk
  std::fstream binary_in(backup_file, std::ios::in | std::ios::binary);
  if (!binary_in.is_open()) {
    std::cerr << "Failed to open file for reading backup!" << backup_file << std::endl;
    exit(EXIT_FAILURE);
  }
  for (node_id_t idx : ids_to_restore) {
    delete this->sketches[idx];
    this->sketches[idx] = new Sketch(num_nodes, seed, binary_in);
  }
}

std::vector<std::set<node_id_t>> CCSketchAlg::connected_components() {
  // DSU check before calling force_flush()
  // TODO! Move this into the needs_query function!
  if (shared_dsu_valid
#ifdef VERIFY_SAMPLES_F
      && !fail_round_2
#endif  // VERIFY_SAMPLES_F
  ) {
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

std::vector<std::set<node_id_t>> CCSketchAlg::cc_from_dsu() {
  // calculate connected components using DSU structure
  std::map<node_id_t, std::set<node_id_t>> temp;
  for (node_id_t i = 0; i < num_nodes; ++i) temp[get_parent(i)].insert(i);
  std::vector<std::set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto &it : temp) retval.push_back(it.second);
  return retval;
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
    bool retval = (get_parent(a) == get_parent(b));
    cc_alg_end = std::chrono::steady_clock::now();
    return retval;
  }

  // if backing up in memory then perform copying in boruvka
  bool except = false;
  std::exception_ptr err;
  bool ret;
  try {
    boruvka_emulation();
    ret = (get_parent(a) == get_parent(b));
  } catch (...) {
    except = true;
    err = std::current_exception();
  }

  // get ready for ingesting more from the stream
  // reset dsu and resume graph workers
  for (node_id_t i = 0; i < num_nodes; i++) {
    sketches[i]->reset_sample_state();
    parent[i] = i;
    size[i] = 1;
  }

  // check if boruvka errored
  if (except) std::rethrow_exception(err);

  return ret;
}

node_id_t CCSketchAlg::get_parent(node_id_t node) {
  if (parent[node] == node) return node;
  return parent[node] = get_parent(parent[node]);
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
