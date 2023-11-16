#include "mincut_graph.h"
#include <graph_worker.h>
#include <types.h>
#include <random>
#include <algorithm>
#include <map>

typedef std::mt19937 MersenneTwister;
static MersenneTwister m_mt;

node_id_t MinCutGraph::k_get_parent(node_id_t node, int k_id) {
  if (parent[(node * k) + k_id] == node) return node;
  return parent[(node * k) + k_id] = k_get_parent(parent[(node * k) + k_id], k_id);
}

std::vector<std::vector<node_id_t>> MinCutGraph::to_merge_and_forest_edges(
    std::vector<Edge> &forest, std::pair<Edge, SampleSketchRet> *query,
    std::vector<node_id_t> &reps, int k_id) {
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
    if (ret_code == ZERO) {
#ifdef VERIFY_SAMPLES_F
      verifier->verify_cc(i);
#endif
      continue;
    }

    // query dsu
    node_id_t a = k_get_parent(edge.src, k_id);
    node_id_t b = k_get_parent(edge.dst, k_id);
    if (a == b) continue;

#ifdef VERIFY_SAMPLES_F
    verifier->verify_edge(edge);
#endif

    // make a the parent of b
    if (size[a] < size[b]) std::swap(a,b);
    parent[(b * k) + k_id] = a;
    size[a] += size[b];

    // add b and any of the nodes to merge with it to a's vector
    to_merge[a].push_back(b);
    to_merge[a].insert(to_merge[a].end(), to_merge[b].begin(), to_merge[b].end());
    to_merge[b].clear();
    modified = true;

    // Update spanning forest
    auto src = std::min(edge.src, edge.dst);
    auto dst = std::max(edge.src, edge.dst);
    forest.push_back({src, dst});
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

inline void MinCutGraph::k_sample_supernodes(std::pair<Edge, SampleSketchRet> *query, std::vector<node_id_t> &reps, int k_id) {
  bool except = false;
  std::exception_ptr err;
  #pragma omp parallel for default(none) shared(query, reps, k_id, except, err)
  for (node_id_t i = 0; i < reps.size(); ++i) { // NOLINT(modernize-loop-convert)
    // wrap in a try/catch because exiting through exception is undefined behavior in OMP
    try {
      query[reps[i]] = supernodes[(reps[i] * k) + k_id]->sample();

    } catch (...) {
      except = true;
      err = std::current_exception();
    }
  }
  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);
}

void MinCutGraph::k_merge_supernodes(Supernode** copy_supernodes, std::vector<node_id_t> &new_reps,
               std::vector<std::vector<node_id_t>> &to_merge, bool make_copy, int k_id) {
  bool except = false;
  std::exception_ptr err;
  // loop over the to_merge vector and perform supernode merging
  #pragma omp parallel for default(shared)
  for (node_id_t i = 0; i < new_reps.size(); i++) { // NOLINT(modernize-loop-convert)
    // OMP requires a traditional for-loop to work
    node_id_t a = new_reps[i];
    try {
      if (make_copy && Graph::config._backup_in_mem) { // make a copy of a
        copy_supernodes[a] = Supernode::makeSupernode(*supernodes[(a * k) + k_id]);
      }

      // perform merging of nodes b into node a
      for (node_id_t b : to_merge[a]) {
        supernodes[(a * k) + k_id]->merge(*supernodes[(b * k) + k_id]);
      }
    } catch (...) {
      except = true;
      err = std::current_exception();
    }
  }

  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);
}

std::vector<Edge> MinCutGraph::get_spanning_forest(int k_id) {
  std::vector<Edge> forest;
  bool first_round = true;
  size_t round = 0;
  Supernode** copy_supernodes = new Supernode*[num_nodes];
  std::pair<Edge, SampleSketchRet> *query = new std::pair<Edge, SampleSketchRet>[num_nodes];
  std::vector<node_id_t> reps(num_nodes);
  std::vector<node_id_t> backed_up;
  std::fill(size, size + num_nodes, 1);
  for (node_id_t i = 0; i < num_nodes; ++i) {
    reps[i] = i;
    copy_supernodes[i] = nullptr;
    parent[(i * k) + k_id] = i;
  }
  try {
    do {
      round++;
      modified = false;
      k_sample_supernodes(query, reps, k_id);

      std::vector<std::vector<node_id_t>> to_merge = to_merge_and_forest_edges(forest, query, reps, k_id);

      // make a copy if necessary
      if (first_round)
        backed_up = reps;

      k_merge_supernodes(copy_supernodes, reps, to_merge, first_round, k_id);

#ifdef VERIFY_SAMPLES_F
      if (!first_round && fail_round_2) throw OutOfQueriesException();
#endif
      first_round = false;
    } while (modified);
  } catch (...) {
    std::cout << "ERROR in get_spanning_forest()" << std::endl;
    std::cout << "round = " << round << std::endl;
    std::rethrow_exception(std::current_exception());
  }
  for (node_id_t i : backed_up) {
    free(supernodes[(i * k) + k_id]);
    supernodes[(i * k) + k_id] = copy_supernodes[i];
  }

  node_id_t num_cc = 0;
  std::set<node_id_t> roots;
  for (node_id_t i = 0; i < num_nodes; i++) {
    node_id_t parent = k_get_parent(i, k_id);
    if (roots.count(parent) == 0) {
      ++num_cc;
      roots.insert(parent);
    }
  }

  delete[] copy_supernodes;
  delete[] query;
  std::cout << "  round = " << round << " cc size = " << num_cc << std::endl;
  return forest;
}

void MinCutGraph::trim_spanning_forest(std::vector<Edge> &forest) {
  //std::cout << "  DELETING THESE EDGES: " << std::endl;
  for (auto e : forest) {
    //std::cout << "{" << e.src << "," << e.dst << "}";
    gts->insert({e.src, e.dst});
    std::swap(e.src, e.dst);
    gts->insert({e.src, e.dst});
  }
  //std::cout << std::endl;
}

// void MinCutGraph::cycle_supernodes() {
//   // if (Supernode::query_rotation == 0)
//   //   Supernode::query_rotation = Supernode::get_max_sketches() - 1;
//   // else
//   //   --Supernode::query_rotation;
//   ++Supernode::query_rotation;
// }

void verify_edges(std::vector<std::vector<Edge>> forests) {
  // Check for duplicate edges and if edges exist in graph
  std::set<Edge> edges;
  for (auto& forest : forests) {
    for (auto& e : forest) {
      if (edges.count(e) == 0) {
        edges.insert(e);
      }
      else {
        std::cerr << "ERROR: duplicate error in forests!" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }
}

void verify_solns(std::vector<std::vector<Edge>> forests) {
  
}

void MinCutGraph::verify_spanning_forests(std::vector<std::vector<Edge>> forests) {
  verify_edges(forests);
  verify_solns(forests);
}

std::vector<std::vector<Edge>> MinCutGraph::k_spanning_forests(size_t k) {
  std::vector<std::vector<Edge>> forests;
  for (size_t i = 0; i < k; i++) {
    gts->force_flush(); // flush everything in guttering system to make final updates
    GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
    cudaDeviceSynchronize();
    cudaGraph->k_applyFlushUpdates();
    std::cout << "Getting spanning forest " << i + 1 << ":" << std::endl;
    forests.push_back(get_spanning_forest(i));

    // get ready for ingesting more from the stream
    // reset supernodes and resume graph workers
    for (node_id_t j = 0; j < num_nodes; j++) {
      supernodes[(j * k) + i]->reset_query_state();
    }
    GraphWorker::unpause_workers();
    trim_spanning_forest(forests[i]);

    // if (i % 4 == 3)
    //   cycle_supernodes();
  }

  verify_spanning_forests(forests);
  return forests;
}

MinCutGraph::~MinCutGraph() {
  // Only delete the extra supernodes that were made due to k
  for (unsigned i=num_nodes;i<num_nodes * k;++i) {
    free(supernodes[i]); // free because memory is malloc'd in make_supernode
  }
}


