#include "k_connected_graph.h"
#include <graph_worker.h>

std::vector<std::vector<node_id_t>> KConnectedGraph::to_merge_and_forest_edges(
    std::vector<Edge> &forest, std::pair<Edge, SampleSketchRet> *query,
    std::vector<node_id_t> &reps) {
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
    node_id_t a = get_parent(edge.src);
    node_id_t b = get_parent(edge.dst);
    if (a == b) continue;

#ifdef VERIFY_SAMPLES_F
    verifier->verify_edge(edge);
#endif

    // make a the parent of b
    if (size[a] < size[b]) std::swap(a,b);
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

std::vector<Edge> KConnectedGraph::get_spanning_forest() {
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
    parent[i] = i;
  }
  try {
    do {
      round++;
      modified = false;
      sample_supernodes(query, reps);
      std::vector<std::vector<node_id_t>> to_merge = to_merge_and_forest_edges(forest, query, reps);
      // make a copy if necessary
      if (first_round)
        backed_up = reps;

      merge_supernodes(copy_supernodes, reps, to_merge, first_round);

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
    free(supernodes[i]);
    supernodes[i] = copy_supernodes[i];
  }

  node_id_t num_cc = 0;
  std::set<node_id_t> roots;
  for (node_id_t i = 0; i < num_nodes; i++) {
    node_id_t parent = get_parent(i);
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

void KConnectedGraph::trim_spanning_forest(std::vector<Edge> &forest) {
  // std::cout << "DELETING THESE EDGES:" << std::endl;
  for (auto e : forest) {
    // std::cout << e.src << " " << e.dst << std::endl;
    update({e, DELETE});    
  }
}

// void KConnectedGraph::cycle_supernodes() {
//   // if (Supernode::query_rotation == 0)
//   //   Supernode::query_rotation = Supernode::get_max_sketches() - 1;
//   // else
//   //   --Supernode::query_rotation;
//   ++Supernode::query_rotation;
// }

void KConnectedGraph::verify_spanning_forests(std::vector<std::vector<Edge>> forests) {
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

std::vector<std::vector<Edge>> KConnectedGraph::k_spanning_forests(size_t k) {
  std::vector<std::vector<Edge>> forests;
  for (size_t i = 0; i < k; i++) {
    gts->force_flush(); // flush everything in guttering system to make final updates
    GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
    cudaDeviceSynchronize();
    cudaGraph->applyFlushUpdates();
    std::cout << "Getting spanning forest " << i + 1 << std::endl;
    forests.push_back(get_spanning_forest());

    // get ready for ingesting more from the stream
    // reset supernodes and resume graph workers
    for (node_id_t i = 0; i < num_nodes; i++) {
      supernodes[i]->reset_query_state();
    }
    GraphWorker::unpause_workers();
    trim_spanning_forest(forests[i]);

    // if (i % 4 == 3)
    //   cycle_supernodes();
  }

  verify_spanning_forests(forests);
  return forests;
}