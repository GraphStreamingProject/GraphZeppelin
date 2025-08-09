#include "min_cut_sketch_alg.h"

#include <chrono>
#include <algorithms/global_mincut/algorithms.h>
#include <algorithms/global_mincut/minimum_cut.h>
#include <data_structure/graph_access.h>
#include <data_structure/mutable_graph.h>

MinCutSketchAlg::MinCutSketchAlg(node_id_t _num_vertices, size_t _seed, MCAlgConfiguration _config)
    : num_vertices(_num_vertices),
      seed(_seed),
      subgraph_seed(col_hash(&seed, sizeof(seed), seed)),
      config(_config),
      max_subgraphs(2 * log2(num_vertices)),
      k(log2(num_vertices) / (config._epsilon * config._epsilon)),
      cur_subgraphs(config._initial_subgraphs),
      sketch_factor(1.3 / (config._epsilon * config._epsilon)),
      sketch_samples(Sketch::calc_cc_samples(num_vertices, sketch_factor)),
      buffer_elms(Sketch::estimate_bytes(Sketch::calc_vector_length(num_vertices), sketch_samples) /
                  sizeof(node_id_t)),
      cc_sketches(new CCSketchAlg *[max_subgraphs]),
      edge_store(seed, num_vertices, buffer_elms * sizeof(node_id_t), max_subgraphs, 1) {
  if (cur_subgraphs > max_subgraphs) {
    std::cerr << "WARNING: MinCutSketchAlg, initial_subgraphs > max_subgraphs. Setting to max."
              << std::endl;
    cur_subgraphs = max_subgraphs;
  }

  cc_config.sketches_factor(sketch_factor);

  for (size_t i = 0; i < cur_subgraphs; i++) {
    cc_sketches[i] = new CCSketchAlg(num_vertices, seed, cc_config);
    if (i > 0) cc_sketches[i]->invalidate_dsu();
  }
}

MinCutSketchAlg::~MinCutSketchAlg() {
  for (size_t i = 0; i < cur_subgraphs; i++) {
    delete cc_sketches[i];
  }
  delete[] cc_sketches;

  if (delta_sketches != nullptr) delete[] delta_sketches;
  if (thread_data != nullptr) delete[] thread_data;
}

void MinCutSketchAlg::allocate_worker_memory(size_t _num_workers) {
  num_workers = _num_workers;
  for (size_t i = 0; i < cur_subgraphs; i++) {
    cc_sketches[i]->allocate_worker_memory(num_workers);
  }

  thread_data = new ThreadData[num_workers];
  for (size_t t = 0; t < num_workers; t++) {
    thread_data[t].cc_buffers.resize(max_subgraphs);
    for (size_t b = 1; b < cur_subgraphs; b++) {
      thread_data[t].cc_buffers[b].resize(buffer_elms);
    }
  }
}

// must hold advance_subgraph_lock when calling this function
void MinCutSketchAlg::advance_cur_subgraph(size_t new_cur_subgraphs) {
  for (size_t i = cur_subgraphs; i < new_cur_subgraphs; i++) {
    cc_sketches[i] = new CCSketchAlg(num_vertices, seed, cc_config); // TODO: DO WE NEED A DIFFERENT SEED?
    cc_sketches[i]->allocate_worker_memory(num_workers);
    cc_sketches[i]->invalidate_dsu();
  }

  for (size_t t = 0; t < num_workers; t++) {
    for (size_t b = cur_subgraphs; b < new_cur_subgraphs; b++) {
      thread_data[t].cc_buffers[b].resize(buffer_elms);
    }
  }

  cur_subgraphs = new_cur_subgraphs;
}

void MinCutSketchAlg::pre_insert(GraphUpdate upd, node_id_t thr_id) {
  // we just pre-insert to the first subgraph
  // TODO: unless there's something more intelligent to do here at some point?
  cc_sketches[0]->pre_insert(upd, thr_id);
}

void MinCutSketchAlg::apply_update_batch(size_t thr_id, node_id_t src_vertex,
                                         const std::vector<node_id_t> &dst_vertices) {
  assert(dst_vertices.size() <= buffer_elms);

  // everything goes in subgraph 0
  cc_sketches[0]->apply_update_batch(thr_id, src_vertex, dst_vertices);

  size_t num_mapped[max_subgraphs];
  std::fill(&num_mapped[0], &num_mapped[max_subgraphs - 1], 0);

  std::vector<std::vector<node_id_t>> &buffers = thread_data[thr_id].cc_buffers;
  std::vector<SubgraphTaggedUpdate> &edge_buf = thread_data[thr_id].edge_store_buffer;
  edge_buf.resize(buffer_elms);

  size_t our_cur_subgraphs = cur_subgraphs;
  
  // map the updates to one of the subgraphs
  for (size_t i = 0; i < dst_vertices.size(); i++) {
    vec_t idx = concat_pairing_fn(src_vertex, dst_vertices[i]);
    node_id_t subgraph_idx = SketchBucket::get_index_depth(idx, subgraph_seed, max_subgraphs - 1) + 1;

    if (subgraph_idx < our_cur_subgraphs) {
      // goes in a sketch!
      assert(num_mapped[subgraph_idx] < buffer_elms);
      assert(buffers[subgraph_idx].size() == buffer_elms);
      buffers[subgraph_idx][num_mapped[subgraph_idx]++] = dst_vertices[i];
    } else {
      // goes in edge store!
      assert(num_mapped[our_cur_subgraphs] < buffer_elms);
      edge_buf[num_mapped[our_cur_subgraphs]++] = {subgraph_idx, dst_vertices[i]};
    }
  }

  for (size_t i = 1; i < our_cur_subgraphs; i++) {
    buffers[i].resize(num_mapped[i]);
    cc_sketches[i]->apply_update_batch(thr_id, src_vertex, buffers[i]);
    buffers[i].resize(buffer_elms);
  }

  edge_buf.resize(num_mapped[our_cur_subgraphs]);
  TaggedUpdateBatch batch = edge_store.insert_adj_edges(src_vertex, our_cur_subgraphs, edge_buf);
  
  while (batch.dsts_data.size() > 0) {
    // we don't have a sketch for this subgraph yet!
    if (batch.edge_store_subgraph > cur_subgraphs) {
      advance_subgraph_lock.lock();
      // double check that we are the thread who will allocate next subgraph
      if (batch.edge_store_subgraph > cur_subgraphs) {
        advance_cur_subgraph(batch.edge_store_subgraph);
      }
      advance_subgraph_lock.unlock();
    }

    std::fill(&num_mapped[0], &num_mapped[max_subgraphs - 1], 0);
    for (auto tagged_edge : batch.dsts_data) {
      size_t subgraph = tagged_edge.subgraph;
      assert(subgraph < cur_subgraphs);

      buffers[subgraph][num_mapped[subgraph]++] = tagged_edge.dst;
      assert(num_mapped[subgraph] <= buffers[subgraph].capacity());

      unlikely_if (num_mapped[subgraph] >= buffer_elms) {
        cc_sketches[subgraph]->apply_update_batch(thr_id, batch.src, buffers[subgraph]);
        num_mapped[subgraph] = 0;
      }
    }

    for (size_t i = 1; i < batch.edge_store_subgraph; i++) {
      if (num_mapped[i] > 0) {
        buffers[i].resize(num_mapped[i]);
        cc_sketches[i]->apply_update_batch(thr_id, batch.src, buffers[i]);
        buffers[i].resize(buffer_elms);
      }
    }

    // check if there are more contractions to perform
    if (edge_store.contract_in_progress())
      batch = edge_store.vertex_advance_subgraph(cur_subgraphs);
    else
      batch.dsts_data.clear();
  }
}

static MinCut run_viecut(node_id_t num_vertices, std::vector<Edge> &edges) {
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

  delete mc;
  return {left, right, cut};
}

MinCut MinCutSketchAlg::calc_minimum_cut() {
#ifdef VERIFY_SAMPLES_F
  create_subgraph_verifiers();
#endif

  auto start = std::chrono::steady_clock::now();
  sf_total_duration = std::chrono::duration<double>(0);
  viecut_duration = std::chrono::duration<double>(0);

  std::cout << "Performing minimum cut query" << std::endl;
  for (size_t i = 0; i < cur_subgraphs; i++) {
    std::cout << "Sketch " << i << " updates = " << cc_sketches[i]->get_num_updates() << std::endl;
  }


  // iterate over our subgraphs to find the correct value
  for (size_t i = 0; i < cur_subgraphs; i++) {
    auto sf_start = std::chrono::steady_clock::now();
    std::vector<SpanningForest> sfs = cc_sketches[i]->calc_disjoint_spanning_forests(k);
    std::vector<Edge> edges;

    for (auto& sf : sfs) {
      auto& sf_edges = sf.get_edges();
      edges.insert(edges.end(), sf_edges.begin(), sf_edges.end());
    }
    sf_total_duration += std::chrono::steady_clock::now() - sf_start;

    auto viecut_start =std::chrono::steady_clock::now();
    MinCut mc = run_viecut(num_vertices, edges);
    size_t adjust_value = mc.value << i;
    std::cout << "Subgraph: " << i + 1 << ", cut value = " << mc.value << ", k = " << k
              << ", adjusted = " << adjust_value << std::endl;
    viecut_duration += std::chrono::steady_clock::now() - viecut_start;

    if (mc.value < k) {
      mc.value = adjust_value;
      total_mc_duration = std::chrono::steady_clock::now() - start;
      return mc;
    }
  }

  // pull from the adjacency list
  std::vector<Edge> adj_edges = edge_store.get_edges();
  MinCut mc = run_viecut(num_vertices, adj_edges);
  std::cout << "Edge Store MinCut = " << mc.value << std::endl;

  // multiply the minimum cut by sampling rate
  // -1 because edge store contains everything in remaining subgraphs, geometric series
  mc.value <<= (cur_subgraphs - 1); 
  std::cout << "Adjusted = " << mc.value << std::endl;
  total_mc_duration = std::chrono::steady_clock::now() - start;
  return mc;
}

#ifdef VERIFY_SAMPLES_F
void MinCutSketchAlg::set_verifier(std::unique_ptr<GraphVerifier> _verifier) {
  verifier = std::make_unique<GraphVerifier>(*_verifier);
  cc_sketches[0]->set_verifier(std::make_unique<GraphVerifier>(*verifier));
}

void MinCutSketchAlg::create_subgraph_verifiers() {
  cc_sketches[0]->set_verifier(std::make_unique<GraphVerifier>(*verifier));
  std::vector<std::unique_ptr<GraphVerifier>> subgraph_verifiers;

  std::cout << "Creating: " << cur_subgraphs + 1 << " verifiers" << std::endl;

  for (size_t i = 0; i <= cur_subgraphs; i++) {
    subgraph_verifiers.emplace_back(new GraphVerifier(num_vertices));
  }

  size_t subgraph_sizes[max_subgraphs];
  std::fill(&subgraph_sizes[0], &subgraph_sizes[max_subgraphs - 1], 0);
  size_t non_zero = 0;

  std::vector<std::vector<bool>> adj_mat = verifier->extract_adj_matrix();
  for (node_id_t i = 0; i < num_vertices; i++) {
    for (node_id_t j = 0; j < num_vertices - i; j++) {
      if (adj_mat[i][j]) {
        node_id_t dst = i + j;

        non_zero++;
        vec_t idx = concat_pairing_fn(i, dst); // edge is 
        node_id_t subgraph_idx = SketchBucket::get_index_depth(idx, subgraph_seed, max_subgraphs - 1) + 1;
        if (subgraph_idx < cur_subgraphs) {
          subgraph_verifiers[subgraph_idx]->edge_update({i, dst});
          subgraph_sizes[subgraph_idx]++;
        } else {
          subgraph_verifiers[cur_subgraphs]->edge_update({i, dst});
        }
      }
    }
  }

  std::cout << "verifier subgraph 0 size = " << non_zero << std::endl;
  for (size_t i = 1; i < cur_subgraphs; i++) {
    cc_sketches[i]->set_verifier(std::make_unique<GraphVerifier>(*subgraph_verifiers[i]));
    std::cout << "verifier subgraph " << i << " size = " << subgraph_sizes[i] << std::endl;
  }
  adj_verifier = std::move(subgraph_verifiers[cur_subgraphs]);
}
#endif
