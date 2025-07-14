#include "min_cut_sketch_alg.h"

#include <algorithms/global_mincut/algorithms.h>
#include <algorithms/global_mincut/minimum_cut.h>
#include <data_structure/graph_access.h>
#include <data_structure/mutable_graph.h>

MinCutSketchAlg::MinCutSketchAlg(node_id_t _num_vertices, size_t _seed, MCAlgConfiguration _config)
    : num_vertices(_num_vertices),
      seed(_seed),
      config(_config),
      max_subgraphs(2 * log2(num_vertices)),
      cur_subgraphs(1),
      sketch_factor(1.3 * 1 / (config._epsilon * config._epsilon)),
      sketch_samples(Sketch::calc_cc_samples(num_vertices, sketch_factor)),
      buffer_elms(Sketch::estimate_bytes(Sketch::calc_vector_length(num_vertices), sketch_samples) /
                  sizeof(node_id_t)),
      cc_sketches(new CCSketchAlg *[max_subgraphs]),
      edge_store(seed, num_vertices, buffer_elms * sizeof(node_id_t), max_subgraphs, 1) {
  cc_config.sketches_factor(sketch_factor);

  cc_sketches[0] = new CCSketchAlg(num_vertices, seed, cc_config);
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
  cc_sketches[0]->allocate_worker_memory(num_workers);

  thread_data = new ThreadData[num_workers];
  for (size_t i = 0; i < num_workers; i++) {
    thread_data[i].cc_buffers.resize(cur_subgraphs);
    for (size_t b = 1; b < cur_subgraphs; b++) {
      thread_data[i].cc_buffers[b].resize(buffer_elms);
    }
  }
}

void MinCutSketchAlg::pre_insert(GraphUpdate upd, node_id_t thr_id) {
  // we just pre-insert to the first subgraph
  // TODO: unless there's something more intelligent to do here at some point?
  cc_sketches[0]->pre_insert(upd, thr_id);
}

void MinCutSketchAlg::apply_update_batch(size_t thr_id, node_id_t src_vertex,
                                         const std::vector<node_id_t> &dst_vertices) {
  // everything goes in subgraph 0
  cc_sketches[0]->apply_update_batch(thr_id, src_vertex, dst_vertices);

  size_t num_mapped[max_subgraphs];

  std::vector<std::vector<node_id_t>> &buffers = thread_data[thr_id].cc_buffers;
  std::vector<SubgraphTaggedUpdate> &edge_buf = thread_data[thr_id].edge_store_buffer;
  
  // map the updates to one of the subgraphs
  for (size_t i = 0; i < dst_vertices.size(); i++) {
    vec_t idx = concat_pairing_fn(src_vertex, dst_vertices[i]);
    node_id_t subgraph_idx = Bucket_Boruvka::get_index_depth(idx, seed, max_subgraphs);
    node_id_t mapped_to = std::min((size_t) subgraph_idx, cur_subgraphs);

    if (subgraph_idx == mapped_to) {
      buffers[mapped_to][num_mapped[mapped_to]++] = dst_vertices[i];
    } else {
      edge_buf[num_mapped[mapped_to]++] = {subgraph_idx, dst_vertices[i]};
    }
  }

  for (size_t i = 1; i < cur_subgraphs; i++) {
    buffers[i].resize(num_mapped[i]);
    cc_sketches[i]->apply_update_batch(thr_id, src_vertex, buffers[i]);
    buffers[i].resize(buffer_elms);
  }

  edge_buf.resize(num_mapped[cur_subgraphs]);
  edge_store.insert_adj_edges(src_vertex, cur_subgraphs, edge_buf);
}

MinCut MinCutSketchAlg::calc_minimum_cut() {
  typedef VieCut::mutable_graph Graph;
  typedef std::shared_ptr<VieCut::mutable_graph> GraphPtr;

  // iterate over our subgraphs to find the correct value
  for (size_t i = 0; i < cur_subgraphs; i++) {
    std::vector<SpanningForest> sfs = cc_sketches[i]->calc_disjoint_spanning_forests(k);

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

    if (cut < k) {
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
  }
}
