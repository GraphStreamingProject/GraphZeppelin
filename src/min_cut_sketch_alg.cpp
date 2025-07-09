#include "min_cut_sketch_alg.h"

MinCutSketchAlg::MinCutSketchAlg(node_id_t _num_vertices, size_t _seed, MCAlgConfiguration _config)
    : num_vertices(_num_vertices),
      seed(_seed),
      config(_config),
      max_subgraphs(2 * log2(num_vertices)),
      cur_subgraphs(1),
      sketch_factor(1.3 * 1 / (config._epsilon * config._epsilon)),
      sketch_samples(Sketch::calc_cc_samples(num_vertices, sketch_factor)),
      cc_sketches(new CCSketchAlg*[max_subgraphs]),
      edge_store(seed, num_vertices, Sketch::estimate_bytes(Sketch::calc_vector_length(num_vertices), sketch_samples), max_subgraphs, 1) {

  cc_config.sketches_factor(sketch_factor);

  cc_sketches[0] = new CCSketchAlg(num_vertices, seed, cc_config);
}

MinCutSketchAlg::~MinCutSketchAlg() {
  for (size_t i = 0; i < cur_subgraphs; i++) {
    delete cc_sketches[i];
  }
  delete[] cc_sketches;

  if (delta_sketches != nullptr) delete[] delta_sketches;
}

void MinCutSketchAlg::allocate_worker_memory(size_t num_workers) {
  num_delta_sketches = num_workers * config._num_subgraphs_use_delta;
  delta_sketches = new Sketch[num_delta_sketches];
  for (size_t i = 0; i < num_delta_sketches; i++) {
    delta_sketches[i] = std::move(Sketch(Sketch::calc_vector_length(num_vertices), seed,
                                         Sketch::calc_cc_samples(num_vertices, sketch_factor)));
  }
}

void MinCutSketchAlg::pre_insert(GraphUpdate upd, node_id_t thr_id) {
  // we just pre-insert to the first subgraph
  // TODO: unless there's something more intelligent to do here at some point?
  cc_sketches[0]->pre_insert(upd, thr_id);
}

void MinCutSketchAlg::apply_update_batch(size_t thr_id, node_id_t src_vertex,
                                         const std::vector<node_id_t> &dst_vertices) {
  for (size_t i = 0; i < dst_vertices.size(); i++) {
    
  }
}

size_t MinCutSketchAlg::calc_minimum_cut() {
  // TODO: Write something here I guess
  return 0;
}
