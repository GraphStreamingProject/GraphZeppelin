#include <stdexcept>
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp>
#include "../include/supernode.h"
#include "../include/graph_worker.h"

uint32_t Supernode::bytes_size;

Supernode::Supernode(uint64_t n, long seed): idx(0), logn(log2(n)),
               n(n), seed(seed), sketch_size(Sketch::sketchSizeof()) {

  // generate logn sketches for each supernode (read: node)
  for (int i = 0; i < logn; ++i) {
    Sketch::makeSketch(get_sketch(i), seed++);
  }
}

Supernode::Supernode(uint64_t n, long seed, std::fstream &binary_in) :
  idx(0), logn(log2(n)), n(n), seed(seed), sketch_size(Sketch::sketchSizeof()) {

  // read logn sketches from file for each supernode (read: node)
  for (int i = 0; i < logn; ++i) {
    Sketch::makeSketch(get_sketch(i), seed++, binary_in);
  }
}

Supernode* Supernode::makeSupernode(uint64_t n, long seed) {
  void *loc = malloc(bytes_size);
  return new (loc) Supernode(n, seed);
}

Supernode* Supernode::makeSupernode(uint64_t n, long seed, std::fstream &binary_in) {
  void *loc = malloc(bytes_size);
  return new (loc) Supernode(n, seed, binary_in);
}

Supernode* Supernode::makeSupernode(void* loc, uint64_t n, long seed) {
  return new (loc) Supernode(n, seed);
}

Supernode::~Supernode() {
}

std::pair<Edge, SampleSketchRet> Supernode::sample() {
  if (idx == logn) throw OutOfQueriesException();

  std::pair<vec_t, SampleSketchRet> query_ret = get_sketch(idx++)->query();
  vec_t idx = query_ret.first;
  SampleSketchRet ret_code = query_ret.second;
  return {inv_nondir_non_self_edge_pairing_fn(idx), ret_code};
}

void Supernode::merge(Supernode &other) {
  idx = max(idx, other.idx);
  for (int i=idx;i<logn;++i) {
    (*get_sketch(i))+=(*other.get_sketch(i));
  }
}

void Supernode::update(vec_t upd) {
  for (int i = 0; i < logn; ++i)
    get_sketch(i)->update(upd);
}

void Supernode::apply_delta_update(const Supernode* delta_node) {
  std::unique_lock<std::mutex> lk(node_mt);
  for (int i = 0; i < logn; ++i) {
    *get_sketch(i) += *delta_node->get_sketch(i);
  }
  lk.unlock();
}

/*
 * Consider fiddling with environment vars
 * OMP_DYNAMIC: whether the OS is allowed to dynamically change the number
 * of threads employed for each parallel section
 * OMP_NUM_THREADS (or set_omp_num_threads): how many threads to spin up for
 * each parallel section. the default is (probably) one per CPU core
 * available, but we may want to set it lower if logn is a nice multiple of
 * a lower number.
 *
 * We may want to use omp option schedule(dynamic) or schedule(guided) if
 * there are very many more iterations of loop than threads. Dynamic
 * scheduling is good if loop iterations are expected to take very much
 * different amounts of time. Refer to
 * http://www.inf.ufsc.br/~bosco.sobral/ensino/ine5645/OpenMP_Dynamic_Scheduling.pdf
 * for a detailed explanation.
 */
/*
 * Current impl uses default threads and parallelism within batched_update.
 * Considered using spin-threads and parallelism within sketch::update, but
 * this was slow (at least on small graph inputs).
 */
void Supernode::delta_supernode(uint64_t n, long seed,
               const vector<vec_t> &updates, void *loc) {
  auto delta_node = makeSupernode(loc, n, seed);
#pragma omp parallel for num_threads(GraphWorker::get_group_size()) default(shared)
  for (int i = 0; i < delta_node->logn; ++i) {
    delta_node->get_sketch(i)->batch_update(updates);
  }
}

void Supernode::write_binary(std::fstream& binary_out) {
  for (int i = 0; i < logn; ++i) {
    get_sketch(i)->write_binary(binary_out);
  }
}
