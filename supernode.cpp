#include <stdexcept>
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp>
#include "include/supernode.h"
#include "include/util.h"
#include "include/graph_worker.h"

constexpr double Supernode::default_bucket_factor;

Supernode::Supernode(uint64_t n, long seed): idx(0), logn(log2(n)),
					     n(n), seed(seed), sketch_size(Sketch::sketchSizeof(n*n, default_bucket_factor)) {
  // generate logn sketches for each supernode (read: node)
  for (int i = 0; i < logn; ++i) {
    Sketch::makeSketch(get_sketch(i), n*n, seed++, default_bucket_factor);
  }
}

Supernode::Supernode(uint64_t n, long seed, size_t sketch_size) :
  idx(0), logn(log2(n)), n(n), seed(seed), sketch_size(sketch_size) {
  // Assume someone else is going to initialize our sketches.
}

Supernode::SupernodeUniquePtr Supernode::makeSupernode(uint64_t n, long seed) {
  void *loc = malloc(sizeof(Supernode) + log2(n) * Sketch::sketchSizeof(n*n, default_bucket_factor) - sizeof(char));
  return SupernodeUniquePtr(makeSupernode(loc, n, seed), [](Supernode* s){ s->~Supernode(); free(s); });
}

Supernode::SupernodeUniquePtr Supernode::makeSupernode(uint64_t n, long seed, std::fstream &binary_in) {
  // Since fstream is stateful, we have to do this awful thing.
  
  // We have to read the first Sketch to figure out how big it's going to be...
  Sketch::SketchUniquePtr first_sketch = Sketch::makeSketch(n*n, seed++, binary_in);
  double bucket_factor = first_sketch->get_bucket_factor();

  // Then we assume they're all the same size...
  size_t sketch_size = Sketch::sketchSizeof(n*n, bucket_factor);
  void *loc = malloc(sizeof(Supernode) + log2(n) * sketch_size - sizeof(char));
  SupernodeUniquePtr ret(new (loc) Supernode(n, seed, sketch_size), [](Supernode* s){ free(s); });

  // Copy the sketch we already read out.
  memcpy((void*)ret->get_sketch(0), (void*)first_sketch.get(), Sketch::sketchSizeof(*first_sketch));
  for (int i = 1; i < ret->logn; ++i) {
    Sketch::makeSketch(ret->get_sketch(i), n*n, seed++, binary_in);

    // All of the bucket factors have to be the same. If they aren't, choke and die.
    if (ret->get_sketch(i)->get_bucket_factor() != bucket_factor) {
      throw new std::runtime_error("Please stop what you're doing");
    }
  }
  return ret;
}

Supernode* Supernode::makeSupernode(void* loc, uint64_t n, long seed) {
  return new (loc) Supernode(n, seed);
}

Supernode::~Supernode() {
}

boost::optional<Edge> Supernode::sample() {
  if (idx == logn) throw OutOfQueriesException();
  vec_t query_idx;
  try {
    query_idx = get_sketch(idx++)->query();
  } catch (AllBucketsZeroException &e) {
    return {};
  }
  return inv_nondir_non_self_edge_pairing_fn(query_idx);
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

Supernode::SupernodeUniquePtr Supernode::delta_supernode(uint64_t n, long seed,
							 const vector<vec_t> &updates) {
  auto delta_node = makeSupernode(n, seed);
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
#pragma omp parallel for num_threads(GraphWorker::get_group_size()) default(shared)
  for (int i = 0; i < delta_node->logn; ++i) {
    delta_node->get_sketch(i)->batch_update(updates);
  }
  return delta_node;
}

void Supernode::write_binary(std::fstream& binary_out) {
  for (int i = 0; i < logn; ++i) {
    get_sketch(i)->write_binary(binary_out);
  }
}
