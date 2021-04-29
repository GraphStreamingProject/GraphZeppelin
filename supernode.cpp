#include <stdexcept>
#include <cmath>
#include <boost/multiprecision/cpp_int.hpp>
#include "include/supernode.h"
#include "include/util.h"

#ifdef EXT_MEM_POST_PROC_F
Supernode::Supernode(uint64_t n, long seed): sketches(log2(n)), idx(0), logn(log2
(n)),  ext_mem_size(1), ext_mem_destroyed(false) {
  // generate logn sketches for each supernode (read: node)
  srand(seed);
  long r = rand();
  for (int i=0;i<logn;++i)
    sketches[i] = new Sketch(n*n, r);
}
#else
Supernode::Supernode(uint64_t n, long seed): sketches(log2(n)), idx(0), logn(log2
(n)){
  // generate logn sketches for each supernode (read: node)
  srand(seed);
  long r = rand();
  for (int i=0;i<logn;++i)
    sketches[i] = new Sketch(n*n, r);
}
#endif

Supernode::~Supernode() {
  for (int i=0;i<logn;++i)
    delete sketches[i];
}

boost::optional<Edge> Supernode::sample() {
  if (idx == logn) throw OutOfQueriesException();
  vec_t query_idx;
  try {
    query_idx = sketches[idx]->query();
  } catch (AllBucketsZeroException &e) {
    ++idx;
    return {};
  }
  ++idx;
  return inv_nondir_non_self_edge_pairing_fn(query_idx);
}

void Supernode::merge(Supernode &other) {
  idx = max(idx, other.idx);
  for (int i=idx;i<logn;++i) {
    (*sketches[i])+=(*other.sketches[i]);
  }
}

void Supernode::update(Edge update) {
  vec_t upd = nondirectional_non_self_edge_pairing_fn(update.first, update.second);
  for (Sketch* s : sketches)
    s->update(upd);
}

void Supernode::batch_update(const std::vector<vec_t>& updates) {
  for (Sketch *s : sketches) {
    s->batch_update(updates);
  }
}
