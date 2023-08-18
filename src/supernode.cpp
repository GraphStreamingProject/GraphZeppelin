#include <stdexcept>
#include <cmath>
#include "../include/supernode.h"
#include "../include/graph_worker.h"

size_t Supernode::max_sketches;
size_t Supernode::bytes_size;
size_t Supernode::serialized_size;

Supernode::Supernode(uint64_t n, uint64_t seed): sample_idx(0),
  n(n), seed(seed), num_sketches(max_sketches),
  merged_sketches(max_sketches), sketch_size(Sketch::sketchSizeof()) {

  size_t sketch_width = Sketch::column_gen(Sketch::get_failure_factor());
  // generate num_sketches sketches for each supernode (read: node)
  for (size_t i = 0; i < num_sketches; ++i) {
    Sketch::makeSketch(get_sketch(i), seed);
    seed += sketch_width;
  }
}

Supernode::Supernode(uint64_t n, uint64_t seed, std::istream &binary_in) :
  sample_idx(0), n(n), seed(seed), sketch_size(Sketch::sketchSizeof()) {

  size_t sketch_width = Sketch::column_gen(Sketch::get_failure_factor());

  SerialType type;
  binary_in.read((char*) &type, sizeof(SerialType));

  uint32_t beg = 0;
  uint32_t num = max_sketches;
  bool sparse = false;
  if (type == PARTIAL) {
    binary_in.read((char*) &beg, sizeof(beg));
    binary_in.read((char*) &num, sizeof(num));
  }
  else if (type == SPARSE) {
    binary_in.read((char*) &beg, sizeof(beg));
    binary_in.read((char*) &num, sizeof(num));
    sparse = true;
  }
  // sample in range [beg, beg + num)
  num_sketches = beg + num;
  merged_sketches = num_sketches;
  sample_idx = beg;

  // create empty sketches, if any
  for (size_t i = 0; i < beg; ++i) {
    Sketch::makeSketch(get_sketch(i), seed);
    seed += sketch_width;
  }
  // build sketches from serialized data
  for (size_t i = beg; i < beg + num; ++i) {
    Sketch::makeSketch(get_sketch(i), seed, binary_in, sparse);
    seed += sketch_width;
  }
  // create empty sketches at end, if any
  for (size_t i = beg + num; i < max_sketches; ++i) {
    Sketch::makeSketch(get_sketch(i), seed);
    seed += sketch_width;
  }
}

Supernode::Supernode(const Supernode& s) : 
  sample_idx(s.sample_idx), n(s.n), seed(s.seed), num_sketches(s.num_sketches), 
  merged_sketches(s.merged_sketches), sketch_size(s.sketch_size) {
  for (size_t i = 0; i < num_sketches; ++i) {
    Sketch::makeSketch(get_sketch(i), *s.get_sketch(i));
  }
}

Supernode* Supernode::makeSupernode(uint64_t n, long seed, void *loc) {
  return new (loc) Supernode(n, seed);
}

Supernode* Supernode::makeSupernode(uint64_t n, long seed, std::istream &binary_in, void *loc) {
  return new (loc) Supernode(n, seed, binary_in);
}

Supernode* Supernode::makeSupernode(const Supernode& s, void *loc) {
  return new (loc) Supernode(s);
}

Supernode::~Supernode() {
}

std::pair<Edge, SampleSketchRet> Supernode::sample() {
  if (out_of_queries()) throw OutOfQueriesException();

  std::pair<vec_t, SampleSketchRet> query_ret = get_sketch(sample_idx++)->query();
  vec_t non_zero = query_ret.first;
  SampleSketchRet ret_code = query_ret.second;
  return {inv_concat_pairing_fn(non_zero), ret_code};
}

std::pair<std::unordered_set<Edge>, SampleSketchRet> Supernode::exhaustive_sample() {
  if (out_of_queries()) throw OutOfQueriesException();

  std::pair<std::unordered_set<vec_t>, SampleSketchRet> query_ret =
      get_sketch(sample_idx++)->exhaustive_query();
  std::unordered_set<Edge> edges(query_ret.first.size());
  for (const auto &query_item: query_ret.first) {
    edges.insert(inv_concat_pairing_fn(query_item));
  }

  SampleSketchRet ret_code = query_ret.second;
  return {edges, ret_code};
}

void Supernode::merge(Supernode &other) {
  sample_idx = std::max(sample_idx, other.sample_idx);
  merged_sketches = std::min(merged_sketches, other.merged_sketches);
  for (size_t i = sample_idx; i < merged_sketches; ++i)
    (*get_sketch(i))+=(*other.get_sketch(i));
}

void Supernode::range_merge(Supernode& other, size_t start_idx, size_t num_merge) {
  // For simplicity we only perform basic error checking here.
  // It's up to the caller to ensure they aren't accessing out of
  // range for a Supernode valid only in a subset of this range.
  if (start_idx >= Supernode::max_sketches) throw OutOfQueriesException();

  sample_idx = std::max(sample_idx, other.sample_idx);
  merged_sketches = start_idx + num_merge;
  for (size_t i = sample_idx; i < merged_sketches; i++)
    (*get_sketch(i))+=(*other.get_sketch(i));
}

void Supernode::update(vec_t upd) {
  for (size_t i = 0; i < num_sketches; ++i)
    get_sketch(i)->update(upd);
}

void Supernode::apply_delta_update(const Supernode* delta_node) {
  std::unique_lock<std::mutex> lk(node_mt);
  for (size_t i = 0; i < num_sketches; ++i) {
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
 * available, but we may want to set it lower if num_sketches is a nice multiple of
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
void Supernode::delta_supernode(uint64_t n, uint64_t seed,
               const std::vector<vec_t> &updates, void *loc) {
  auto delta_node = makeSupernode(n, seed, loc);
  for (size_t i = 0; i < delta_node->num_sketches; ++i) {
    delta_node->get_sketch(i)->batch_update(updates);
  }
}

void Supernode::write_binary(std::ostream& binary_out, bool sparse) {
  SerialType type = FULL;
  binary_out.write((char*) &type, sizeof(type));
  for (size_t i = 0; i < num_sketches; ++i) {
    if (sparse)
      get_sketch(i)->write_sparse_binary(binary_out);
    else
      get_sketch(i)->write_binary(binary_out);
  }
}

void Supernode::write_binary_range(std::ostream &binary_out, uint32_t beg, uint32_t num,
                                   bool sparse) {
  if (beg >= num_sketches) beg = num_sketches - 1;
  if (beg + num > num_sketches) num = num_sketches - beg;
  if (num == 0) num = 1;

  SerialType type = sparse ? SPARSE : PARTIAL;
  binary_out.write((char*) &type, sizeof(type));
  binary_out.write((char*) &beg, sizeof(beg));
  binary_out.write((char*) &num, sizeof(num));
  for (size_t i = beg; i < beg + num; ++i)
    if (sparse)
      get_sketch(i)->write_sparse_binary(binary_out);
    else
      get_sketch(i)->write_binary(binary_out);
}
