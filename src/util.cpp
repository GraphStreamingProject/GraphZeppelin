#include <stdexcept>
#include "../include/util.h"
#include "../include/graph_worker.h"
#include "../include/graph.h"

typedef uint32_t ul;
typedef uint64_t ull;

const ull ULLMAX = std::numeric_limits<ull>::max();
const uint8_t num_bits = sizeof(node_id_t) * 8;

unsigned long long int double_to_ull(double d, double epsilon) {
  return (unsigned long long) (d + epsilon);
}

uint64_t nondirectional_non_self_edge_pairing_fn(uint32_t i, uint32_t j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  ull jm = j-1ull;
  if ((j & 1ull) == 0ull) j>>=1ull;
  else jm>>=1ull;
//  if (ULLMAX/j < jm)
//    throw std::overflow_error("Computation would overflow unsigned long long max");
  j*=jm;
//  if (ULLMAX-j < i)
//    throw std::overflow_error("Computation would overflow unsigned long long max");
  return i+j;
}

std::pair<uint32_t , uint32_t> inv_nondir_non_self_edge_pairing_fn(uint64_t idx) {
  // we ignore possible overflow
  ull eidx = 8ull*idx + 1ull;
  eidx = sqrt(eidx)+1ull;
  eidx/=2ull;
  ull i,j = (ull) eidx;
  if ((j & 1ull) == 0ull) i = idx-(j>>1ull)*(j-1ull);
  else i = idx-j*((j-1ull)>>1ull);
  return {i, j};
}

ull concat_pairing_fn(ul i, ul j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  return ((ull)i << num_bits) | j;
}

std::pair<ul, ul> inv_concat_pairing_fn(ull idx) {
  ul j = idx & 0xFFFFFFFF;
  ul i = idx >> num_bits;
  return {i, j};
}
