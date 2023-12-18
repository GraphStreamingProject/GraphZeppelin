//#include <stdexcept>
#include <limits>
#include <cmath>
#include <cassert>
#include <graph_zeppelin_common.h>
#include "../include/util.h"

typedef uint32_t ul;
typedef uint64_t ull;

constexpr ull ULLMAX = std::numeric_limits<ull>::max();
constexpr uint8_t num_bits = sizeof(node_id_t) * 8;

unsigned long long int double_to_ull(double d, double epsilon) {
  return (unsigned long long) (d + epsilon);
}

edge_id_t nondirectional_non_self_edge_pairing_fn(node_id_t i, node_id_t j) {
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

Edge inv_nondir_non_self_edge_pairing_fn(uint64_t idx) {
  // we ignore possible overflow
  ull eidx = 8ull*idx + 1ull;
  eidx = sqrt(eidx)+1ull;
  eidx/=2ull;
  ull i,j = (ull) eidx;
  if ((j & 1ull) == 0ull) i = idx-(j>>1ull)*(j-1ull);
  else i = idx-j*((j-1ull)>>1ull);
  return {(node_id_t)i, (node_id_t)j};
}

edge_id_t concat_pairing_fn(node_id_t i, node_id_t j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  return ((edge_id_t)i << num_bits) | j;
}

Edge inv_concat_pairing_fn(ull idx) {
  node_id_t j = idx & 0xFFFFFFFF;
  node_id_t i = idx >> num_bits;
  return {i, j};
}
