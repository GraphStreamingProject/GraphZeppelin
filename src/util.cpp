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

ull nondirectional_non_self_edge_pairing_fn(ul i, ul j) {
  // swap i,j if necessary
  if (i > j) {
    std::swap(i,j);
  }
  return ((ull)i << num_bits) | j;
}

std::pair<ul, ul> inv_nondir_non_self_edge_pairing_fn(ull idx) {
  ul j = idx & 0xFFFFFFFF;
  ul i = idx >> num_bits;
  return {i, j};
}
