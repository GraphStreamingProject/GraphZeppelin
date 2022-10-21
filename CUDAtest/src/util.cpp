#include "../include/util.h"

typedef uint32_t ul;
typedef uint64_t ull;

const uint8_t num_bits = sizeof(node_id_t) * 8;

unsigned long long int double_to_ull(double d, double epsilon) {
    return (unsigned long long) (d + epsilon);
}

std::pair<ul, ul> inv_concat_pairing_fn(ull idx) {
    ul j = idx & 0xFFFFFFFF;
    ul i = idx >> num_bits;
    return {i, j};
}