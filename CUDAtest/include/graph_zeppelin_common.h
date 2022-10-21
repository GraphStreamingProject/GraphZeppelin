#pragma once

#include <cstdint>

// graph
typedef uint32_t node_id_t; // Max graph vertices is 2^32 - 1 = 4 294 967 295
typedef uint64_t edge_id_t; // Max number edges

// sketching
typedef uint64_t vec_t; //Max sketch vector size is 2^64 - 1
typedef uint32_t vec_hash_t;
