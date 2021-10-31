#pragma once
#include <xxhash.h>

typedef uint64_t node_t; //Max graph vertices is 6 074 001 000
typedef uint64_t vec_t; //Max sketch vector size is 2^64 - 1
typedef uint32_t node_id_t;
typedef uint32_t vec_hash_t;
typedef uint32_t col_hash_t;
static const auto& vec_hash = XXH32;
static const auto& col_hash = XXH32;

enum UpdateType {
  INSERT = 0,
  DELETE = 1,
};
