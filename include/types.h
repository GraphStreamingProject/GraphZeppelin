#pragma once
#include <xxhash.h>
#include <graph_zeppelin_common.h>

typedef uint64_t col_hash_t;
static const auto& vec_hash = XXH32;
static const auto& col_hash = XXH64;

enum UpdateType {
  INSERT = 0,
  DELETE = 1,
};
