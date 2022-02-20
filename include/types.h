#pragma once
#include <graph_zeppelin_common.h>
#include "l0_sampling/pmshash.h"

typedef uint64_t col_hash_t;
static const auto& vec_hash = mmp_hash_32;
static const auto& col_hash = mmp_hash_64;

enum UpdateType {
  INSERT = 0,
  DELETE = 1,
};
