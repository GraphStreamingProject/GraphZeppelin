#pragma once
#include <xxhash.h>
#include <graph_zeppelin_common.h>
#include <functional>
#include <graph_stream.h>

typedef uint64_t col_hash_t;
static const auto& vec_hash = XXH3_64bits_withSeed;
static const auto& col_hash = XXH3_64bits_withSeed;

// Graph Stream Updates are parsed into the GraphUpdate type for more convinient processing
struct GraphUpdate {
  Edge edge;
  UpdateType type;
};
