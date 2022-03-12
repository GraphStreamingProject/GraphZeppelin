#pragma once
#include <xxhash.h>
#include <graph_zeppelin_common.h>

typedef uint64_t col_hash_t;

static unsigned int XXH3_32(const void* a, unsigned long b, unsigned
int c) {
  return (unsigned int) XXH3_64bits_withSeed(a,b,c);
}
static const auto& vec_hash = XXH3_32;
static const auto& col_hash = XXH3_64bits_withSeed;

enum UpdateType {
  INSERT = 0,
  DELETE = 1,
};
