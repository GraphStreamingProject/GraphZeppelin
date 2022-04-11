#pragma once
#include <xxhash.h>
#include <graph_zeppelin_common.h>

typedef uint64_t col_hash_t;
static const auto& vec_hash = XXH32;
static const auto& col_hash = XXH64;

enum UpdateType {
  INSERT = 0,
  DELETE = 1,
  END_OF_FILE = 2
}; // special type to indicate that there is no more data in the stream

typedef int64_t bucket_t;
typedef uint64_t ubucket_t;
typedef struct {
  uint64_t low64;
  uint64_t high64;
} uint128_t;
#ifdef __SIZEOF_INT128__
typedef __int128 bucket_prod_t;
typedef unsigned __int128 ubucket_prod_t;
#else
typedef uint128_t bucket_prod_t;
#endif