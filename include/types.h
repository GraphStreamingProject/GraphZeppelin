#pragma once
#include <xxhash.h>

#ifdef USE_NATIVE_F
//Small types
typedef uint32_t node_t; //Max graph vertices is 92682
typedef uint32_t vec_t; //Max sketch vector size is 2^32 - 1
typedef uint32_t vec_hash_t;
typedef uint32_t col_hash_t;
#else //USE_NATIVE_F
//Big types
typedef uint64_t node_t; //Max graph vertices is 6074001000
typedef uint64_t vec_t; //Max sketch vector size is 2^64 - 1
typedef uint32_t vec_hash_t;
typedef uint32_t col_hash_t;
#endif //USE_NATIVE_F
