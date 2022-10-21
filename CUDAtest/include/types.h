#pragma once
#include <xxhash.h>
#include "./graph_zeppelin_common.h"

typedef uint64_t col_hash_t;
static const auto& vec_hash = XXH32;
static const auto& col_hash = XXH64;

// Is a stream update an insertion or a deletion
// NXT_QUERY: special type to indicate getting more data would cross a query boundary
// END_OF_FILE: special type to indicate that there is no more data in the stream
enum UpdateType {
    INSERT = 0,
    DELETE = 1,
    NXT_QUERY = 2,
    END_OF_FILE = 3
}; 