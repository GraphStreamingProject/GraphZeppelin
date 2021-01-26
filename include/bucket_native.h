#pragma once
#include <xxhash.h>
#include <iostream>
#include <string>
using namespace std;

/**
 * Native representation of a bucket in a sketch.
 */
struct Bucket{
  int64_t a = 0;
  int64_t b = 0;
  int64_t c = 0;

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @return true if the index is NOT hashed to zero.
   */
  static bool contains(uint64_t index, XXH64_hash_t bucket_seed, long
  guess_nonzero) {
    index++;
    XXH64_hash_t hash = XXH64(&index, sizeof(index), bucket_seed);
    if (hash % guess_nonzero == 0)
      return true;
    return false;
  }
};
