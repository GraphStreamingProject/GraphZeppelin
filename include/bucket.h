#pragma once
#include <xxhash.h>
#include <iostream>
#include <string>
#include <boost/multiprecision/cpp_int.hpp>
#include "int127.h"

using namespace std;

/*
 * nodes: 2^20
 * n: 2^40, n^2: 2^80 > LONG_LONG_MAX \approx 2^63
 */

/**
 * Represents a bucket in a sketch.
 */
struct Bucket_Boruvka{
  int127 a = 0;
  int127 b = 0;
  boost::multiprecision::uint128_t c = 0;
  long long vc = 0;

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @param index
   * @param bucket_seed
   * @param guess_nonzero the number of nonzero elements we assume.
   * @return true if the index is NOT hashed to zero.
   */
  static bool contains(uint128_t index, XXH64_hash_t bucket_seed, long
  guess_nonzero) {
    XXH64_hash_t hash = XXH64(&index, 8, bucket_seed);
    if (hash % guess_nonzero == 0)
      return true;
    return false;
  }
};
