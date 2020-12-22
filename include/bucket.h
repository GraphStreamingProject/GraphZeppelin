#pragma once
#include <xxhash.h>
#include <iostream>
#include <string>
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;

/*
 * nodes: 2^20
 * n: 2^40, n^2: 2^80 > LONG_LONG_MAX \appro 2^63
 */

/**
 * Represents a bucket in a sketch.
 */
struct Bucket_Boruvka{
// may be n^2/4 = 2^78
  boost::multiprecision::int128_t a = 0;
  boost::multiprecision::int128_t b = 0;
  boost::multiprecision::uint128_t c = 0;

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @return true if the index is NOT hashed to zero.
   */

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @param index
   * @param bucket_seed
   * @param guess_nonzero the number of nonzero elements we assume.
   * @return true if the index is NOT hashed to zero.
   */
  static bool contains(unsigned long long int index, XXH64_hash_t bucket_seed, long
  guess_nonzero) {
    XXH64_hash_t hash = XXH64(&index, 8, bucket_seed);
    if (hash % guess_nonzero == 0)
      return true;
    return false;
  }
};
