#pragma once
#include <vector>
#include <xxhash.h>
#include "types.h"
#include "update.h"

/*
 * nodes: 2^20
 * n: 2^40, n^2: 2^80 > LONG_LONG_MAX \approx 2^63
 */

/**
 * Represents a bucket in a sketch.
 */
struct Bucket_Boruvka {
  bucket_t a = 0;
  bucket_t b = 0;
  ubucket_t c = 0;

  static XXH64_hash_t gen_bucket_seed(const unsigned bucket_id, long seed);
  static ubucket_t gen_r(const XXH64_hash_t& bucket_seed, const ubucket_t& large_prime);
  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @param index
   * @param bucket_seed
   * @param guess_nonzero
   * @return true if the index is NOT hashed to zero.
   */
  bool contains(const vec_t& index, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const;
  bool is_good(const vec_t& n, const ubucket_t& large_prime, const XXH64_hash_t& bucket_seed, const ubucket_t& r, const vec_t& guess_nonzero) const;
  void update(const Update& update, const ubucket_t& large_prime, const ubucket_t& r);
  void cached_update(const Update& update, const ubucket_t& large_prime, const std::vector<ubucket_t>& r_sq_cache);
};

