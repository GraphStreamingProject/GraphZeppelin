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
  vec_t a = 0;
  XXH64_hash_t c = 0;

  /**
   * Generates this Bucket's seed.
   * @param bucket_id The id of this bucket.
   * @param seed The seed of the sketch containing this bucket.
   * @return This Bucket's seed.
   */
  inline static XXH64_hash_t gen_bucket_seed(const unsigned bucket_id, long seed) {
    return XXH64(&bucket_id, sizeof(bucket_id), seed);
  }

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @param index
   * @param bucket_seed
   * @param guess_nonzero
   * @return true if the index is NOT hashed to zero.
   */
  bool contains(const vec_t& index, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const;

  /**
   * Checks whether this Bucket is good.
   * TODO
   * @param n Size of the vector being sketched.
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   * @param guess_nonzero The guess of nonzero elements in the vector being sketched.
   * @return true if this bucket is good, else false.
   */
  bool is_good(const vec_t& n, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero, const long& sketch_seed) const;

  /**
   * Updates this Bucket with the given Update
   * TODO
   * @param update
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   */
  void update(const vec_t& update_idx, const XXH64_hash_t& update_hash);
};

