#pragma once
#include <vector>
#include "xxhash.h"
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
   * Generates this Bucket's r, in the range [2, large_prime - 2].
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   * @param large_prime Modulus to use in c caluclation.
   * @return This Bucket's r.
   */
  inline static ubucket_t gen_r(const XXH64_hash_t& bucket_seed, const ubucket_t& large_prime) {
    return 2 + bucket_seed % (large_prime - 3);
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
   * @param n Size of the vector being sketched.
   * @param large_prime Modulus to use in c caluclation.
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   * @param r This Bucket's r, generated with gen_r.
   * @param guess_nonzero The guess of nonzero elements in the vector being sketched.
   * @return true if this bucket is good, else false.
   */
  bool is_good(const vec_t& n, const ubucket_t& large_prime, const XXH64_hash_t& bucket_seed, const ubucket_t& r, const vec_t& guess_nonzero) const;

  /**
   * Updates this Bucket with the given Update
   * @param update
   * @param large_prime Modulus to use in c caluclation.
   * @param r This Bucket's r, generated with gen_r.
   */
  void update(const Update& update, const ubucket_t& large_prime, const ubucket_t& r);

  /**
   * Updates this Bucket with the given Update, using cached powers of r
   * @param update
   * @param large_prime Modulus to use in c caluclation.
   * @param r_sq_cache A vector, where r_sq_cache[i] = r^2^i. Generated with
   * PrimeGenerator::gen_sq_cache
   */
  void cached_update(const Update& update, const ubucket_t& large_prime, const std::vector<ubucket_t>& r_sq_cache);

  friend bool operator== (const Bucket_Boruvka &bucket1, const Bucket_Boruvka &bucket2);
  friend bool operator!= (const Bucket_Boruvka &bucket1, const Bucket_Boruvka &bucket2);
};

const Bucket_Boruvka BUCKET_ZERO = {0,0,0};
