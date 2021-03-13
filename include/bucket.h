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
  inline static XXH64_hash_t gen_bucket_seed(const unsigned bucket_id, long seed);

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @param index
   * @param bucket_seed
   * @param guess_nonzero
   * @return true if the index is NOT hashed to zero.
   */
  inline bool contains(const vec_t& index, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const;

  /**
   * Checks whether this Bucket is good.
   * TODO
   * @param n Size of the vector being sketched.
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   * @param guess_nonzero The guess of nonzero elements in the vector being sketched.
   * @return true if this bucket is good, else false.
   */
  inline bool is_good(const vec_t& n, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const;

  /**
   * Updates this Bucket with the given Update
   * TODO
   * @param update
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   */
  inline void update(const vec_t& update_idx, const XXH64_hash_t& update_hash);
};


inline XXH64_hash_t Bucket_Boruvka::gen_bucket_seed(const unsigned bucket_id, long seed) {
  return bucket_id * ((seed & ~7ULL) | 5ULL) + ((seed & 7ULL) << 2 | 1);
}

inline bool Bucket_Boruvka::contains(const vec_t& index, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const {
  XXH64_hash_t hash = index * ((bucket_seed & ~7ULL) | 5ULL) + ((bucket_seed & 7ULL) << 2 | 1);
  if (hash % guess_nonzero == 0)
    return true;
  return false;
}

inline bool Bucket_Boruvka::is_good(const vec_t& n, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const {
  return a < n
    && c == a * ((bucket_seed & ~7ULL) | 5ULL) + ((bucket_seed & 7ULL) << 1 | 0x11)
    && contains(a, bucket_seed, guess_nonzero);
}

inline void Bucket_Boruvka::update(const vec_t& update_idx, const XXH64_hash_t& bucket_seed) {
  a ^= update_idx;
  c ^= update_idx * ((bucket_seed & ~7ULL) | 5ULL) + ((bucket_seed & 7ULL) << 1 | 0x11);
}
