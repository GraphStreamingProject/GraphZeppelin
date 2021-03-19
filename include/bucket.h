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

  inline static XXH64_hash_t col_index_hash(const unsigned bucket_col, const vec_t& update_idx, const long sketch_seed);

  inline static XXH64_hash_t index_hash(const vec_t& index, long seed);

  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @param index
   * @param bucket_seed
   * @param guess_nonzero
   * @return true if the index is NOT hashed to zero.
   */
  inline bool contains(const XXH64_hash_t& index_hash, const vec_t& guess_nonzero) const;

  /**
   * Checks whether this Bucket is good.
   * TODO
   * @param n Size of the vector being sketched.
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   * @param guess_nonzero The guess of nonzero elements in the vector being sketched.
   * @return true if this bucket is good, else false.
   */
  inline bool is_good(const vec_t& n, const unsigned bucket_col, const vec_t& guess_nonzero, const long& sketch_seed) const;

  /**
   * Updates this Bucket with the given Update
   * TODO
   * @param update
   * @param bucket_seed This Bucket's seed, generated with gen_bucket_seed.
   */
  inline void update(const vec_t& update_idx, const XXH64_hash_t& update_hash);
};

inline XXH64_hash_t Bucket_Boruvka::col_index_hash(const unsigned bucket_col, const vec_t& update_idx, const long sketch_seed) {
  struct {
    unsigned bucket_col;
    vec_t update_idx;
  } __attribute__((packed)) buf = {bucket_col, update_idx};
  return XXH64(&buf, sizeof(buf), sketch_seed);
}

inline XXH64_hash_t Bucket_Boruvka::index_hash(const vec_t& index, long sketch_seed) {
  return XXH64(&index, sizeof(index), sketch_seed);
}

inline bool Bucket_Boruvka::contains(const XXH64_hash_t& col_index_hash, const vec_t& guess_nonzero) const {
  return col_index_hash % guess_nonzero == 0;
}

inline bool Bucket_Boruvka::is_good(const vec_t& n, const unsigned bucket_col, const vec_t& guess_nonzero, const long& sketch_seed) const {
  return a < n && c == index_hash(a, sketch_seed)
    && contains(col_index_hash(bucket_col, a, sketch_seed), guess_nonzero);
}

inline void Bucket_Boruvka::update(const vec_t& update_idx, const XXH64_hash_t& update_hash) {
  a ^= update_idx;
  c ^= update_hash;
}
