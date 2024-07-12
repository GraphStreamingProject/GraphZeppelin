#pragma once
#include <vector>
#include <xxhash.h>
#include <iostream>
#include "types.h"

#pragma pack(push,1)
struct Bucket {
  vec_t alpha;
  vec_hash_t gamma;
};
#pragma pack(pop)

namespace Bucket_Boruvka {
  static constexpr size_t col_hash_bits = sizeof(col_hash_t) * 8;
  /**
   * Hashes the column index and the update index together to determine the depth of an update
   * This is used as a parameter to Bucket::contains.
   * @param update_idx    Vector index to update
   * @param seed_and_col  Combination of seed and column
   * @param max_depth  The maximum depth to return
   * @return              The hash of update_idx using seed_and_col as a seed.
   */
  inline static col_hash_t get_index_depth(const vec_t update_idx, const long seed_and_col,
   const vec_hash_t max_depth);

  /**
   * Hashes the index for checksumming
   * This is used to as a parameter to Bucket::update
   * @param index      Vector index to update
   * @param seed       The seed of the Sketch this Bucket belongs to
   * @return           The depth of the bucket to update
   */
  inline static vec_hash_t get_index_hash(const vec_t index, const long sketch_seed);

  /**
   * Checks whether a Bucket is good, assuming the Bucket contains all elements.
   * @param bucket       The bucket to check
   * @param sketch_seed  The seed of the Sketch this Bucket belongs to.
   * @return             true if this Bucket is good, else false.
   */
  inline static bool is_good(const Bucket &bucket, const long sketch_seed);

  /**
   * Updates a Bucket with the given update index
   * @param bucket      The bucket to update
   * @param update_idx  The update index
   * @param update_hash The hash of the update index, generated with Bucket::index_hash.
   */
  inline static void update(Bucket& bucket, const vec_t update_idx,
                            const vec_hash_t update_hash);
} // namespace Bucket_Boruvka

inline col_hash_t Bucket_Boruvka::get_index_depth(const vec_t update_idx, const long seed_and_col,
                                                  const vec_hash_t max_depth) {
  col_hash_t depth_hash = col_hash(&update_idx, sizeof(vec_t), seed_and_col);
  depth_hash |= (1ull << max_depth); // assert not > max_depth by ORing
  return __builtin_ctzll(depth_hash);
}

inline vec_hash_t Bucket_Boruvka::get_index_hash(const vec_t update_idx, const long sketch_seed) {
  return vec_hash(&update_idx, sizeof(vec_t), sketch_seed);
}

inline bool Bucket_Boruvka::is_good(const Bucket &bucket, const long sketch_seed) {
  return bucket.gamma == get_index_hash(bucket.alpha, sketch_seed);
}

inline void Bucket_Boruvka::update(Bucket& bucket, const vec_t update_idx,
                                   const vec_hash_t update_hash) {
  bucket.alpha ^= update_idx;
  bucket.gamma ^= update_hash;
}
