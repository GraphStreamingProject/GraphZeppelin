#pragma once
#include <vector>
#include <xxhash.h>
#include <iostream>
#include <bitset>
#include "types.h"

#pragma pack(push,1)
struct Bucket {
  vec_t alpha;
  vec_hash_t gamma;
  Bucket operator^(const Bucket &rhs) {
    return {alpha ^= rhs.alpha,
            gamma ^= rhs.gamma};
  };
  void operator^=(const Bucket &rhs) {
    alpha ^= rhs.alpha;
    gamma ^= rhs.gamma;
  };
  bool operator==(const Bucket &rhs) const {
    return alpha == rhs.alpha && gamma == rhs.gamma;
  };
  bool operator!=(const Bucket &rhs) const {
    return alpha != rhs.alpha || gamma != rhs.gamma;
  };
  friend std::ostream &operator<<(std::ostream &os, const Bucket &b) {
    os << "(a: " << b.alpha << " g: " << b.gamma << ")";
    return os;
  }
};
#pragma pack(pop)

namespace Bucket_Boruvka {
  static constexpr size_t col_hash_bits = sizeof(col_hash_t) * 8;

  /**
   * Returns whether or not a bucket is empty.
   * @param bucket    Bucket to check for empty.
   * @return          With high probability, return whether or not a given bucket is empty. 
   */
  inline static bool is_empty(const Bucket &bucket);
  /**
   * Hashes the column index and the update index together to determine the depth of an update
   * This is used as a parameter to Bucket::contains.
   * @param update_idx    Vector index to update
   * @param seed_and_col  Combination of seed and column
   * @param max_depth  The maximum depth to return
   * @return              The hash of update_idx using seed_and_col as a seed.
   */
  inline static col_hash_t get_index_depth(const vec_t update_idx, const long seed, const long col,
   const vec_hash_t max_depth);

  inline static void get_all_index_depths(
    const vec_t update_idx,
    uint32_t *depths_buffer, 
    const long seed, 
    const long num_columns,
    const vec_hash_t max_depth
    ) {
    if (num_columns == 0) return;
    XXH128_hash_t *hashes = (XXH128_hash_t*) depths_buffer;
    #pragma omp simd
    for (int col = 0; col <= num_columns -4; col+=4) {
      auto hash = XXH3_128bits_withSeed(&update_idx, sizeof(vec_t), seed + 5 * (col / 4) );
      hashes[col / 4] = hash;
    }
    for (int col = 0;  col < num_columns - 4; col+=4) {
      auto hash = hashes[col / 4];
      depths_buffer[col] = (uint32_t) (hash.low64 >> 32);
      depths_buffer[col+1] = (uint32_t) (hash.low64 & 0xFFFFFFFF);
      depths_buffer[col+2] = (uint32_t) (hash.high64 >> 32);
      depths_buffer[col+3] = (uint32_t) (hash.high64 & 0xFFFFFFFF);
    }
    int col=0;
    for (; col < num_columns -4; col++) {
      depths_buffer[col] |= (uint32_t) (1ull << max_depth); // assert not > max_depth by ORing
      depths_buffer[col] = (uint32_t) (__builtin_ctzll(depths_buffer[col]));
    }
    col-= 1;
    for (; col< num_columns; col++) {
      depths_buffer[col] = (uint32_t) get_index_depth(update_idx, seed, col, max_depth);
    }
  }

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

inline bool Bucket_Boruvka::is_empty(const Bucket &bucket) {
  return (bucket.alpha | bucket.gamma) == 0;
}

inline col_hash_t Bucket_Boruvka::get_index_depth(const vec_t update_idx, const long seed, const long col,
                                                  const vec_hash_t max_depth) {
  auto hash = XXH3_128bits_withSeed(&update_idx, sizeof(vec_t), seed + 5 * (col / 4) );
  // auto hash = XXH3_128bits_withSeed(&update_idx, sizeof(vec_t), seed + 5 * (col) );
  col_hash_t depth_hash = 0;
  int offset = col % 4;
  if (offset == 0) {
    depth_hash = (uint32_t) (hash.low64 >> 32);
  } else if (offset == 1) {
    depth_hash = (uint32_t) (hash.low64 & 0xFFFFFFFF);
  } else if (offset == 2) {
    depth_hash = (uint32_t) (hash.high64 >> 32);
  } else if (offset == 3) {
    depth_hash = (uint32_t) (hash.high64 & 0xFFFFFFFF);
  }
  // col_hash_t depth_hash = hash.low64;
  // std::cout << "depth_hash: " << std::bitset<32>(depth_hash) << std::endl;
  depth_hash |= (1ull << max_depth); // assert not > max_depth by ORing
  return __builtin_ctzll(depth_hash);
}

inline vec_hash_t Bucket_Boruvka::get_index_hash(const vec_t update_idx, const long sketch_seed) {
  return (XXH3_128bits_withSeed (&update_idx, sizeof(vec_t), sketch_seed)).low64;
}


inline bool Bucket_Boruvka::is_good(const Bucket &bucket, const long sketch_seed) {
  return !Bucket_Boruvka::is_empty(bucket) && bucket.gamma == get_index_hash(bucket.alpha, sketch_seed);
}

inline void Bucket_Boruvka::update(Bucket& bucket, const vec_t update_idx,
                                   const vec_hash_t update_hash) {
  bucket.alpha ^= update_idx;
  bucket.gamma ^= update_hash;
}
