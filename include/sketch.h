#pragma once
#include <cmath>
#include <exception>
#include <iostream>
#include <vector>
#include "bucket.h"
#include "prime_generator.h"
#include "types.h"
#include "update.h"
#include "util.h"
#include <gtest/gtest_prod.h>

#define bucket_gen(x, c) double_to_ull((c)*(log2(x)+1))
#define guess_gen(x) double_to_ull(log2(x)+2)

/**
 * An implementation of a "sketch" as defined in the L0 algorithm.
 * Note a sketch may only be queried once. Attempting to query multiple times will
 * raise an error.
 */
class Sketch {
  const long seed;
  const vec_t n;
  const double num_bucket_factor;
  std::vector<Bucket_Boruvka> buckets;
  bool already_quered = false;

  FRIEND_TEST(SketchTestSuite, TestExceptions);
  FRIEND_TEST(SketchTestSuite, TestBatchUpdate);

  //Initialize a sketch of a vector of size n
public:
  Sketch(vec_t n, long seed, double num_bucket_factor = 1);

  /**
   * Update a sketch based on information about one of its indices.
   * @param update the point update.
   */
  void update(const vec_t& update_idx);

  /**
   * Update a sketch given a batch of updates
   * @param begin a ForwardIterator to the first update
   * @param end a ForwardIterator to after the last update
   */
  template <typename ForwardIterator>
  void batch_update(ForwardIterator begin, ForwardIterator end);

  /**
   * Function to query a sketch.
   * @return                        an index in the form of an Update.
   * @throws MultipleQueryException if the sketch has already been queried.
   * @throws NoGoodBucketException  if there are no good buckets to choose an
   *                                index from.
   */
  vec_t query();

  friend Sketch operator+ (const Sketch &sketch1, const Sketch &sketch2);
  friend Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2);
  friend Sketch operator* (const Sketch &sketch1, long scaling_factor );
  friend std::ostream& operator<< (std::ostream &os, const Sketch &sketch);
};

class AllBucketsZeroException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "All buckets zero";
  }
};

class MultipleQueryException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "This sketch has already been sampled!";
  }
};

class NoGoodBucketException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "Found no good bucket!";
  }
};

template <typename ForwardIterator>
void Sketch::batch_update(ForwardIterator begin, ForwardIterator end) {
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      Bucket_Boruvka& bucket = buckets[bucket_id];
      XXH64_hash_t bucket_seed = Bucket_Boruvka::gen_bucket_seed(bucket_id, seed);
      for (auto it = begin; it != end; it++) {
        const vec_t& update_idx = *it;
        if (bucket.contains(update_idx, bucket_seed, 1 << j)) {
          XXH64_hash_t update_hash = XXH64(&update_idx, sizeof(update_idx), seed);
          bucket.update(update_idx, update_hash);
        }
      }
    }
  }
}

