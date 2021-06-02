#pragma once
#include <cmath>
#include <exception>
#include <iostream>
#include <vector>
#include "bucket.h"
#include "types.h"
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
  // Seed used for hashing operations in this sketch.
  const long seed;
  // Length of the vector this is sketching.
  const vec_t n;
  // Factor for how many buckets there are in this sketch.
  const double num_bucket_factor;
  // Buckets of this sketch.
  // Length is bucket_gen(n, num_bucket_factor) * guess_gen(n).
  // For buckets[i * guess_gen(n) + j], the bucket has a 1/2^j probability
  // of containing an index.
  std::vector<vec_t> bucket_a;
  std::vector<vec_hash_t> bucket_c;
  // Flag to keep track if this sketch has already been queried.
  bool already_quered = false;

  FRIEND_TEST(SketchTestSuite, TestExceptions);
  FRIEND_TEST(EXPR_Parallelism, N10kU100k);


  friend void test_continuous(unsigned long vec_size, unsigned long updates_per_sample,
    unsigned long samples, double num_bucket_factor);

public:
  /**
   * Construct a sketch of a vector of size n
   * @param n Length of the vector to sketch.
   * @param seed Seed to use for hashing operations
   * @param num_bucket_factor Factor to scale the number of buckets in this sketch
   */
  Sketch(vec_t n, long seed, double num_bucket_factor = 1);

  /**
   * Update a sketch based on information about one of its indices.
   * @param update the point update.
   */
  void update(const vec_t& update_idx);

  /**
   * Update a sketch given a batch of updates
   * @param updates A vector of updates
   */
  void batch_update(const std::vector<vec_t>& updates);

  /**
   * Function to query a sketch.
   * @return                        an index.
   * @throws MultipleQueryException if the sketch has already been queried.
   * @throws NoGoodBucketException  if there are no good buckets to choose an
   *                                index from.
   */
  vec_t query();

  friend Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2);
  friend bool operator== (const Sketch &sketch1, const Sketch &sketch2);
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

