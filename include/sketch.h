#pragma once
#include <vector>
#include <exception>
#include "update.h"
#include "bucket.h"
#include "util.h"
#include <gtest/gtest_prod.h>

#define bucket_gen(x) double_to_ull(log2(x)+1)
#define guess_gen(x) double_to_ull(log2(x)+2)

using namespace std;

/**
 * An implementation of a "sketch" as defined in the L0 algorithm.
 * Note a sketch may only be queried once. Attempting to query multiple times will
 * raise an error.
 */
class Sketch {
  const long seed;
  const uint64_t n;
  std::vector<Bucket_Boruvka> buckets;
  const uint128_t large_prime;
  bool already_quered = false;

  FRIEND_TEST(SketchTestSuite, TestExceptions);

  //Initialize a sketch of a vector of size n
public:
  Sketch(uint64_t n, long seed);

  /**
   * Update a sketch based on information about one of its indices.
   * @param update the point update.
   */
  void update(Update update);

  /**
   * Update a sketch given a batch of updates
   * @param begin a ForwardIterator to the first update
   * @param end a ForwardIterator to after the last update
   */
  template <typename ForwardIterator>
  void Sketch::batch_update(ForwardIterator begin, ForwardIterator end) {
    const unsigned long long int num_buckets = bucket_gen(n);
    const unsigned long long int num_guesses = guess_gen(n);
    for (unsigned i = 0; i < num_buckets; ++i) {
      for (unsigned j = 0; j < num_guesses; ++j) {
        unsigned bucket_id = i * num_guesses + j;
        XXH64_hash_t bucket_seed = XXH64(&bucket_id, sizeof(bucket_id), seed);
        int128_t r = 2 +  bucket_seed % (large_prime - 3);
        for (auto it = begin; it != end; it++) {
          const Update& update = *it;
          Bucket& bucket = buckets[bucket_id];
          if (bucket.contains(update.index+1, bucket_seed, 1 << j)){
            bucket.a += update.delta;
            bucket.b += update.delta*(update.index+1); // deals with updates whose indices are 0
            bucket.c = static_cast<uint128_t>(
                  (static_cast<int128_t>(buckets.c)
                  + static_cast<int128_t>(large_prime)
                  + (update.delta*PrimeGenerator::power(r,(uint128_t) update
                  .index+1, large_prime) % static_cast<int128_t>(large_prime)))
                  % large_prime
                  );
          }
        }
      }
    }
  }


  /**
   * Function to query a sketch.
   * @return                        an index in the form of an Update.
   * @throws MultipleQueryException if the sketch has already been queried.
   * @throws NoGoodBucketException  if there are no good buckets to choose an
   *                                index from.
   */
  Update query();

  friend Sketch operator+ (const Sketch &sketch1, const Sketch &sketch2);
  friend Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2);
  friend Sketch operator* (const Sketch &sketch1, long scaling_factor );
  friend std::ostream& operator<< (std::ostream &os, const Sketch &sketch);
};

class AllBucketsZeroException : public exception {
public:
  virtual const char* what() const throw() {
    return "All buckets zero";
  }
};

class MultipleQueryException : public exception {
public:
  virtual const char* what() const throw() {
    return "This sketch has already been sampled!";
  }
};

class NoGoodBucketException : public exception {
public:
  virtual const char* what() const throw() {
    return "Found no good bucket!";
  }
};
