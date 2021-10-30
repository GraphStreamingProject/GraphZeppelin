#pragma once
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <vector>
#include <memory>
#include <mutex>
#include "../bucket.h"
#include "../types.h"
#include "../util.h"
#include <gtest/gtest_prod.h>

#define bucket_gen(x, c) double_to_ull((c)*(log2(x)+1))
#define guess_gen(x) double_to_ull(log2(x)+1)

/**
 * An implementation of a "sketch" as defined in the L0 algorithm.
 * Note a sketch may only be queried once. Attempting to query multiple times will
 * raise an error.
 */
class Sketch {
private:
  static double num_bucket_factor; // Factor for how many buckets there are in this sketch.
  static vec_t n;                  // Length of the vector this is sketching.
  static size_t num_elems;         // length of our actual arrays in number of elements
  static size_t num_buckets;       // Portion of array length, number of buckets
  static size_t num_guesses;       // Portion of array length, number of guesses

  // Seed used for hashing operations in this sketch.
  const long seed;
  // pointers to buckets
  vec_t*      bucket_a;
  vec_hash_t* bucket_c;

  // Flag to keep track if this sketch has already been queried.
  bool already_quered = false;

  FRIEND_TEST(SketchTestSuite, TestExceptions);
  FRIEND_TEST(EXPR_Parallelism, N10kU100k);

  
  // Buckets of this sketch.
  // Length is bucket_gen(n, num_bucket_factor) * guess_gen(n).
  // For buckets[i * guess_gen(n) + j], the bucket has a 1/2^j probability
  // of containing an index. The first two are pointers into the buckets array.
  char buckets[1];

  /**
   * Construct a sketch of a vector of size n
   * @param n Length of the vector to sketch. (static variable)
   * @param seed Seed to use for hashing operations
   * @param num_bucket_factor Factor to scale the number of buckets in this sketch (static variable)
   */
  Sketch(long seed);
  Sketch(long seed, std::fstream &binary_in);
  Sketch(const Sketch& s);

public:
  static Sketch* makeSketch(void* loc, long seed);
  static Sketch* makeSketch(void* loc, long seed, std::fstream &binary_in);
  static Sketch* makeSketch(void* loc, long seed, double num_bucket_factor, std::fstream &binary_in);
  static Sketch* makeSketch(void* loc, const Sketch& s);
  
  // configure the static variables of sketches
  inline static void configure(size_t _n, double _factor) {
    n = _n;
    num_bucket_factor = _factor;
    num_buckets = bucket_gen(n, num_bucket_factor);
    num_guesses = guess_gen(n);
    num_elems = num_buckets * num_guesses + 1;
  }

  inline static size_t sketchSizeof()
  { return sizeof(Sketch) + num_elems * (sizeof(vec_t) + sizeof(vec_hash_t)) - sizeof(char); }
  
  inline static double get_bucket_factor() 
  { return num_bucket_factor; }
  /**
   * Update a sketch based on information about one of its indices.
   * @param update the point update.
   */
  void update(const vec_t& update_idx);

  /**
   * Update a sketch given a batch of updates.
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

  /**
   * Operator to add a sketch to another one in-place. Guaranteed to be
   * thread-safe for the sketch being added to. It is up to the user to
   * handle concurrency of the other sketch.
   * @param sketch1 the one being added to.
   * @param sketch2 the one being added.
   * @return a reference to the combined sketch.
   */
  friend Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2);
  friend bool operator== (const Sketch &sketch1, const Sketch &sketch2);
  friend std::ostream& operator<< (std::ostream &os, const Sketch &sketch);

  /**
   * Serialize the sketch to a binary output stream.
   * @param out the stream to write to.
   */
  void write_binary(std::fstream& binary_out);
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
