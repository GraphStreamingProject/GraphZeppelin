#pragma once
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <vector>
#include <memory>
#include <mutex>
#include <utility>
#include "bucket.h"
#include "../types.h"
#include "../util.h"
#include <gtest/gtest_prod.h>

// max number of non-zeroes in vector is n/2*n/2=n^2/4
#define guess_gen(x) double_to_ull(log2(x) - 2)
#define bucket_gen(d) double_to_ull((log2(d)+1))

enum SampleSketchRet {
  GOOD,  // querying this sketch returned a single non-zero value
  ZERO,  // querying this sketch returned that there are no non-zero values
  FAIL   // querying this sketch failed to produce a single non-zero value
};

/**
 * An implementation of a "sketch" as defined in the L0 algorithm.
 * Note a sketch may only be queried once. Attempting to query multiple times will
 * raise an error.
 */
class Sketch {
private:
  static vec_t failure_factor;     // Failure factor determines number of columns in sketch. Pr(failure) = 1 / factor
  static vec_t n;                  // Length of the vector this is sketching.
  static size_t num_elems;         // length of our actual arrays in number of elements
  static size_t num_buckets;       // Portion of array length, number of buckets
  static size_t num_guesses;       // Portion of array length, number of guesses

  // Seed used for hashing operations in this sketch.
  const uint64_t seed;
  const ubucket_t large_prime;

  Bucket_Boruvka* buckets;

  // Flag to keep track if this sketch has already been queried.
  bool already_queried = false;

  FRIEND_TEST(SketchTestSuite, TestExceptions);
  FRIEND_TEST(EXPR_Parallelism, N10kU100k);

  
  // Buckets of this sketch.
  // Length is bucket_gen(failure_factor) * guess_gen(n).
  // For buckets[i * guess_gen(n) + j], the bucket has a 1/2^j probability
  // of containing an index. The first two are pointers into the buckets array.
  char _bucket_data[1];

  // private constructors -- use makeSketch
  Sketch(uint64_t seed);
  Sketch(uint64_t seed, std::istream &binary_in);
  Sketch(const Sketch& s);

public:
  /**
   * Construct a sketch of a vector of size n
   * The optional parameters are used when building a sketch from a file
   * @param loc        A pointer to a location in memory where the caller would like the sketch constructed
   * @param seed       Seed to use for hashing operations
   * @param binary_in  (Optional) A file which holds an encoding of a sketch
   * @return           A pointer to a newly constructed sketch 
   */
  static Sketch* makeSketch(void* loc, uint64_t seed);
  static Sketch* makeSketch(void* loc, uint64_t seed, std::istream &binary_in);
  
  /**
   * Copy constructor to create a sketch from another
   * @param loc   A pointer to a location in memory where the caller would like the sketch constructed
   * @param s     A sketch to make a copy of
   * @return      A pointer to a newly constructed sketch 
   */
  static Sketch* makeSketch(void* loc, const Sketch& s);
  
  /* configure the static variables of sketches
   * @param n               Length of the vector to sketch. (static variable)
   * @param failure_factor  The rate at which an individual sketch is allowed to fail (determines column width)
   * @return nothing
   */
  inline static void configure(vec_t _n, vec_t _factor) {
    n = _n;
    failure_factor = _factor;
    num_buckets = bucket_gen(failure_factor);
    num_guesses = guess_gen(n);
    num_elems = num_buckets * num_guesses + 1;
  }

  inline static size_t sketchSizeof()
  { return sizeof(Sketch) + num_elems * sizeof(Bucket_Boruvka) - sizeof(char); }
  
  inline static vec_t get_failure_factor() 
  { return failure_factor; }

  inline void reset_queried() 
  { already_queried = false; }

  /**
   * Update a sketch based on information about one of its indices.
   * @param update the point update.
   */
  void update(const Update& update);

  /**
   * Update a sketch given a batch of updates.
   * @param updates A vector of updates
   */
  void batch_update(const std::vector<Update>& updates);


  /**
   * Function to refresh the pointers in the object. Only useful if you're
   * doing illegal things. So don't use this.
   *
   * (please)
   * @return
   */
  inline Sketch* fixed() {
    buckets = reinterpret_cast<Bucket_Boruvka *>(_bucket_data);
    return this;
  }

  /**
   * Function to query a sketch.
   * @return   A pair with the result index and a code indicating if the type of result.
   */
  std::pair<bucket_t , SampleSketchRet> query();

  inline uint64_t get_seed() const {
    return seed;
  }

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
  void write_binary(std::ostream& binary_out);
  void write_binary(std::ostream& binary_out) const;
};

class MultipleQueryException : public std::exception {
public:
  virtual const char* what() const throw() {
    return "This sketch has already been sampled!";
  }
};
