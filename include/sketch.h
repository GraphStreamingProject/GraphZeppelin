#pragma once
#include <graph_zeppelin_common.h>
#include <gtest/gtest_prod.h>
#include <sys/mman.h>

#include <fstream>
#include <unordered_set>
#include <cmath>
#include <mutex>

#include "util.h"
#include "bucket.h"

enum SerialType {
  FULL,
  RANGE,
  SPARSE,
};

enum SampleSketchRet {
  GOOD,  // querying this sketch returned a single non-zero value
  ZERO,  // querying this sketch returned that there are no non-zero values
  FAIL   // querying this sketch failed to produce a single non-zero value
};

/**
 * Sketch for graph processing, either CubeSketch or CameoSketch.
 * Sub-linear representation of a vector.
 */
class Sketch {
 private:
  const uint64_t seed;     // seed for random number generators
  size_t num_samples;      // number of samples we can perform
  size_t cols_per_sample;  // number of columns to use on each sample
  size_t num_columns;      // Total number of columns. (product of above 2)
  size_t num_guesses;      // number of buckets per column
  size_t num_buckets;      // number of total buckets (product of above 2)

  size_t sample_idx = 0;   // number of samples performed so far

  // bucket data
  Bucket* buckets;

 public:
  // Constructors for the sketch object
  // 0 means that we should use the default value for samples = log_3/2(n) and cols = 2
  Sketch(node_id_t n, uint64_t seed, size_t num_samples = 0, size_t cols_per_sample = 0);
  Sketch(node_id_t n, uint64_t seed, std::istream& binary_in, size_t num_samples = 0,
         size_t cols_per_sample = 0);
  Sketch(const Sketch& s);

  ~Sketch();

  /**
   * Update a sketch based on information about one of its indices.
   * @param update   the point update.
   */
  void update(const vec_t update);

  /**
   * Function to sample from the sketch.
   * cols_per_sample determines the number of columns we allocate to this query
   * @return   A pair with the result index and a code indicating the type of result.
   */
  std::pair<vec_t, SampleSketchRet> sample();

  /**
   * Function to sample from the appropriate columns to return 1 or more non-zero indices
   * @return   A pair with the result indices and a code indicating the type of result.
   */
  std::pair<std::unordered_set<vec_t>, SampleSketchRet> exhaustive_sample();

  std::mutex mutex; // lock the sketch for applying updates in multithreaded processing

  /**
   * In-place merge function.
   * @param other  Sketch to merge into caller
   */
  void merge(Sketch& other);

  /**
   * Perform an in-place merge function without another Sketch and instead
   * use a raw bucket memory.
   * 
   * We also allow for only a portion of the buckets to be merge at once
   */
  void merge_raw_bucket_buffer(vec_t *buckets, size_t start_sample, size_t num_samples);

  /**
   * Zero out all the buckets of a sketch.
   */
  void zero_contents();

  /**
   * In-place range merge function. Updates the caller Sketch.
   * The range merge only merges some of the Sketches
   * This function should only be used if you know what you're doing
   * @param other       Sketch to merge into caller
   * @param start_idx   Index of first sample to merge
   * @param num_merge   How many samples to merge
   */
  void range_merge(Sketch& other, size_t start_idx, size_t num_merge);

  friend bool operator==(const Sketch& sketch1, const Sketch& sketch2);
  friend std::ostream& operator<<(std::ostream& os, const Sketch& sketch);

  /**
   * Serialize the sketch to a binary output stream.
   * @param binary_out   the stream to write to.
   */
  void serialize(std::ostream& binary_out) const;

  inline void reset_sample_state() {
    sample_idx = 0;
  }

  // return the size of the sketching datastructure in bytes (just the buckets, not the metadata)
  inline size_t sketch_bytes() const { return num_buckets * sizeof(Bucket); }

  inline const char* get_bucket_memory_buffer() const { return (char*)buckets; }
  inline uint64_t get_seed() const { return seed; }
  inline size_t column_seed(size_t column_idx) const { return seed + column_idx * 5; }
  inline size_t checksum_seed() const { return seed; }
  inline size_t get_columns() const { return num_columns; }

  // max number of non-zeroes in vector is n/2*n/2=n^2/4
  static size_t guess_gen(size_t n) { return 2 * log2(n) - 2; }
  static size_t samples_gen(size_t n) { return ceil(log2(n) / num_samples_div); }

#ifdef L0_SAMPLING
  static constexpr size_t default_cols_per_sample = 7;
  static constexpr double num_samples_div = log2(3) - 1;
#else
  static constexpr size_t default_cols_per_sample = 2;
  static constexpr double num_samples_div = log2(3) - 1;
#endif
};

class OutOfQueriesException : public std::exception {
 public:
  virtual const char* what() const throw() {
    return "This supernode cannot be sampled more times!";
  }
};
