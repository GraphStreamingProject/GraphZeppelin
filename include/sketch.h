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

// TODO: Do we want to use row major or column major order?
// TODO: How do we want to handle raw_bucket_merge() and get_readonly_bucket_ptr()?
//       These functions are nice for performance because we can skip serialization but aren't
//       strictly necessary.
// TODO: It would be nice to preallocate the structure if we know how big its probably going to be.
//       This would be helpful for delta sketches for example. 
// TODO: What are we doing with the num_buckets variable? Could be nice to just be the size of 
//       buckets array. Could also be upperbound on the size.

// A strategy that could work well would be to allocate a chunk of memory some of which is given to
// the dense region of the sketch and 3 * sizeof(uint64_t) are given to sparse region.
// 3 -> position, alpha, gamma (could save a little more space by using 16 bits for position)

// enum SerialType {
//   FULL,
//   RANGE,
//   SPARSE,
// };

enum SampleResult {
  GOOD,  // sampling this sketch returned a single non-zero value
  ZERO,  // sampling this sketch returned that there are no non-zero values
  FAIL   // sampling this sketch failed to produce a single non-zero value
};

struct SketchSample {
  vec_t idx;
  SampleResult result;
};

struct ExhaustiveSketchSample {
  std::unordered_set<vec_t> idxs;
  SampleResult result;
};

/**
 * Sketch for graph processing, either CubeSketch or CameoSketch.
 * Sub-linear representation of a vector.
 */
class Sketch {
 private:
  const uint64_t seed;     // seed for hash functions
  size_t num_samples;      // number of samples we can perform
  size_t cols_per_sample;  // number of columns to use on each sample
  size_t num_columns;      // Total number of columns. (product of above 2)
  size_t bkt_per_col;      // maximum number of buckets per column (max number of rows)
  size_t num_buckets;      // number of total buckets (product of above 2)

  size_t sample_idx = 0;   // number of samples performed so far

  // bucket data, stored densely
  Bucket* buckets;

#ifndef L0_FULLY_DENSE
  size_t num_dense_rows = 4;

  // sparse representation of lower levels of Matrix
  // TODO: Evaluate if this is shit. It probably is
  std::vector<std::unordered_map<size_t, Bucket>> bucket_buffer;
  size_t number_of_sparse_buckets = 0;
  size_t sparse_capacity = 2 * num_columns; // TODO: evaluate implications of this constant

  /**
   * Reallocates the dense region of the sketch to have a different number of rows
   * @param new_num_rows  the new number of rows to store densely
   */
  void reallocate_dense_region(size_t new_num_rows);
#endif

  inline Bucket& get_deterministic_bucket() {
    // TODO: implement this
  }

 public:
  /**
   * The below constructors use vector length as their input. However, in graph sketching our input
   * is the number of vertices. This function converts from number of graph vertices to vector
   * length.
   * @param num_vertices  Number of graph vertices
   * @return              The length of the vector to sketch
   */
  static vec_t calc_vector_length(node_id_t num_vertices) {
    return ceil(double(num_vertices) * (num_vertices - 1) / 2);
  }

  /**
   * This function computes the number of samples a Sketch should support in order to solve
   * connected components. Optionally, can increase or decrease the number of samples by a
   * multiplicative factor.
   * @param num_vertices   Number of graph vertices
   * @param f              Multiplicative sample factor
   * @return               The number of samples
   */
  static size_t calc_cc_samples(node_id_t num_vertices, double f) {
    return std::max(size_t(18), (size_t) ceil(f * log2(num_vertices) / num_samples_div));
  }

  /**
   * Construct a sketch object
   * @param vector_len       Length of the vector we are sketching
   * @param seed             Random seed of the sketch
   * @param num_samples      [Optional] Number of samples this sketch supports (default = 1)
   * @param cols_per_sample  [Optional] Number of sketch columns for each sample (default = 1)
   */
  Sketch(vec_t vector_len, uint64_t seed, size_t num_samples = 1,
         size_t cols_per_sample = default_cols_per_sample);

  /**
   * Construct a sketch from a serialized stream
   * @param vector_len       Length of the vector we are sketching
   * @param seed             Random seed of the sketch
   * @param binary_in        Stream holding serialized sketch object
   * @param num_samples      [Optional] Number of samples this sketch supports (default = 1)
   * @param cols_per_sample  [Optional] Number of sketch columns for each sample (default = 1)
   */
  Sketch(vec_t vector_len, uint64_t seed, std::istream& binary_in, size_t num_samples = 1,
         size_t cols_per_sample = default_cols_per_sample);

  /**
   * Sketch copy constructor
   * @param s  The sketch to copy.
   */
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
  SketchSample sample();

  /**
   * Function to sample from the appropriate columns to return 1 or more non-zero indices
   * @return   A pair with the result indices and a code indicating the type of result.
   */
  ExhaustiveSketchSample exhaustive_sample();

  std::mutex mutex; // lock the sketch for applying updates in multithreaded processing

  /**
   * In-place merge function.
   * @param other  Sketch to merge into caller
   */
  void merge(const Sketch &other);

  /**
   * In-place range merge function. Updates the caller Sketch.
   * The range merge only merges some of the Sketches
   * This function should only be used if you know what you're doing
   * @param other         Sketch to merge into caller
   * @param start_sample  Index of first sample to merge
   * @param n_samples     Number of samples to merge
   */
  void range_merge(const Sketch &other, size_t start_sample, size_t n_samples);

  /**
   * Perform an in-place merge function without another Sketch and instead
   * use a raw bucket memory.
   * We also allow for only a portion of the buckets to be merge at once
   * @param raw_bucket    Raw bucket data to merge into this sketch
   */
  void merge_raw_bucket_buffer(const Bucket *raw_buckets);

  /**
   * Zero out all the buckets of a sketch.
   */
  void zero_contents();

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
  inline size_t bucket_array_bytes() const { return num_buckets * sizeof(Bucket); }

  inline const Bucket* get_readonly_bucket_ptr() const { return (const Bucket*) buckets; }
  inline uint64_t get_seed() const { return seed; }
  inline size_t column_seed(size_t column_idx) const { return seed + column_idx * 5; }
  inline size_t checksum_seed() const { return seed; }
  inline size_t get_columns() const { return num_columns; }
  inline size_t get_buckets() const { return num_buckets; }
  inline size_t get_num_samples() const { return num_samples; }

  static size_t calc_bkt_per_col(size_t n) { return ceil(log2(n)) + 1; }

#ifdef L0_SAMPLING
  static constexpr size_t default_cols_per_sample = 7;
  // NOTE: can improve this but leaving for comparison purposes
  static constexpr double num_samples_div = log2(3) - 1;
#else
  static constexpr size_t default_cols_per_sample = 1;
  static constexpr double num_samples_div = 1 - log2(2 - 0.8);
#endif
};

class OutOfSamplesException : public std::exception {
 private:
  std::string err_msg;
 public:
  OutOfSamplesException(size_t seed, size_t num_samples, size_t sample_idx)
      : err_msg("This sketch (seed=" + std::to_string(seed) +
                ", max samples=" + std::to_string(num_samples) +
                ") cannot be sampled more times (cur idx=" + std::to_string(sample_idx) + ")!") {}
  virtual const char* what() const throw() {
    return err_msg.c_str();
  }
};
