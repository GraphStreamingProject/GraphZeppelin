#pragma once
#include <graph_zeppelin_common.h>
#include <gtest/gtest_prod.h>
#include <sys/mman.h>

#include <fstream>
#include <unordered_set>
#include <cmath>
#include <cassert>
#include <mutex>

#include "util.h"
#include "bucket.h"
#include "sketch_types.h"

/**
 * Sketch for graph processing, either CubeSketch or CameoSketch.
 * Sub-linear representation of a vector.
 */
class DenseSketch {
 private:
  const uint64_t seed;     // seed for hash functions
  size_t num_samples;      // number of samples we can perform
  size_t cols_per_sample;  // number of columns to use on each sample
  size_t num_columns;      // Total number of columns. (product of above 2)
  size_t bkt_per_col;      // maximum number of buckets per column (max number of rows)
  size_t num_buckets;      // number of total buckets product of above two
  size_t sample_idx = 0;   // number of samples performed so far

  // Allocated buckets
  Bucket* buckets;

  inline Bucket& deterministic_bucket() {
    return buckets[0];
  }
  inline const Bucket& deterministic_bucket() const {
    return buckets[0];
  }

  // return the bucket at a particular index in bucket array
  inline Bucket& bucket(size_t col, size_t row) {
    return buckets[col * bkt_per_col + row + 1];
  }
  inline const Bucket& bucket(size_t col, size_t row) const {
    return buckets[col * bkt_per_col + row + 1];
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
  DenseSketch(vec_t vector_len, uint64_t seed, size_t num_samples = 1,
         size_t cols_per_sample = default_cols_per_sample);

  /**
   * Construct a sketch from a serialized stream
   * @param vector_len       Length of the vector we are sketching
   * @param seed             Random seed of the sketch
   * @param binary_in        Stream holding serialized sketch object
   * @param num_samples      [Optional] Number of samples this sketch supports (default = 1)
   * @param cols_per_sample  [Optional] Number of sketch columns for each sample (default = 1)
   */
  DenseSketch(vec_t vector_len, uint64_t seed, std::istream& binary_in, size_t num_samples = 1,
         size_t cols_per_sample = default_cols_per_sample);

  /**
   * Sketch copy constructor
   * @param s  The sketch to copy.
   */
  DenseSketch(const DenseSketch& s);

  ~DenseSketch();

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
  void merge(const DenseSketch &other);

  /**
   * In-place range merge function. Updates the caller Sketch.
   * The range merge only merges some of the Sketches
   * This function should only be used if you know what you're doing
   * @param other         Sketch to merge into caller
   * @param start_sample  Index of first sample to merge
   * @param n_samples     Number of samples to merge
   */
  void range_merge(const DenseSketch &other, size_t start_sample, size_t n_samples);

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

  friend bool operator==(const DenseSketch& sketch1, const DenseSketch& sketch2);
  friend std::ostream& operator<<(std::ostream& os, const DenseSketch& sketch);

  /**
   * Serialize the sketch to a binary output stream.
   * @param binary_out   the stream to write to.
   */
  void serialize(std::ostream& binary_out) const;

  inline void reset_sample_state() {
    sample_idx = 0;
  }

  // return the size of the sketching datastructure in bytes (just the buckets, not the metadata)
  inline size_t bucket_array_bytes() const { 
    return num_buckets * sizeof(Bucket); 
  }

  inline const Bucket* get_readonly_bucket_ptr() const { return (const Bucket*) buckets; }
  inline uint64_t get_seed() const { return seed; }
  inline size_t column_seed(size_t column_idx) const { return seed + column_idx * 5; }
  inline size_t checksum_seed() const { return seed; }
  inline size_t get_columns() const { return num_columns; }
  inline size_t get_buckets() const { return num_buckets; }
  inline size_t get_num_samples() const { return num_samples; }

  static size_t calc_bkt_per_col(size_t n) { return ceil(log2(n)) + 1; }

  static constexpr size_t default_cols_per_sample = 1;
  static constexpr double num_samples_div = 1 - log2(2 - 0.8);
};
