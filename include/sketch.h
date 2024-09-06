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

#include "cuckoohash_map.hh"

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
 public:
  size_t num_columns;      // Total number of columns. (product of above 2)
  size_t bkt_per_col;      // number of buckets per column
 private:

  // TODO - decringe this
  // should be 4 * 32 = 128 minimum
  // should be one-per-thread
  // TODO - also figure out why 128 bits isnt enough
  uint32_t depth_buffer[256];
  const uint64_t seed;     // seed for hash functions
  size_t num_samples;      // number of samples we can perform
  size_t cols_per_sample;  // number of columns to use on each sample
  // size_t num_columns;      // Total number of columns. (product of above 2)
  // size_t bkt_per_col;      // number of buckets per column
  size_t num_buckets;      // number of total buckets (product of above 2)

  size_t sample_idx = 0;   // number of samples performed so far

  // bucket data
  Bucket* buckets;
  // TEMPORARY
  // libcuckoo::cuckoohash_map<vec_t, size_t, std::hash<vec_t>, std::equal_to<vec_t>, std::allocator<std::pair<const vec_t, size_t>>>
  // libcuckoo::cuckoohash_map<vec_t, size_t>
  //  bucket_map(
  //   // n=32, // initial number of buckets
  //   // std::hash<vec_t>(), // hash function for keys
  //   // std::equal_to<vec_t>(), // equal function for keys
  //   // std::allocator<std::pair<const vec_t, size_t>>(), // allocator for the map
  // );
  std::unordered_map<vec_t, bool> bucket_map;
  // PER BUCKET 
  std::function<void(vec_t)> evict_fn = [this](vec_t update){
    // interface: update is the index that's being pushed,
    bucket_map.emplace(update, 0);
    bucket_map[update] ^= 1;
  };
  std::function<std::vector<vec_t>()> get_evicted_fn = [this](){
    std::vector<vec_t> ret;
    for (auto it = bucket_map.begin(); it != bucket_map.end(); it++) {
      if (it->second == 1) {
        ret.push_back(it->first);
      }
    }
    return ret;
  };

  // flags

#ifdef EAGER_BUCKET_CHECK
  vec_t *nonempty_buckets;
  /**
   * Updates the nonempty flags in a given range by recalculating the is_empty() call.
   * @param col_idx   The column to update
   * @param start_row The depth of the first bucket in the column to check the emptyness of.
   * @param end_row   The depth of the first bucket not to check the emptyness (i.e., an exclusive bound)
    */
  void recalculate_flags(size_t col_idx, size_t start_row, size_t end_row);
#endif
 private:
  inline Bucket& get_deterministic_bucket() const {
    return buckets[num_buckets - 1];
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
   * Construct a sketch from a (potentially compressed) serialized stream
   * @param vector_len       Length of the vector we are sketching
   * @param seed             Random seed of the sketch
   * @param binary_in        Stream holding serialized sketch object
   * @param num_samples      [Optional] Number of samples this sketch supports (default = 1)
   * @param cols_per_sample  [Optional] Number of sketch columns for each sample (default = 1)
   * @param compressed       Whether or not to use the compression (default = true) 
   */
  Sketch(vec_t vector_len, uint64_t seed, bool compressed, std::istream& binary_in, size_t num_samples = 1,
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
   * Get the bucket at a specific column and depth
   */
  inline Bucket& get_bucket(size_t col_idx, size_t depth) const {
#ifdef ROW_MAJOR_SKETCHES
    // contiguous by bucket depth
    return buckets[depth * num_columns + col_idx];
#else 
    // contiguous by column
    return buckets[col_idx * bkt_per_col + depth];
#endif
  }

  /**
   * Occupies the contents of an empty sketch with input from a stream that contains
   * the compressed version.
   * @param binary_in   Stream holding serialized/compressed sketch object.
   */
  void compressed_deserialize(std::istream& binary_in);


  /**
   * Update a sketch based on information about one of its indices.
   * @param update   the point update.
   */
  void update(const vec_t update);


#ifdef EAGER_BUCKET_CHECK
  /**
   * TODO - make this less silly
   */

  void unsafe_update();
#endif

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
   * Gives the cutoff index such that all non-empty buckets are strictly above. 
   * @param col_idx The column to find the cutoff index of.
   * @return  The depth of the non-zero'th bucket + 1. If the bucket is entirely empty, returns 0
   */
  uint8_t effective_size(size_t col_idx) const;


  /**
   * Gives the cutoff index such that all non-empty buckets are strictly above for ALL columns
   * @return Depth of the deepest non-zero'th bucket + 1. 0 if all buckets are empty.
   */
  uint8_t effective_depth() const;

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

  /**
   * Serialize the sketch to a binary output stream, with a compressed representation.
   * takes significantly less space for mostly-empty sketches.
   * @param binary_out   the stream to write to.
   */
  void compressed_serialize(std::ostream& binary_out) const;

  inline void reset_sample_state() {
    sample_idx = 0;
  }

  // return the size of the sketching datastructure in bytes (just the buckets, not the metadata)
  inline size_t bucket_array_bytes() const { 
#ifdef EAGER_BUCKET_CHECK
    return (num_buckets * sizeof(Bucket)) + (num_columns * sizeof(vec_t)); 
#else
    return num_buckets * sizeof(Bucket);
#endif
  }

  inline const Bucket* get_readonly_bucket_ptr() const { return (const Bucket*) buckets; }
  inline uint64_t get_seed() const { return seed; }
  inline size_t column_seed(size_t column_idx) const { return seed + column_idx * 5; }
  inline size_t checksum_seed() const { return seed; }
  inline size_t get_columns() const { return num_columns; }
  inline size_t get_buckets() const { return num_buckets; }
  inline size_t get_num_samples() const { return num_samples; }

  static size_t calc_bkt_per_col(size_t n) { return ceil(log2(n)) + 4;}

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
