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

#pragma pack(push,1)
struct SparseBucket {
  uint8_t next; // index of next sparse bucket in this column
  uint8_t row;  // row of sparse bucket
  Bucket bkt;   // actual bucket content
};
#pragma pack(pop)

// TODO: Do we want to use row major or column major order?
//       So the advantage of row-major is that we can update faster. Most updates will only touch
//       first few rows of data-structure. However, could slow down queries. (Although most query
//       answers will probably be in sparse data-structure). OH! Also, range_merge is important here
//       if column-major then the column we are merging is contig, if not, then not.
// A: Keep column-major for the moment, performance evaluation later.

/* Memory Allocation of a SparseSketch. Contiguous (only roughly to scale). 
   Where z is number of non-zero elements in vector we are sketching.
 _________________________________________________________________________________________________
| Dense                                           | Sparse                     | Linked List      |
| Bucket                                          | Bucket                     | Metadata         |
| Region                                          | Region                     | for Sparse bkts  |
| log n * log z buckets                           | clog n buckets             | clogn/16 buckets |
|_________________________________________________|____________________________|__________________|
*/

/**
 * SparseSketch for graph processing
 * Sub-linear representation of a vector.
 */
class SparseSketch {
 private:
  const uint64_t seed;           // seed for hash functions
  const size_t num_samples;      // number of samples we can perform
  const size_t cols_per_sample;  // number of columns to use on each sample
  const size_t num_columns;      // Total number of columns. (product of above 2)
  const size_t bkt_per_col;      // maximum number of buckets per column (max number of rows)

  size_t num_buckets;            // number of total buckets (col * dense_rows + sparse_capacity)
  size_t sample_idx = 0;         // number of samples performed so far

  // Allocated buckets
  Bucket* buckets;

  static constexpr size_t min_num_dense_rows = 4;
  size_t num_dense_rows = min_num_dense_rows;

  // Variables for sparse representation of lower levels of bucket Matrix
  // TODO: evaluate implications of this constant
  static constexpr double sparse_bucket_constant = 3;            // constant factor c (see diagram)
  SparseBucket* sparse_buckets;                                  // a pointer into the buckets array
  uint8_t *ll_metadata;                                          // pointer to heads of column LLs
  size_t number_of_sparse_buckets = 0;                           // cur number of sparse buckets
  size_t sparse_capacity = sparse_bucket_constant * num_columns; // max number of sparse buckets

  /**
   * Reallocates the bucket array if necessary to either grow or shrink the dense region
   */
  void reallocate_if_needed(int delta);
  void dense_realloc(size_t new_num_dense_rows);

  // These variables let us know how many Buckets to allocate to make space for the SparseBuckets
  // and the LL metadata that will use that space
  size_t sparse_data_size = ceil(double(sparse_capacity) * sizeof(SparseBucket) / sizeof(Bucket));
  size_t ll_metadata_size = ceil((double(num_columns) + 1) * sizeof(uint8_t) / sizeof(Bucket));

  void update_sparse(uint8_t col, SparseBucket to_add, bool realloc_if_needed = true);
  SketchSample sample_sparse(size_t first_col, size_t end_col);

  inline Bucket& deterministic_bucket() {
    return buckets[0];
  }
  inline const Bucket& deterministic_bucket() const {
    return buckets[0];
  }

  inline size_t position_func(size_t col, size_t row, size_t num_rows) const {
    return col * num_rows + row + 1;
  }

  // return the bucket at a particular index in bucket array
  inline Bucket& bucket(size_t col, size_t row) {
    assert(row < num_dense_rows);
    return buckets[position_func(col, row, num_dense_rows)];
  }
  inline const Bucket& bucket(size_t col, size_t row) const {
    assert(row < num_dense_rows);
    return buckets[position_func(col, row, num_dense_rows)];
  }

  size_t calc_num_buckets(size_t new_num_dense_rows) {
    return num_columns * new_num_dense_rows + sparse_data_size + ll_metadata_size + 1;
  }

  size_t calc_sparse_index(size_t rows) {
    return num_columns * rows + 1;
  }
  
  size_t calc_metadata_index(size_t rows) {
    return num_columns * rows + sparse_data_size + 1;
  }

  void upd_sparse_ptrs() {
    sparse_buckets = (SparseBucket *) &buckets[calc_sparse_index(num_dense_rows)];
    ll_metadata = (uint8_t *) &buckets[calc_metadata_index(num_dense_rows)];
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
  SparseSketch(vec_t vector_len, uint64_t seed, size_t num_samples = 1,
         size_t cols_per_sample = default_cols_per_sample);

  /**
   * Construct a sketch from a serialized stream
   * @param vector_len       Length of the vector we are sketching
   * @param seed             Random seed of the sketch
   * @param binary_in        Stream holding serialized sketch object
   * @param num_buckets      Number of buckets in serialized sketch (dense + sparse_capacity)
   * @param num_samples      [Optional] Number of samples this sketch supports (default = 1)
   * @param cols_per_sample  [Optional] Number of sketch columns for each sample (default = 1)
   */
  SparseSketch(vec_t vector_len, uint64_t seed, std::istream& binary_in, size_t num_buckets,
               size_t num_samples = 1, size_t cols_per_sample = default_cols_per_sample);

  /**
   * SparseSketch copy constructor
   * @param s  The sketch to copy.
   */
  SparseSketch(const SparseSketch& s);

  ~SparseSketch();

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
  void merge(const SparseSketch &other);

  /**
   * In-place range merge function. Updates the caller Sketch.
   * The range merge only merges some of the Sketches
   * This function should only be used if you know what you're doing
   * @param other         Sketch to merge into caller
   * @param start_sample  Index of first sample to merge
   * @param n_samples     Number of samples to merge
   */
  void range_merge(const SparseSketch &other, size_t start_sample, size_t n_samples);

  /**
   * Perform an in-place merge function without another Sketch and instead
   * use a raw bucket memory.
   * We also allow for only a portion of the buckets to be merge at once
   * @param raw_bucket     Raw bucket data to merge into this sketch
   * @param n_raw_buckets  Size of raw_buckets in number of Bucket data-structures
   */
  void merge_raw_bucket_buffer(const Bucket *raw_buckets, size_t n_raw_buckets);

  /**
   * Zero out all the buckets of a sketch.
   */
  void zero_contents();

  friend bool operator==(const SparseSketch& sketch1, const SparseSketch& sketch2);
  friend std::ostream& operator<<(std::ostream& os, const SparseSketch& sketch);

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
  inline size_t get_num_dense_rows() const { return num_dense_rows; }

  static size_t calc_bkt_per_col(size_t n) { return ceil(log2(n)) + 1; }

  static constexpr size_t default_cols_per_sample = 1;
  static constexpr double num_samples_div = 1 - log2(2 - 0.8);
};
