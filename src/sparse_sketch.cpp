#include "sparse_sketch.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>

SparseSketch::SparseSketch(vec_t vector_len, uint64_t seed, size_t _samples, size_t _cols)
    : seed(seed),
      num_samples(_samples),
      cols_per_sample(_cols),
      num_columns(cols_per_sample * num_samples),
      bkt_per_col(calc_bkt_per_col(vector_len)) {

  // plus 1, deterministic bucket
  num_buckets = num_columns * num_dense_rows + sparse_data_size + 1;
  buckets = new Bucket[num_buckets];
  sparse_buckets = (SparseBucket *) &buckets[num_columns * num_dense_rows + 1];

  // initialize bucket values
  for (size_t i = 0; i < num_buckets; ++i) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }
}

SparseSketch::SparseSketch(vec_t vector_len, uint64_t seed, std::istream &binary_in, size_t _samples,
               size_t _cols)
    : seed(seed),
      num_samples(_samples),
      cols_per_sample(_cols),
      num_columns(cols_per_sample * num_samples),
      bkt_per_col(calc_bkt_per_col(vector_len)) {

  // TODO: Make this actually work for sparse-sketch
  num_buckets = num_columns * bkt_per_col + 1; // plus 1 for deterministic bucket
  buckets = new Bucket[num_buckets];

  // Read the serialized Sketch contents
  binary_in.read((char *)buckets, bucket_array_bytes()); // TODO: Figure out bucket_array_bytes() in this context
}

SparseSketch::SparseSketch(const SparseSketch &s) 
    : seed(s.seed),
      num_samples(s.num_samples),
      cols_per_sample(s.cols_per_sample),
      num_columns(s.num_columns),
      bkt_per_col(s.bkt_per_col) {
  num_buckets = s.num_buckets;
  buckets = new Bucket[num_buckets];

  std::memcpy(buckets, s.buckets, bucket_array_bytes());
}

SparseSketch::~SparseSketch() { delete[] buckets; }


// Helper functions for interfacing with SparseBuckets
void SparseSketch::reallocate_if_needed() {
  if (num_dense_rows <= min_num_dense_rows) return; // do not reallocate
  if (number_of_sparse_buckets > num_columns && number_of_sparse_buckets < sparse_capacity)
    return; // do not reallocate

  // we are performing a reallocation
  std::cout << "Reallocating!" << std::endl;
  std::cout << "num_sparse: " << number_of_sparse_buckets << std::endl;
  std::cout << "capacity:   " << sparse_capacity << std::endl;
  const size_t old_buckets = num_buckets;
  const size_t old_rows = num_dense_rows;
  SparseBucket *old_sparse_pointer = sparse_buckets;
  Bucket *new_buckets;

  if (number_of_sparse_buckets < num_columns) {
    // shrink dense region by 1 row
    // Scan over deepest row of dense region and add all those buckets to sparse
    size_t depth = num_dense_rows - 1;
    for (size_t c = 0; c < num_columns; c++) {
      Bucket bkt = bucket(c, depth);
      if (!Bucket_Boruvka::is_empty(bkt)) {
        uint16_t sparse_position = (c << 8) + depth;
        update_sparse(sparse_position, bkt.alpha, bkt.gamma);
      }
    }

    // Allocate new memory
    --num_dense_rows;
    num_buckets = num_columns * num_dense_rows + sparse_data_size + 1;
    new_buckets = new Bucket[num_buckets];
  } else {
    // grow dense region by 1 row
    // Allocate new memory
    ++num_dense_rows;
    num_buckets = num_columns * num_dense_rows + sparse_data_size + 1;
    new_buckets = new Bucket[num_buckets];
  }
  sparse_buckets = (SparseBucket *) &new_buckets[num_columns * num_dense_rows + 1];

  // Copy dense content
  for (size_t c = 0; c < num_columns; c++) {
    for (size_t r = 0; r < std::min(num_dense_rows, old_rows); r++) {
      new_buckets[position_func(c, r, num_dense_rows)] = buckets[position_func(c, r, old_rows)];
    }
  }
  // sparse contents
  memcpy(sparse_buckets, old_sparse_pointer, sparse_capacity * sizeof(SparseBucket));


  if (num_buckets > old_buckets) {
    // We shrinking
    // Scan sparse buckets and move all updates of depth num_dense_rows-1
    // to the new dense row
    uint16_t depth_mask = 0xFFFF;
    for (size_t i = 0; i < sparse_capacity; i++) {
      if ((sparse_buckets[i].position & depth_mask) == num_dense_rows - 1) {
        size_t column = sparse_buckets[i].position >> 8;
        bucket(column, num_dense_rows - 1) = sparse_buckets[i].bkt;
        sparse_buckets[i].position = uint16_t(-1); // tombstone
        number_of_sparse_buckets -= 1;
      }
    }
  }

  // 4. Clean up
  std::swap(buckets, new_buckets);
  delete[] new_buckets;
}

// Update a bucket value
// Changes number_of_sparse_buckets as follows:
//    +1 if we added a new bucket value
//     0 if the bucket was found and update (but not cleared)
//    -1 if the bucket was found and cleared of all content
void SparseSketch::update_sparse(uint16_t pos, vec_t update_idx, vec_hash_t checksum) {
  SparseBucket *tombstone = nullptr;
  uint16_t tombstone_pos = uint16_t(-1);
  for (size_t i = 0; i < num_buckets; i++) {
    auto &sparse_bucket = sparse_buckets[i];
    if (sparse_bucket.position == 0 || sparse_bucket.position == pos) {
      // We apply our update here!
      if (sparse_bucket.position == pos) {
        // we update bucket
        Bucket_Boruvka::update(sparse_bucket.bkt, update_idx, checksum);

        // did we clear it out?
        if (Bucket_Boruvka::is_empty(sparse_bucket.bkt)) {
          sparse_bucket.position = tombstone_pos; // mark it as tombstone
          number_of_sparse_buckets -= 1;
        }
        return;
      } else {
        if (tombstone != nullptr) {
          // use the tombstone
          tombstone->position = pos;
          Bucket_Boruvka::update(tombstone->bkt, update_idx, checksum);
        } else {
          sparse_bucket.position = pos;
          Bucket_Boruvka::update(sparse_bucket.bkt, update_idx, checksum);
        }

        // we created a new sparse bucket
        number_of_sparse_buckets += 1;
        return;
      }
    } else if (sparse_bucket.position == tombstone_pos && tombstone == nullptr) {
      tombstone = &sparse_bucket;
      number_of_sparse_buckets += 1;
      return;
    }
  }
  // this is an error!
  std::cout << "num_sparse: " << number_of_sparse_buckets << std::endl;
  std::cout << "capacity:   " << sparse_capacity << std::endl;
  throw std::runtime_error("update_sparse(): Failed to find update location!");
}

// sample a good bucket from the sparse region if one exists. 
// Additionally, specify the column to query from
// TODO: Do we want to include this column thing?
SketchSample SparseSketch::sample_sparse(size_t first_col, size_t end_col) {
  for (size_t i = 0; i < sparse_capacity; i++) {
    if (size_t(sparse_buckets[i].position >> 8) >= first_col &&
        size_t(sparse_buckets[i].position >> 8) < end_col &&
        Bucket_Boruvka::is_good(sparse_buckets[i].bkt, checksum_seed())) {
      return {sparse_buckets[i].bkt.alpha, GOOD};
    }
  }

  // We could not find a good bucket
  return {0, FAIL};
}


void SparseSketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, checksum_seed());

  // Update depth 0 bucket
  Bucket_Boruvka::update(deterministic_bucket(), update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_columns; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, column_seed(i), bkt_per_col);
    likely_if(depth < bkt_per_col) {
      likely_if(depth < num_dense_rows) {
        Bucket_Boruvka::update(bucket(i, depth), update_idx, checksum);
      } else {
        update_sparse((i << 8) | depth, update_idx, checksum);

        // based upon this update to sparse matrix, check if we need to reallocate dense region
        reallocate_if_needed();
      }
    }
  }
}

// TODO: Switch the L0_SAMPLING flag to instead affect query procedure. 
// (Only use deepest bucket. We don't need the alternate update procedure in the code anymore.)

void SparseSketch::zero_contents() {
  // TODO: Should we also set the size of this bucket back to an initial state?
  for (size_t i = 0; i < num_buckets; i++) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }
  reset_sample_state();
}

SketchSample SparseSketch::sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  if (Bucket_Boruvka::is_empty(deterministic_bucket()))
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return

  if (Bucket_Boruvka::is_good(deterministic_bucket(), checksum_seed()))
    return {deterministic_bucket().alpha, GOOD};

  for (size_t i = 0; i < cols_per_sample; ++i) {
    for (size_t j = 0; j < num_dense_rows; ++j) {
      if (Bucket_Boruvka::is_good(bucket(i + first_column, j), checksum_seed()))
        return {bucket(i + first_column, j).alpha, GOOD};
    }
  }

  // Sample sparse region
  return sample_sparse(first_column, first_column + cols_per_sample);
}

ExhaustiveSketchSample SparseSketch::exhaustive_sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }
  std::unordered_set<vec_t> ret;

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  unlikely_if (deterministic_bucket().alpha == 0 && deterministic_bucket().gamma == 0)
    return {ret, ZERO}; // the "first" bucket is deterministic so if zero then no edges to return

  unlikely_if (Bucket_Boruvka::is_good(deterministic_bucket(), checksum_seed())) {
    ret.insert(deterministic_bucket().alpha);
    return {ret, GOOD};
  }

  for (size_t i = 0; i < cols_per_sample; ++i) {
    for (size_t j = 0; j < bkt_per_col; ++j) {
      unlikely_if (Bucket_Boruvka::is_good(bucket(i + first_column, j), checksum_seed())) {
        ret.insert(bucket(i + first_column, j).alpha);
      }
    }
  }

  // TODO: How do we do exhaustive sampling properly here?
  SketchSample sample = sample_sparse(first_column, first_column + cols_per_sample);
  if (sample.result == GOOD) {
    ret.insert(sample.idx);
  }

  unlikely_if (ret.size() == 0)
    return {ret, FAIL};
  return {ret, GOOD};
}

void SparseSketch::merge(const SparseSketch &other) {
  for (size_t i = 0; i < num_buckets; ++i) {
    buckets[i].alpha ^= other.buckets[i].alpha;
    buckets[i].gamma ^= other.buckets[i].gamma;
  }

  // TODO: Handle sparse stuff!
}

void SparseSketch::range_merge(const SparseSketch &other, size_t start_sample, size_t n_samples) {
  if (start_sample + n_samples > num_samples) {
    assert(false);
    sample_idx = num_samples; // sketch is in a fail state!
    return;
  }

  // update sample idx to point at beginning of this range if before it
  sample_idx = std::max(sample_idx, start_sample);

  // merge deterministic buffer
  deterministic_bucket().alpha ^= other.deterministic_bucket().alpha;
  deterministic_bucket().gamma ^= other.deterministic_bucket().gamma;

  // merge other buckets
  size_t start_column = start_sample * cols_per_sample;
  size_t end_column = (start_sample + n_samples) * cols_per_sample;

  for (size_t i = start_column; i < end_column; i++) {
    for (size_t j = 0; j < bkt_per_col; j++) {
      bucket(i, j).alpha ^= other.bucket(i, j).alpha;
      bucket(i, j).gamma ^= other.bucket(i, j).gamma;
    }
  }

  // TODO: Handle sparse!
}

void SparseSketch::merge_raw_bucket_buffer(const Bucket *raw_buckets) {
  for (size_t i = 0; i < num_buckets; i++) {
    buckets[i].alpha ^= raw_buckets[i].alpha;
    buckets[i].gamma ^= raw_buckets[i].gamma;
  }

  // TODO: Handle sparse
}

void SparseSketch::serialize(std::ostream &binary_out) const {
  binary_out.write((char*) buckets, bucket_array_bytes());

  // TODO: Handle sparse
}

bool operator==(const SparseSketch &sketch1, const SparseSketch &sketch2) {
  if (sketch1.num_buckets != sketch2.num_buckets || sketch1.seed != sketch2.seed)
    return false;

  for (size_t i = 0; i < sketch1.num_buckets; ++i) {
    if (sketch1.buckets[i].alpha != sketch2.buckets[i].alpha ||
        sketch1.buckets[i].gamma != sketch2.buckets[i].gamma) {
      return false;
    }
  }

  // TODO: Handle sparse

  return true;
}

std::ostream &operator<<(std::ostream &os, const SparseSketch &sketch) {
  Bucket bkt = sketch.buckets[sketch.num_buckets - 1];
  bool good = Bucket_Boruvka::is_good(bkt, sketch.checksum_seed());
  vec_t a = bkt.alpha;
  vec_hash_t c = bkt.gamma;

  os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;

  for (unsigned i = 0; i < sketch.num_columns; ++i) {
    for (unsigned j = 0; j < sketch.bkt_per_col; ++j) {
      Bucket bkt = sketch.bucket(i, j);
      vec_t a = bkt.alpha;
      vec_hash_t c = bkt.gamma;
      bool good = Bucket_Boruvka::is_good(bkt, sketch.checksum_seed());

      os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;
    }
    os << std::endl;
  }
  return os;
}
