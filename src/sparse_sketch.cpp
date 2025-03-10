#include "sparse_sketch.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

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

SparseSketch::SparseSketch(vec_t vector_len, uint64_t seed, std::istream &binary_in,
                           size_t num_buckets, size_t _samples, size_t _cols)
    : seed(seed),
      num_samples(_samples),
      cols_per_sample(_cols),
      num_columns(cols_per_sample * num_samples),
      bkt_per_col(calc_bkt_per_col(vector_len)),
      num_buckets(num_buckets) {
  buckets = new Bucket[num_buckets];
  num_dense_rows = (num_buckets - sparse_data_size) / num_columns;
  sparse_buckets = (SparseBucket *) &buckets[num_columns * num_dense_rows + 1]; 

  // Read the serialized Sketch contents
  binary_in.read((char *)buckets, bucket_array_bytes());
}

SparseSketch::SparseSketch(const SparseSketch &s) 
    : seed(s.seed),
      num_samples(s.num_samples),
      cols_per_sample(s.cols_per_sample),
      num_columns(s.num_columns),
      bkt_per_col(s.bkt_per_col),
      num_buckets(s.num_buckets),
      num_dense_rows(s.num_dense_rows) {
  buckets = new Bucket[num_buckets];
  sparse_buckets = (SparseBucket *) &buckets[num_columns * num_dense_rows + 1];

  std::memcpy(buckets, s.buckets, bucket_array_bytes());
}

SparseSketch::~SparseSketch() {
  // std::cout << "Deleting sketch! buckets = " << buckets << std::endl;
  delete[] buckets; 
}


// Helper functions for interfacing with SparseBuckets
void SparseSketch::dense_realloc(size_t new_num_dense_rows) {
  // we are performing a reallocation
  const size_t old_rows = num_dense_rows;
  SparseBucket *old_sparse_pointer = sparse_buckets;
  Bucket *new_buckets;

  if (new_num_dense_rows < min_num_dense_rows) {
    throw std::runtime_error("new_num_dense_rows too small!");
  }

  if (new_num_dense_rows < num_dense_rows) {
    // std::cout << "Shrinking to " << new_num_dense_rows << " from " << old_rows << std::endl;
    // shrink dense region
    // Scan over the rows we are removing and add all those buckets to sparse
    for (size_t c = 0; c < num_columns; c++) {
      for (size_t r = new_num_dense_rows; r < old_rows; r++) {
        Bucket bkt = bucket(c, r);
        if (!Bucket_Boruvka::is_empty(bkt)) {
          SparseBucket new_sparse;
          new_sparse.set_col(c);
          new_sparse.set_row(r);
          new_sparse.bkt = bkt;
          update_sparse(new_sparse, false);
        }
      }
    }

    // Allocate new memory
    num_dense_rows = new_num_dense_rows;
    num_buckets = num_columns * num_dense_rows + sparse_data_size + 1;
    new_buckets = new Bucket[num_buckets];
  } else {
    // std::cout << "Growing to " << new_num_dense_rows << " from " << old_rows << std::endl;
    // grow dense region by 1 row
    // Allocate new memory
    num_dense_rows = new_num_dense_rows;
    num_buckets = num_columns * num_dense_rows + sparse_data_size + 1;
    new_buckets = new Bucket[num_buckets];

    // initialize new rows to zero
    for (size_t c = 0; c < num_columns; c++) {
      for (size_t r = old_rows; r < num_dense_rows; r++) {
        new_buckets[position_func(c, r, num_dense_rows)] = {0, 0};
      }
    }
  }
  sparse_buckets = (SparseBucket *) &new_buckets[num_columns * num_dense_rows + 1];

  // Copy dense content
  new_buckets[0] = deterministic_bucket();
  for (size_t c = 0; c < num_columns; c++) {
    for (size_t r = 0; r < std::min(num_dense_rows, old_rows); r++) {
      new_buckets[position_func(c, r, num_dense_rows)] = buckets[position_func(c, r, old_rows)];
    }
  }
  // sparse contents
  memcpy(sparse_buckets, old_sparse_pointer, sparse_capacity * sizeof(SparseBucket));


  if (num_dense_rows > old_rows) {
    // We growing
    // Scan sparse buckets and move all updates of depth num_dense_rows-1
    // to the new dense row
    for (size_t i = 0; i < sparse_capacity; i++) {
      // std::cout << "sparse_bucket = " << sparse_buckets[i].col() << ", " << sparse_buckets[i].row()
      //           << ": " << sparse_buckets[i].bkt.alpha << ", " << sparse_buckets[i].bkt.gamma
      //           << std::endl;
      if (sparse_buckets[i].row() < num_dense_rows && sparse_buckets[i].position != 0) {
        size_t col = sparse_buckets[i].col();
        size_t row = sparse_buckets[i].row();
        assert(Bucket_Boruvka::is_empty(new_buckets[position_func(col, row, num_dense_rows)]));
        new_buckets[position_func(col, row, num_dense_rows)] = sparse_buckets[i].bkt;
        sparse_buckets[i].position = uint16_t(-1); // tombstone
        sparse_buckets[i].bkt = {0, 0}; // clear out tombstone
        number_of_sparse_buckets -= 1;
        // std::cout << "Moving to dense!" << std::endl;
      }
    }
  }

  // 4. Clean up
  std::swap(buckets, new_buckets);
  delete[] new_buckets;
}

void SparseSketch::reallocate_if_needed(int delta) {
  // if we're currently adding something, don't shrink
  if (delta == 1 && number_of_sparse_buckets <= num_columns / 4) {
    return;
  }

  // while we need to reallocate, attempt to do so. If realloc doesn't solve problem. Do it again.
  while ((delta == -1 && number_of_sparse_buckets <= num_columns / 4 &&
          num_dense_rows > min_num_dense_rows) ||
         (delta == 1 && number_of_sparse_buckets == sparse_capacity)) {
    if (number_of_sparse_buckets >= sparse_capacity) {
      dense_realloc(num_dense_rows + 1);
    } else {
      dense_realloc(num_dense_rows - 1);
    }
  }
}

// Update a bucket value
// Changes number_of_sparse_buckets as follows:
//    +1 if we added a new bucket value
//     0 if the bucket was found and update (but not cleared)
//    -1 if the bucket was found and cleared of all content
void SparseSketch::update_sparse(SparseBucket to_add, bool realloc_if_needed) {
  SparseBucket *tombstone = nullptr;
  uint16_t tombstone_pos = uint16_t(-1);
  for (size_t i = 0; i < sparse_capacity; i++) {
    auto &sparse_bucket = sparse_buckets[i];
    if (sparse_bucket.position == 0 || sparse_bucket.position == to_add.position) {
      // We apply our update here!
      if (sparse_bucket.position == to_add.position) {
        // we update bucket
        sparse_bucket.bkt.alpha ^= to_add.bkt.alpha;
        sparse_bucket.bkt.gamma ^= to_add.bkt.gamma;

        // did we clear it out?
        if (Bucket_Boruvka::is_empty(sparse_bucket.bkt)) {
          sparse_bucket.position = tombstone_pos; // mark it as tombstone
          number_of_sparse_buckets -= 1;
          if (realloc_if_needed) reallocate_if_needed(-1);
        }
        return;
      } else {
        if (tombstone != nullptr) {
          // use the tombstone
          *tombstone = to_add;
        } else {
          sparse_bucket = to_add;
        }

        // we created a new sparse bucket
        number_of_sparse_buckets += 1;
        if (realloc_if_needed) reallocate_if_needed(1);
        return;
      }
    } else if (sparse_bucket.position == tombstone_pos && tombstone == nullptr) {
      tombstone = &sparse_bucket;
    }
  }
  if (tombstone != nullptr) {
    // use the tombstone
    *tombstone = to_add;
    number_of_sparse_buckets += 1; // we created a new sparse bucket
    if (realloc_if_needed) reallocate_if_needed(1);
    return;
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
    if (size_t(sparse_buckets[i].col()) >= first_col &&
        size_t(sparse_buckets[i].col()) < end_col &&
        Bucket_Boruvka::is_good(sparse_buckets[i].bkt, checksum_seed())) {
      // std::cout << "Found GOOD sparse bucket" << std::endl;
      return {sparse_buckets[i].bkt.alpha, GOOD};
    }
  }

  // We could not find a good bucket
  // std::cout << "Sketch FAIL" << std::endl;
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
        update_sparse({uint16_t((i << 8) | depth), {update_idx, checksum}});
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
  number_of_sparse_buckets = 0;
}

SketchSample SparseSketch::sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  // std::cout << "Sampling sketch" << std::endl;
  // std::cout << "first_col = " << first_column << std::endl;
  // std::cout << "end_col =   " << first_column + cols_per_sample << std::endl;
  // std::cout << *this << std::endl;

  if (Bucket_Boruvka::is_empty(deterministic_bucket())) {
    // std::cout << "ZERO!" << std::endl;
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  }

  if (Bucket_Boruvka::is_good(deterministic_bucket(), checksum_seed())) {
    // std::cout << "Deterministic GOOD" << std::endl;
    return {deterministic_bucket().alpha, GOOD};
  }

  // Sample sparse region
  SketchSample sample = sample_sparse(first_column, first_column + cols_per_sample);
  if (sample.result == GOOD) {
    return sample;
  }

  for (size_t c = 0; c < cols_per_sample; ++c) {
    for (size_t r = 0; r < num_dense_rows; ++r) {
      if (Bucket_Boruvka::is_good(bucket(c + first_column, r), checksum_seed())) {
        // std::cout << "Found GOOD dense bucket" << std::endl;
        return {bucket(c + first_column, r).alpha, GOOD};
      }
    }
  }

  // Sample sparse region
  return {0, FAIL};
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

  for (size_t c = 0; c < cols_per_sample; ++c) {
    for (size_t r = 0; r < num_dense_rows; ++r) {
      unlikely_if (Bucket_Boruvka::is_good(bucket(c + first_column, r), checksum_seed())) {
        ret.insert(bucket(c + first_column, r).alpha);
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
  deterministic_bucket().alpha ^= other.deterministic_bucket().alpha;
  deterministic_bucket().gamma ^= other.deterministic_bucket().gamma;

  // merge all dense buckets from other sketch into this one
  for (size_t c = 0; c < num_columns; c++) {
    for (size_t r = 0; r < other.num_dense_rows; ++r) {
      if (r < num_dense_rows) {
        bucket(c, r).alpha ^= other.bucket(c, r).alpha;
        bucket(c, r).gamma ^= other.bucket(c, r).gamma;
      } else if (!Bucket_Boruvka::is_empty(other.bucket(c, r))) {
        SparseBucket sparse_bkt;
        sparse_bkt.set_col(c);
        sparse_bkt.set_row(r);
        sparse_bkt.bkt = other.bucket(c, r);
        update_sparse(sparse_bkt);
      }
    }
  }

  // Merge all sparse buckets from other sketch into this one
  for (size_t i = 0; i < other.sparse_capacity; i++) {
    const auto &oth_sparse_bkt = other.sparse_buckets[i];
    if (oth_sparse_bkt.position != uint16_t(-1) && oth_sparse_bkt.position != 0) {
      if (oth_sparse_bkt.row() < num_dense_rows) {
        auto &bkt = bucket(oth_sparse_bkt.col(), oth_sparse_bkt.row());
        bkt.alpha ^= oth_sparse_bkt.bkt.alpha;
        bkt.gamma ^= oth_sparse_bkt.bkt.gamma;
      } else {
        update_sparse(oth_sparse_bkt);
      }
    }
  }
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

  // merge all their dense buckets into us
  for (size_t c = start_column; c < end_column; c++) {
    for (size_t r = 0; r < other.num_dense_rows; r++) {
      if (r < num_dense_rows) {
        bucket(c, r).alpha ^= other.bucket(c, r).alpha;
        bucket(c, r).gamma ^= other.bucket(c, r).gamma;
      } else if (!Bucket_Boruvka::is_empty(other.bucket(c, r))) {
        SparseBucket sparse_bkt;
        sparse_bkt.set_col(c);
        sparse_bkt.set_row(r);
        sparse_bkt.bkt = other.bucket(c, r);
        update_sparse(sparse_bkt);
      }
    }
  }

  // Merge all sparse buckets from other sketch's columns into this one
  for (size_t i = 0; i < other.sparse_capacity; i++) {
    const auto &oth_sparse_bkt = other.sparse_buckets[i];
    if (oth_sparse_bkt.position != uint16_t(-1) && oth_sparse_bkt.position != 0 &&
        oth_sparse_bkt.col() >= start_column && oth_sparse_bkt.col() < end_column) {
      if (oth_sparse_bkt.row() < num_dense_rows) {
        auto &bkt = bucket(oth_sparse_bkt.col(), oth_sparse_bkt.row());
        bkt.alpha ^= oth_sparse_bkt.bkt.alpha;
        bkt.gamma ^= oth_sparse_bkt.bkt.gamma;
      } else {
        update_sparse(oth_sparse_bkt);
      }
    }
  }
}

void SparseSketch::merge_raw_bucket_buffer(const Bucket *raw_buckets, size_t n_raw_buckets) {
  size_t num_merge_dense_rows = (n_raw_buckets - sparse_data_size - 1) / num_columns;
  const SparseBucket *raw_sparse =
      (const SparseBucket *) &raw_buckets[num_columns * num_merge_dense_rows + 1];

  deterministic_bucket().alpha ^= raw_buckets[0].alpha;
  deterministic_bucket().gamma ^= raw_buckets[0].gamma;

  for (size_t c = 0; c < num_columns; c++) {
    for (size_t r = 0; r < num_merge_dense_rows; r++) {
      if (r < num_dense_rows) {
        bucket(c, r).alpha ^= raw_buckets[position_func(c, r, num_merge_dense_rows)].alpha;
        bucket(c, r).gamma ^= raw_buckets[position_func(c, r, num_merge_dense_rows)].gamma;
      } else if (!Bucket_Boruvka::is_empty(
                     raw_buckets[position_func(c, r, num_merge_dense_rows)])) {
        SparseBucket sparse_bkt;
        sparse_bkt.set_col(c);
        sparse_bkt.set_row(r);
        sparse_bkt.bkt = raw_buckets[position_func(c, r, num_merge_dense_rows)];
        update_sparse(sparse_bkt);
      }
    }
  }

  for (size_t i = 0; i < sparse_capacity; i++) {
    const auto &oth_sparse_bkt = raw_sparse[i];
    if (oth_sparse_bkt.position != uint16_t(-1) && oth_sparse_bkt.position != 0) {
      if (oth_sparse_bkt.row() < num_dense_rows) {
        auto &bkt = bucket(oth_sparse_bkt.col(), oth_sparse_bkt.row());
        bkt.alpha ^= oth_sparse_bkt.bkt.alpha;
        bkt.gamma ^= oth_sparse_bkt.bkt.gamma;
      } else {
        update_sparse(oth_sparse_bkt);
      }
    }
  }
}

void SparseSketch::serialize(std::ostream &binary_out) const {
  binary_out.write((char*) buckets, bucket_array_bytes());
}

bool operator==(const SparseSketch &sketch1, const SparseSketch &sketch2) {
  if (sketch1.num_buckets != sketch2.num_buckets || sketch1.seed != sketch2.seed)
    return false;

  return memcmp(sketch1.buckets, sketch2.buckets, sketch1.bucket_array_bytes()) == 0;
}

std::ostream &operator<<(std::ostream &os, const SparseSketch &sketch) {
  Bucket bkt = sketch.buckets[sketch.num_buckets - 1];
  bool good = Bucket_Boruvka::is_good(bkt, sketch.checksum_seed());
  vec_t a = bkt.alpha;
  vec_hash_t c = bkt.gamma;

  os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;

  for (unsigned i = 0; i < sketch.num_columns; ++i) {
    for (unsigned j = 0; j < sketch.num_dense_rows; ++j) {
      Bucket bkt = sketch.bucket(i, j);
      vec_t a = bkt.alpha;
      vec_hash_t c = bkt.gamma;
      bool good = Bucket_Boruvka::is_good(bkt, sketch.checksum_seed());

      os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;
    }
    os << std::endl;
  }

  os << "Sparse Buckets" << std::endl;
  const auto sparse_buckets = sketch.sparse_buckets;
  for (size_t i = 0; i < sketch.sparse_capacity; i++) {
    bool good = Bucket_Boruvka::is_good(sparse_buckets[i].bkt, sketch.checksum_seed());
    os << " p:" << sparse_buckets[i].col() << ", " << sparse_buckets[i].row()
       << ":= a:" << sparse_buckets[i].bkt.alpha << " c:" << sparse_buckets[i].bkt.gamma
       << (good ? " good" : " bad") << std::endl;
  }
  os << std::endl;
  return os;
}
