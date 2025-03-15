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
  num_buckets = calc_num_buckets(num_dense_rows);
  buckets = new Bucket[num_buckets];
  upd_sparse_ptrs();

  // initialize bucket values
  for (size_t i = 0; i < num_buckets; ++i) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }

  // initialize sparse bucket linked lists
  // every bucket is currently free, so each points to next
  for (size_t i = 0; i < sparse_capacity; i++) {
    sparse_buckets[i].next = i + 1;
  }
  sparse_buckets[sparse_capacity - 1].next = uint8_t(-1);

  // initialize LL metadata
  for (size_t i = 0; i < num_columns; i++) {
    ll_metadata[i] = uint8_t(-1); // head of each column points nowhere (empty)
  }
  ll_metadata[num_columns] = 0; // free list head
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
  upd_sparse_ptrs();

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
  upd_sparse_ptrs();

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
  Bucket *old_buckets = buckets;

  if (new_num_dense_rows < min_num_dense_rows) {
    throw std::runtime_error("new_num_dense_rows too small!");
  }

  if (new_num_dense_rows < num_dense_rows) {
    // std::cerr << "Shrinking to " << new_num_dense_rows << " from " << old_rows << std::endl;
    // shrink dense region
    // Scan over the rows we are removing and add all those buckets to sparse
    for (size_t c = 0; c < num_columns; c++) {
      for (size_t r = new_num_dense_rows; r < old_rows; r++) {
        Bucket bkt = bucket(c, r);
        if (!Bucket_Boruvka::is_empty(bkt)) {
          SparseBucket new_sparse;
          new_sparse.row = r;
          new_sparse.bkt = bkt;
          update_sparse(c, new_sparse, false);
        }
      }
    }

    // Allocate new memory
    num_dense_rows = new_num_dense_rows;
    num_buckets = calc_num_buckets(num_dense_rows);
    buckets = new Bucket[num_buckets];
  } else {
    // std::cerr << "Growing to " << new_num_dense_rows << " from " << old_rows << std::endl;
    // grow dense region by 1 row
    // Allocate new memory
    num_dense_rows = new_num_dense_rows;
    num_buckets = calc_num_buckets(num_dense_rows);
    buckets = new Bucket[num_buckets];

    // initialize new rows to zero
    for (size_t c = 0; c < num_columns; c++) {
      for (size_t r = old_rows; r < num_dense_rows; r++) {
        buckets[position_func(c, r, num_dense_rows)] = {0, 0};
      }
    }
  }
  upd_sparse_ptrs();

  // Copy dense content
  buckets[0] = old_buckets[0];
  for (size_t c = 0; c < num_columns; c++) {
    for (size_t r = 0; r < std::min(num_dense_rows, old_rows); r++) {
      buckets[position_func(c, r, num_dense_rows)] = old_buckets[position_func(c, r, old_rows)];
    }
  }
  // sparse contents
  memcpy(sparse_buckets, old_sparse_pointer,
         (sparse_data_size + ll_metadata_size) * sizeof(Bucket));

  if (num_dense_rows > old_rows) {
    // We growing
    // Scan sparse buckets and move all updates of depth num_dense_rows-1
    // to the new dense row
    for (size_t c = 0; c < num_columns; c++) {
      while (ll_metadata[c] != uint8_t(-1) && sparse_buckets[ll_metadata[c]].row < num_dense_rows) {
        // remove this bucket from column ll
        uint8_t idx = ll_metadata[c];
        ll_metadata[c] = sparse_buckets[ll_metadata[c]].next;
        number_of_sparse_buckets -= 1;

        // add this bucket to dense region
        bucket(c, sparse_buckets[idx].row) = sparse_buckets[idx].bkt;

        // add this sparse_bucket to free list
        sparse_buckets[idx].bkt = {0, 0};
        sparse_buckets[idx].row = 0;
        sparse_buckets[idx].next = ll_metadata[num_columns];
        ll_metadata[num_columns] = idx;
      }
    }
  }

  // 4. Clean up
  delete[] old_buckets;
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
void SparseSketch::update_sparse(uint8_t col, SparseBucket to_add, bool realloc_if_needed) {
  uint8_t next_ptr = ll_metadata[col];
  uint8_t prev = uint8_t(-1);
  while (next_ptr != uint8_t(-1)) {
    if (sparse_buckets[next_ptr].row == to_add.row) {
      sparse_buckets[next_ptr].bkt.alpha ^= to_add.bkt.alpha;
      sparse_buckets[next_ptr].bkt.gamma ^= to_add.bkt.gamma;
      if (Bucket_Boruvka::is_empty(sparse_buckets[next_ptr].bkt)) {
        // remove this bucket from column list
        if (prev == uint8_t(-1)) {
          ll_metadata[col] = sparse_buckets[next_ptr].next;
        } else {
          sparse_buckets[prev].next = sparse_buckets[next_ptr].next;
        }
        number_of_sparse_buckets -= 1;

        // add this bucket to free list
        sparse_buckets[next_ptr].next = ll_metadata[num_columns];
        ll_metadata[num_columns] = next_ptr;

        if (realloc_if_needed) reallocate_if_needed(-1);
      }
      return; // we've done it!
    } else if (sparse_buckets[next_ptr].row > to_add.row) {
      break;
    }
    prev = next_ptr;
    next_ptr = sparse_buckets[next_ptr].next;
  }

  // pull a bucket off the free list and set it equal to to_add
  uint8_t free_bucket = ll_metadata[num_columns];
  // std::cerr << "free bucket = " << size_t(free_bucket) << std::endl;
  // std::cerr << "next bucket = " << size_t(next_ptr) << std::endl;
  if (free_bucket == uint8_t(-1)) {
    throw std::runtime_error("Found invalid bucket index in LL");
  }
  ll_metadata[num_columns] = sparse_buckets[free_bucket].next;
  // std::cerr << "free head = " << size_t(ll_metadata[num_columns]) << std::endl;

  // update buffer
  sparse_buckets[free_bucket] = to_add;
  sparse_buckets[free_bucket].next = next_ptr;
  number_of_sparse_buckets += 1;
  // std::cerr << "new bucket " << size_t(sparse_buckets[free_bucket].row) << " n = " << size_t(sparse_buckets[free_bucket].next) << std::endl;

  // update column ll
  if (prev == uint8_t(-1)) {
    ll_metadata[col] = free_bucket;
    // std::cerr << "Set column head to new bucket " << size_t(ll_metadata[col]) << std::endl;
  } else {
    sparse_buckets[prev].next = free_bucket;
    // std::cerr << "Placed new bucket in column " << size_t(prev) << "->" << size_t(sparse_buckets[prev].next) << "->" << size_t(sparse_buckets[free_bucket].next) << std::endl;
  }

  if (realloc_if_needed) reallocate_if_needed(1);
}

// sample a good bucket from the sparse region if one exists. 
// Additionally, specify the column to query from
SketchSample SparseSketch::sample_sparse(size_t first_col, size_t end_col) {
  // std::cerr << "sample_sparse" << std::endl;
  for (size_t c = first_col; c < end_col; c++) {
    uint8_t idx = ll_metadata[c];
    while (idx != uint8_t(-1)) {
      if (Bucket_Boruvka::is_good(sparse_buckets[idx].bkt, checksum_seed())) {
        return {sparse_buckets[idx].bkt.alpha, GOOD};
      }
      idx = sparse_buckets[idx].next;
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
        update_sparse(i, {uint8_t(-1), uint8_t(depth), {update_idx, checksum}});
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

  // initialize sparse bucket linked lists
  // every bucket is currently free, so each points to next
  for (size_t i = 0; i < sparse_capacity; i++) {
    sparse_buckets[i].next = i + 1;
  }
  sparse_buckets[sparse_capacity - 1].next = uint8_t(-1);

  // initialize LL metadata
  for (size_t i = 0; i < num_columns; i++) {
    ll_metadata[i] = uint8_t(-1); // head of each column points nowhere (empty)
  }
  ll_metadata[num_columns] = 0; // free list head
  
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
    for (int r = num_dense_rows - 1; r >= 0; --r) {
      if (Bucket_Boruvka::is_good(bucket(c + first_column, r), checksum_seed())) {
        // std::cout << "Found GOOD dense bucket" << std::endl;
        return {bucket(c + first_column, r).alpha, GOOD};
      }
    }
  }

  // Sample sparse region
  // std::cout << "Sketch is bad" << std::endl;
  // std::cout << *this << std::endl;
  return {0, FAIL};
}

ExhaustiveSketchSample SparseSketch::exhaustive_sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }
  std::unordered_set<vec_t> ret;

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  unlikely_if (Bucket_Boruvka::is_empty(deterministic_bucket()))
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
  // std::cerr << "PERFORMING A MERGE" << std::endl;
  // std::cerr << *this << std::endl;

  // std::cerr << "MERGE SKETCH" << std::endl;
  // std::cerr << other << std::endl;

  // merge the deterministic bucket
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
        sparse_bkt.row = r;
        sparse_bkt.bkt = other.bucket(c, r);
        update_sparse(c, sparse_bkt);
      }
    }
  }

  // Merge all sparse buckets from other sketch into this one
  for (size_t c = 0; c < num_columns; c++) {
    uint8_t this_idx = ll_metadata[c];
    uint8_t oth_idx = other.ll_metadata[c];

    while (oth_idx != uint8_t(-1)) {
      if (other.sparse_buckets[oth_idx].row < num_dense_rows) {
        auto &bkt = bucket(c, other.sparse_buckets[oth_idx].row);
        bkt.alpha ^= other.sparse_buckets[oth_idx].bkt.alpha;
        bkt.gamma ^= other.sparse_buckets[oth_idx].bkt.gamma;
      } else {
        // TODO: This can be made faster by utilizing this_idx and performing a merge operation
        update_sparse(c, other.sparse_buckets[oth_idx]);
      }
      oth_idx = other.sparse_buckets[oth_idx].next;
    }
  }
}

void SparseSketch::range_merge(const SparseSketch &other, size_t start_sample, size_t n_samples) {
  if (start_sample + n_samples > num_samples) {
    assert(false);
    sample_idx = num_samples; // sketch is in a fail state!
    return;
  }
  // std::cerr << "SKETCH BEFORE MERGE" << std::endl;
  // std::cerr << *this << std::endl;

  // std::cerr << "SKETCH WE MERGE WITH" << std::endl;
  // std::cerr << other << std::endl;

  // update sample idx to point at beginning of this range if before it
  sample_idx = std::max(sample_idx, start_sample);

  // Columns we be merging
  size_t start_column = start_sample * cols_per_sample;
  size_t end_column = (start_sample + n_samples) * cols_per_sample;

  // merge deterministic buffer
  deterministic_bucket().alpha ^= other.deterministic_bucket().alpha;
  deterministic_bucket().gamma ^= other.deterministic_bucket().gamma;

  // merge all their dense buckets into us
  for (size_t c = start_column; c < end_column; c++) {
    for (size_t r = 0; r < other.num_dense_rows; r++) {
      if (r < num_dense_rows) {
        bucket(c, r).alpha ^= other.bucket(c, r).alpha;
        bucket(c, r).gamma ^= other.bucket(c, r).gamma;
      } else if (!Bucket_Boruvka::is_empty(other.bucket(c, r))) {
        SparseBucket sparse_bkt;
        sparse_bkt.row = r;
        sparse_bkt.bkt = other.bucket(c, r);
        update_sparse(c, sparse_bkt);
      }
    }
  }

  // Merge all sparse buckets from other sketch into this one
  for (size_t c = start_column; c < end_column; c++) {
    uint8_t this_idx = ll_metadata[c];
    uint8_t oth_idx = other.ll_metadata[c];

    while (oth_idx != uint8_t(-1)) {
      if (other.sparse_buckets[oth_idx].row < num_dense_rows) {
        auto &bkt = bucket(c, other.sparse_buckets[oth_idx].row);
        bkt.alpha ^= other.sparse_buckets[oth_idx].bkt.alpha;
        bkt.gamma ^= other.sparse_buckets[oth_idx].bkt.gamma;
      } else {
        // TODO: This can be made faster by utilizing this_idx and performing a merge operation
        update_sparse(c, other.sparse_buckets[oth_idx]);
      }
      oth_idx = other.sparse_buckets[oth_idx].next;
    }
  }
  // std::cerr << "SKETCH AFTER MERGE" << std::endl;
  // std::cerr << *this << std::endl;
}

void SparseSketch::merge_raw_bucket_buffer(const Bucket *raw_buckets, size_t n_raw_buckets) {
  size_t raw_rows = (n_raw_buckets - sparse_data_size - ll_metadata_size - 1) / num_columns;
  const SparseBucket *raw_sparse = (const SparseBucket *) &raw_buckets[calc_sparse_index(raw_rows)];
  const uint8_t *raw_metadata = (const uint8_t *) &raw_buckets[calc_metadata_index(raw_rows)];

  deterministic_bucket().alpha ^= raw_buckets[0].alpha;
  deterministic_bucket().gamma ^= raw_buckets[0].gamma;

  for (size_t c = 0; c < num_columns; c++) {
    for (size_t r = 0; r < raw_rows; r++) {
      if (r < num_dense_rows) {
        bucket(c, r).alpha ^= raw_buckets[position_func(c, r, raw_rows)].alpha;
        bucket(c, r).gamma ^= raw_buckets[position_func(c, r, raw_rows)].gamma;
      } else if (!Bucket_Boruvka::is_empty(
                     raw_buckets[position_func(c, r, raw_rows)])) {
        SparseBucket sparse_bkt;
        sparse_bkt.row = r;
        sparse_bkt.bkt = raw_buckets[position_func(c, r, raw_rows)];
        update_sparse(c, sparse_bkt);
      }
    }
  }

  // Merge all sparse buckets from other sketch into this one
  for (size_t c = 0; c < num_columns; c++) {
    uint8_t this_idx = ll_metadata[c];
    uint8_t oth_idx = raw_metadata[c];

    while (oth_idx != uint8_t(-1)) {
      if (raw_sparse[oth_idx].row < num_dense_rows) {
        auto &bkt = bucket(c, raw_sparse[oth_idx].row);
        bkt.alpha ^= raw_sparse[oth_idx].bkt.alpha;
        bkt.gamma ^= raw_sparse[oth_idx].bkt.gamma;
      } else {
        // TODO: This can be made faster by utilizing this_idx and performing a merge operation
        update_sparse(c, raw_sparse[oth_idx]);
      }
      oth_idx = raw_sparse[oth_idx].next;
    }
  }
}

void SparseSketch::serialize(std::ostream &binary_out) const {
  binary_out.write((char*) buckets, bucket_array_bytes());
}

bool operator==(const SparseSketch &sketch1, const SparseSketch &sketch2) {
  if (sketch1.num_buckets != sketch2.num_buckets || sketch1.seed != sketch2.seed)
    return false;

  return memcmp(sketch1.buckets, sketch2.buckets,
                sketch1.bucket_array_bytes() - sketch1.ll_metadata_size * sizeof(Bucket)) == 0;
}

std::ostream &operator<<(std::ostream &os, const SparseSketch &sketch) {
  Bucket bkt = sketch.deterministic_bucket();
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
  for (size_t c = 0; c < sketch.num_columns; c++) {
    uint8_t idx = sketch.ll_metadata[c];
    while (idx != uint8_t(-1)) {
      bool good = Bucket_Boruvka::is_good(sparse_buckets[idx].bkt, sketch.checksum_seed());
      os << "i: " << size_t(idx) << " n: " << size_t(sparse_buckets[idx].next) << " p:" << c << ", "
         << size_t(sparse_buckets[idx].row) << " := a:" << sparse_buckets[idx].bkt.alpha
         << " c:" << sparse_buckets[idx].bkt.gamma << (good ? " good" : " bad") << std::endl;
      if (idx == sketch.sparse_buckets[idx].next) {
        os << "LL error!" << std::endl;
        return os;
      }
      idx = sketch.sparse_buckets[idx].next;
    }
  }
  os << std::endl;
  return os;
}
