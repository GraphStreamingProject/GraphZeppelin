#include "dense_sketch.h"

#include <cassert>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

DenseSketch::DenseSketch(vec_t vector_len, uint64_t seed, size_t _samples, size_t _cols)
    : seed(seed),
      num_samples(_samples),
      cols_per_sample(_cols),
      num_columns(cols_per_sample * num_samples),
      bkt_per_col(calc_bkt_per_col(vector_len)) {

  num_buckets = num_columns * bkt_per_col + 1; // plus 1, deterministic bucket
  buckets = new Bucket[num_buckets];

  // initialize bucket values
  for (size_t i = 0; i < num_buckets; ++i) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }
}

DenseSketch::DenseSketch(vec_t vector_len, uint64_t seed, std::istream &binary_in,
                         size_t num_buckets, size_t _samples, size_t _cols)
    : seed(seed),
      num_samples(_samples),
      cols_per_sample(_cols),
      num_columns(cols_per_sample * num_samples),
      bkt_per_col(calc_bkt_per_col(vector_len)),
      num_buckets(num_buckets) {
  if (num_buckets != num_columns * bkt_per_col + 1) {
    throw std::invalid_argument("Serial Constructor: Number of buckets does not match expectation");
  }
  num_buckets = num_columns * bkt_per_col + 1; // plus 1 for deterministic bucket
  buckets = new Bucket[num_buckets];

  // Read the serialized Sketch contents
  binary_in.read((char *)buckets, bucket_array_bytes());
}

DenseSketch::DenseSketch(const DenseSketch &s) 
    : seed(s.seed),
      num_samples(s.num_samples),
      cols_per_sample(s.cols_per_sample),
      num_columns(s.num_columns),
      bkt_per_col(s.bkt_per_col) {
  num_buckets = s.num_buckets;
  buckets = new Bucket[num_buckets];

  std::memcpy(buckets, s.buckets, bucket_array_bytes());
}

DenseSketch::~DenseSketch() { delete[] buckets; }


void DenseSketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, checksum_seed());

  // Update depth 0 bucket
  Bucket_Boruvka::update(deterministic_bucket(), update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_columns; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, column_seed(i), bkt_per_col);
    likely_if(depth < bkt_per_col) {
      Bucket_Boruvka::update(bucket(i, depth), update_idx, checksum);
    }
  }
}

static void is_empty(DenseSketch &skt) {
  const Bucket* buckets = skt.get_readonly_bucket_ptr();
  for (size_t i = 0; i < skt.get_buckets(); i++) {
    if (!Bucket_Boruvka::is_empty(buckets[i])) {
      std::cerr << "FOUND NOT EMPTY BUCKET!" << std::endl;
    }
  }
}

// TODO: Switch the L0_SAMPLING flag to instead affect query procedure.
// (Only use deepest bucket. We don't need the alternate update procedure in the code anymore.)

void DenseSketch::zero_contents() {
  for (size_t i = 0; i < num_buckets; i++) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }
  reset_sample_state();
}

SketchSample DenseSketch::sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  // std::cout << "Sampling: " << first_column  << ", " << first_column + cols_per_sample << std::endl;

  // std::cout << *this << std::endl;

  if (Bucket_Boruvka::is_empty(deterministic_bucket())) {
    is_empty(*this);
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  }

  if (Bucket_Boruvka::is_good(deterministic_bucket(), checksum_seed()))
    return {deterministic_bucket().alpha, GOOD};

  for (size_t i = 0; i < cols_per_sample; ++i) {
    for (size_t j = 0; j < bkt_per_col; ++j) {
      if (Bucket_Boruvka::is_good(bucket(i + first_column, j), checksum_seed()))
        return {bucket(i + first_column, j).alpha, GOOD};
    }
  }
  return {0, FAIL};
}

ExhaustiveSketchSample DenseSketch::exhaustive_sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }
  std::vector<vec_t> ret;

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  unlikely_if (deterministic_bucket().alpha == 0 && deterministic_bucket().gamma == 0)
    return {ret, ZERO}; // the "first" bucket is deterministic so if zero then no edges to return

  unlikely_if (Bucket_Boruvka::is_good(deterministic_bucket(), checksum_seed())) {
    ret.push_back(deterministic_bucket().alpha);
    return {ret, GOOD};
  }

  for (size_t i = 0; i < cols_per_sample; ++i) {
    for (size_t j = 0; j < bkt_per_col; ++j) {
      unlikely_if (Bucket_Boruvka::is_good(bucket(i + first_column, j), checksum_seed())) {
        ret.push_back(bucket(i + first_column, j).alpha);
      }
    }
  }

  unlikely_if (ret.size() == 0)
    return {ret, FAIL};
  return {ret, GOOD};
}

void DenseSketch::merge(const DenseSketch &other) {
  for (size_t i = 0; i < num_buckets; ++i) {
    buckets[i].alpha ^= other.buckets[i].alpha;
    buckets[i].gamma ^= other.buckets[i].gamma;
  }
}

void DenseSketch::range_merge(const DenseSketch &other, size_t start_sample, size_t n_samples) {
  if (start_sample + n_samples > num_samples) {
    assert(false);
    sample_idx = num_samples; // sketch is in a fail state!
    return;
  }

  // std::cout << "MERGING THIS" << std::endl;
  // std::cout << *this << std::endl;
  // std::cout << "WITH THIS" << std::endl;
  // std::cout << other << std::endl;

  // update sample idx to point at beginning of this range if before it
  sample_idx = std::max(sample_idx, start_sample);

  // merge deterministic bucket
  // TODO: I don't like this. Repeated calls to range_merge on same sketches will potentially cause us issues
  deterministic_bucket().alpha ^= other.deterministic_bucket().alpha;
  deterministic_bucket().gamma ^= other.deterministic_bucket().gamma;

  // merge other buckets
  size_t start_column = start_sample * cols_per_sample;
  size_t end_column = (start_sample + n_samples) * cols_per_sample;

  // std::cout << start_column << ", " << end_column << std::endl;
  for (size_t i = start_column; i < end_column; i++) {
    for (size_t j = 0; j < bkt_per_col; j++) {
      bucket(i, j).alpha ^= other.bucket(i, j).alpha;
      bucket(i, j).gamma ^= other.bucket(i, j).gamma;
    }
  }
  
  // std::cout << "RESULT" << std::endl;
  // std::cout << *this << std::endl;
}

void DenseSketch::merge_raw_bucket_buffer(const Bucket *raw_buckets, size_t n_raw_buckets) {
  if (n_raw_buckets != num_buckets) {
    throw std::invalid_argument("Raw bucket buffer is not the same size as DenseSketch");
  }

  for (size_t i = 0; i < num_buckets; i++) {
    buckets[i].alpha ^= raw_buckets[i].alpha;
    buckets[i].gamma ^= raw_buckets[i].gamma;
  }
}

void DenseSketch::serialize(std::ostream &binary_out) const {
  binary_out.write((char*) buckets, bucket_array_bytes());
}

bool operator==(const DenseSketch &sketch1, const DenseSketch &sketch2) {
  if (sketch1.num_buckets != sketch2.num_buckets || sketch1.seed != sketch2.seed)
    return false;

  for (size_t i = 0; i < sketch1.num_buckets; ++i) {
    if (sketch1.buckets[i].alpha != sketch2.buckets[i].alpha ||
        sketch1.buckets[i].gamma != sketch2.buckets[i].gamma) {
      return false;
    }
  }

  return true;
}

std::ostream &operator<<(std::ostream &os, const DenseSketch &sketch) {
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
