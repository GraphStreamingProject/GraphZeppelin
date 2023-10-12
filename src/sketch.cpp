#include "sketch.h"
#include "bucket.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>

Sketch::Sketch(node_id_t n, uint64_t seed, size_t _samples, size_t _cols) : seed(seed) {
  num_samples = _samples == 0 ? samples_gen(n) : _samples;
  cols_per_sample = _cols == 0 ? default_cols_per_sample : _cols;
  num_columns = num_samples * cols_per_sample;
  num_guesses = guess_gen(n);
  num_buckets = num_columns * num_guesses + 1; // plus 1 for deterministic bucket

  // all bucket data is stored in contiguous array with bucket_a at the beginning
  // and bucket_c at the end. This allows for easy copying/serialization of the data
  size_t num_vec_slots = ceil(num_buckets * (1 + double(sizeof(vec_hash_t)) / sizeof(vec_t)));
  assert(num_vec_slots * sizeof(vec_t) >= num_buckets * (sizeof(vec_t) + sizeof(vec_hash_t)));
  bucket_memory = new vec_t[num_vec_slots];
  bucket_a = bucket_memory;
  bucket_c = (vec_hash_t *)&bucket_a[num_buckets];

  // initialize bucket values
  for (size_t i = 0; i < num_buckets; ++i) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }
}

Sketch::Sketch(node_id_t n, uint64_t seed, std::istream &binary_in, SerialType type,
               size_t _samples, size_t _cols) : seed(seed) {
  num_samples = _samples == 0 ? samples_gen(n) : _samples;
  cols_per_sample = _cols == 0 ? default_cols_per_sample : _cols;
  num_columns = num_samples * cols_per_sample;
  num_guesses = guess_gen(n);
  num_buckets = num_columns * num_guesses + 1; // plus 1 for deterministic bucket

  // all bucket data is stored in contiguous array with bucket_a at the beginning
  // and bucket_c at the end. This allows for easy copying/serialization of the data
  size_t num_vec_slots = ceil(num_buckets * (1 + double(sizeof(vec_hash_t)) / sizeof(vec_t)));
  assert(num_vec_slots * sizeof(vec_t) >= num_buckets * (sizeof(vec_t) + sizeof(vec_hash_t)));
  bucket_memory = new vec_t[num_vec_slots];
  bucket_a = bucket_memory;
  bucket_c = (vec_hash_t *)&bucket_a[num_buckets];

  // Read the serialized Sketch contents
  if (type == FULL) {
    binary_in.read((char *)bucket_memory, num_buckets * (sizeof(vec_t) + sizeof(vec_hash_t)));
  } else if (type == PARTIAL) {
    for (size_t i = 0; i < num_buckets; ++i) {
      bucket_a[i] = 0;
      bucket_c[i] = 0;
    }

    // TODO!
    exit(EXIT_FAILURE);
  } else {
    for (size_t i = 0; i < num_buckets; ++i) {
      bucket_a[i] = 0;
      bucket_c[i] = 0;
    }

    uint16_t idx;
    binary_in.read((char *)&idx, sizeof(idx));
    while (idx < num_buckets - 1) {
      binary_in.read((char *)&bucket_a[idx], sizeof(bucket_a[idx]));
      binary_in.read((char *)&bucket_c[idx], sizeof(bucket_c[idx]));
      binary_in.read((char *)&idx, sizeof(idx));
    }
    // finally handle the level 0 bucket (num_buckets - 1)
    binary_in.read((char *)&bucket_a[idx], sizeof(bucket_a[idx]));
    binary_in.read((char *)&bucket_c[idx], sizeof(bucket_c[idx]));
  }
}

Sketch::Sketch(const Sketch &s) : seed(s.seed) {
  num_samples = s.num_samples;
  cols_per_sample = s.cols_per_sample;
  num_columns = s.num_columns;
  num_guesses = s.num_guesses;
  num_buckets = s.num_buckets;

  size_t num_vec_slots = ceil(num_buckets * (1 + double(sizeof(vec_hash_t)) / sizeof(vec_t)));
  bucket_memory = new vec_t[num_vec_slots];
  bucket_a = bucket_memory;
  bucket_c = (vec_hash_t *)&bucket_a[num_buckets];

  std::memcpy(bucket_memory, s.bucket_memory, num_vec_slots * sizeof(vec_t));
}

Sketch::~Sketch() { delete[] bucket_memory; }

#ifdef L0_SAMPLING
void Sketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, checksum_seed());

  // Update depth 0 bucket
  Bucket_Boruvka::update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_columns; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, column_seed(i), num_guesses);
    likely_if(depth < num_guesses) {
      for (col_hash_t j = 0; j <= depth; ++j) {
        size_t bucket_id = i * num_guesses + j;
        Bucket_Boruvka::update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, checksum);
      }
    }
  }
}
#else  // Use support finding algorithm instead. Faster but no guarantee of uniform sample.
void Sketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, checksum_seed());

  // Update depth 0 bucket
  Bucket_Boruvka::update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_columns; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, column_seed(i), num_guesses);
    size_t bucket_id = i * num_guesses + depth;
    likely_if(depth < num_guesses) {
      Bucket_Boruvka::update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, checksum);
    }
  }
}
#endif

void Sketch::zero_contents() {
  for (size_t i = 0; i < num_buckets; i++) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }
}

std::pair<vec_t, SampleSketchRet> Sketch::sample() {
  if (sample_idx >= num_samples) {
    throw OutOfQueriesException();
  }

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  if (bucket_a[num_buckets - 1] == 0 && bucket_c[num_buckets - 1] == 0)
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return

  if (Bucket_Boruvka::is_good(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1],
                              checksum_seed()))
    return {bucket_a[num_buckets - 1], GOOD};

  for (unsigned i = 0; i < cols_per_sample; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = (i + first_column) * num_guesses + j;
      if (Bucket_Boruvka::is_good(bucket_a[bucket_id], bucket_c[bucket_id], checksum_seed()))
        return {bucket_a[bucket_id], GOOD};
    }
  }
  return {0, FAIL};
}

std::pair<std::unordered_set<vec_t>, SampleSketchRet> Sketch::exhaustive_sample() {
  // TODO!
  exit(EXIT_FAILURE);
}

void Sketch::merge(Sketch &other) {
  if (other.bucket_a[num_buckets-1] == 0 && other.bucket_c[num_buckets-1] == 0) {
    // other sketch is empty so just return
    return;
  }

  // perform the merge
  for (size_t i = 0; i < num_buckets; ++i) {
    bucket_a[i] ^= other.bucket_a[i];
    bucket_c[i] ^= other.bucket_c[i];
  }
}

void Sketch::range_merge(Sketch &other, size_t start_idx, size_t num_merge) {
  // TODO!
  exit(EXIT_FAILURE);
}

void Sketch::serialize(std::ostream &binary_out, SerialType type) const {
  if (type == FULL) {
    binary_out.write((char*) bucket_memory, num_buckets * (sizeof(vec_t) + sizeof(vec_hash_t)));
  }
  else if (type == PARTIAL) {
    // TODO!
    exit(EXIT_FAILURE);
  }
  else {
    // TODO!
    exit(EXIT_FAILURE);
  }
}

bool operator==(const Sketch &sketch1, const Sketch &sketch2) {
  if (sketch1.num_buckets != sketch2.num_buckets || sketch1.seed != sketch2.seed) {
    std::cout << "sketch1 = " << sketch1 << std::endl;
    std::cout << "sketch2 = " << sketch2 << std::endl;
    return false;
  }

  for (size_t i = 0; i < sketch1.num_buckets; ++i) {
    if (sketch1.bucket_a[i] != sketch2.bucket_a[i] || sketch1.bucket_c[i] != sketch2.bucket_c[i]) {
      std::cout << i << std::endl;
      std::cout << "sketch1 = " << sketch1 << std::endl;
      std::cout << "sketch2 = " << sketch2 << std::endl;
      return false;
    }
  }

  return true;
}

std::ostream &operator<<(std::ostream &os, const Sketch &sketch) {
  vec_t a = sketch.bucket_a[sketch.num_buckets - 1];
  vec_hash_t c = sketch.bucket_c[sketch.num_buckets - 1];
  bool good = Bucket_Boruvka::is_good(a, c, sketch.checksum_seed());

  os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;

  for (unsigned i = 0; i < sketch.num_columns; ++i) {
    for (unsigned j = 0; j < sketch.num_guesses; ++j) {
      unsigned bucket_id = i * sketch.num_guesses + j;
      vec_t a = sketch.bucket_a[bucket_id];
      vec_hash_t c = sketch.bucket_c[bucket_id];
      bool good = Bucket_Boruvka::is_good(a, c, sketch.checksum_seed());

      os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;
    }
    os << std::endl;
  }
  return os;
}
