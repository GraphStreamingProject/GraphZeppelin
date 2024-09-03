#include "sketch.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>


inline static void set_bit(vec_t &t, int position) {
  t |= 1 << position;
}

inline static void clear_bit(vec_t &t, int position) {
  t &= ~(1 << position);
}

Sketch::Sketch(vec_t vector_len, uint64_t seed, size_t _samples, size_t _cols) : seed(seed) {
  num_samples = _samples;
  cols_per_sample = _cols;
  num_columns = num_samples * cols_per_sample;
  bkt_per_col = calc_bkt_per_col(vector_len);
  num_buckets = num_columns * bkt_per_col + 1; // plus 1 for deterministic bucket
#ifdef EAGER_BUCKET_CHECK
  buckets = (Bucket*) (new char[bucket_array_bytes()]);
  nonempty_buckets = (vec_t*) (buckets + num_buckets);
#else
  buckets = new Bucket[num_buckets];
#endif

  // initialize bucket values
  for (size_t i = 0; i < num_buckets; ++i) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }

#ifdef EAGER_BUCKET_CHECK
  for (size_t i = 0; i < num_columns; ++i) {
    nonempty_buckets[i] = 0;
  }
#endif

}


Sketch::Sketch(vec_t vector_len, uint64_t seed, bool compressed, std::istream &binary_in,
               size_t _samples, size_t _cols)
    : seed(seed) {
  num_samples = _samples;
  cols_per_sample = _cols;
  num_columns = num_samples * cols_per_sample;
  bkt_per_col = calc_bkt_per_col(vector_len);
  num_buckets = num_columns * bkt_per_col + 1; // plus 1 for deterministic bucket
  buckets = (Bucket*) new char[bucket_array_bytes()];
#ifdef EAGER_BUCKET_CHECK
  nonempty_buckets = (vec_t*) (buckets + num_buckets);
#endif
  if (compressed) {
    compressed_deserialize(binary_in);
  }
  else {
    binary_in.read((char *)buckets, bucket_array_bytes());
  }
}


  /**
   * Occupies the contents of an empty sketch with input from a stream that contains
   * the compressed version.
   * @param binary_in   Stream holding serialized/compressed sketch object.
   */
void Sketch::compressed_deserialize(std::istream& binary_in) {
  uint8_t sizes[num_columns];
  // Read the serialized Sketch contents
  // first, read in the sizes
  binary_in.read((char *) sizes, sizeof(uint8_t) * num_columns);
#ifdef EAGER_BUCKET_CHECK
  binary_in.read((char *) nonempty_buckets, sizeof(vec_t) * num_columns);  
#endif
  // grab the deterministic bucket:
  binary_in.read((char *) (buckets + num_buckets -1), sizeof(Bucket));
  for (size_t col_idx=0; col_idx < num_columns; col_idx++) {
    Bucket *current_column = buckets + (col_idx * bkt_per_col);
    binary_in.read((char *) current_column, sizeof(Bucket) * sizes[col_idx]);
  }
}

Sketch::Sketch(vec_t vector_len, uint64_t seed, std::istream &binary_in, size_t _samples,
               size_t _cols):
    seed(seed) {
  num_samples = _samples;
  cols_per_sample = _cols;
  num_columns = num_samples * cols_per_sample;
  bkt_per_col = calc_bkt_per_col(vector_len);
  num_buckets = num_columns * bkt_per_col + 1; // plus 1 for deterministic bucket
  buckets = (Bucket*) new char[bucket_array_bytes()];
#ifdef EAGER_BUCKET_CHECK
  nonempty_buckets = (vec_t*) (buckets + num_buckets);
#endif
  binary_in.read((char *)buckets, bucket_array_bytes());

    }

Sketch::Sketch(const Sketch &s) : seed(s.seed) {
  num_samples = s.num_samples;
  cols_per_sample = s.cols_per_sample;
  num_columns = s.num_columns;
  bkt_per_col = s.bkt_per_col;
  num_buckets = s.num_buckets;
  buckets = (Bucket*) new char[bucket_array_bytes()];
  // buckets = new Bucket[num_buckets];

  std::memcpy(buckets, s.buckets, bucket_array_bytes());

  #ifdef EAGER_BUCKET_CHECK
  nonempty_buckets = (vec_t*) (buckets + num_buckets);
  #endif
}

Sketch::~Sketch() { 
  delete[] buckets;
 }



#ifdef L0_SAMPLING
void Sketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, checksum_seed());

  // Update depth 0 bucket
  Bucket_Boruvka::update(buckets[num_buckets - 1], update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_columns; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, column_seed(i), bkt_per_col);
    likely_if(depth < bkt_per_col) {
      for (col_hash_t j = 0; j <= depth; ++j) {
        size_t bucket_id = i * bkt_per_col + j;
        Bucket_Boruvka::update(buckets[bucket_id], update_idx, checksum);
      }
#ifdef EAGER_BUCKET_CHECK
      recalculate_flags(i, 0, depth);
#endif
    }
  }
}
#else  // Use support finding algorithm instead. Faster but no guarantee of uniform sample.
void Sketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, checksum_seed());

  // Update depth 0 bucket
  Bucket_Boruvka::update(buckets[num_buckets - 1], update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_columns; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, column_seed(i), bkt_per_col);
    size_t bucket_id = i * bkt_per_col + depth;
    likely_if(depth < bkt_per_col) {
      Bucket_Boruvka::update(buckets[bucket_id], update_idx, checksum);
      #ifdef EAGER_BUCKET_CHECK
      likely_if(!Bucket_Boruvka::is_empty(buckets[bucket_id])) {
        set_bit(nonempty_buckets[i], depth);
      } else {
        clear_bit(nonempty_buckets[i], depth);
      }
      #endif
    }
  }
}
#endif

void Sketch::zero_contents() {
  for (size_t i = 0; i < num_buckets; i++) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }
  reset_sample_state();
}

SketchSample Sketch::sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  if (Bucket_Boruvka::is_empty(buckets[num_buckets - 1]))
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return

  if (Bucket_Boruvka::is_good(buckets[num_buckets - 1], checksum_seed()))
    return {buckets[num_buckets - 1].alpha, GOOD};


  for (size_t col = first_column; col < first_column + cols_per_sample; ++col) {
    int row = int(effective_size(col))-1;
    int window_size = 6;  // <- log_2(64), the maximum sketch depth
    // now that we've found a non-zero bucket check next if next 6 buckets good
    int stop = std::max(row - window_size, 0);
    for (; row >= stop; row--) {
      if (Bucket_Boruvka::is_good(buckets[col * bkt_per_col + row], checksum_seed()))
        return {buckets[col * bkt_per_col + row].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

ExhaustiveSketchSample Sketch::exhaustive_sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }
  std::unordered_set<vec_t> ret;

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  unlikely_if (Bucket_Boruvka::is_empty(buckets[num_buckets -1]))
    return {ret, ZERO}; // the "first" bucket is deterministic so if zero then no edges to return

  unlikely_if (Bucket_Boruvka::is_good(buckets[num_buckets - 1], checksum_seed())) {
    ret.insert(buckets[num_buckets - 1].alpha);
    return {ret, GOOD};
  }

  for (size_t col = first_column; col < first_column + cols_per_sample; ++col) {
    int row = effective_size(col)-1;
    int window_size = 6;
    int stop = std::max(row - window_size, 0);
    for (; row >= stop; row--) {
      if (Bucket_Boruvka::is_good(buckets[col * bkt_per_col + row], checksum_seed()))
        ret.insert(buckets[col * bkt_per_col + row].alpha);
    }
  }

  unlikely_if (ret.size() == 0)
    return {ret, FAIL};
  return {ret, GOOD};
}


void Sketch::merge(const Sketch &other) {
  for (size_t i=0; i < num_columns; ++i) {
    Bucket *current_col = buckets + (i* bkt_per_col);
    Bucket *other_col = other.buckets + (i * bkt_per_col);
    size_t other_effective_size = other.effective_size(i);
    // size_t other_effective_size = bkt_per_col;
    #pragma omp simd
    for (size_t bucket_id=0; bucket_id < other_effective_size; bucket_id++) {
      current_col[bucket_id].alpha ^= other_col[bucket_id].alpha;
      current_col[bucket_id].gamma ^= other_col[bucket_id].gamma;
    }
#ifdef EAGER_BUCKET_CHECK
    recalculate_flags(i, 0, other_effective_size);
#endif
  }

  // seperately update the deterministic bucket
  buckets[num_buckets-1].alpha ^= other.buckets[num_buckets-1].alpha;
  buckets[num_buckets-1].gamma ^= other.buckets[num_buckets-1].gamma;
}


#ifdef EAGER_BUCKET_CHECK
void Sketch::recalculate_flags(size_t col_idx, size_t start_idx, size_t end_idx) {
  Bucket *current_col = buckets + (col_idx * bkt_per_col);
  assert(end_idx >= start_idx);
  vec_t clear_mask = (~0) >> (8*sizeof(vec_t) - (end_idx - start_idx));
  clear_mask = ~(clear_mask << start_idx);
  vec_t col_nonempty_buckets = 0;
  // vec_t col_nonempty_buckets = ~0;
  #pragma omp simd
  for (size_t bucket_id=start_idx; bucket_id < end_idx; bucket_id++) {
    likely_if(!Bucket_Boruvka::is_empty(current_col[bucket_id])) set_bit(col_nonempty_buckets, bucket_id);
    // unlikely_if(Bucket_Boruvka::is_empty(current_col[bucket_id])) clear_bit(col_nonempty_buckets,bucket_id);
  }
  nonempty_buckets[col_idx] = (nonempty_buckets[col_idx] & clear_mask) | (col_nonempty_buckets & ~clear_mask);
}
#endif



void Sketch::range_merge(const Sketch &other, size_t start_sample, size_t n_samples) {
  if (start_sample + n_samples > num_samples) {
    assert(false);
    sample_idx = num_samples; // sketch is in a fail state!
    return;
  }

  // update sample idx to point at beginning of this range if before it
  sample_idx = std::max(sample_idx, start_sample);

  // merge deterministic buffer
  buckets[num_buckets - 1].alpha ^= other.buckets[num_buckets - 1].alpha;
  buckets[num_buckets - 1].gamma ^= other.buckets[num_buckets - 1].gamma;

  // merge other buckets
  size_t start_bucket_id = start_sample * cols_per_sample * bkt_per_col;
  size_t n_buckets = n_samples * cols_per_sample * bkt_per_col;

  for (size_t i = 0; i < n_buckets; i++) {
    size_t bucket_id = start_bucket_id + i;
    buckets[bucket_id].alpha ^= other.buckets[bucket_id].alpha;
    buckets[bucket_id].gamma ^= other.buckets[bucket_id].gamma;
  }
#ifdef EAGER_BUCKET_CHECK
  size_t start_col_id = start_sample * cols_per_sample;
  size_t end_col_id = (start_sample + n_samples) * cols_per_sample;
  for (size_t i=start_col_id; i < end_col_id; i++ ) {
    recalculate_flags(i, 0, other.effective_size(i));
  }
#endif
}

void Sketch::merge_raw_bucket_buffer(const Bucket *raw_buckets) {
  for (size_t i = 0; i < num_buckets; i++) {
    buckets[i].alpha ^= raw_buckets[i].alpha;
    buckets[i].gamma ^= raw_buckets[i].gamma;
  }
#ifdef EAGER_BUCKET_CHECK
  for (size_t col_idx=0; col_idx < num_columns; col_idx++) {
    recalculate_flags(col_idx, 0, bkt_per_col);
  }
#endif
}

uint8_t Sketch::effective_size(size_t col_idx) const
{
  // first, check for emptyness
  Bucket *current_row = buckets + (col_idx * bkt_per_col);
  if (Bucket_Boruvka::is_empty(buckets[num_buckets - 1]))
  {
    return 0;
  }
// #ifdef EAGER_BUCKET_CHECK
//   unlikely_if(nonempty_buckets[col_idx] == 0) return 0;
//   return (uint8_t)((sizeof(unsigned long long) * 8) - __builtin_clzll(nonempty_buckets[col_idx]));
// #else
  uint8_t idx = bkt_per_col - 1;
  while (idx != 0 && Bucket_Boruvka::is_empty(current_row[idx]))
  {
    idx--;
  }
  unlikely_if(idx == 0 && Bucket_Boruvka::is_empty(current_row[idx])) return 0;
  else return idx + 1;
// #endif
}

void Sketch::compressed_serialize(std::ostream &binary_out) const {
  uint8_t sizes[num_columns];
  for (size_t i=0; i < num_columns; i++) {
    sizes[i] = effective_size(i);
  }
  binary_out.write((char*) sizes, sizeof(uint8_t) * num_columns);
  #ifdef EAGER_BUCKET_CHECK
  binary_out.write((char *) nonempty_buckets, sizeof(vec_t) * num_columns);
  #endif
  binary_out.write((char *) (buckets + num_buckets-1), sizeof(Bucket));
  for (size_t i=0; i < num_columns; i++) {
    Bucket *current_column = buckets + (i * bkt_per_col);
    binary_out.write((char *) current_column, sizeof(Bucket) * sizes[i]);
  }
}

void Sketch::serialize(std::ostream &binary_out) const {
  // note that these will include the flag bits, if used.
  binary_out.write((char*) buckets, bucket_array_bytes());
}

bool operator==(const Sketch &sketch1, const Sketch &sketch2) {
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

std::ostream &operator<<(std::ostream &os, const Sketch &sketch) {
  Bucket bkt = sketch.buckets[sketch.num_buckets - 1];
  bool good = Bucket_Boruvka::is_good(bkt, sketch.checksum_seed());
  vec_t a = bkt.alpha;
  vec_hash_t c = bkt.gamma;

  os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;

  for (unsigned i = 0; i < sketch.num_columns; ++i) {
    for (unsigned j = 0; j < sketch.bkt_per_col; ++j) {
      unsigned bucket_id = i * sketch.bkt_per_col + j;
      Bucket bkt = sketch.buckets[bucket_id];
      vec_t a = bkt.alpha;
      vec_hash_t c = bkt.gamma;
      bool good = Bucket_Boruvka::is_good(bkt, sketch.checksum_seed());

      os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;
    }
    os << std::endl;
  }
  return os;
}
