#include "sketch.h"

#include <cstring>
#include <iostream>
#include <vector>
#include <cassert>

Sketch::Sketch(vec_t vector_len, uint64_t seed, size_t _samples, size_t _cols) : seed(seed) {
  num_samples = _samples;
  cols_per_sample = _cols;
  num_columns = num_samples * cols_per_sample;
  bkt_per_col = calc_bkt_per_col(vector_len);
  num_buckets = num_columns * bkt_per_col + 1; // plus 1 for deterministic bucket
  buckets = new Bucket[num_buckets];

  // initialize bucket values
  for (size_t i = 0; i < num_buckets; ++i) {
    buckets[i].alpha = 0;
    buckets[i].gamma = 0;
  }

  #ifdef EAGER_BUCKET_CHECK
  good_buckets = new vec_t[num_columns];
  for (size_t i = 0; i < num_columns; ++i) {
    good_buckets[i] = 0;
  }
  #endif

}

Sketch::Sketch(vec_t vector_len, uint64_t seed, std::istream &binary_in, size_t _samples,
               size_t _cols)
    : seed(seed) {
  num_samples = _samples;
  cols_per_sample = _cols;
  num_columns = num_samples * cols_per_sample;
  bkt_per_col = calc_bkt_per_col(vector_len);
  num_buckets = num_columns * bkt_per_col + 1; // plus 1 for deterministic bucket
  buckets = new Bucket[num_buckets];

  // Read the serialized Sketch contents
  binary_in.read((char *)buckets, bucket_array_bytes());
}

Sketch::Sketch(const Sketch &s) : seed(s.seed) {
  num_samples = s.num_samples;
  cols_per_sample = s.cols_per_sample;
  num_columns = s.num_columns;
  bkt_per_col = s.bkt_per_col;
  num_buckets = s.num_buckets;
  buckets = new Bucket[num_buckets];

  std::memcpy(buckets, s.buckets, bucket_array_bytes());

  #ifdef EAGER_BUCKET_CHECK
  good_buckets = new vec_t[num_columns];
  std::memcpy(good_buckets, s.good_buckets, sizeof(vec_t) * num_columns);
  #endif
}

Sketch::~Sketch() { 
  delete[] buckets;
  #ifdef EAGER_BUCKET_CHECK
  delete[] good_buckets;
  #endif
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
      #if defined EAGER_BUCKET_CHECK
      // bool bucket_was_empty = Bucket_Boruvka::is_empty(buckets[bucket_id], checksum_seed());
      // bool bucket_contained_entry = buckets[bucket_id].alpha == update_idx && buckets[bucket_id].gamma == checksum;
      Bucket_Boruvka::update(buckets[bucket_id], update_idx, checksum);
      // unlikely_if(bucket_was_empty && ) {
      #if EAGER_BUCKET_CHECK == EmptyOnly
      likely_if(!Bucket_Boruvka::is_empty(buckets[bucket_id])) {
      #else 
      unlikely_if(Bucket_Boruvka::is_good(buckets[bucket_id], checksum_seed())) {
      #endif
        // set the bit to 1
        good_buckets[i] = (1 << depth) ^ ( good_buckets[i]  & ~(1 << depth) );
      }
      else {
        good_buckets[i] &= ~(1 << depth);
      }
      #else
      Bucket_Boruvka::update(buckets[bucket_id], update_idx, checksum);
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

SketchSample Sketch::fast_sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }

  size_t idx = sample_idx++;

  if (Bucket_Boruvka::is_empty(buckets[num_buckets-1]))
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  if (Bucket_Boruvka::is_good(buckets[num_buckets - 1], checksum_seed()))
    return {buckets[num_buckets - 1].alpha, GOOD};


  int window_size = 4+(sizeof(unsigned long long)*8 - __builtin_clzll(bkt_per_col))/4;
  // int window_size = 6;


  for (size_t col=0; col< num_columns; col++) {

    Bucket* current_column = buckets + ((idx * cols_per_sample + col)  * bkt_per_col);

    // for (size_t idx=0; idx < 2; idx++) {
    //   if (Bucket_Boruvka::is_good(current_column[idx], checksum_seed()))
    //     return {current_column[idx].alpha, GOOD};
    // }
    if (!Bucket_Boruvka::is_empty(buckets[std::max(((int) bkt_per_col)- 2*window_size, 0)])) {
      size_t row = bkt_per_col-1;
      while (Bucket_Boruvka::is_empty(current_column[row]) && row >= 0) {
        row--;
      }
      for (size_t idx=row; idx > row - window_size && idx >= 0; idx-- ) {
        if (Bucket_Boruvka::is_good(current_column[idx], checksum_seed()))
          return {current_column[idx].alpha, GOOD};
      }
      continue;
    }

    if (Bucket_Boruvka::is_empty(buckets[std::min((size_t) 2*window_size, bkt_per_col-1)])) {
      size_t row = std::max((size_t) 2*window_size, bkt_per_col-1);
      while (Bucket_Boruvka::is_empty(current_column[row]) && row >= 0) {
        row--;
      }
      for (size_t idx=row; idx >= 0; idx-- ) {
        if (Bucket_Boruvka::is_good(current_column[idx], checksum_seed()))
          return {current_column[idx].alpha, GOOD};
      }
      continue;
    }
    // NOTE - we want to take advantage of signed types here
    int lo=1, hi=bkt_per_col-1;

    // while (lo + window_size < bkt_per_col && !(
    //   !Bucket_Boruvka::is_empty(current_column[lo]) && Bucket_Boruvka::is_empty(current_column[lo+window_size])
    //   )) {
    //     lo *= 2;
    // }
    // hi = std::min(lo+2*window_size, hi);
    // lo /= 2;
    int midpt = (lo+hi)/2;
    static size_t max_search_iterations = 3;
    size_t search_itr = 0;
    while (hi - lo >= window_size && midpt + window_size <= bkt_per_col && midpt - window_size >= 0 && search_itr++ < max_search_iterations) {
      if (!Bucket_Boruvka::is_empty(current_column[midpt]) && Bucket_Boruvka::is_empty(current_column[midpt+window_size])) {
        // lo = midpt-window_size;
        // hi = midpt+ (2*window_size);
        lo = midpt;
        hi = midpt + window_size;
        break;
      }
      else if (Bucket_Boruvka::is_empty(current_column[midpt]) && !Bucket_Boruvka::is_empty(current_column[midpt-window_size]) ) {
        // lo = midpt - 2*window_size + 1;
        // hi = midpt + window_size;
        hi = midpt;
        lo = midpt - window_size;
        break;
      }
      else if (Bucket_Boruvka::is_empty(current_column[midpt])){
        hi = midpt;
      }
      else {
        lo = midpt;
      }
      midpt = (lo+hi)/2;
    };
    lo = lo - window_size;
    hi = hi + window_size;
    lo = std::max(0, lo);
    hi = std::min(hi, (int) bkt_per_col-1);
    // std::cout << "lo: " << lo << " hi: " << hi << " max: " << bkt_per_col << " window size: " << window_size << std::endl;
    int row = hi;
    while (Bucket_Boruvka::is_empty(current_column[row]) && row > lo) {
    // for (int j = bkt_per_col-1; j >= 0; --j) {
      --row;
    }

    // now that we've found a non-zero bucket check next if next 4 buckets good
    int stop = std::max(row - window_size, lo);
    for (; row >= stop; row--) {
      // if (!Bucket_Boruvka::is_empty(buckets[bucket_id]))
      //   window_ctr=0;
      // else 
      //   window_ctr++;
      if (Bucket_Boruvka::is_good(current_column[row], checksum_seed()))
        return {current_column[row].alpha, GOOD};
    }
    // // for (size_t i=lo; i < hi; i++) {
    // // NEEDS TO BE SIGNED FOR THIS TO WORK. TODO - fix
    // for (int i=hi-1; i >= lo; --i) {
    //   // std::cout << i << " \n";
    //   if (Bucket_Boruvka::is_good(current_column[i], checksum_seed()))
    //     return {current_column[i].alpha, GOOD};
    // }
  }
  return {0, FAIL};
}

SketchSample Sketch::doubling_sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }

  size_t idx = sample_idx++;

  if (Bucket_Boruvka::is_empty(buckets[num_buckets-1]))
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return
  if (Bucket_Boruvka::is_good(buckets[num_buckets - 1], checksum_seed()))
    return {buckets[num_buckets - 1].alpha, GOOD};


  int window_size = 4+(sizeof(unsigned long long)*8 - __builtin_clzll(bkt_per_col))/4;
  // int window_size = 6;


  for (size_t col=0; col< num_columns; col++) {

    Bucket* current_column = buckets + ((idx * cols_per_sample + col)  * bkt_per_col);
    int dist_from_end = 1;
    while (dist_from_end < bkt_per_col && Bucket_Boruvka::is_empty(current_column[bkt_per_col-dist_from_end])) {
      dist_from_end *= 2;
    }
    dist_from_end /= 2;
    int row = bkt_per_col-dist_from_end-1;
    while (Bucket_Boruvka::is_empty(current_column[row]) && row > 0) {
      --row;
    }
    int stop = std::max(row - window_size, 0);
    for (; row >= stop; row--) {
      if (Bucket_Boruvka::is_good(buckets[col * bkt_per_col + row], checksum_seed()))
        return {buckets[col * bkt_per_col + row].alpha, GOOD};
    }
  }
  return {0, FAIL};
  
}

SketchSample Sketch::sample() {
  // return fast_sample();
  // return doubling_sample();
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  #if defined EAGER_BUCKET_CHECK && EAGER_BUCKET_CHECK != EmptyOnly
    for (size_t col = first_column; col < first_column + cols_per_sample; ++col) {
      vec_t col_good_buckets = good_buckets[col];
      if (col_good_buckets == 0)
        return {0, FAIL};
      // int idx = (int) ((sizeof(unsigned long long) * 8) - __builtin_clzll(col_good_buckets)) - 1;
      int idx = __builtin_ctzll(col_good_buckets);
      // std::cout << "idx: " << idx << "flag array: " << col_good_buckets << std::endl;
      likely_if (idx >= 0 && idx < bkt_per_col ) {
        // TODO - replace with just returning once we're sure it works
        // if (Bucket_Boruvka::is_good(buckets[col * num_columns + idx], checksum_seed())) {
          return {buckets[col * num_columns + idx].alpha, GOOD};
        // }
      }
    }
    return {0, FAIL};
  #endif

  if (Bucket_Boruvka::is_empty(buckets[num_buckets - 1]))
    return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return

  if (Bucket_Boruvka::is_good(buckets[num_buckets - 1], checksum_seed()))
    return {buckets[num_buckets - 1].alpha, GOOD};


  for (size_t col = first_column; col < first_column + cols_per_sample; ++col) {
    int row = bkt_per_col - 1;
    int window_size = 6;
    #if defined EAGER_BUCKET_CHECK && EAGER_BUCKET_CHECK == EmptyOnly
    // std::cout << "AYO";
    if (good_buckets[col] == 0) {
      continue;
    }
    // this is the deepest non-empty:
    row = ((int) ((sizeof(unsigned long long) * 8) - __builtin_clzll(good_buckets[col]))) - 1;
    row = std::min(row, (int) bkt_per_col-1);
    # else
    while (Bucket_Boruvka::is_empty(buckets[col * bkt_per_col + row]) && row > 0) {
      --row;
    }
    #endif

    // now that we've found a non-zero bucket check next if next 4 buckets good
    int stop = std::max(row - window_size, 0);
    for (; row >= stop; row--) {
      if (Bucket_Boruvka::is_good(buckets[col * bkt_per_col + row], checksum_seed()))
        return {buckets[col * bkt_per_col + row].alpha, GOOD};
    }
  }
  return {0, FAIL};
}

// SketchSample Sketch::sample() {
//   if (sample_idx >= num_samples) {
//     throw OutOfSamplesException(seed, num_samples, sample_idx);
//   }

//   size_t idx = sample_idx++;
//   size_t first_column = idx * cols_per_sample;

//   if (buckets[num_buckets - 1].alpha == 0 && buckets[num_buckets - 1].gamma == 0)
//     return {0, ZERO};  // the "first" bucket is deterministic so if all zero then no edges to return

//   if (Bucket_Boruvka::is_good(buckets[num_buckets - 1], checksum_seed()))
//     return {buckets[num_buckets - 1].alpha, GOOD};

//   for (size_t i = 0; i < cols_per_sample; ++i) {
//     for (size_t j = 0; j < bkt_per_col; ++j) {
//       size_t bucket_id = (i + first_column) * bkt_per_col + j;
//       if (Bucket_Boruvka::is_good(buckets[bucket_id], checksum_seed()))
//         return {buckets[bucket_id].alpha, GOOD};
//     }
//   }
//   return {0, FAIL};
// }

ExhaustiveSketchSample Sketch::exhaustive_sample() {
  if (sample_idx >= num_samples) {
    throw OutOfSamplesException(seed, num_samples, sample_idx);
  }
  std::unordered_set<vec_t> ret;

  size_t idx = sample_idx++;
  size_t first_column = idx * cols_per_sample;

  unlikely_if (buckets[num_buckets - 1].alpha == 0 && buckets[num_buckets - 1].gamma == 0)
    return {ret, ZERO}; // the "first" bucket is deterministic so if zero then no edges to return

  unlikely_if (Bucket_Boruvka::is_good(buckets[num_buckets - 1], checksum_seed())) {
    ret.insert(buckets[num_buckets - 1].alpha);
    return {ret, GOOD};
  }

  for (size_t i = 0; i < cols_per_sample; ++i) {
    for (size_t j = 0; j < bkt_per_col; ++j) {
    // for (size_t j = bkt_per_col-1; j >= 0; --j) {
      size_t bucket_id = (i + first_column) * bkt_per_col + j;
      unlikely_if (Bucket_Boruvka::is_good(buckets[bucket_id], checksum_seed())) {
        ret.insert(buckets[bucket_id].alpha);
      }
    }
  }

  unlikely_if (ret.size() == 0)
    return {ret, FAIL};
  return {ret, GOOD};
}

void Sketch::merge(const Sketch &other) {
  // #pragma omp simd 
  // for (size_t i = 0; i < num_buckets; ++i) {
  //   buckets[i].alpha ^= other.buckets[i].alpha;
  //   buckets[i].gamma ^= other.buckets[i].gamma;
  // }
  for (size_t i=0; i < num_columns; ++i) {
    Bucket *current_col = buckets + (i* bkt_per_col);
    Bucket *other_col = other.buckets + (i * bkt_per_col);

    #ifdef EAGER_BUCKET_CHECK
    vec_t good_buck_status = 0;    
    #endif 
    // #pragma omp simd
    for (size_t bucket_id=0; bucket_id < bkt_per_col; bucket_id++) {

      #if defined EAGER_BUCKET_CHECK && !EAGER_BUCKET_CHECK == EmptyOnly
      // bool this_good = (good_buckets[i] >> bucket_id) & 1;
      // bool other_good = (other.good_buckets[i] >> bucket_id) & 1;
      // bool this_empty = Bucket_Boruvka::is_empty(current_col[bucket_id]);
      // bool other_empty = Bucket_Boruvka::is_empty(other_col[bucket_id]);
      // bool same_item_if_good = (current_col[bucket_id].alpha == other_col[bucket_id].alpha);
      // bool updated_known_good = ((this_good && other_empty) ^ (other_good && this_empty));
      // bool updated_known_bad = (this_good && other_good); //if both are good, then it either becomes empty or 2 elements
      // current_col[bucket_id].alpha ^= other_col[bucket_id].alpha;
      // current_col[bucket_id].gamma ^= other_col[bucket_id].gamma;
      // unlikely_if (updated_known_good || !updated_known_bad && Bucket_Boruvka::is_good(current_col[bucket_id], checksum_seed()) ) {
      //   good_buck_status |= 1 << bucket_id;
      // }
      good_buck_status &= (!!Bucket_Boruvka::is_good(current_col[bucket_id], checksum_seed())) << bucket_id;
      #elif defined EAGER_BUCKET_CHECK && EAGER_BUCKET_CHECK == EmptyOnly
      good_buck_status &= (!Bucket_Boruvka::is_empty(current_col[bucket_id])) << bucket_id;
      #else 
      current_col[bucket_id].alpha ^= other_col[bucket_id].alpha;
      current_col[bucket_id].gamma ^= other_col[bucket_id].gamma;
      #endif 
    }
    #ifdef EAGER_BUCKET_CHECK
    good_buckets[i] = good_buck_status;
    #endif
  }
}

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

  #pragma omp simd
  for (size_t i = 0; i < n_buckets; i++) {
    size_t bucket_id = start_bucket_id + i;
    buckets[bucket_id].alpha ^= other.buckets[bucket_id].alpha;
    buckets[bucket_id].gamma ^= other.buckets[bucket_id].gamma;
  }
}

void Sketch::merge_raw_bucket_buffer(const Bucket *raw_buckets) {
  for (size_t i = 0; i < num_buckets; i++) {
    buckets[i].alpha ^= raw_buckets[i].alpha;
    buckets[i].gamma ^= raw_buckets[i].gamma;
  }
}

void Sketch::serialize(std::ostream &binary_out) const {
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