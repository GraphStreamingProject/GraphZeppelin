#include "../../include/l0_sampling/sketch.h"
#include <cassert>
#include <cstring>
#include <iostream>

vec_t Sketch::failure_factor = 100;
vec_t Sketch::n;
size_t Sketch::num_elems;
size_t Sketch::num_buckets;
size_t Sketch::num_guesses;

/*
 * Static functions for creating sketches with a provided memory location.
 * We use these in the production system to keep supernodes virtually contiguous.
 */
Sketch* Sketch::makeSketch(void* loc, uint64_t seed) {
  return new (loc) Sketch(seed);
}

Sketch* Sketch::makeSketch(void* loc, uint64_t seed, std::istream &binary_in) {
  return new (loc) Sketch(seed, binary_in);
}

Sketch* Sketch::makeSketch(void* loc, const Sketch& s) {
  return new (loc) Sketch(s);
}

Sketch::Sketch(uint64_t seed): seed(seed) {
  // establish the bucket_a and bucket_c locations
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  // initialize bucket values
  for (size_t i = 0; i < num_elems; ++i) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }
}

Sketch::Sketch(uint64_t seed, std::istream &binary_in): seed(seed) {
  // establish the bucket_a and bucket_c locations
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  binary_in.read((char*)bucket_a, num_elems * sizeof(vec_t));
  binary_in.read((char*)bucket_c, num_elems * sizeof(vec_hash_t));
}

Sketch::Sketch(const Sketch& s) : seed(s.seed) {
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  std::memcpy(bucket_a, s.bucket_a, num_elems * sizeof(vec_t));
  std::memcpy(bucket_c, s.bucket_c, num_elems * sizeof(vec_hash_t));
}

void Sketch::update(const vec_t update_idx) {
  vec_hash_t checksum = Bucket_Boruvka::get_index_hash(update_idx, seed);
  
  // Update depth 0 bucket
  Bucket_Boruvka::update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], update_idx, checksum);

  // Update higher depth buckets
  for (unsigned i = 0; i < num_buckets; ++i) {
    col_hash_t depth = Bucket_Boruvka::get_index_depth(update_idx, seed + i, num_guesses);
    size_t bucket_id = i * num_guesses + depth;
    bucket_id *= (bool)(depth!=0); // if depth is 0 then "update" null bucket -> bucket[0]

    Bucket_Boruvka::update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, checksum);
  }
}

void Sketch::batch_update(const std::vector<vec_t>& updates) {
  for (const auto& update_idx : updates) {
    update(update_idx);
  }
}

std::pair<vec_t, SampleSketchRet> Sketch::query() {
  if (already_queried) {
    throw MultipleQueryException();
  }
  already_queried = true;

  if (bucket_a[num_elems - 1] == 0 && bucket_c[num_elems - 1] == 0) {
    return {0, ZERO}; // the "first" bucket is deterministic so if it is all zero then there are no edges to return
  }
  if (Bucket_Boruvka::is_good(bucket_a[num_elems - 1], bucket_c[num_elems - 1], seed)) {
    return {bucket_a[num_elems - 1], GOOD};
  }
  for (unsigned i = 0; i < num_buckets; ++i) {
    // bucket[0] is null
    for (unsigned j = begin_nonnull; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (Bucket_Boruvka::is_good(bucket_a[bucket_id], bucket_c[bucket_id], seed)) {
        return {bucket_a[bucket_id], GOOD};
      }
    }
  }
  return {0, FAIL};
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.seed == sketch2.seed);
  for (unsigned i = Sketch::begin_nonnull; i < Sketch::num_elems; i++) {
    sketch1.bucket_a[i] ^= sketch2.bucket_a[i];
    sketch1.bucket_c[i] ^= sketch2.bucket_c[i];
  }
  sketch1.already_queried = sketch1.already_queried || sketch2.already_queried;
  return sketch1;
}

bool operator== (const Sketch &sketch1, const Sketch &sketch2) {
  if (sketch1.seed != sketch2.seed || sketch1.already_queried != sketch2.already_queried) 
    return false;

  for (size_t i = Sketch::begin_nonnull; i < Sketch::num_elems; ++i) {
    if (sketch1.bucket_a[i] != sketch2.bucket_a[i]) return false;
  }

  for (size_t i = Sketch::begin_nonnull; i < Sketch::num_elems; ++i) {
    if (sketch1.bucket_c[i] != sketch2.bucket_c[i]) return false;
  }

  return true;
}

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  vec_t a      = sketch.bucket_a[Sketch::num_elems - 1];
  vec_hash_t c = sketch.bucket_c[Sketch::num_elems - 1];
  bool good    = Bucket_Boruvka::is_good(a, c, sketch.seed);

  os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;

  for (unsigned i = 0; i < Sketch::num_buckets; ++i) {
    for (unsigned j = Sketch::begin_nonnull; j < Sketch::num_guesses; ++j) {
      unsigned bucket_id = i * Sketch::num_guesses + j;
      vec_t a      = sketch.bucket_a[bucket_id];
      vec_hash_t c = sketch.bucket_c[bucket_id];
      bool good    = Bucket_Boruvka::is_good(a, c, sketch.seed);

      os << " a:" << a << " c:" << c << (good ? " good" : " bad") << std::endl;
    }
    os << std::endl;
  }
  return os;
}

void Sketch::write_binary(std::ostream& binary_out) {
  const_cast<const Sketch*>(this)->write_binary(binary_out);
}

void Sketch::write_binary(std::ostream &binary_out) const {
  // Write out the bucket values to the stream.
  // Do not include the null bucket
  binary_out.write((char*)bucket_a, num_elems * sizeof(vec_t));
  binary_out.write((char*)bucket_c, num_elems * sizeof(vec_hash_t));
}
