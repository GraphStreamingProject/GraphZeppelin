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
Sketch* Sketch::makeSketch(void* loc, long seed) {
  return new (loc) Sketch(seed);
}

Sketch* Sketch::makeSketch(void* loc, long seed, std::fstream &binary_in) {
  return new (loc) Sketch(seed, binary_in);
}

Sketch* Sketch::makeSketch(void* loc, const Sketch& s) {
  return new (loc) Sketch(s);
}

Sketch::Sketch(long seed): seed(seed) {
  // establish the bucket_a and bucket_c locations
  bucket_a = reinterpret_cast<vec_t*>(buckets);
  bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  // initialize bucket values
  for (size_t i = 0; i < num_elems; ++i) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }
}

Sketch::Sketch(long seed, std::fstream &binary_in): seed(seed) {
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

void Sketch::update(const vec_t& update_idx) {
  vec_hash_t update_hash = Bucket_Boruvka::index_hash(update_idx, seed);
  Bucket_Boruvka::update(bucket_a[num_elems - 1], bucket_c[num_elems - 1], update_idx, update_hash);
  for (unsigned i = 0; i < num_buckets; ++i) {
    col_hash_t col_index_hash = Bucket_Boruvka::col_index_hash(update_idx, seed + i);
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (Bucket_Boruvka::contains(col_index_hash, ((col_hash_t)1) << j)){
        Bucket_Boruvka::update(bucket_a[bucket_id], bucket_c[bucket_id], update_idx, update_hash);
      } else break;
    }
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
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (Bucket_Boruvka::is_good(bucket_a[bucket_id], bucket_c[bucket_id], i, 1 << j, seed)) {
        return {bucket_a[bucket_id], GOOD};
      }
    }
  }
  return {0, FAIL};
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.seed == sketch2.seed);
  for (unsigned i = 0; i < Sketch::num_elems; i++) {
    sketch1.bucket_a[i] ^= sketch2.bucket_a[i];
    sketch1.bucket_c[i] ^= sketch2.bucket_c[i];
  }
  sketch1.already_queried = sketch1.already_queried || sketch2.already_queried;
  return sketch1;
}

bool operator== (const Sketch &sketch1, const Sketch &sketch2) {
  if (sketch1.seed != sketch2.seed || sketch1.already_queried != sketch2.already_queried) 
    return false;

  for (size_t i = 0; i < Sketch::num_elems; ++i) {
    if (sketch1.bucket_a[i] != sketch2.bucket_a[i]) return false;
  }

  for (size_t i = 0; i < Sketch::num_elems; ++i) {
    if (sketch1.bucket_c[i] != sketch2.bucket_c[i]) return false;
  }

  return true;
}

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  for (unsigned k = 0; k < Sketch::n; k++) {
    os << '1';
  }
  os << std::endl
     << "a:" << sketch.bucket_a[Sketch::num_buckets * Sketch::num_guesses] << std::endl
     << "c:" << sketch.bucket_c[Sketch::num_buckets * Sketch::num_guesses] << std::endl
     << (Bucket_Boruvka::is_good(sketch.bucket_a[Sketch::num_buckets * Sketch::num_guesses], sketch.bucket_c[Sketch::num_buckets * Sketch::num_guesses], sketch.seed) ? "good" : "bad") << std::endl;

  for (unsigned i = 0; i < Sketch::num_buckets; ++i) {
    for (unsigned j = 0; j < Sketch::num_guesses; ++j) {
      unsigned bucket_id = i * Sketch::num_guesses + j;
      for (unsigned k = 0; k < Sketch::n; k++) {
        os << (Bucket_Boruvka::contains(Bucket_Boruvka::col_index_hash(k, sketch.seed + 1), 1 << j) ? '1' : '0');
      }
      os << std::endl
         << "a:" << sketch.bucket_a[bucket_id] << std::endl
         << "c:" << sketch.bucket_c[bucket_id] << std::endl
         << (Bucket_Boruvka::is_good(sketch.bucket_a[bucket_id], sketch.bucket_c[bucket_id], i, 1 << j, sketch.seed) ? "good" : "bad") << std::endl;
    }
  }
  return os;
}

void Sketch::write_binary(std::fstream& binary_out) {
  binary_out.write((char*)bucket_a, num_elems * sizeof(vec_t));
  binary_out.write((char*)bucket_c, num_elems * sizeof(vec_hash_t));
}
