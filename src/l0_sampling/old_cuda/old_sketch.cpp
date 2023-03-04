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
  h_bucket_a = reinterpret_cast<vec_t*>(buckets);
  h_bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));
  h_bucket_debug = reinterpret_cast<vec_t*>(buckets);

  // initialize bucket values
  for (size_t i = 0; i < num_elems; ++i) {
    h_bucket_a[i] = 0;
    h_bucket_c[i] = 0;
    h_bucket_debug[i] = 0;
  }

  cudaMalloc(&d_bucket_a, num_elems * sizeof(vec_t));
  cudaMalloc(&d_bucket_c, num_elems * sizeof(vec_hash_t));
  cudaMalloc(&d_col_index_hash, num_buckets * sizeof(col_hash_t));
  cudaMalloc(&d_bucket_debug, num_elems * sizeof(vec_t));

  cudaMemcpy(d_bucket_a, h_bucket_a, num_elems* sizeof(vec_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bucket_c, h_bucket_c, num_elems* sizeof(vec_hash_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bucket_debug, h_bucket_debug, num_elems* sizeof(vec_t), cudaMemcpyHostToDevice);
}

Sketch::Sketch(uint64_t seed, std::istream &binary_in): seed(seed) {
  // establish the bucket_a and bucket_c locations
  h_bucket_a = reinterpret_cast<vec_t*>(buckets);
  h_bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  binary_in.read((char*)h_bucket_a, num_elems * sizeof(vec_t));
  binary_in.read((char*)h_bucket_c, num_elems * sizeof(vec_hash_t));

  cudaMalloc(&d_bucket_a, num_elems * sizeof(vec_t));
  cudaMalloc(&d_bucket_c, num_elems * sizeof(vec_hash_t));
  cudaMalloc(&d_col_index_hash, num_buckets * sizeof(col_hash_t));

  cudaMemcpy(d_bucket_a, h_bucket_a, num_elems* sizeof(vec_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bucket_c, h_bucket_c, num_elems* sizeof(vec_hash_t), cudaMemcpyHostToDevice);
}

Sketch::Sketch(const Sketch& s) : seed(s.seed) {

  // establish the bucket_a and bucket_c locations
  h_bucket_a = reinterpret_cast<vec_t*>(buckets);
  h_bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

  cudaMalloc(&d_col_index_hash, num_buckets * sizeof(col_hash_t));

  cudaMemcpy(d_bucket_a, s.d_bucket_a, num_elems* sizeof(vec_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_bucket_c, s.d_bucket_c, num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToDevice);

  //std::memcpy(bucket_a, s.bucket_a, num_elems * sizeof(vec_t));
  //std::memcpy(bucket_c, s.bucket_c, num_elems * sizeof(vec_hash_t));
}

void Sketch::update(const vec_t& update_idx) {
  CudaSketch cudaSketch(num_elems, num_buckets, num_guesses, d_bucket_a, d_bucket_c, seed);
  cudaSketch.update(d_col_index_hash, update_idx, d_bucket_debug);
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

  cudaMemcpy(h_bucket_a, d_bucket_a, num_elems* sizeof(vec_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bucket_c, d_bucket_c, num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

  if (h_bucket_a[num_elems - 1] == 0 && h_bucket_c[num_elems - 1] == 0) {
    return {0, ZERO}; // the "first" bucket is deterministic so if it is all zero then there are no edges to return
  }
  if (Bucket_Boruvka::is_good(h_bucket_a[num_elems - 1], h_bucket_c[num_elems - 1], seed)) {
    return {h_bucket_a[num_elems - 1], GOOD};
  }

  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      if (Bucket_Boruvka::is_good(h_bucket_a[bucket_id], h_bucket_c[bucket_id], i, 1 << j, seed)) {
        return {h_bucket_a[bucket_id], GOOD};
      }
    }
  }
  return {0, FAIL};
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.seed == sketch2.seed);

  cudaMemcpy(sketch1.h_bucket_a, sketch1.d_bucket_a, sketch1.num_elems* sizeof(vec_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(sketch1.h_bucket_c, sketch1.d_bucket_c, sketch1.num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

  cudaMemcpy(sketch2.h_bucket_a, sketch2.d_bucket_a, sketch2.num_elems* sizeof(vec_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(sketch2.h_bucket_c, sketch2.d_bucket_c, sketch2.num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

  for (unsigned i = 0; i < Sketch::num_elems; i++) {
    sketch1.h_bucket_a[i] ^= sketch2.h_bucket_a[i];
    sketch1.h_bucket_c[i] ^= sketch2.h_bucket_c[i];
  }

  cudaMemcpy(sketch1.d_bucket_a, sketch1.h_bucket_a, sketch1.num_elems* sizeof(vec_t), cudaMemcpyHostToDevice);
  cudaMemcpy(sketch1.d_bucket_c, sketch1.h_bucket_c, sketch1.num_elems* sizeof(vec_hash_t), cudaMemcpyHostToDevice);

  sketch1.already_queried = sketch1.already_queried || sketch2.already_queried;
  return sketch1;
}

bool operator== (const Sketch &sketch1, const Sketch &sketch2) {
  if (sketch1.seed != sketch2.seed || sketch1.already_queried != sketch2.already_queried) 
    return false;

  cudaMemcpy(sketch1.h_bucket_a, sketch1.d_bucket_a, sketch1.num_elems* sizeof(vec_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(sketch1.h_bucket_c, sketch1.d_bucket_c, sketch1.num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

  cudaMemcpy(sketch2.h_bucket_a, sketch2.d_bucket_a, sketch2.num_elems* sizeof(vec_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(sketch2.h_bucket_c, sketch2.d_bucket_c, sketch2.num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < Sketch::num_elems; ++i) {
    if (sketch1.h_bucket_a[i] != sketch2.h_bucket_a[i]) return false;
  }

  for (size_t i = 0; i < Sketch::num_elems; ++i) {
    if (sketch1.h_bucket_c[i] != sketch2.h_bucket_c[i]) return false;
  }

  return true;
}

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  for (unsigned k = 0; k < Sketch::n; k++) {
    os << '1';
  }

  cudaMemcpy(sketch.h_bucket_a, sketch.d_bucket_a, sketch.num_elems* sizeof(vec_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(sketch.h_bucket_c, sketch.d_bucket_c, sketch.num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

  os << std::endl
     << "a:" << sketch.h_bucket_a[Sketch::num_buckets * Sketch::num_guesses] << std::endl
     << "c:" << sketch.h_bucket_c[Sketch::num_buckets * Sketch::num_guesses] << std::endl
     << (Bucket_Boruvka::is_good(sketch.h_bucket_a[Sketch::num_buckets * Sketch::num_guesses], sketch.h_bucket_c[Sketch::num_buckets * Sketch::num_guesses], sketch.seed) ? "good" : "bad") << std::endl;

  for (unsigned i = 0; i < Sketch::num_buckets; ++i) {
    for (unsigned j = 0; j < Sketch::num_guesses; ++j) {
      unsigned bucket_id = i * Sketch::num_guesses + j;
      for (unsigned k = 0; k < Sketch::n; k++) {
        os << (Bucket_Boruvka::contains(Bucket_Boruvka::col_index_hash(k, sketch.seed + 1), 1 << j) ? '1' : '0');
      }
      os << std::endl
         << "a:" << sketch.h_bucket_a[bucket_id] << std::endl
         << "c:" << sketch.h_bucket_c[bucket_id] << std::endl
         << (Bucket_Boruvka::is_good(sketch.h_bucket_a[bucket_id], sketch.h_bucket_c[bucket_id], i, 1 << j, sketch.seed) ? "good" : "bad") << std::endl;
    }
  }
  return os;
}

void Sketch::write_binary(std::ostream& binary_out) {
  const_cast<const Sketch*>(this)->write_binary(binary_out);
}

void Sketch::write_binary(std::ostream &binary_out) const {
  cudaMemcpy(h_bucket_a, d_bucket_a, num_elems* sizeof(vec_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bucket_c, d_bucket_c, num_elems* sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

  binary_out.write((char*)h_bucket_a, num_elems * sizeof(vec_t));
  binary_out.write((char*)h_bucket_c, num_elems * sizeof(vec_hash_t));

  cudaMemcpy(d_bucket_a, h_bucket_a, num_elems* sizeof(vec_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bucket_c, h_bucket_c, num_elems* sizeof(vec_hash_t), cudaMemcpyHostToDevice);
}
