#include "../../include/l0_sampling/sketch.h"
#include "prime_generator.h"
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

Sketch::Sketch(uint64_t seed): seed(seed), large_prime(PrimeGenerator::generate_prime(static_cast<ubucket_t>(n) * n)) {
  // zero buckets
  std::memset(_bucket_data, 0, num_elems * sizeof(Bucket_Boruvka));

  buckets = reinterpret_cast<Bucket_Boruvka *>(_bucket_data);
}

Sketch::Sketch(uint64_t seed, std::istream &binary_in): seed(seed), large_prime(PrimeGenerator::generate_prime(static_cast<ubucket_t>(n) * n)) {
  binary_in.read(_bucket_data, num_elems * sizeof(Bucket_Boruvka));
  buckets = reinterpret_cast<Bucket_Boruvka *>(_bucket_data);
}

Sketch::Sketch(const Sketch& s) : seed(s.seed), large_prime(s.large_prime) {
  std::memcpy(_bucket_data, s._bucket_data, num_elems * sizeof(Bucket_Boruvka));
}

void Sketch::update(const Update& update) {
  auto cbucket_seed = Bucket_Boruvka::gen_bucket_seed(num_elems - 1, seed);
  auto cr = Bucket_Boruvka::gen_r(cbucket_seed, large_prime);
  buckets[num_elems - 1].update(update, large_prime, cr);

  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      auto bucket_seed = Bucket_Boruvka::gen_bucket_seed(bucket_id, seed);
      auto r = Bucket_Boruvka::gen_r(bucket_seed, large_prime);
      auto& bucket = buckets[bucket_id];
      if (bucket.contains(update.index, bucket_seed, 2 << j)) {
        bucket.update(update, large_prime, r);
      }
    }
  }
}

void Sketch::batch_update(const std::vector<Update>& updates) {
  for (const auto& upd : updates) {
    update(upd);
  }
}

std::pair<vec_t, SampleSketchRet> Sketch::query() {
  if (already_queried) {
    throw MultipleQueryException();
  }
  already_queried = true;

  auto& determ_bucket = buckets[num_elems - 1];
  if (determ_bucket == BUCKET_ZERO) {
    return {0, ZERO}; // the "first" bucket is deterministic so if it is all zero then there are no edges to return
  }

  auto cbucket_seed = Bucket_Boruvka::gen_bucket_seed(num_elems - 1, seed);
  auto cr = Bucket_Boruvka::gen_r(num_elems - 1, large_prime);
  if (determ_bucket.is_good(n, large_prime, cbucket_seed, cr, 1)) {
    return {determ_bucket.a, GOOD};
  }
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      auto bucket_seed = Bucket_Boruvka::gen_bucket_seed(bucket_id, seed);
      auto r = Bucket_Boruvka::gen_r(bucket_seed, large_prime);
      auto& bucket = buckets[bucket_id];
      if (bucket.is_good(n, large_prime, bucket_seed, r, 2 << j)) {
        return {bucket.a, GOOD};
      }
    }
  }
  return {0, FAIL};
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.seed == sketch2.seed);
  assert (sketch1.large_prime == sketch2.large_prime);
  for (unsigned i = 0; i < Sketch::num_elems; i++) {
    auto& bucket1 = sketch1.buckets[i];
    const auto& bucket2 = sketch2.buckets[i];
    bucket1.a += bucket2.a;
    bucket1.b += bucket2.b;
    bucket1.c += bucket2.c;
    bucket1.c %= sketch1.large_prime;
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

void Sketch::write_binary(std::ostream& binary_out) {
  const_cast<const Sketch*>(this)->write_binary(binary_out);
}

void Sketch::write_binary(std::ostream &binary_out) const {
  binary_out.write(_bucket_data, num_elems * sizeof(Bucket_Boruvka));
}
