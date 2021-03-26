#include "../include/sketch.h"
#include <cassert>

#ifdef __AVX2__
#include <immintrin.h>
#endif

Sketch::Sketch(vec_t n, long seed, double num_bucket_factor):
    seed(seed), n(n), num_bucket_factor(num_bucket_factor) {
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
#ifdef __AVX2__
  size_t total_bucket_size = num_buckets * num_guesses * sizeof(Bucket_Boruvka);
  buckets = (Bucket_Boruvka *)_mm_malloc(total_bucket_size, 32);
  memset(buckets, 0, total_bucket_size);
#else
  buckets = std::vector<Bucket_Boruvka>(num_buckets * num_guesses);
#endif
}

Sketch::~Sketch() {
#ifdef __AVX2__
  _mm_free(buckets);
#endif
}

void Sketch::update(const vec_t& update_idx) {
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
  XXH64_hash_t update_hash = Bucket_Boruvka::index_hash(update_idx, seed);
#ifdef __AVX2__
  __m256i update_reg = _mm256_set_epi64x(update_hash, update_idx, update_hash, update_idx);
#endif
  for (unsigned i = 0; i < num_buckets; ++i) {
    XXH64_hash_t col_index_hash = Bucket_Boruvka::col_index_hash(i, update_idx, seed);
    unsigned num_updates = __builtin_ctzll(col_index_hash | 1ULL << (num_guesses - 1)) + 1;
#ifdef __AVX2__
    unsigned j = 0;
//    printf("%p:", buckets);
//    fflush(stdout);
    for (; j + 1 < num_updates; j += 2) {
      unsigned bucket_id = i * num_guesses + j;
//      printf("%p:", &buckets[i]);
//      fflush(stdout);
      *(__m256i *)&buckets[bucket_id] = _mm256_xor_si256(
          *(__m256i *)&buckets[bucket_id], update_reg);
    }
    if (j < num_updates) {
      unsigned bucket_id = i * num_guesses + j;
      buckets[bucket_id].update(update_idx, update_hash);
    }
//    puts("");
#else
    for (unsigned j = 0; j + 1 < num_updates; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      buckets[bucket_id].update(update_idx, update_hash);
    }
#endif
  }
}

void Sketch::batch_update(const std::vector<vec_t>& updates) {
  for (const auto update_idx : updates) {
    update(update_idx);
  }
}

vec_t Sketch::query() {
  if (already_quered) {
    throw MultipleQueryException();
  }
  already_quered = true;
  bool all_buckets_zero = true;
  const unsigned num_buckets = bucket_gen(n, num_bucket_factor);
  const unsigned num_guesses = guess_gen(n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      const Bucket_Boruvka& bucket = buckets[bucket_id];
      if (bucket.a != 0 || bucket.c != 0) {
        all_buckets_zero = false;
      }
      if (bucket.is_good(n, i, 1 << j, seed)) {
        return bucket.a;
      }
    }
  }
  if (all_buckets_zero) {
    throw AllBucketsZeroException();
  } else {
    throw NoGoodBucketException();
  }
}

Sketch operator+ (const Sketch &sketch1, const Sketch &sketch2){
  assert (sketch1.n == sketch2.n);
  assert (sketch1.num_bucket_factor == sketch2.num_bucket_factor);
  assert (sketch1.seed == sketch2.seed);
  assert (sketch1.num_bucket_factor == sketch2.num_bucket_factor);
  Sketch result = Sketch(sketch1.n, sketch1.seed, sketch1.num_bucket_factor);
  const unsigned total_buckets = bucket_gen(sketch1.n, sketch1.num_bucket_factor) * guess_gen(sketch1.n);
  for (unsigned i = 0; i < total_buckets; i++){
    Bucket_Boruvka& b = result.buckets[i];
    b.a = sketch1.buckets[i].a ^ sketch2.buckets[i].a;
    b.c = sketch1.buckets[i].c ^ sketch2.buckets[i].c;
  }
  return result;
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.n == sketch2.n);
  assert (sketch1.num_bucket_factor == sketch2.num_bucket_factor);
  assert (sketch1.seed == sketch2.seed);
  assert (sketch1.num_bucket_factor == sketch2.num_bucket_factor);
  const unsigned total_buckets = bucket_gen(sketch1.n, sketch1.num_bucket_factor) * guess_gen(sketch1.n);
  for (unsigned i = 0; i < total_buckets; i++){
    sketch1.buckets[i].a ^= sketch2.buckets[i].a;
    sketch1.buckets[i].c ^= sketch2.buckets[i].c;
  }
  sketch1.already_quered = sketch1.already_quered || sketch2.already_quered;
  return sketch1;
}

//Probably doesn't work anymore
/*
Sketch operator* (const Sketch &sketch1, long scaling_factor){
  Sketch result = Sketch(sketch1.n,sketch1.seed);
  for (unsigned int i = 0; i < result.buckets.size(); i++){
    Bucket_Boruvka& b = result.buckets[i];
    b.a = sketch1.buckets[i].a * scaling_factor;
    b.b = sketch1.buckets[i].b * scaling_factor;
    b.c = (sketch1.buckets[i].c * scaling_factor)% result.large_prime;
  }
  return result;
}
*/

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  const unsigned long long int num_buckets = bucket_gen(sketch.n, sketch.num_bucket_factor);
  const unsigned long long int num_guesses = guess_gen(sketch.n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      const Bucket_Boruvka& bucket = sketch.buckets[bucket_id];
      for (unsigned k = 0; k < sketch.n; k++) {
        os << (bucket.contains(Bucket_Boruvka::col_index_hash(i, k, sketch.seed), 1 << j) ? '1' : '0');
      }
      os << std::endl
         << "a:" << bucket.a << std::endl
         << "c:" << bucket.c << std::endl
         << (bucket.is_good(sketch.n, i, 1 << j, sketch.seed) ? "good" : "bad") << std::endl;
    }
  }
  return os;
}
