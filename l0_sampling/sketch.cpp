#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>
#include <string>
#include "../include/prime_generator.h"
#include "../include/sketch.h"

Sketch::Sketch(uint64_t n, long seed): seed(seed), n(n), large_prime
(PrimeGenerator::generate_prime((uint128_t)n*n)) {
  const unsigned long long int num_buckets = bucket_gen(n);
  const unsigned long long int num_guesses = guess_gen(n);
  buckets = std::vector<Bucket_Boruvka>(num_buckets * num_guesses);
}

void Sketch::update(Update update ) {
  const unsigned long long int num_buckets = bucket_gen(n);
  const unsigned long long int num_guesses = guess_gen(n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      long bucket_id = i * num_guesses + j;
      XXH64_hash_t bucket_seed = XXH64(&bucket_id, 8, seed);
      int128_t r = 2 +  bucket_seed % (large_prime - 3);
      if (buckets[bucket_id].contains(update.index+1, bucket_seed, 1 << j)){
        buckets[bucket_id].a += update.delta;
        buckets[bucket_id].b += update.delta*(update.index+1); // deals with updates whose indices are 0
        buckets[bucket_id].c = static_cast<uint128_t>((buckets[bucket_id].c
              + large_prime
              + update.delta*PrimeGenerator::power(r,(uint128_t) update.index+1, large_prime)) % large_prime);
      }
    }
  }
}

Update Sketch::query() {
  if (already_quered) {
    throw MultipleQueryException();
  }
  already_quered = true;
  bool all_buckets_zero = true;
  const unsigned long long int num_buckets = bucket_gen(n);
  const unsigned long long int num_guesses = guess_gen(n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      Bucket_Boruvka& b = buckets[i*num_guesses+j];
      if (b.a != 0 || b.b != 0 || b.c != 0) {
        all_buckets_zero = false;
      }
      long bucket_id = i * num_guesses + j;
      XXH64_hash_t bucket_seed = XXH64(&bucket_id, 8, seed);
      uint128_t r = 2 + bucket_seed % (large_prime - 3);
      if (b.a != 0 && b.b % b.a == 0  && b.b / b.a > 0 && b.b / b.a <= n
            && Bucket_Boruvka::contains((b.b / b.a).toBoostUInt128(), bucket_seed,1 << j)
            && (static_cast<int128_t>(b.c) - b.a.toBoostInt128() *
            PrimeGenerator::power
            (r,(b.b /b.a).toBoostUInt128(),large_prime)) % large_prime == 0
         ) {
        // TODO: update Update to return 128 bit types
        return {(int64_t) (b.b / b.a - 1).toBoostInt128(),
                (long) b.a.toBoostInt128()}; // 0-index adjustment
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
  assert (sketch1.seed == sketch2.seed);
  assert (sketch1.large_prime == sketch2.large_prime);
  Sketch result = Sketch(sketch1.n,sketch1.seed);
  for (unsigned i = 0; i < result.buckets.size(); i++){
    Bucket_Boruvka& b = result.buckets[i];
    b.a = sketch1.buckets[i].a + sketch2.buckets[i].a;
    b.b = sketch1.buckets[i].b + sketch2.buckets[i].b;
    b.c = (sketch1.buckets[i].c + sketch2.buckets[i].c)%result.large_prime;
  }
  return result;
}

Sketch &operator+= (Sketch &sketch1, const Sketch &sketch2) {
  assert (sketch1.n == sketch2.n);
  assert (sketch1.seed == sketch2.seed);
  assert (sketch1.large_prime == sketch2.large_prime);
  for (unsigned i = 0; i < sketch1.buckets.size(); i++){
    sketch1.buckets[i].a += sketch2.buckets[i].a;
    sketch1.buckets[i].b += sketch2.buckets[i].b;
    sketch1.buckets[i].c += sketch2.buckets[i].c;
    sketch1.buckets[i].c %= sketch1.large_prime;
  }
  return sketch1;
}


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

std::ostream& operator<< (std::ostream &os, const Sketch &sketch) {
  os << sketch.large_prime << std::endl;
  const unsigned long long int num_buckets = bucket_gen(sketch.n);
  const unsigned long long int num_guesses = guess_gen(sketch.n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      long bucket_id = i * num_guesses + j;
      XXH64_hash_t bucket_seed = XXH64(&bucket_id, 8, sketch.seed);
      uint128_t r = 2 + bucket_seed % (sketch.large_prime - 3);
      const Bucket_Boruvka& bucket = sketch.buckets[i];
      for (unsigned k = 0; k < sketch.n; k++) {
        os << (Bucket_Boruvka::contains(k+1,bucket_seed,1<<j) ? '1' : '0');
      }
      os << std::endl
         << "a:" << bucket.a << std::endl
         << "b:" << bucket.b << std::endl
         << "c:" << bucket.c << std::endl
         << "r:" << r << std::endl
         << (bucket.a != 0 && bucket.b % bucket.a == 0
              && (bucket.c - bucket.a.toBoostInt128()*PrimeGenerator::power(r, (bucket
              .b/bucket.a).toBoostUInt128(), sketch.large_prime)) % sketch.large_prime == 0 ?
              "good" : "bad")
         << std::endl;
    }
  }
  return os;
}
