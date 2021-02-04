#include "../include/sketch.h"

Sketch::Sketch(vec_t n, long seed): seed(seed), n(n),
    large_prime(PrimeGenerator::generate_prime(static_cast<ubucket_t>(n) * n)) {
  const unsigned num_buckets = bucket_gen(n);
  const unsigned num_guesses = guess_gen(n);
  buckets = std::vector<Bucket_Boruvka>(num_buckets * num_guesses);
}

void Sketch::update(Update update) {
  const unsigned num_buckets = bucket_gen(n);
  const unsigned num_guesses = guess_gen(n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      Bucket_Boruvka& bucket = buckets[bucket_id];
      XXH64_hash_t bucket_seed = Bucket_Boruvka::gen_bucket_seed(bucket_id, seed);
      ubucket_t r = Bucket_Boruvka::gen_r(bucket_seed, large_prime);
      if (bucket.contains(update.index, bucket_seed, 1 << j)){
        bucket.update(update, large_prime, r);
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
  const unsigned num_buckets = bucket_gen(n);
  const unsigned num_guesses = guess_gen(n);
  for (unsigned i = 0; i < num_buckets; ++i) {
    for (unsigned j = 0; j < num_guesses; ++j) {
      unsigned bucket_id = i * num_guesses + j;
      const Bucket_Boruvka& bucket = buckets[bucket_id];
      XXH64_hash_t bucket_seed = Bucket_Boruvka::gen_bucket_seed(bucket_id, seed);
      ubucket_t r = Bucket_Boruvka::gen_r(bucket_seed, large_prime);
      if (bucket.a != 0 || bucket.b != 0 || bucket.c != 0) {
        all_buckets_zero = false;
      }
      if (bucket.is_good(n, large_prime, bucket_seed, r, 1 << j)) {
        return {static_cast<vec_t>(bucket.b / bucket.a - 1), // 0-index adjustment
                static_cast<long>(bucket.a)};
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
  sketch1.already_quered = sketch1.already_quered || sketch2.already_quered;
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
      unsigned bucket_id = i * num_guesses + j;
      const Bucket_Boruvka& bucket = sketch.buckets[bucket_id];
      XXH64_hash_t bucket_seed = Bucket_Boruvka::gen_bucket_seed(bucket_id, sketch.seed);
      ubucket_t r = Bucket_Boruvka::gen_r(bucket_seed, sketch.large_prime);
      for (unsigned k = 0; k < sketch.n; k++) {
        os << (bucket.contains(k, bucket_seed, 1 << j) ? '1' : '0');
      }
      os << std::endl
         << "a:" << bucket.a << std::endl
         << "b:" << bucket.b << std::endl
         << "c:" << bucket.c << std::endl
         << "r:" << r << std::endl
         << (bucket.is_good(sketch.n, sketch.large_prime, bucket_seed, r, 1 << j) ? "good" : "bad") << std::endl;
    }
  }
  return os;
}
