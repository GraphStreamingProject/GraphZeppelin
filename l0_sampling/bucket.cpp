#include "../include/bucket.h"
#include "../include/prime_generator.h"

bool Bucket_Boruvka::contains(const vec_t& index, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const {
  vec_t buf = index + 1;
  XXH64_hash_t hash = XXH64(&buf, sizeof(buf), bucket_seed);
  if (hash % guess_nonzero == 0)
    return true;
  return false;
}

bool Bucket_Boruvka::is_good(const vec_t& n, const ubucket_t& large_prime, const XXH64_hash_t& bucket_seed, const ubucket_t& r, const vec_t& guess_nonzero) const {
  return a != 0 && b % a == 0  && b / a > 0 && b / a <= n
      && contains(static_cast<vec_t>(b / a - 1), bucket_seed, guess_nonzero)
      && ((static_cast<bucket_prod_t>(c) + static_cast<bucket_prod_t>(large_prime) - static_cast<bucket_prod_t>(a)
      * PrimeGenerator::powermod(r, b / a, large_prime)) % large_prime) % large_prime == 0;
}

void Bucket_Boruvka::update(const Update& update, const ubucket_t& large_prime, const ubucket_t& r) {
  a += update.delta;
  b += update.delta * static_cast<bucket_t>(update.index + 1); // deals with updates whose indices are 0
  c = static_cast<ubucket_t>(
      (static_cast<bucket_prod_t>(c)
      + static_cast<bucket_prod_t>(large_prime)
      + (static_cast<bucket_prod_t>(update.delta) * PrimeGenerator::powermod(r, update.index + 1, large_prime) % large_prime))
      % large_prime);
}

void Bucket_Boruvka::cached_update(const Update& update, const ubucket_t& large_prime, const std::vector<ubucket_t>& r_sq_cache) {
  a += update.delta;
  b += update.delta * static_cast<bucket_t>(update.index + 1); // deals with updates whose indices are 0
  c = static_cast<ubucket_t>(
      (static_cast<bucket_prod_t>(c)
      + static_cast<bucket_prod_t>(large_prime)
      + (static_cast<bucket_prod_t>(update.delta) * PrimeGenerator::cached_powermod(r_sq_cache, update.index + 1, large_prime) % large_prime))
      % large_prime);
}

