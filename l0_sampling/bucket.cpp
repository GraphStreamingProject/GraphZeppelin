#include "../include/bucket.h"
#include "../include/prime_generator.h"

bool Bucket_Boruvka::contains(const vec_t& index, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const {
  XXH64_hash_t hash = XXH64(&index, sizeof(index), bucket_seed);
  if (hash % guess_nonzero == 0)
    return true;
  return false;
}

bool Bucket_Boruvka::is_good(const vec_t& n, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero, const long& sketch_seed) const {
  return a < n && contains(a, bucket_seed, guess_nonzero)
    && c == XXH64(&a, sizeof(a), sketch_seed);
}

void Bucket_Boruvka::update(const vec_t& update_idx, const XXH64_hash_t& update_hash) {
  a ^= update_idx;
  c ^= update_hash;
}
