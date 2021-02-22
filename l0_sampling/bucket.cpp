#include "../include/bucket.h"
#include "../include/prime_generator.h"

bool Bucket_Boruvka::contains(const vec_t& index, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const {
  vec_t buf = index + 1;
  XXH64_hash_t hash = XXH64(&buf, sizeof(buf), bucket_seed);
  if (hash % guess_nonzero == 0)
    return true;
  return false;
}

bool Bucket_Boruvka::is_good(const vec_t& n, const XXH64_hash_t& bucket_seed, const vec_t& guess_nonzero) const {
  if (a == 0 || b % a != 0 || b / a <= 0) return false;
  vec_t update_idx = b / a - 1;
  return update_idx < n && contains(update_idx, bucket_seed, guess_nonzero)
    && c == XXH64(&update_idx, sizeof(update_idx), bucket_seed);
}

void Bucket_Boruvka::update(const Update& update, const XXH64_hash_t& bucket_seed) {
  vec_t update_idx = update.index;
  a += update.delta;
  b += update.delta * static_cast<bucket_t>(update_idx + 1); // deals with updates whose indices are 0
  c ^= XXH64(&update_idx, sizeof(update_idx), bucket_seed);
}
