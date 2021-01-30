#include "../include/bucket.h"

XXH64_hash_t Bucket_Boruvka::gen_bucket_seed(const unsigned bucket_id, long seed) {
  return XXH64(&bucket_id, sizeof(bucket_id), seed);
}

mp::uint128_t Bucket_Boruvka::gen_r(const XXH64_hash_t& bucket_seed, const mp::uint128_t& large_prime) {
  //TODO: XXH64_hash_t is 64 bits, but we want to generate a 128 bit number in [2,large_prime-2]
  return 2 + bucket_seed % (large_prime - 3);
}

bool Bucket_Boruvka::contains(const uint64_t& index, const XXH64_hash_t& bucket_seed, const uint64_t& guess_nonzero) const {
  uint64_t buf = index + 1;
  XXH64_hash_t hash = XXH64(&buf, sizeof(buf), bucket_seed);
  if (hash % guess_nonzero == 0)
    return true;
  return false;
}

bool Bucket_Boruvka::is_good(const uint64_t& n, const mp::uint128_t& large_prime, const Montgomery::Ctx& large_prime_ctx, const XXH64_hash_t& bucket_seed, const mp::uint128_t& r, const uint64_t& guess_nonzero) const {
  return a != 0 && b % a == 0  && b / a > 0 && b / a <= n
      && contains(static_cast<uint64_t>((b / a - 1).toBoostUInt128()), bucket_seed, guess_nonzero)
      && ((static_cast<mp::int256_t>(c) + static_cast<mp::int256_t>(large_prime) - static_cast<mp::int256_t>(a.toBoostInt128())
//      * PrimeGenerator::powermod(r, (b / a).toBoostUInt128(), large_prime_ctx))
      * PrimeGenerator::powermod(r, (b / a).toBoostUInt128(), large_prime))
      % large_prime) % large_prime == 0;
}

void Bucket_Boruvka::update(const Update& update, const mp::uint128_t& large_prime, const Montgomery::Ctx& large_prime_ctx, const mp::uint128_t& r) {
  a += update.delta;
  b += update.delta * static_cast<mp::int128_t>(update.index + 1); // deals with updates whose indices are 0
  c = static_cast<mp::uint128_t>(
      (static_cast<mp::int256_t>(c)
      + static_cast<mp::int256_t>(large_prime)
//      + (static_cast<mp::int256_t>(update.delta) * PrimeGenerator::powermod(r, update.index + 1, large_prime_ctx) % large_prime))
      + (static_cast<mp::int256_t>(update.delta) * PrimeGenerator::powermod(r, update.index + 1, large_prime) % large_prime))
      % large_prime);
}

void Bucket_Boruvka::cached_update(const Update& update, const mp::uint128_t& large_prime, const Montgomery::Ctx& large_prime_ctx, const std::vector<mp::uint128_t>& r_sq_cache) {
  a += update.delta;
  b += update.delta * static_cast<mp::int128_t>(update.index + 1); // deals with updates whose indices are 0
  c = static_cast<mp::uint128_t>(
      (static_cast<mp::int256_t>(c)
      + static_cast<mp::int256_t>(large_prime)
      + (static_cast<mp::int256_t>(update.delta) * PrimeGenerator::cached_powermod(r_sq_cache, update.index + 1, large_prime) % large_prime))
      % large_prime);
}

