#pragma once
#include <xxhash.h>
#include <iostream>
#include <string>
#include <boost/multiprecision/cpp_int.hpp>
#include "int127.h"
#include "montgomery.h"
#include "prime_generator.h"
#include "update.h"

namespace mp = boost::multiprecision;

/*
 * nodes: 2^20
 * n: 2^40, n^2: 2^80 > LONG_LONG_MAX \approx 2^63
 */

/**
 * Represents a bucket in a sketch.
 */
struct Bucket_Boruvka {
  int127 a = 0;
  int127 b = 0;
  mp::uint128_t c = 0;

  static XXH64_hash_t gen_bucket_seed(const unsigned bucket_id, long seed);
  static mp::uint128_t gen_r(const XXH64_hash_t& bucket_seed, const mp::uint128_t& large_prime);
  /**
   * Checks whether the hash associated with the Bucket hashes the index to 0.
   * @param index
   * @param bucket_seed
   * @param guess_nonzero
   * @return true if the index is NOT hashed to zero.
   */
  bool contains(const uint64_t& index, const XXH64_hash_t& bucket_seed, const uint64_t& guess_nonzero) const;
  bool is_good(const uint64_t& n, const mp::uint128_t& large_prime, const Montgomery::Ctx& large_prime_ctx, const XXH64_hash_t& bucket_seed, const mp::uint128_t& r, const uint64_t& guess_nonzero) const;
  void update(const Update& update, const mp::uint128_t& large_prime, const Montgomery::Ctx& large_prime_ctx, const mp::uint128_t& r);
};
