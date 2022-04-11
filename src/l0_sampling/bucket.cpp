#include "../../include/l0_sampling/bucket.h"
#include "../../include/l0_sampling/prime_generator.h"

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

bool operator== (const Bucket_Boruvka &bucket1, const Bucket_Boruvka &bucket2) {
  return (bucket1.a == bucket2.a && bucket1.b == bucket2.b && bucket1.c == bucket2.c);
}


// yoinked from xxhash
uint128_t mult64to128(xxh_u64 lhs, xxh_u64 rhs)
{
/*
 * GCC/Clang __uint128_t method.
 *
 * On most 64-bit targets, GCC and Clang define a __uint128_t type.
 * This is usually the best way as it usually uses a native long 64-bit
 * multiply, such as MULQ on x86_64 or MUL + UMULH on aarch64.
 *
 * Usually.
 *
 * Despite being a 32-bit platform, Clang (and emscripten) define this type
 * despite not having the arithmetic for it. This results in a laggy
 * compiler builtin call which calculates a full 128-bit multiply.
 * In that case it is best to use the portable one.
 * https://github.com/Cyan4973/xxHash/issues/211#issuecomment-515575677
 */
#if defined(__GNUC__) && !defined(__wasm__) \
    && defined(__SIZEOF_INT128__) \
    || (defined(_INTEGRAL_MAX_BITS) && _INTEGRAL_MAX_BITS >= 128)

__uint128_t const product = (__uint128_t)lhs * (__uint128_t)rhs;
XXH128_hash_t r128;
r128.low64  = (xxh_u64)(product);
r128.high64 = (xxh_u64)(product >> 64);
return r128;

/*
 * MSVC for x64's _umul128 method.
 *
 * xxh_u64 _umul128(xxh_u64 Multiplier, xxh_u64 Multiplicand, xxh_u64 *HighProduct);
 *
 * This compiles to single operand MUL on x64.
 */
#elif defined(_M_X64) || defined(_M_IA64)

#ifndef _MSC_VER
#   pragma intrinsic(_umul128)
#endif
    xxh_u64 product_high;
    xxh_u64 const product_low = _umul128(lhs, rhs, &product_high);
    XXH128_hash_t r128;
    r128.low64  = product_low;
    r128.high64 = product_high;
    return r128;

#else
    /*
     * Portable scalar method. Optimized for 32-bit and 64-bit ALUs.
     *
     * This is a fast and simple grade school multiply, which is shown below
     * with base 10 arithmetic instead of base 0x100000000.
     *
     *           9 3 // D2 lhs = 93
     *         x 7 5 // D2 rhs = 75
     *     ----------
     *           1 5 // D2 lo_lo = (93 % 10) * (75 % 10) = 15
     *         4 5 | // D2 hi_lo = (93 / 10) * (75 % 10) = 45
     *         2 1 | // D2 lo_hi = (93 % 10) * (75 / 10) = 21
     *     + 6 3 | | // D2 hi_hi = (93 / 10) * (75 / 10) = 63
     *     ---------
     *         2 7 | // D2 cross = (15 / 10) + (45 % 10) + 21 = 27
     *     + 6 7 | | // D2 upper = (27 / 10) + (45 / 10) + 63 = 67
     *     ---------
     *       6 9 7 5 // D4 res = (27 * 10) + (15 % 10) + (67 * 100) = 6975
     *
     * The reasons for adding the products like this are:
     *  1. It avoids manual carry tracking. Just like how
     *     (9 * 9) + 9 + 9 = 99, the same applies with this for UINT64_MAX.
     *     This avoids a lot of complexity.
     *
     *  2. It hints for, and on Clang, compiles to, the powerful UMAAL
     *     instruction available in ARM's Digital Signal Processing extension
     *     in 32-bit ARMv6 and later, which is shown below:
     *
     *         void UMAAL(xxh_u32 *RdLo, xxh_u32 *RdHi, xxh_u32 Rn, xxh_u32 Rm)
     *         {
     *             xxh_u64 product = (xxh_u64)*RdLo * (xxh_u64)*RdHi + Rn + Rm;
     *             *RdLo = (xxh_u32)(product & 0xFFFFFFFF);
     *             *RdHi = (xxh_u32)(product >> 32);
     *         }
     *
     *     This instruction was designed for efficient long multiplication, and
     *     allows this to be calculated in only 4 instructions at speeds
     *     comparable to some 64-bit ALUs.
     *
     *  3. It isn't terrible on other platforms. Usually this will be a couple
     *     of 32-bit ADD/ADCs.
     */

    /* First calculate all of the cross products. */
    xxh_u64 const lo_lo = XXH_mult32to64(lhs & 0xFFFFFFFF, rhs & 0xFFFFFFFF);
    xxh_u64 const hi_lo = XXH_mult32to64(lhs >> 32,        rhs & 0xFFFFFFFF);
    xxh_u64 const lo_hi = XXH_mult32to64(lhs & 0xFFFFFFFF, rhs >> 32);
    xxh_u64 const hi_hi = XXH_mult32to64(lhs >> 32,        rhs >> 32);

    /* Now add the products together. These will never overflow. */
    xxh_u64 const cross = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFF) + lo_hi;
    xxh_u64 const upper = (hi_lo >> 32) + (cross >> 32)        + hi_hi;
    xxh_u64 const lower = (cross << 32) | (lo_lo & 0xFFFFFFFF);

    XXH128_hash_t r128;
    r128.low64  = lower;
    r128.high64 = upper;
    return r128;
#endif
}