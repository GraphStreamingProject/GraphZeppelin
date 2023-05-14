#pragma once
#include <stdint.h>
#include <assert.h>
#include "../src/cuda_library.cu"

#define CUDA_XXH_CPU_LITTLE_ENDIAN 1
#define CUDA_XXH_REROLL_CUDA_XXH64 0
#define CUDA_XXH_SECRET_DEFAULT_SIZE 192
#define CUDA_XXH3_SECRET_SIZE_MIN 136
#define CUDA_XXH __device__

#define CUDA_XXH3_accumulate_512 CUDA_XXH3_accumulate_512_scalar
#define CUDA_XXH3_scrambleAcc    CUDA_XXH3_scrambleAcc_scalar
#define CUDA_XXH3_initCustomSecret CUDA_XXH3_initCustomSecret_scalar

#define CUDA_XXH_TARGET_SSE2  /* disable attribute target */

/* note: use after variable declarations */
#define CUDA_XXH_STATIC_ASSERT(c)  do { enum { CUDA_XXH_sa = 1/(int)(!!(c)) }; } while (0)

#define CUDA_XXH_rotl64(x,r) (((x) << (r)) | ((x) >> (64 - (r))))

#define CUDA_XXH_ASSERT(c)   assert(c)
#define CUDA_XXH_ACC_ALIGN 8
#define CUDA_XXH_likely(x) (x)

typedef uint32_t CUDA_XXH32_hash_t;
typedef CUDA_XXH32_hash_t cuda_xxh_u32;
typedef uint8_t cuda_xxh_u8;

typedef uint64_t CUDA_XXH64_hash_t;
typedef CUDA_XXH64_hash_t cuda_xxh_u64;
typedef uint8_t cuda_xxh_u8;

typedef CUDA_XXH64_hash_t (*CUDA_XXH3_hashLong64_f)(const void*, size_t, CUDA_XXH64_hash_t, const cuda_xxh_u8*, size_t);

typedef void (*CUDA_XXH3_f_accumulate_512)(void*, const void*, const void*);
typedef void (*CUDA_XXH3_f_scrambleAcc)(void*, const void*);
typedef void (*CUDA_XXH3_f_initCustomSecret)(void*, cuda_xxh_u64);

typedef struct {
 CUDA_XXH64_hash_t low64;
 CUDA_XXH64_hash_t high64;
} CUDA_XXH128_hash_cu_t;

/* Pseudorandom secret taken directly from FARSH */
CUDA_XXH static const cuda_xxh_u8 CUDA_XXH3_kSecret[CUDA_XXH_SECRET_DEFAULT_SIZE] = {
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
    0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
    0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
    0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
    0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
    0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
};

/* =======     Long Keys     ======= */

#define CUDA_XXH_STRIPE_LEN 64
#define CUDA_XXH_SECRET_CONSUME_RATE 8   /* nb of secret bytes consumed at each accumulation */
#define CUDA_XXH_ACC_NB (CUDA_XXH_STRIPE_LEN / sizeof(cuda_xxh_u64))

/* *******************************************************************
*  32-bit hash functions
*********************************************************************/
static const cuda_xxh_u32 CUDA_XXH_PRIME32_1 = 0x9E3779B1U;   /* 0b10011110001101110111100110110001 */
static const cuda_xxh_u32 CUDA_XXH_PRIME32_2 = 0x85EBCA77U;   /* 0b10000101111010111100101001110111 */
static const cuda_xxh_u32 CUDA_XXH_PRIME32_3 = 0xC2B2AE3DU;   /* 0b11000010101100101010111000111101 */
static const cuda_xxh_u32 CUDA_XXH_PRIME32_4 = 0x27D4EB2FU;   /* 0b00100111110101001110101100101111 */
static const cuda_xxh_u32 CUDA_XXH_PRIME32_5 = 0x165667B1U;   /* 0b00010110010101100110011110110001 */

/*******   cuda_xxh64   *******/

static const cuda_xxh_u64 CUDA_XXH_PRIME64_1 = 0x9E3779B185EBCA87ULL;   /* 0b1001111000110111011110011011000110000101111010111100101010000111 */
static const cuda_xxh_u64 CUDA_XXH_PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;   /* 0b1100001010110010101011100011110100100111110101001110101101001111 */
static const cuda_xxh_u64 CUDA_XXH_PRIME64_3 = 0x165667B19E3779F9ULL;   /* 0b0001011001010110011001111011000110011110001101110111100111111001 */
static const cuda_xxh_u64 CUDA_XXH_PRIME64_4 = 0x85EBCA77C2B2AE63ULL;   /* 0b1000010111101011110010100111011111000010101100101010111001100011 */
static const cuda_xxh_u64 CUDA_XXH_PRIME64_5 = 0x27D4EB2F165667C5ULL;   /* 0b0010011111010100111010110010111100010110010101100110011111000101 */

#define CUDA_XXH_mult32to64(x, y) ((cuda_xxh_u64)(cuda_xxh_u32)(x) * (cuda_xxh_u64)(cuda_xxh_u32)(y))

/*
 * Calculates a 64->128-bit long multiply.
 *
 * Uses __uint128_t and _umul128 if available, otherwise uses a scalar version.
 */
CUDA_XXH static CUDA_XXH128_hash_cu_t CUDA_XXH_mult64to128(cuda_xxh_u64 lhs, cuda_xxh_u64 rhs){
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
     *         void UMAAL(cuda_xxh_u32 *RdLo, cuda_xxh_u32 *RdHi, cuda_xxh_u32 Rn, cuda_xxh_u32 Rm)
     *         {
     *             cuda_xxh_u64 product = (cuda_xxh_u64)*RdLo * (cuda_xxh_u64)*RdHi + Rn + Rm;
     *             *RdLo = (cuda_xxh_u32)(product & 0xFFFFFFFF);
     *             *RdHi = (cuda_xxh_u32)(product >> 32);
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
    cuda_xxh_u64 const lo_lo = CUDA_XXH_mult32to64(lhs & 0xFFFFFFFF, rhs & 0xFFFFFFFF);
    cuda_xxh_u64 const hi_lo = CUDA_XXH_mult32to64(lhs >> 32,        rhs & 0xFFFFFFFF);
    cuda_xxh_u64 const lo_hi = CUDA_XXH_mult32to64(lhs & 0xFFFFFFFF, rhs >> 32);
    cuda_xxh_u64 const hi_hi = CUDA_XXH_mult32to64(lhs >> 32,        rhs >> 32);

    /* Now add the products together. These will never overflow. */
    cuda_xxh_u64 const cross = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFF) + lo_hi;
    cuda_xxh_u64 const upper = (hi_lo >> 32) + (cross >> 32)        + hi_hi;
    cuda_xxh_u64 const lower = (cross << 32) | (lo_lo & 0xFFFFFFFF);

    CUDA_XXH128_hash_cu_t r128;
    r128.low64  = lower;
    r128.high64 = upper;
    return r128;
}

CUDA_XXH static cuda_xxh_u64 CUDA_XXH3_mul128_fold64(cuda_xxh_u64 lhs, cuda_xxh_u64 rhs)
{
    CUDA_XXH128_hash_cu_t product = CUDA_XXH_mult64to128(lhs, rhs);
    return product.low64 ^ product.high64;
}

CUDA_XXH static cuda_xxh_u32 CUDA_XXH_read32(const void* memPtr)
{
    cuda_xxh_u32 val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

CUDA_XXH static cuda_xxh_u32 CUDA_XXH_swap32 (cuda_xxh_u32 x)
{
    return  ((x << 24) & 0xff000000 ) |
            ((x <<  8) & 0x00ff0000 ) |
            ((x >>  8) & 0x0000ff00 ) |
            ((x >> 24) & 0x000000ff );
}


CUDA_XXH cuda_xxh_u32 CUDA_XXH_readLE32(const void* ptr)
{
    return CUDA_XXH_CPU_LITTLE_ENDIAN ? CUDA_XXH_read32(ptr) : CUDA_XXH_swap32(CUDA_XXH_read32(ptr));
}


CUDA_XXH static cuda_xxh_u64 CUDA_XXH_read64(const void* memPtr)
{
    cuda_xxh_u64 val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

CUDA_XXH static cuda_xxh_u64 CUDA_XXH_swap64 (cuda_xxh_u64 x)
{
    return  ((x << 56) & 0xff00000000000000ULL) |
            ((x << 40) & 0x00ff000000000000ULL) |
            ((x << 24) & 0x0000ff0000000000ULL) |
            ((x << 8)  & 0x000000ff00000000ULL) |
            ((x >> 8)  & 0x00000000ff000000ULL) |
            ((x >> 24) & 0x0000000000ff0000ULL) |
            ((x >> 40) & 0x000000000000ff00ULL) |
            ((x >> 56) & 0x00000000000000ffULL);
}

CUDA_XXH cuda_xxh_u64 CUDA_XXH_readLE64(const void* ptr)
{
    return CUDA_XXH_CPU_LITTLE_ENDIAN ? CUDA_XXH_read64(ptr) : CUDA_XXH_swap64(CUDA_XXH_read64(ptr));
}

CUDA_XXH void CUDA_XXH_writeLE64(void* dst, cuda_xxh_u64 v64)
{
    if (!CUDA_XXH_CPU_LITTLE_ENDIAN) v64 = CUDA_XXH_swap64(v64);
    memcpy(dst, &v64, sizeof(v64));
}

CUDA_XXH cuda_xxh_u64 CUDA_XXH3_mix2Accs(const cuda_xxh_u64* acc, const cuda_xxh_u8* secret)
{
    return CUDA_XXH3_mul128_fold64(
               acc[0] ^ CUDA_XXH_readLE64(secret),
               acc[1] ^ CUDA_XXH_readLE64(secret+8) );
}

/* Seems to produce slightly better code on GCC for some reason. */
CUDA_XXH cuda_xxh_u64 CUDA_XXH_xorshift64(cuda_xxh_u64 v64, int shift)
{
    CUDA_XXH_ASSERT(0 <= shift && shift < 64);
    return v64 ^ (v64 >> shift);
}

/*
 * This is a fast avalanche stage,
 * suitable when input bits are already partially mixed
 */
CUDA_XXH static CUDA_XXH64_hash_t CUDA_XXH3_avalanche(cuda_xxh_u64 h64)
{
    h64 = CUDA_XXH_xorshift64(h64, 37);
    h64 *= 0x165667919E3779F9ULL;
    h64 = CUDA_XXH_xorshift64(h64, 32);
    return h64;
}

CUDA_XXH static cuda_xxh_u64 CUDA_XXH64_avalanche(cuda_xxh_u64 h64)
{
    h64 ^= h64 >> 33;
    h64 *= CUDA_XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= CUDA_XXH_PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}

CUDA_XXH static CUDA_XXH64_hash_t CUDA_XXH3_mergeAccs(const cuda_xxh_u64* acc, const cuda_xxh_u8* secret, cuda_xxh_u64 start)
{
    cuda_xxh_u64 result64 = start;
    size_t i = 0;

    for (i = 0; i < 4; i++) {
        result64 += CUDA_XXH3_mix2Accs(acc+2*i, secret + 16*i);
    }

    return CUDA_XXH3_avalanche(result64);
}

CUDA_XXH void CUDA_XXH3_accumulate(cuda_xxh_u64* acc,
                const cuda_xxh_u8* input,
                const cuda_xxh_u8* secret,
                size_t nbStripes,
                CUDA_XXH3_f_accumulate_512 f_acc512)
{
    size_t n;
    for (n = 0; n < nbStripes; n++ ) {
        const cuda_xxh_u8* const in = input + n*CUDA_XXH_STRIPE_LEN;
        //CUDA_XXH_PREFETCH(in + CUDA_XXH_PREFETCH_DIST);
        f_acc512(acc,
                 in,
                 secret + n*CUDA_XXH_SECRET_CONSUME_RATE);
    }
}

CUDA_XXH void CUDA_XXH3_hashLong_internal_loop(cuda_xxh_u64* acc,
                      const cuda_xxh_u8* input, size_t len,
                      const cuda_xxh_u8* secret, size_t secretSize,
                      CUDA_XXH3_f_accumulate_512 f_acc512,
                      CUDA_XXH3_f_scrambleAcc f_scramble)
{
    size_t const nbStripesPerBlock = (secretSize - CUDA_XXH_STRIPE_LEN) / CUDA_XXH_SECRET_CONSUME_RATE;
    size_t const block_len = CUDA_XXH_STRIPE_LEN * nbStripesPerBlock;
    size_t const nb_blocks = (len - 1) / block_len;

    size_t n;

    CUDA_XXH_ASSERT(secretSize >= CUDA_XXH3_SECRET_SIZE_MIN);

    for (n = 0; n < nb_blocks; n++) {
        CUDA_XXH3_accumulate(acc, input + n*block_len, secret, nbStripesPerBlock, f_acc512);
        f_scramble(acc, secret + secretSize - CUDA_XXH_STRIPE_LEN);
    }

    /* last partial block */
    CUDA_XXH_ASSERT(len > CUDA_XXH_STRIPE_LEN);
    {   size_t const nbStripes = ((len - 1) - (block_len * nb_blocks)) / CUDA_XXH_STRIPE_LEN;
        CUDA_XXH_ASSERT(nbStripes <= (secretSize / CUDA_XXH_SECRET_CONSUME_RATE));
        CUDA_XXH3_accumulate(acc, input + nb_blocks*block_len, secret, nbStripes, f_acc512);

        /* last stripe */
        {   const cuda_xxh_u8* const p = input + len - CUDA_XXH_STRIPE_LEN;

#define CUDA_XXH_SECRET_LASTACC_START 7  /* not aligned on 8, last secret is different from acc & scrambler */
            f_acc512(acc, p, secret + secretSize - CUDA_XXH_STRIPE_LEN - CUDA_XXH_SECRET_LASTACC_START);
    }   }
}

#define CUDA_XXH3_INIT_ACC { CUDA_XXH_PRIME32_3, CUDA_XXH_PRIME64_1, CUDA_XXH_PRIME64_2, CUDA_XXH_PRIME64_3, \
                        CUDA_XXH_PRIME64_4, CUDA_XXH_PRIME32_2, CUDA_XXH_PRIME64_5, CUDA_XXH_PRIME32_1 }

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_hashLong_64b_internal(const void* input, size_t len,
                           const void* secret, size_t secretSize,
                           CUDA_XXH3_f_accumulate_512 f_acc512,
                           CUDA_XXH3_f_scrambleAcc f_scramble)
{
    cuda_xxh_u64 acc[CUDA_XXH_ACC_NB] = CUDA_XXH3_INIT_ACC;

    CUDA_XXH3_hashLong_internal_loop(acc, (const cuda_xxh_u8*)input, len, (const cuda_xxh_u8*)secret, secretSize, f_acc512, f_scramble);

    /* converge into final hash */
    CUDA_XXH_STATIC_ASSERT(sizeof(acc) == 64);
    /* do not align on 8, so that the secret is different from the accumulator */
#define CUDA_XXH_SECRET_MERGEACCS_START 11
    CUDA_XXH_ASSERT(secretSize >= sizeof(acc) + CUDA_XXH_SECRET_MERGEACCS_START);
    return CUDA_XXH3_mergeAccs(acc, (const cuda_xxh_u8*)secret + CUDA_XXH_SECRET_MERGEACCS_START, (cuda_xxh_u64)len * CUDA_XXH_PRIME64_1);
}

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_hashLong_64b_withSeed_internal(const void* input, size_t len,
                                    CUDA_XXH64_hash_t seed,
                                    CUDA_XXH3_f_accumulate_512 f_acc512,
                                    CUDA_XXH3_f_scrambleAcc f_scramble,
                                    CUDA_XXH3_f_initCustomSecret f_initSec)
{
    if (seed == 0)
        return CUDA_XXH3_hashLong_64b_internal(input, len,
                                          CUDA_XXH3_kSecret, sizeof(CUDA_XXH3_kSecret),
                                          f_acc512, f_scramble);
    {   cuda_xxh_u8 secret[CUDA_XXH_SECRET_DEFAULT_SIZE];
        f_initSec(secret, seed);
        return CUDA_XXH3_hashLong_64b_internal(input, len, secret, sizeof(secret),
                                          f_acc512, f_scramble);
    }
}

CUDA_XXH void CUDA_XXH3_accumulate_512_scalar(void* acc,
                     const void* input,
                     const void* secret)
{
    cuda_xxh_u64* const xacc = (cuda_xxh_u64*) acc; /* presumed aligned */
    const cuda_xxh_u8* const xinput  = (const cuda_xxh_u8*) input;  /* no alignment restriction */
    const cuda_xxh_u8* const xsecret = (const cuda_xxh_u8*) secret;   /* no alignment restriction */
    size_t i;
    CUDA_XXH_ASSERT(((size_t)acc & (CUDA_XXH_ACC_ALIGN-1)) == 0);
    for (i=0; i < CUDA_XXH_ACC_NB; i++) {
        cuda_xxh_u64 const data_val = CUDA_XXH_readLE64(xinput + 8*i);
        cuda_xxh_u64 const data_key = data_val ^ CUDA_XXH_readLE64(xsecret + i*8);
        xacc[i ^ 1] += data_val; /* swap adjacent lanes */
        xacc[i] += CUDA_XXH_mult32to64(data_key & 0xFFFFFFFF, data_key >> 32);
    }
}

CUDA_XXH void CUDA_XXH3_scrambleAcc_scalar(void* acc, const void* secret)
{
    cuda_xxh_u64* const xacc = (cuda_xxh_u64*) acc;   /* presumed aligned */
    const cuda_xxh_u8* const xsecret = (const cuda_xxh_u8*) secret;   /* no alignment restriction */
    size_t i;
    CUDA_XXH_ASSERT((((size_t)acc) & (CUDA_XXH_ACC_ALIGN-1)) == 0);
    for (i=0; i < CUDA_XXH_ACC_NB; i++) {
        cuda_xxh_u64 const key64 = CUDA_XXH_readLE64(xsecret + 8*i);
        cuda_xxh_u64 acc64 = xacc[i];
        acc64 = CUDA_XXH_xorshift64(acc64, 47);
        acc64 ^= key64;
        acc64 *= CUDA_XXH_PRIME32_1;
        xacc[i] = acc64;
    }
}

CUDA_XXH void CUDA_XXH3_initCustomSecret_scalar(void* customSecret, cuda_xxh_u64 seed64)
{
    /*
     * We need a separate pointer for the hack below,
     * which requires a non-const pointer.
     * Any decent compiler will optimize this out otherwise.
     */
    const cuda_xxh_u8* kSecretPtr = CUDA_XXH3_kSecret;
    CUDA_XXH_STATIC_ASSERT((CUDA_XXH_SECRET_DEFAULT_SIZE & 15) == 0);
    /*
     * Note: in debug mode, this overrides the asm optimization
     * and Clang will emit MOVK chains again.
     */
    CUDA_XXH_ASSERT(kSecretPtr == CUDA_XXH3_kSecret);

    {   int const nbRounds = CUDA_XXH_SECRET_DEFAULT_SIZE / 16;
        int i;
        for (i=0; i < nbRounds; i++) {
            /*
             * The asm hack causes Clang to assume that kSecretPtr aliases with
             * customSecret, and on aarch64, this prevented LDP from merging two
             * loads together for free. Putting the loads together before the stores
             * properly generates LDP.
             */
            cuda_xxh_u64 lo = CUDA_XXH_readLE64(kSecretPtr + 16*i)     + seed64;
            cuda_xxh_u64 hi = CUDA_XXH_readLE64(kSecretPtr + 16*i + 8) - seed64;
            CUDA_XXH_writeLE64((cuda_xxh_u8*)customSecret + 16*i,     lo);
            CUDA_XXH_writeLE64((cuda_xxh_u8*)customSecret + 16*i + 8, hi);
    }   }
}

/*
 * It's important for performance that CUDA_XXH3_hashLong is not inlined.
 */
CUDA_XXH static CUDA_XXH64_hash_t CUDA_XXH3_hashLong_64b_withSeed(const void* input, size_t len,
                           CUDA_XXH64_hash_t seed, const cuda_xxh_u8* secret, size_t secretLen)
{
    (void)secret; (void)secretLen;
    return CUDA_XXH3_hashLong_64b_withSeed_internal(input, len, seed,
                CUDA_XXH3_accumulate_512, CUDA_XXH3_scrambleAcc, CUDA_XXH3_initCustomSecret);
}

/*
 * This is a stronger avalanche,
 * inspired by Pelle Evensen's rrmxmx
 * preferable when input has not been previously mixed
 */
CUDA_XXH static CUDA_XXH64_hash_t CUDA_XXH3_rrmxmx(cuda_xxh_u64 h64, cuda_xxh_u64 len)
{
    /* this mix is inspired by Pelle Evensen's rrmxmx */
    h64 ^= CUDA_XXH_rotl64(h64, 49) ^ CUDA_XXH_rotl64(h64, 24);
    h64 *= 0x9FB21C651E98DF25ULL;
    h64 ^= (h64 >> 35) + len ;
    h64 *= 0x9FB21C651E98DF25ULL;
    return CUDA_XXH_xorshift64(h64, 28);
}

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_len_1to3_64b(const cuda_xxh_u8* input, size_t len, const cuda_xxh_u8* secret, CUDA_XXH64_hash_t seed)
{
    CUDA_XXH_ASSERT(input != NULL);
    CUDA_XXH_ASSERT(1 <= len && len <= 3);
    CUDA_XXH_ASSERT(secret != NULL);
    /*
     * len = 1: combined = { input[0], 0x01, input[0], input[0] }
     * len = 2: combined = { input[1], 0x02, input[0], input[1] }
     * len = 3: combined = { input[2], 0x03, input[0], input[1] }
     */
    {   cuda_xxh_u8  const c1 = input[0];
        cuda_xxh_u8  const c2 = input[len >> 1];
        cuda_xxh_u8  const c3 = input[len - 1];
        cuda_xxh_u32 const combined = ((cuda_xxh_u32)c1 << 16) | ((cuda_xxh_u32)c2  << 24)
                               | ((cuda_xxh_u32)c3 <<  0) | ((cuda_xxh_u32)len << 8);
        cuda_xxh_u64 const bitflip = (CUDA_XXH_readLE32(secret) ^ CUDA_XXH_readLE32(secret+4)) + seed;
        cuda_xxh_u64 const keyed = (cuda_xxh_u64)combined ^ bitflip;
        return CUDA_XXH64_avalanche(keyed);
    }
}

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_len_4to8_64b(const cuda_xxh_u8* input, size_t len, const cuda_xxh_u8* secret, CUDA_XXH64_hash_t seed)
{
    CUDA_XXH_ASSERT(input != NULL);
    CUDA_XXH_ASSERT(secret != NULL);
    CUDA_XXH_ASSERT(4 <= len && len < 8);
    seed ^= (cuda_xxh_u64)CUDA_XXH_swap32((cuda_xxh_u32)seed) << 32;
    {   cuda_xxh_u32 const input1 = CUDA_XXH_readLE32(input);
        cuda_xxh_u32 const input2 = CUDA_XXH_readLE32(input + len - 4);
        cuda_xxh_u64 const bitflip = (CUDA_XXH_readLE64(secret+8) ^ CUDA_XXH_readLE64(secret+16)) - seed;
        cuda_xxh_u64 const input64 = input2 + (((cuda_xxh_u64)input1) << 32);
        cuda_xxh_u64 const keyed = input64 ^ bitflip;
        return CUDA_XXH3_rrmxmx(keyed, len);
    }
}

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_len_9to16_64b(const cuda_xxh_u8* input, size_t len, const cuda_xxh_u8* secret, CUDA_XXH64_hash_t seed)
{
    CUDA_XXH_ASSERT(input != NULL);
    CUDA_XXH_ASSERT(secret != NULL);
    CUDA_XXH_ASSERT(8 <= len && len <= 16);
    {   cuda_xxh_u64 const bitflip1 = (CUDA_XXH_readLE64(secret+24) ^ CUDA_XXH_readLE64(secret+32)) + seed;
        cuda_xxh_u64 const bitflip2 = (CUDA_XXH_readLE64(secret+40) ^ CUDA_XXH_readLE64(secret+48)) - seed;
        cuda_xxh_u64 const input_lo = CUDA_XXH_readLE64(input)           ^ bitflip1;
        cuda_xxh_u64 const input_hi = CUDA_XXH_readLE64(input + len - 8) ^ bitflip2;
        cuda_xxh_u64 const acc = len
                          + CUDA_XXH_swap64(input_lo) + input_hi
                          + CUDA_XXH3_mul128_fold64(input_lo, input_hi);
        return CUDA_XXH3_avalanche(acc);
    }
}

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_len_0to16_64b(const cuda_xxh_u8* input, size_t len, const cuda_xxh_u8* secret, CUDA_XXH64_hash_t seed)
{
    CUDA_XXH_ASSERT(len <= 16);
    {   if (CUDA_XXH_likely(len >  8)) return CUDA_XXH3_len_9to16_64b(input, len, secret, seed);
        if (CUDA_XXH_likely(len >= 4)) return CUDA_XXH3_len_4to8_64b(input, len, secret, seed);
        if (len) return CUDA_XXH3_len_1to3_64b(input, len, secret, seed);
        return CUDA_XXH64_avalanche(seed ^ (CUDA_XXH_readLE64(secret+56) ^ CUDA_XXH_readLE64(secret+64)));
    }
}

CUDA_XXH cuda_xxh_u64 CUDA_XXH3_mix16B(const cuda_xxh_u8* input,
                                     const cuda_xxh_u8* secret, cuda_xxh_u64 seed64)
{
    cuda_xxh_u64 const input_lo = CUDA_XXH_readLE64(input);
    cuda_xxh_u64 const input_hi = CUDA_XXH_readLE64(input+8);
    return CUDA_XXH3_mul128_fold64(
        input_lo ^ (CUDA_XXH_readLE64(secret)   + seed64),
        input_hi ^ (CUDA_XXH_readLE64(secret+8) - seed64)
    );
}

/* For mid range keys, CUDA_XXH3 uses a Mum-hash variant. */
CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_len_17to128_64b(const cuda_xxh_u8* input, size_t len,
                     const cuda_xxh_u8* secret, size_t secretSize,
                     CUDA_XXH64_hash_t seed)
{
    CUDA_XXH_ASSERT(secretSize >= CUDA_XXH3_SECRET_SIZE_MIN); (void)secretSize;
    CUDA_XXH_ASSERT(16 < len && len <= 128);

    {   cuda_xxh_u64 acc = len * CUDA_XXH_PRIME64_1;
        if (len > 32) {
            if (len > 64) {
                if (len > 96) {
                    acc += CUDA_XXH3_mix16B(input+48, secret+96, seed);
                    acc += CUDA_XXH3_mix16B(input+len-64, secret+112, seed);
                }
                acc += CUDA_XXH3_mix16B(input+32, secret+64, seed);
                acc += CUDA_XXH3_mix16B(input+len-48, secret+80, seed);
            }
            acc += CUDA_XXH3_mix16B(input+16, secret+32, seed);
            acc += CUDA_XXH3_mix16B(input+len-32, secret+48, seed);
        }
        acc += CUDA_XXH3_mix16B(input+0, secret+0, seed);
        acc += CUDA_XXH3_mix16B(input+len-16, secret+16, seed);

        return CUDA_XXH3_avalanche(acc);
    }
}

#define CUDA_XXH3_MIDSIZE_MAX 240

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_len_129to240_64b(const cuda_xxh_u8* input, size_t len,
                      const cuda_xxh_u8* secret, size_t secretSize,
                      CUDA_XXH64_hash_t seed)
{
    CUDA_XXH_ASSERT(secretSize >= CUDA_XXH3_SECRET_SIZE_MIN); (void)secretSize;
    CUDA_XXH_ASSERT(128 < len && len <= CUDA_XXH3_MIDSIZE_MAX);

    #define CUDA_XXH3_MIDSIZE_STARTOFFSET 3
    #define CUDA_XXH3_MIDSIZE_LASTOFFSET  17

    cuda_xxh_u64 acc = len * CUDA_XXH_PRIME64_1;
    int const nbRounds = (int)len / 16;
    int i;
    for (i=0; i<8; i++) {
        acc += CUDA_XXH3_mix16B(input+(16*i), secret+(16*i), seed);
    }
    acc = CUDA_XXH3_avalanche(acc);
    CUDA_XXH_ASSERT(nbRounds >= 8);

    for (i=8 ; i < nbRounds; i++) {
        acc += CUDA_XXH3_mix16B(input+(16*i), secret+(16*(i-8)) + CUDA_XXH3_MIDSIZE_STARTOFFSET, seed);
    }
    /* last bytes */
    acc += CUDA_XXH3_mix16B(input + len - 16, secret + CUDA_XXH3_SECRET_SIZE_MIN - CUDA_XXH3_MIDSIZE_LASTOFFSET, seed);
    return CUDA_XXH3_avalanche(acc);
    
}

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH3_64bits_internal(const void* input, size_t len,
                     CUDA_XXH64_hash_t seed64, const void* secret, size_t secretLen,
                     CUDA_XXH3_hashLong64_f f_hashLong)
{
    CUDA_XXH_ASSERT(secretLen >= CUDA_XXH3_SECRET_SIZE_MIN);
    /*
     * If an action is to be taken if `secretLen` condition is not respected,
     * it should be done here.
     * For now, it's a contract pre-condition.
     * Adding a check and a branch here would cost performance at every hash.
     * Also, note that function signature doesn't offer room to return an error.
     */
    if (len <= 16)
        return CUDA_XXH3_len_0to16_64b((const cuda_xxh_u8*)input, len, (const cuda_xxh_u8*)secret, seed64);
    if (len <= 128)
        return CUDA_XXH3_len_17to128_64b((const cuda_xxh_u8*)input, len, (const cuda_xxh_u8*)secret, secretLen, seed64);
    if (len <= CUDA_XXH3_MIDSIZE_MAX)
        return CUDA_XXH3_len_129to240_64b((const cuda_xxh_u8*)input, len, (const cuda_xxh_u8*)secret, secretLen, seed64);
    return f_hashLong(input, len, seed64, (const cuda_xxh_u8*)secret, secretLen);
}

CUDA_XXH CUDA_XXH64_hash_t CUDA_XXH64 (const void* input, size_t len, CUDA_XXH64_hash_t seed)
{
    //return CUDA_XXH64_endian_align((const cuda_xxh_u8*)input, len, seed, CUDA_XXH_unaligned);
    return CUDA_XXH3_64bits_internal(input, len, seed, CUDA_XXH3_kSecret, sizeof(CUDA_XXH3_kSecret), CUDA_XXH3_hashLong_64b_withSeed);
}