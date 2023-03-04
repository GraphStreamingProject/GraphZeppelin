#pragma once
#include <stdint.h>
#include <assert.h>

#include "../../src/cuda_library.cu"

#define XXH_CPU_LITTLE_ENDIAN 1
#define XXH_REROLL 0
#define XXH_ASSERT(c)   assert(c)
#define CUDA_XXH __device__

typedef uint32_t XXH32_hash_t;
typedef XXH32_hash_t xxh_u32;
typedef uint8_t xxh_u8;
typedef enum { XXH_aligned, XXH_unaligned } XXH_alignment;

/* *******************************************************************
*  32-bit hash functions
*********************************************************************/
static const xxh_u32 XXH_PRIME32_1 = 0x9E3779B1U;   /* 0b10011110001101110111100110110001 */
static const xxh_u32 XXH_PRIME32_2 = 0x85EBCA77U;   /* 0b10000101111010111100101001110111 */
static const xxh_u32 XXH_PRIME32_3 = 0xC2B2AE3DU;   /* 0b11000010101100101010111000111101 */
static const xxh_u32 XXH_PRIME32_4 = 0x27D4EB2FU;   /* 0b00100111110101001110101100101111 */
static const xxh_u32 XXH_PRIME32_5 = 0x165667B1U;   /* 0b00010110010101100110011110110001 */

CUDA_XXH static xxh_u32 XXH_read32(const void* memPtr)
{
    xxh_u32 val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

CUDA_XXH static xxh_u32 XXH_swap32 (xxh_u32 x)
{
    return  ((x << 24) & 0xff000000 ) |
            ((x <<  8) & 0x00ff0000 ) |
            ((x >>  8) & 0x0000ff00 ) |
            ((x >> 24) & 0x000000ff );
}

CUDA_XXH xxh_u32 XXH_readLE32(const void* ptr)
{
    return XXH_CPU_LITTLE_ENDIAN ? XXH_read32(ptr) : XXH_swap32(XXH_read32(ptr));
}

CUDA_XXH xxh_u32 XXH_readLE32_align(const void* ptr, XXH_alignment align)
{
    if (align==XXH_unaligned) {
        return XXH_readLE32(ptr);
    } else {
        return XXH_CPU_LITTLE_ENDIAN ? *(const xxh_u32*)ptr : XXH_swap32(*(const xxh_u32*)ptr);
    }
}

#define XXH_get32bits(p) XXH_readLE32_align(p, align)
#define XXH_rotl32(x,r) (((x) << (r)) | ((x) >> (32 - (r))))

CUDA_XXH static xxh_u32 XXH32_round(xxh_u32 acc, xxh_u32 input)
{
    acc += input * XXH_PRIME32_2;
    acc  = XXH_rotl32(acc, 13);
    acc *= XXH_PRIME32_1;
    return acc;
}

/* mix all bits */
CUDA_XXH static xxh_u32 XXH32_avalanche(xxh_u32 h32)
{
    h32 ^= h32 >> 15;
    h32 *= XXH_PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= XXH_PRIME32_3;
    h32 ^= h32 >> 16;
    return(h32);
}

CUDA_XXH static xxh_u32 XXH32_finalize(xxh_u32 h32, const xxh_u8* ptr, size_t len, XXH_alignment align)
{
#define XXH_PROCESS1 do {                           \
    h32 += (*ptr++) * XXH_PRIME32_5;                \
    h32 = XXH_rotl32(h32, 11) * XXH_PRIME32_1;      \
} while (0)

#define XXH_PROCESS4 do {                           \
    h32 += XXH_get32bits(ptr) * XXH_PRIME32_3;      \
    ptr += 4;                                   \
    h32  = XXH_rotl32(h32, 17) * XXH_PRIME32_4;     \
} while (0)

    /* Compact rerolled version */
    if (XXH_REROLL) {
        len &= 15;
        while (len >= 4) {
            XXH_PROCESS4;
            len -= 4;
        }
        while (len > 0) {
            XXH_PROCESS1;
            --len;
        }
        return XXH32_avalanche(h32);
    } else {
         switch(len&15) /* or switch(bEnd - p) */ {
           case 12:      XXH_PROCESS4;
                         /* fallthrough */
           case 8:       XXH_PROCESS4;
                         /* fallthrough */
           case 4:       XXH_PROCESS4;
                         return XXH32_avalanche(h32);

           case 13:      XXH_PROCESS4;
                         /* fallthrough */
           case 9:       XXH_PROCESS4;
                         /* fallthrough */
           case 5:       XXH_PROCESS4;
                         XXH_PROCESS1;
                         return XXH32_avalanche(h32);

           case 14:      XXH_PROCESS4;
                         /* fallthrough */
           case 10:      XXH_PROCESS4;
                         /* fallthrough */
           case 6:       XXH_PROCESS4;
                         XXH_PROCESS1;
                         XXH_PROCESS1;
                         return XXH32_avalanche(h32);

           case 15:      XXH_PROCESS4;
                         /* fallthrough */
           case 11:      XXH_PROCESS4;
                         /* fallthrough */
           case 7:       XXH_PROCESS4;
                         /* fallthrough */
           case 3:       XXH_PROCESS1;
                         /* fallthrough */
           case 2:       XXH_PROCESS1;
                         /* fallthrough */
           case 1:       XXH_PROCESS1;
                         /* fallthrough */
           case 0:       return XXH32_avalanche(h32);
        }
        XXH_ASSERT(0);
        return h32;   /* reaching this point is deemed impossible */
    }
}

CUDA_XXH xxh_u32 XXH32_endian_align(const xxh_u8* input, size_t len, xxh_u32 seed, XXH_alignment align)
{
    const xxh_u8* bEnd = input + len;
    xxh_u32 h32;

    if (len>=16) {
        const xxh_u8* const limit = bEnd - 15;
        xxh_u32 v1 = seed + XXH_PRIME32_1 + XXH_PRIME32_2;
        xxh_u32 v2 = seed + XXH_PRIME32_2;
        xxh_u32 v3 = seed + 0;
        xxh_u32 v4 = seed - XXH_PRIME32_1;

        do {
            v1 = XXH32_round(v1, XXH_get32bits(input)); input += 4;
            v2 = XXH32_round(v2, XXH_get32bits(input)); input += 4;
            v3 = XXH32_round(v3, XXH_get32bits(input)); input += 4;
            v4 = XXH32_round(v4, XXH_get32bits(input)); input += 4;
        } while (input < limit);

        h32 = XXH_rotl32(v1, 1)  + XXH_rotl32(v2, 7)
            + XXH_rotl32(v3, 12) + XXH_rotl32(v4, 18);
    } else {
        h32  = seed + XXH_PRIME32_5;
    }

    h32 += (xxh_u32)len;

    return XXH32_finalize(h32, input, len&15, align);
}

CUDA_XXH XXH32_hash_t CUDA_XXH32 (const void* input, size_t len, XXH32_hash_t seed) {
    return XXH32_endian_align((const xxh_u8*)input, len, seed, XXH_unaligned);
}