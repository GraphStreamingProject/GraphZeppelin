#include "../../include/l0_sampling/pmshash.h"

const uint64_t A1 = 3388879740877228053UL;
const uint64_t B = 987987350100107189UL;

const uint64_t A1_1 = 10892479947228793041UL;
const uint64_t B_1 = 15161517132367261381UL;

const uint64_t A1_2 = 14826349932123903041UL;
const uint64_t B_2 = 11875562970292602379UL;

const uint64_t xor_mask = 0xFFFFFFFFFFFFFFFF;

// https://arxiv.org/pdf/1504.06804.pdf 3.5: Strongly universal hash
inline uint32_t pms_hash(uint64_t x, uint64_t a1, uint64_t a2, uint64_t b) {
  return ((a1+x)*(a2+(x>>32))+b) >> 32;
}

uint32_t mmp_hash_32(uint64_t x, uint64_t seed) {
  return pms_hash(x, A1, seed, B);
}

uint64_t mmp_hash_64(uint64_t x, uint64_t seed) {
  return (((uint64_t) pms_hash(x, A1_1, seed, B_1)) << 32)
         | pms_hash(x, A1_2, seed^xor_mask, B_2);
}