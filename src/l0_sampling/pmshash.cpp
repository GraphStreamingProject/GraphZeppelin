#include "../../include/l0_sampling/pmshash.h"

const uint64_t A1_1 = 10892479947228793040UL;
const uint64_t A2_1 = 5324285833102856563UL;
const uint64_t B_1 = 15161517132367261381UL;

const uint64_t A1_2 = 14826349932123903041UL;
const uint64_t A2_2 = 15419701087670201850UL;
const uint64_t B_2 = 11875562970292602379UL;


// https://arxiv.org/pdf/1504.06804.pdf 3.5: Strongly universal hash
uint32_t pms_hash(uint64_t x, uint64_t a1, uint64_t a2, uint64_t b) {
  // hashes 64-bit x strongly universally into l<=32 bits
  // using the random seeds a1, a2, and b.
  return ((a1+x)*(a2+(x>>32))+b) >> 32;
}

uint32_t mmp_hash_32(uint64_t x, uint64_t seed) {
  return pms_hash(x, A1_1, seed^A2_1, B_1);
}

uint64_t mmp_hash_64(uint64_t x, uint64_t seed) {
  return (((uint64_t) pms_hash(x, A1_1, seed^A2_1, B_1)) << 32)
         | pms_hash(x, A1_2, seed^A2_2, B_2);
}
