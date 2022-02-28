#include "../../include/l0_sampling/pmshash.h"

const uint64_t A1_1 = 10892479947228793040UL;
const uint64_t A2_1 = 5324285833102856563UL;
const uint64_t B_1 = 15161517132367261381UL;

const uint64_t A1_2 = 14826349932123903041UL;
const uint64_t A2_2 = 15419701087670201850UL;
const uint64_t B_2 = 11875562970292602379UL;

static uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static uint64_t XXH3_avalanche(uint64_t h64) {
  h64 = rotl(h64, 37);
  h64 *= 0x165667919E3779F9ULL;
  h64 = rotl(h64, 32);
  return h64;
}

uint64_t XXPMS64(const uint64_t input, uint64_t seed) {
  auto const bitflip1 = 0xb9e942ea7b738267 + seed;
  auto const bitflip2 = 0x3a5296093bbc56af - seed;
  auto const input_lo = input ^ bitflip1;
  auto const input_hi = input ^ bitflip2;
  // https://arxiv.org/pdf/1504.06804.pdf 3.5: Strongly universal hash
  auto const acc = (input_lo+input)*(input_hi+(input>>32)) + 0x107b0c29033513cdUL;
  return XXH3_avalanche(acc);
}

uint32_t XXPMS32(uint64_t x, uint64_t seed) {
  return (uint32_t) XXPMS64(x, seed);
}
