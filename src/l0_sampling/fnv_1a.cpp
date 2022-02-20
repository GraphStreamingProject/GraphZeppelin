//
// Created by victor on 2/19/22.
//

#include "../../include/l0_sampling/fnv_1a.h"

const uint32_t p32 = 16777619;
const uint32_t s32 = 2166136261;
const uint64_t p64 = 1099511628211UL;
const uint64_t s64 = 14695981039346656037UL;

uint32_t fnv1a_32(uint64_t x, uint64_t seed) {
  uint32_t val = s32 ^ seed;
  for (int i = 0; i < 4; ++i) {
    val ^= (x >> 8*i) & 255;
    val *= p32;
  }
  return val;
}

inline uint64_t fnv1a_64(uint64_t x, uint64_t seed) {
  uint64_t val = s64 ^ seed;

  val ^= x & 65535;
  val *= p64;

  val ^= (x >> 16) & 65535;
  val *= p64;

  val ^= (x >> 32) & 65535;
  val *= p64;

  val ^= (x >> 48) & 65535;
  val *= p64;

  return val;
}

/*
 * algorithm fnv-1a is
    hash := FNV_offset_basis

    for each byte_of_data to be hashed do
        hash := hash XOR byte_of_data
        hash := hash × FNV_prime

    return hash
 */