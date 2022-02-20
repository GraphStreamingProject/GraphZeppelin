#pragma once
#include <cstdint>

uint32_t mmp_hash_32(uint64_t x, uint64_t seed);

uint64_t mmp_hash_64(uint64_t x, uint64_t seed);
