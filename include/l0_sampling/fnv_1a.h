#pragma once

#include <cstdint>

uint32_t fnv1a_32(uint64_t x, uint64_t seed);

uint64_t fnv1a_64(uint64_t x, uint64_t seed);