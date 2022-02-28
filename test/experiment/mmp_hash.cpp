#include <iostream>
#include <random>
#include <vector>
#include <bitset>
#include "../../include/l0_sampling/pmshash.h"

const uint64_t A1_1 = 10892479947228793040UL;
const uint64_t A2_1 = 5324285833102856563UL;
const uint64_t B_1 = 15161517132367261381UL;

const uint64_t A1_2 = 14826349932123903041UL;
const uint64_t A2_2 = 15419701087670201850UL;
const uint64_t B_2 = 11875562970292602379UL;

const uint64_t ee = 0x49e3a9c184bd9173;

std::mt19937_64 rng (std::random_device{}());


// https://arxiv.org/pdf/1504.06804.pdf 3.5: Strongly universal hash
uint32_t pms_hash(uint64_t x, uint64_t a1, uint64_t a2, uint64_t b) {
  // hashes 64-bit x strongly universally into l<=32 bits
  // using the random seeds a1, a2, and b.
  return ((a1+x)*(a2+(x>>32))+b) >> 32;
}

uint64_t hasher(uint64_t x) {
  return (((uint64_t) pms_hash(x, A1_1, A2_1, B_1)) << 32)
         | pms_hash(x, A1_2, A2_2, B_2);
}

uint32_t mmp_hash_32(uint64_t x, uint64_t seed) {
  return pms_hash(x, A1_1, seed^A2_1, B_1);
}

uint64_t mmp_hash_64(uint64_t x, uint64_t seed) {
  return (((uint64_t) pms_hash(x, A1_1, seed^A2_1, B_1)) << 32)
    | pms_hash(x, A1_2, seed^A2_2, B_2);
}

// find number of bits that match between 2 32-bit hashes. a good hash should
// give expected 16 matches
int bit_matches(uint32_t orig_hash, uint32_t altered_hash) {
  // popcount
  uint32_t t = orig_hash ^ altered_hash;
  int popcount = 0;
  while (t) {
    if (t & 1) ++popcount;
    t >>= 1;
  }
  return 32 - popcount;
}

int bit_matches(uint64_t orig_hash, uint64_t altered_hash) {
  // popcount
  uint64_t t = orig_hash ^ altered_hash;
  int popcount = 0;
  while (t) {
    if (t & 1) ++popcount;
    t >>= 1;
  }
  return 64 - popcount;
}

int least_set_bit(uint64_t n) {
  int retval = 0;
  if (n == 0) return 64;
  while ((n & 1) == 0) {
    ++retval;
    n >>= 1;
  }
  return retval;
}

void collision_expr(uint32_t (*hash_func)(uint64_t, uint64_t, uint64_t,
      uint64_t), uint64_t a1, uint64_t a2, uint64_t b) {
  unsigned num_reps = 1000000;
  unsigned num_bits_diff = 4;
  std::vector<uint64_t> vec (num_reps);
  for (unsigned i = 0; i < num_reps; ++i) {
    vec[i] = rng();
  }
  std::vector<int> num_matches (1 << num_bits_diff);
  for (size_t i = 0; i < num_reps; ++i) {
    for (uint32_t j = 1; j < (1 << num_bits_diff); ++j) {
      num_matches[j] += bit_matches(hash_func(vec[i], a1, a2, b), hash_func
      (vec[i] + j, a1, a2, b));
    }
  }
  for (int i = 1; i < (1 << num_bits_diff); ++i) {
    num_matches[i] /= num_reps;
  }

  std::cout << "--------------- COLLISION TEST RESULTS ---------------\n";
  std::cout << "a1: " << a1 << ", a2: " << a2 << ", b: " << b << std::endl;
  for (int i = 1; i < (1 << num_bits_diff); ++i) {
    std::cout << "diff[" << i << "]: " << num_matches[i] << "\n";
  }
}

void zeroes_expr(uint64_t (*uint64_hasher)(uint64_t)) {
  unsigned num_reps = 1e8;
  std::vector<uint64_t> vec (num_reps);
  std::vector<int> trailing_zero (65, 0);
  for (unsigned i = 0; i < num_reps; ++i) {
    vec[i] = rng();
  }
  for (size_t i = 0; i < num_reps; ++i) {
    ++trailing_zero[least_set_bit(uint64_hasher(vec[i]))];
  }

  std::cout << "--------------- TRAILING ZERO TEST RESULTS ---------------\n";
  for (int i = 0; i<= 64; ++i) {
    std::cout << "trail[" << i << "]: " << trailing_zero[i]
      << "\t EXP " << (num_reps >> (i+1)) << "\n";
  }
}

void seed_expr(uint64_t (*hash_func)(uint64_t x, uint64_t seed)) {
  unsigned num_reps = 1e7;
  std::vector<uint64_t> vec (num_reps);
  uint64_t x = rng();
  uint64_t zero_seed = hash_func(x, 0);

  std::vector<int> trailing_zero (65, 0);
  std::vector<int> xor_popcount (65, 0);
  for (unsigned i = 0; i < num_reps; ++i) {
    vec[i] = rng();
  }
  for (size_t i = 0; i < num_reps; ++i) {
    uint64_t val = hash_func(x, vec[i]);
    ++trailing_zero[least_set_bit(val)];
    ++xor_popcount[bit_matches(zero_seed, val)];
  }

  std::cout << "---------------- SEED: ZERO TEST RESULTS ----------------\n";
  for (int i = 0; i<= 64; ++i) {
    std::cout << "trail[" << i << "]:\t" << trailing_zero[i]
              << "\t EXP " << (num_reps >> (i+1)) << "\n";
  }
  std::cout << "--------------- SEED: POPCNT TEST RESULTS ---------------\n";
  for (int i = 0; i<= 64; ++i) {
    std::cout << "pop[" << i << "]:\t" << xor_popcount[i] << "\n";
  }
}

void plot_vals(int num_vals, uint64_t seed) {
  for (int i = 0; i < num_vals; ++i) {
    std::cout << std::dec << i << "\t";
    auto val = XXPMS64(i, seed);
    std::cout << std::dec << (val >> 32) << "\t" << (uint32_t) val << "\n";  }
}

void plot_seeds(int num_vals, uint64_t x) {
  for (int i = 0; i < num_vals; ++i) {
    std::cout << std::dec << i << "\t";
    auto val = XXPMS64(x,i);
    std::cout << std::dec << (val >> 32) << "\t" << (uint32_t) val << "\n";
  }
}

void collision_counter(int num_vals) {
  std::vector<uint32_t> vals(num_vals);
  for (int i = 0; i < num_vals; ++i) {
    vals[i] = XXPMS64(14, i) & 0xffffffff;
  }
  int num_collisions = 0;
  for (int i = 0; i < num_vals; ++i) {
    for (int j = i + 1; j < num_vals; ++j) {
      if (vals[i] == vals[j]) {
        ++num_collisions;
        std::cout << i << "," << j << "\n";
      }
    }
  }
  std::cout << "Collisions: " << num_collisions << std::endl;
}

int main() {
  // initialization phase
  uint64_t temp {};
  for (int i = 0; i < 700000; ++i) {
    temp += rng();
  }

//  collision_expr(pms_hash, rng(), rng(), rng());
//  zeroes_expr(hasher);
//  seed_expr(mmp_hash_64);
//  plot_vals(200, 602439);
  collision_counter(20000);
}
