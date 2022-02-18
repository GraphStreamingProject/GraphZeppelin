#include <iostream>
#include <random>
#include <vector>

// TODO: set these
const uint64_t A1_1 = 10892479947228793040UL;
const uint64_t A2_1 = 4032793241538373843UL;
const uint64_t B_1 = 15161517132367261381UL;

const uint64_t A1_2 = 14826349932123903041UL;
const uint64_t A2_2 = 15419701087670201850UL;
const uint64_t B_2 = 11875562970292602379UL;


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

uint32_t hash_32(uint64_t x, uint64_t seed) {
  return pms_hash(x, A1_1, seed, B_1);
}

uint64_t hash_64(uint64_t x, uint64_t seed) {
  // TODO make the hashes independent
  return (((uint64_t) pms_hash(x, A1_1, seed, B_1)) << 32)
    | pms_hash(x, A1_2, seed, B_2);
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

int main() {
  // initialization phase
  uint64_t temp {};
//  for (int i = 0; i < 700000; ++i) {
//    temp += rng();
//  }

//  collision_expr(pms_hash, rng(), rng(), rng());
//  zeroes_expr(hasher);
  seed_expr(hash_64);
}

/*
--------------- COLLISION TEST RESULTS ---------------
a1: 10892479947228793040, a2: 4032793241538373843, b: 15161517132367261381
diff[1]: 17
diff[2]: 17
diff[3]: 16
diff[4]: 17
diff[5]: 15
diff[6]: 16
diff[7]: 17
diff[8]: 17
diff[9]: 16
diff[10]: 15
diff[11]: 16
diff[12]: 16
diff[13]: 15
diff[14]: 17
diff[15]: 15
--------------- COLLISION TEST RESULTS ---------------
a1: 14826349932123903041, a2: 15419701087670201850, b: 11875562970292602379
diff[1]: 17
diff[2]: 17
diff[3]: 15
diff[4]: 17
diff[5]: 15
diff[6]: 16
diff[7]: 16
diff[8]: 17
diff[9]: 19
diff[10]: 15
diff[11]: 15
diff[12]: 15
diff[13]: 15
diff[14]: 16
diff[15]: 15
--------------- TRAILING ZERO TEST RESULTS ---------------
trail[0]: 49997230	 EXP 50000000
trail[1]: 25003145	 EXP 25000000
trail[2]: 12502016	 EXP 12500000
trail[3]: 6248179	 EXP 6250000
trail[4]: 3127404	 EXP 3125000
trail[5]: 1561049	 EXP 1562500
trail[6]: 781161	 EXP 781250
trail[7]: 390469	 EXP 390625
trail[8]: 194454	 EXP 195312
trail[9]: 97294	 EXP 97656
trail[10]: 49056	 EXP 48828
trail[11]: 24438	 EXP 24414
trail[12]: 12091	 EXP 12207
trail[13]: 6142	 EXP 6103
trail[14]: 2927	 EXP 3051
trail[15]: 1479	 EXP 1525
trail[16]: 751	 EXP 762
trail[17]: 367	 EXP 381
trail[18]: 168	 EXP 190
trail[19]: 97	 EXP 95
trail[20]: 36	 EXP 47
trail[21]: 24	 EXP 23
trail[22]: 9	 EXP 11
trail[23]: 9	 EXP 5
trail[24]: 2	 EXP 2
trail[25]: 2	 EXP 1
trail[26]: 1	 EXP 0
trail[27]: 0	 EXP 0
trail[28]: 0	 EXP 0
trail[29]: 0	 EXP 0
trail[30]: 0	 EXP 0
trail[31]: 0	 EXP 100000000
trail[32]: 0	 EXP 50000000
trail[33]: 0	 EXP 25000000
trail[34]: 0	 EXP 12500000
trail[35]: 0	 EXP 6250000
trail[36]: 0	 EXP 3125000
trail[37]: 0	 EXP 1562500
trail[38]: 0	 EXP 781250
trail[39]: 0	 EXP 390625
trail[40]: 0	 EXP 195312
trail[41]: 0	 EXP 97656
trail[42]: 0	 EXP 48828
trail[43]: 0	 EXP 24414
trail[44]: 0	 EXP 12207
trail[45]: 0	 EXP 6103
trail[46]: 0	 EXP 3051
trail[47]: 0	 EXP 1525
trail[48]: 0	 EXP 762
trail[49]: 0	 EXP 381
trail[50]: 0	 EXP 190
trail[51]: 0	 EXP 95
trail[52]: 0	 EXP 47
trail[53]: 0	 EXP 23
trail[54]: 0	 EXP 11
trail[55]: 0	 EXP 5
trail[56]: 0	 EXP 2
trail[57]: 0	 EXP 1
trail[58]: 0	 EXP 0
trail[59]: 0	 EXP 0
trail[60]: 0	 EXP 0
trail[61]: 0	 EXP 0
trail[62]: 0	 EXP 0
trail[63]: 0	 EXP 100000000
trail[64]: 0	 EXP 50000000
---------------- SEED: ZERO TEST RESULTS ----------------
trail[0]: 5000709	 EXP 5000000
trail[1]: 2498885	 EXP 2500000
trail[2]: 1249793	 EXP 1250000
trail[3]: 625539	 EXP 625000
trail[4]: 312472	 EXP 312500
trail[5]: 155914	 EXP 156250
trail[6]: 78460	 EXP 78125
trail[7]: 39053	 EXP 39062
trail[8]: 19551	 EXP 19531
trail[9]: 9877	 EXP 9765
trail[10]: 4912	 EXP 4882
trail[11]: 2449	 EXP 2441
trail[12]: 1166	 EXP 1220
trail[13]: 598	 EXP 610
trail[14]: 300	 EXP 305
trail[15]: 168	 EXP 152
trail[16]: 84	 EXP 76
trail[17]: 30	 EXP 38
trail[18]: 23	 EXP 19
trail[19]: 9	 EXP 9
trail[20]: 2	 EXP 4
trail[21]: 3	 EXP 2
trail[22]: 3	 EXP 1
trail[23]: 0	 EXP 0
trail[24]: 0	 EXP 0
trail[25]: 0	 EXP 0
trail[26]: 0	 EXP 0
trail[27]: 0	 EXP 0
trail[28]: 0	 EXP 0
trail[29]: 0	 EXP 0
trail[30]: 0	 EXP 0
trail[31]: 0	 EXP 10000000
trail[32]: 0	 EXP 5000000
trail[33]: 0	 EXP 2500000
trail[34]: 0	 EXP 1250000
trail[35]: 0	 EXP 625000
trail[36]: 0	 EXP 312500
trail[37]: 0	 EXP 156250
trail[38]: 0	 EXP 78125
trail[39]: 0	 EXP 39062
trail[40]: 0	 EXP 19531
trail[41]: 0	 EXP 9765
trail[42]: 0	 EXP 4882
trail[43]: 0	 EXP 2441
trail[44]: 0	 EXP 1220
trail[45]: 0	 EXP 610
trail[46]: 0	 EXP 305
trail[47]: 0	 EXP 152
trail[48]: 0	 EXP 76
trail[49]: 0	 EXP 38
trail[50]: 0	 EXP 19
trail[51]: 0	 EXP 9
trail[52]: 0	 EXP 4
trail[53]: 0	 EXP 2
trail[54]: 0	 EXP 1
trail[55]: 0	 EXP 0
trail[56]: 0	 EXP 0
trail[57]: 0	 EXP 0
trail[58]: 0	 EXP 0
trail[59]: 0	 EXP 0
trail[60]: 0	 EXP 0
trail[61]: 0	 EXP 0
trail[62]: 0	 EXP 0
trail[63]: 0	 EXP 10000000
trail[64]: 0	 EXP 5000000
--------------- SEED: POPCNT TEST RESULTS ---------------
pop[0]: 0
pop[1]: 0
pop[2]: 0
pop[3]: 0
pop[4]: 0
pop[5]: 0
pop[6]: 0
pop[7]: 0
pop[8]: 0
pop[9]: 0
pop[10]: 0
pop[11]: 0
pop[12]: 0
pop[13]: 5
pop[14]: 25
pop[15]: 83
pop[16]: 260
pop[17]: 732
pop[18]: 1937
pop[19]: 4641
pop[20]: 10654
pop[21]: 22239
pop[22]: 44063
pop[23]: 79325
pop[24]: 135652
pop[25]: 218561
pop[26]: 326320
pop[27]: 458255
pop[28]: 606778
pop[29]: 753912
pop[30]: 877864
pop[31]: 962251
pop[32]: 993098
pop[33]: 962289
pop[34]: 878973
pop[35]: 754516
pop[36]: 607139
pop[37]: 457662
pop[38]: 325353
pop[39]: 217682
pop[40]: 135711
pop[41]: 79535
pop[42]: 43622
pop[43]: 22292
pop[44]: 10640
pop[45]: 4699
pop[46]: 2044
pop[47]: 790
pop[48]: 260
pop[49]: 101
pop[50]: 26
pop[51]: 8
pop[52]: 3
pop[53]: 0
pop[54]: 0
pop[55]: 0
pop[56]: 0
pop[57]: 0
pop[58]: 0
pop[59]: 0
pop[60]: 0
pop[61]: 0
pop[62]: 0
pop[63]: 0
pop[64]: 0
 */