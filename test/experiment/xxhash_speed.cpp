#include <xxhash.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Uncomment this to check the quality of the generated hash
// #define CHECK_QUALITY

typedef uint32_t vec_hash_t;
typedef uint64_t vec_t;
static const auto& hash32_func  = XXH32;
static const auto& hash64_func  = XXH3_64bits_withSeed;

inline vec_hash_t get_hash32_direct(const vec_t& index, long seed) {
  return hash32_func(&index, sizeof(index), seed);
}

inline vec_hash_t get_hash64_direct(const vec_t& index, long seed) {
  return hash64_func(&index, sizeof(index), seed);
}

inline std::vector<vec_hash_t> get_hash32_vector(const vec_t& index, long seed, uint64_t nhash) {
  std::vector<vec_hash_t> ret(nhash);
  for (uint64_t i = 0; i < nhash; i++)
    ret[i] = hash32_func(&index, sizeof(index), seed++);
  return ret;
}

inline std::vector<vec_hash_t> get_hash64_vector(const vec_t& index, long seed, uint64_t nhash) {
  std::vector<vec_hash_t> ret(nhash);
  for (uint64_t i = 0; i < nhash; i+=2) {
    uint64_t hash = hash64_func(&index, sizeof(index), seed++);
    ret[i] = hash >> 32;
    if (i + 1 < nhash) ret[i + 1] = hash & 0xFFFFFFFF;
  }
  return ret;
}

void run_32bit_direct_test(uint64_t num_items, uint64_t num_hashes, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);
  vec_hash_t hash_xor = 0;

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      for (uint64_t c = 0; c < num_hashes; c++) {
        vec_hash_t hash = get_hash32_direct(index, seed + c);
	hash_xor ^= hash;

#ifdef CHECK_QUALITY	
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;
        zeros[j-1]++;
#endif
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  32 bit hash direct ======" << std::endl;
  std::cout << "hash xor " << hash_xor << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;
  std::cout << "Hash calls per second: " << num_items * num_hashes * num_trials / time_taken << std::endl;

#ifdef CHECK_QUALITY
  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * num_hashes * num_trials) << ", ";
  }
  std::cout << std::endl;
#endif
}

void run_64bit_direct_test(uint64_t num_items, uint64_t num_hashes, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);
  vec_hash_t hash_xor = 0;

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      for (uint64_t c = 0; c < num_hashes; c++) {
        vec_hash_t hash = get_hash64_direct(index, seed + c);
	hash_xor ^= hash;
      
#ifdef CHECK_QUALITY	
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;
        zeros[j-1]++;
#endif
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  64 bit hash direct ======" << std::endl;
  std::cout << "hash xor " << hash_xor << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;
  std::cout << "Hash calls per second: " << num_items * num_hashes * num_trials / time_taken << std::endl;

#ifdef CHECK_QUALITY
  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * num_hashes * num_trials) << ", ";
  }
  std::cout << std::endl;
#endif
}

void run_32bit_vector_test(uint64_t num_items, uint64_t num_hashes, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);
  vec_hash_t hash_xor = 0;

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      std::vector<vec_hash_t> hashes = get_hash32_vector(index, seed, num_hashes);
      for (vec_hash_t hash : hashes) {
	hash_xor ^= hash;
#ifdef CHECK_QUALITY	
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;
        zeros[j-1]++;
#endif
      }
    }
    
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  32 bit hash vector ======" << std::endl;
  std::cout << "hash xor " << hash_xor << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;
  std::cout << "Hash calls per second: " << num_items * num_hashes * num_trials / time_taken << std::endl;

#ifdef CHECK_QUALITY
  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * num_hashes * num_trials) << ", ";
  }
  std::cout << std::endl;
#endif
}

void run_64bit_vector_test(uint64_t num_items, uint64_t num_hashes, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);
  vec_hash_t hash_xor = 0;

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      std::vector<vec_hash_t> hashes = get_hash64_vector(index, seed, num_hashes);
      for (vec_hash_t hash : hashes) {
	hash_xor ^= hash;
#ifdef CHECK_QUALITY	
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;
        zeros[j-1]++;
#endif
      }
    }
    
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  64 bit hash vector ======" << std::endl;
  std::cout << "hash xor " << hash_xor << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;
  std::cout << "Hash calls per second: " << num_items * num_hashes * num_trials / time_taken << std::endl;

#ifdef CHECK_QUALITY
  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * num_hashes * num_trials) << ", ";
  }
  std::cout << std::endl;
#endif
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Number of arguments is incorrect. 3 required" << std::endl;
    std::cout << "Arguments: num_items, num_hashes, num_trials" << std::endl;
    std::cout << "num_items:  The number of items to hash, randomly generated" << std::endl;
    std::cout << "num_hashes: Hash the item with this many unique hash seeds" << std::endl;
    std::cout << "num_trials: Repeat the experiment this many times" << std::endl;
    exit(1); 
  }

  uint64_t num_items = std::atoll(argv[1]);
  uint64_t num_hashes = std::atoll(argv[2]);
  int num_trials = std::atoi(argv[3]);

  /*
   * 32 bit tests
   */
  run_32bit_direct_test(num_items, num_hashes, num_trials);
  run_32bit_vector_test(num_items, num_hashes, num_trials);
  
  /*
   * 64 bit tests
   */
  run_64bit_direct_test(num_items, num_hashes, num_trials);  
  run_64bit_vector_test(num_items, num_hashes, num_trials);
}
