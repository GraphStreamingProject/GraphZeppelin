#include <xxhash.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

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

void run_32bit_direct_test(uint64_t num_items, uint64_t col_width, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      for (uint64_t c = 0; c < col_width; c++) {
        vec_hash_t hash = get_hash32_direct(index, seed + c);
      
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;

        zeros[j-1]++;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  32 bit hash direct ======" << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;

  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * col_width * num_trials) << ", ";
  }
  std::cout << std::endl;
}

void run_64bit_direct_test(uint64_t num_items, uint64_t col_width, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      for (uint64_t c = 0; c < col_width; c++) {
        vec_hash_t hash = get_hash64_direct(index, seed + c);
      
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;

        zeros[j-1]++;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  64 bit hash direct ======" << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;

  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * col_width * num_trials) << ", ";
  }
  std::cout << std::endl;
}

void run_32bit_vector_test(uint64_t num_items, uint64_t col_width, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      std::vector<vec_hash_t> hashes = get_hash32_vector(index, seed, col_width);
      for (vec_hash_t hash : hashes) {
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;

        zeros[j-1]++;
      }
    }
    
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  32 bit hash vector ======" << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;

  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * col_width * num_trials) << ", ";
  }
  std::cout << std::endl;
}

void run_64bit_vector_test(uint64_t num_items, uint64_t col_width, int num_trials) {
  std::vector<long> zeros(32);
  long seed = 0xBAAABAAA;
  std::mt19937_64 rand(seed);

  auto start = std::chrono::steady_clock::now();
  for (int t = 0; t < num_trials; t++) {
    for (uint64_t i = 0; i < num_items; i++) {
      vec_t index = rand();
      std::vector<vec_hash_t> hashes = get_hash64_vector(index, seed, col_width);
      for (vec_hash_t hash : hashes) {
        int j = 1;
        while(hash % (1 << j) == 0 && j <= 32) j++;

        zeros[j-1]++;
      }
    }
    
  }
  auto end = std::chrono::steady_clock::now();
  long double time_taken = static_cast<std::chrono::duration<long double>>(end - start).count();
  std::cout << "======  64 bit hash vector ======" << std::endl;

  long sum = 0;
  for (int i = 31; i >= 0; i--) {
    sum += zeros[i];
    zeros[i] = sum;
  }

  std::cout << "Average time: " << time_taken / num_trials << std::endl;

  std::cout << "Col depth: ";
  for (int i = 0; i < 32; i++) {
    std::cout << (double)(zeros[i]) / (double)(num_items * col_width * num_trials) << ", ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Number of arguments is incorrect. 3 required" << std::endl;
    std::cout << "num_items, col_width, num_trials" << std::endl;
    exit(1); 
  }

  uint64_t num_items = std::atoll(argv[1]);
  uint64_t col_width = std::atoll(argv[2]);
  int num_trials = std::atoi(argv[3]);

  /*
   * 32 bit tests
   */
  run_32bit_direct_test(num_items, col_width, num_trials);
  run_32bit_vector_test(num_items, col_width, num_trials);
  
  /*
   * 64 bit tests
   */
  run_64bit_direct_test(num_items, col_width, num_trials);  
  run_64bit_vector_test(num_items, col_width, num_trials);
}