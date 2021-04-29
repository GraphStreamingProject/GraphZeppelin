#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "../../include/supernode.h"

void parallel_test(unsigned long vec_size, unsigned long num_updates) {
  srand(time(NULL));
  std::vector<vec_t> updates(num_updates);
  for (unsigned long i = 0; i < num_updates; i++) {
    updates[i] = static_cast<vec_t>(rand() % vec_size);
  }
  auto seed = rand();
  Supernode supernode(vec_size, seed);
  Supernode supernode_batch(vec_size, seed);
  auto start_time = omp_get_wtime();
  for (const auto& update : updates) {
    supernode.update(update);
  }
  long double one, batch;
  one = omp_get_wtime() - start_time;
  std::cout << "One by one updates took " << one << std::endl;
  start_time = omp_get_wtime();
  supernode_batch.batch_update(updates);
  batch = omp_get_wtime() - start_time;
  std::cout << "Batched updates took " << batch << std::endl;
  std::cout << "Fraction of time used: " << batch/one << std::endl;
}

TEST(EXPR_Parallelism, Experiment) {
  for (int i = 2; i <=10; i+=2) {
    unsigned long vec_size = (unsigned long) pow(2,i) + 1;
    for (int j = 3; j <= 7; ++j) {
      unsigned long num_updates = (unsigned long) pow(10,j);
      std::cout << "-------<logn:" << i << ", upd:10e" << j << ">-------" << endl;
      parallel_test(vec_size,num_updates);
      std::cout << std::endl;
    }
  }
}
