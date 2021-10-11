#include <gtest/gtest.h>
#include <fstream>
#include "../include/work_queue.h"
#include "../include/cache_buffer_tree.h"

#include <experimental/random>
#include <chrono>
#include <iomanip>

TEST(WorkQueueTestSuite, TestSpeed) {
  const auto count = 10000000;
  const auto node_count = 1000000;
  std::vector<std::pair<node_id_t, node_id_t>> pairs;
  for (size_t i = 0; i < count; ++i)
  {
    auto update = std::make_pair<node_id_t, node_id_t>(std::experimental::randint(0, node_count), std::experimental::randint(0, node_count));
    pairs.push_back(update);
  }
  WorkQueue wq(100000, node_count + 1, 2);
  const auto start = std::chrono::system_clock::now();
  for (size_t i = 0; i < count; ++i)
  {
    wq.insert(pairs[i]);
  }
  const auto end = std::chrono::system_clock::now();
  std::cout << std::fixed << "Edges / sec: " << static_cast<double>(count) / std::chrono::duration<double>(end - start).count() << std::endl;

  //CacheAwareBufferTree<16384, 64> cabt(100000, node_count + 1, 2);
  CacheAwareBufferTree<32768, 64> cabt(100000, node_count + 1, 2);
  //CacheAwareBufferTree<1048576, 64> cabt(100000, node_count + 1, 2);
  const auto start2 = std::chrono::system_clock::now();
  for (size_t i = 0; i < count; ++i)
  {
    cabt.insert(pairs[i]);
  }
  const auto end2 = std::chrono::system_clock::now();
  std::cout << std::fixed << "Edges / sec: " << static_cast<double>(count) / std::chrono::duration<double>(end2 - start2).count() << std::endl;
}
