#include <gtest/gtest.h>
#include <fstream>
#include "../include/work_queue.h"

#include <experimental/random>
#include <chrono>
#include <iomanip>

TEST(WorkQueueTestSuite, TestSpeed) {
  const auto count = 10000000;
  const auto node_count = 10000;
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
}
