#include <ascii_file_stream.h>
#include <binary_file_stream.h>
#include <dynamic_erdos_generator.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>

#include "min_cut_sketch_alg.h"
#include "graph_sketch_driver.h"
#include "graph_verifier.h"

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  size_t s = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  std::cout << "Seed = " << s << std::endl;
  return s;
}

TEST(MinCutTest, Construction) {
  MinCutSketchAlg mc_alg(1024, get_seed());
}
