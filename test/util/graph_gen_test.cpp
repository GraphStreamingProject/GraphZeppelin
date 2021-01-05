#include <gtest/gtest.h>
#include "graph_gen.h"

TEST(GraphGenTestSuite, TestGeneration) {
  std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  generate_stream();
  struct stat buffer;
  ASSERT_FALSE(stat("./sample.txt", &buffer));
  ASSERT_FALSE(stat("./cum_sample.txt", &buffer));
}
