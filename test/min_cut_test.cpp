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

// Test for constructing mincut and driver. Just to ensure everything work
TEST(MinCutTest, Construction) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  AsciiFileStream stream{curr_dir + "/res/multiples_graph_1024.txt", false};

  MinCutSketchAlg mc_alg(1024, get_seed());
  GraphSketchDriver<MinCutSketchAlg> driver(&mc_alg, &stream, DriverConfiguration());
}

TEST(MinCutTest, DisconnectedMinCut) {
  auto driver_config = DriverConfiguration();
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  AsciiFileStream stream{curr_dir + "/res/multiples_graph_1024.txt", false};
  node_id_t num_nodes = stream.vertices();

  MinCutSketchAlg mc_alg(1024, get_seed());

  GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(MINIMUMCUT);
  driver.check_verifier(GraphVerifier(1024, curr_dir + "/res/multiples_graph_1024.txt"));

  ASSERT_EQ(0, mc_alg.minimum_cut().value);
}

