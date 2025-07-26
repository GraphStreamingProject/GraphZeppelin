#include <ascii_file_stream.h>
#include <binary_file_stream.h>
#include <static_erdos_generator.h>
#include <dynamic_erdos_generator.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>

#include "min_cut_sketch_alg.h"
#include "graph_sketch_driver.h"
#include "graph_verifier.h"

// helper function to generate a dynamic binary stream and its cumulative insert only stream
void generate_stream(size_t seed, node_id_t num_vertices, double density, std::string stream_name) {
  // generate new stream files
  StaticErdosGenerator stream(seed, num_vertices, density);
  stream.to_binary_file(stream_name);
}

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
  node_id_t num_vertices = stream.vertices();

  MinCutSketchAlg mc_alg(num_vertices, get_seed());

  GraphSketchDriver<MinCutSketchAlg> driver(&mc_alg, &stream, driver_config);
  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(MINIMUMCUT);
  driver.check_verifier(GraphVerifier(num_vertices, curr_dir + "/res/multiples_graph_1024.txt"));

  ASSERT_EQ(0, mc_alg.calc_minimum_cut().value);
}

TEST(MinCutTest, CompleteGraphMinCut) {
  auto driver_config = DriverConfiguration().worker_threads(8);
  double epsilon = 0.8;
  generate_stream(get_seed(), 1 << 13, 1, "./complete_stream.bin");
  size_t true_mc = (1 << 13) - 1;

  BinaryFileStream stream{"./complete_stream.bin"};
  node_id_t num_vertices = stream.vertices();

  MinCutSketchAlg mc_alg(num_vertices, get_seed(), MCAlgConfiguration().epsilon(0.8));

  GraphSketchDriver<MinCutSketchAlg> driver(&mc_alg, &stream, driver_config);
  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(MINIMUMCUT);

  MinCut ret = mc_alg.calc_minimum_cut();

  ASSERT_GT(true_mc, ret.value);
  ASSERT_LT(true_mc * (1 - epsilon), ret.value);
  std::remove("./complete_stream.bin");
}

TEST(MinCutTest, DynamicMinCut) {
  auto driver_config = DriverConfiguration().worker_threads(8);
  double epsilon = 0.8;
  node_id_t num_vertices = 1 << 13;
  {
    DynamicErdosGenerator gen(get_seed(), num_vertices, 0.5, 0.25, 0.25, 2);
    gen.to_binary_file("./dynamic_stream.bin");
  }
  
  size_t true_mc = num_vertices / 2; // probably about half of complete graph min-cut

  BinaryFileStream stream{"./dynamic_stream.bin"};

  MinCutSketchAlg mc_alg(num_vertices, get_seed(), MCAlgConfiguration().epsilon(0.8));

  GraphSketchDriver<MinCutSketchAlg> driver(&mc_alg, &stream, driver_config);
  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query(MINIMUMCUT);

  MinCut ret = mc_alg.calc_minimum_cut();

  ASSERT_GT(true_mc, ret.value);
  ASSERT_LT(true_mc * (1 - epsilon), ret.value);
  std::remove("./dynamic_stream.bin");
}

