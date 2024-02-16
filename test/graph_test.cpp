#include <ascii_file_stream.h>
#include <binary_file_stream.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>

#include <KEdgeConnect.h>
#include "cc_sketch_alg.h"
#include "file_graph_verifier.h"
#include "graph_gen.h"
#include "graph_sketch_driver.h"
#include "mat_graph_verifier.h"

/**
 * For many of these tests (especially for those upon very sparse and small graphs)
 * we allow for a certain number of failures per test.
 * This is because the responsibility of these tests is to quickly alert us
 * to “this code is very wrong” whereas the statistical testing is responsible
 * for a more fine grained analysis.
 * In this context a false positive is much worse than a false negative.
 * With 2 failures allowed per test our entire testing suite should fail 1/5000 runs.
 */

// We create this class and instantiate a paramaterized test suite so that we
// can run these tests both with the GutterTree and with StandAloneGutters
class GraphTest : public testing::TestWithParam<GutterSystem> {};
INSTANTIATE_TEST_SUITE_P(GraphTestSuite, GraphTest,
                         testing::Values(GUTTERTREE, STANDALONE, CACHETREE));

/*************************************
 * TODO: These are not finished -- come back to fix this later
TEST_P(GraphTest, SmallGraphForestDeletion) {
    auto driver_config = DriverConfiguration().gutter_sys(GetParam());
    const std::string fname = __FILE__;
    size_t pos = fname.find_last_of("\\/");
    const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
    AsciiFileStream stream{curr_dir + "/res/multiples_graph_1024.txt", false};
    node_id_t num_nodes = stream.vertices();

    CCSketchAlg cc_alg{num_nodes};
    cc_alg.set_verifier(
            std::make_unique<FileGraphVerifier>(1024, curr_dir + "/res/multiples_graph_1024.txt"));

    GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();
    ASSERT_EQ(78, cc_alg.connected_components().size());
}
***********************************************/

TEST_P(GraphTest, SmallGraphConnectivity) {
  auto driver_config = DriverConfiguration().gutter_sys(GetParam());
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  AsciiFileStream stream{curr_dir + "/res/multiples_graph_1024.txt", false};
  node_id_t num_nodes = stream.vertices();

  CCSketchAlg cc_alg{num_nodes};
  cc_alg.set_verifier(
      std::make_unique<FileGraphVerifier>(1024, curr_dir + "/res/multiples_graph_1024.txt"));

  GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query();
  ASSERT_EQ(78, cc_alg.connected_components().size());
}

TEST_P(GraphTest, TestCorrectnessOnSmallRandomGraphs) {
  auto driver_config = DriverConfiguration().gutter_sys(GetParam());
  int num_trials = 5;
  while (num_trials--) {
    generate_stream();
    AsciiFileStream stream{"./sample.txt"};
    node_id_t num_nodes = stream.vertices();

    CCSketchAlg cc_alg{num_nodes};
    cc_alg.set_verifier(std::make_unique<FileGraphVerifier>(1024, "./cumul_sample.txt"));

    GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();

    cc_alg.connected_components();
  }
}

TEST_P(GraphTest, TestCorrectnessOnSmallSparseGraphs) {
  auto driver_config = DriverConfiguration().gutter_sys(GetParam());
  int num_trials = 5;
  while (num_trials--) {
    generate_stream({1024, 0.002, 0.5, 0, "./sample.txt", "./cumul_sample.txt"});
    AsciiFileStream stream{"./sample.txt"};
    node_id_t num_nodes = stream.vertices();

    CCSketchAlg cc_alg{num_nodes};
    cc_alg.set_verifier(std::make_unique<FileGraphVerifier>(1024, "./cumul_sample.txt"));

    GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();

    cc_alg.connected_components();
  }
}

TEST_P(GraphTest, TestCorrectnessOfReheating) {
  auto driver_config = DriverConfiguration().gutter_sys(GetParam());
  int num_trials = 5;
  while (num_trials--) {
    generate_stream({1024, 0.002, 0.5, 0, "./sample.txt", "./cumul_sample.txt"});

    AsciiFileStream stream{"./sample.txt"};
    node_id_t num_nodes = stream.vertices();

    CCSketchAlg cc_alg{num_nodes};
    cc_alg.set_verifier(std::make_unique<FileGraphVerifier>(1024, "./cumul_sample.txt"));

    GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();

    cc_alg.write_binary("./out_temp.txt");
    std::vector<std::set<node_id_t>> orig_cc;
    orig_cc = cc_alg.connected_components();
    printf("number of CC = %lu\n", orig_cc.size());

    CCSketchAlg *reheat_alg = CCSketchAlg::construct_from_serialized_data("./out_temp.txt");
    reheat_alg->set_verifier(std::make_unique<FileGraphVerifier>(1024, "./cumul_sample.txt"));
    auto reheat_cc = reheat_alg->connected_components();
    printf("number of reheated CC = %lu\n", reheat_cc.size());
    ASSERT_EQ(orig_cc.size(), reheat_cc.size());
    delete reheat_alg;
  }
}

// Test the multithreaded system by using multiple worker threads
TEST_P(GraphTest, MultipleWorkers) {
  auto driver_config = DriverConfiguration().gutter_sys(GetParam()).worker_threads(8);
  int num_trials = 5;
  while (num_trials--) {
    generate_stream({1024, 0.002, 0.5, 0, "./sample.txt", "./cumul_sample.txt"});
    AsciiFileStream stream{"./sample.txt"};
    node_id_t num_nodes = stream.vertices();

    CCSketchAlg cc_alg{num_nodes};
    cc_alg.set_verifier(std::make_unique<FileGraphVerifier>(1024, "./cumul_sample.txt"));

    GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();
    cc_alg.connected_components();
  }
}

TEST_P(GraphTest, TestPointQuery) {
  auto driver_config = DriverConfiguration().gutter_sys(GetParam());
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  AsciiFileStream stream{curr_dir + "/res/multiples_graph_1024.txt", false};
  node_id_t num_nodes = stream.vertices();

  CCSketchAlg cc_alg{num_nodes};
  cc_alg.set_verifier(
      std::make_unique<FileGraphVerifier>(1024, curr_dir + "/res/multiples_graph_1024.txt"));

  GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query();

  std::vector<std::set<node_id_t>> ret = cc_alg.connected_components();
  std::vector<node_id_t> ccid(num_nodes);
  for (node_id_t i = 0; i < ret.size(); ++i) {
    for (const node_id_t node : ret[i]) {
      ccid[node] = i;
    }
  }
  for (node_id_t i = 0; i < std::min(10u, num_nodes); ++i) {
    for (node_id_t j = 0; j < std::min(10u, num_nodes); ++j) {
      cc_alg.set_verifier(
          std::make_unique<FileGraphVerifier>(1024, curr_dir + "/res/multiples_graph_1024.txt"));
      ASSERT_EQ(cc_alg.point_query(i, j), ccid[i] == ccid[j]);
    }
  }
}

TEST(GraphTest, TestQueryDuringStream) {
  auto driver_config = DriverConfiguration().gutter_sys(STANDALONE);
  auto cc_config = CCAlgConfiguration();
  generate_stream({1024, 0.002, 0.5, 0, "./sample.txt", "./cumul_sample.txt"});
  std::ifstream in{"./sample.txt"};
  AsciiFileStream stream{"./sample.txt"};
  node_id_t num_nodes = stream.vertices();
  edge_id_t num_edges = stream.edges();
  edge_id_t tenth     = num_edges / 10;

  CCSketchAlg cc_alg{num_nodes, cc_config};
  GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config);
  MatGraphVerifier verify(num_nodes);


  int type;
  node_id_t a, b;

  // read header from verify stream
  in >> a >> b;

  for (int j = 0; j < 9; j++) {
    for (edge_id_t i = 0; i < tenth; i++) {
      in >> type >> a >> b;
      verify.edge_update(a, b);
    }
    verify.reset_cc_state();

    driver.process_stream_until(tenth * (j+1));
    driver.prep_query();
    cc_alg.set_verifier(std::make_unique<MatGraphVerifier>(verify));
    cc_alg.connected_components();
  }
  num_edges -= 9 * tenth;
  while (num_edges--) {
    in >> type >> a >> b;
    verify.edge_update(a, b);
  }
  verify.reset_cc_state();

  driver.process_stream_until(END_OF_STREAM);
  driver.prep_query();
  cc_alg.set_verifier(std::make_unique<MatGraphVerifier>(verify));
  cc_alg.connected_components();
}

TEST(GraphTest, EagerDSUTest) {
  node_id_t num_nodes = 100;
  CCSketchAlg cc_alg{num_nodes};
  MatGraphVerifier verify(num_nodes);

  // This should be a spanning forest edge
  cc_alg.update({{1, 2}, INSERT});
  verify.edge_update(1, 2);
  verify.reset_cc_state();
  cc_alg.set_verifier(std::make_unique<decltype(verify)>(verify));
  cc_alg.connected_components();

  // This should be a spanning forest edge
  cc_alg.update({{2, 3}, INSERT});
  verify.edge_update(2, 3);
  verify.reset_cc_state();
  cc_alg.set_verifier(std::make_unique<decltype(verify)>(verify));
  cc_alg.connected_components();

  // This should be an edge within a component
  cc_alg.update({{1, 3}, INSERT});
  verify.edge_update(1, 3);
  verify.reset_cc_state();
  cc_alg.set_verifier(std::make_unique<decltype(verify)>(verify));
  cc_alg.connected_components();

  // This should delete an edge within a component
  cc_alg.update({{1, 3}, DELETE});
  verify.edge_update(1, 3);
  verify.reset_cc_state();
  cc_alg.set_verifier(std::make_unique<decltype(verify)>(verify));
  cc_alg.connected_components();

  // This one should delete a spanning forest edge and cause a rebuild
  cc_alg.update({{2, 3}, DELETE});
  verify.edge_update(2, 3);
  verify.reset_cc_state();
  cc_alg.set_verifier(std::make_unique<decltype(verify)>(verify));
  cc_alg.connected_components();

  // This one should be a normal edge
  cc_alg.update({{2, 3}, INSERT});
  verify.edge_update(2, 3);
  verify.reset_cc_state();
  cc_alg.set_verifier(std::make_unique<decltype(verify)>(verify));
  cc_alg.connected_components();
}

TEST(GraphTest, MTStreamWithMultipleQueries) {
  for (int t = 1; t <= 3; t++) {
    auto driver_config = DriverConfiguration().gutter_sys(STANDALONE);

    const std::string fname = __FILE__;
    size_t pos = fname.find_last_of("\\/");
    const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
    BinaryFileStream stream{curr_dir + "/res/multiples_graph_1024_stream.data"};
    BinaryFileStream verify_stream{curr_dir + "/res/multiples_graph_1024_stream.data"};

    node_id_t num_nodes = stream.vertices();
    edge_id_t num_edges = stream.edges();

    std::cerr << num_nodes << " " << num_edges << std::endl;

    CCSketchAlg cc_alg{num_nodes};
    GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, driver_config, 4);
    MatGraphVerifier verify(num_nodes);

    size_t num_queries = 10;
    size_t upd_per_query = num_edges / num_queries;
    for (size_t i = 0; i < num_queries-1; i++) {
      for (size_t j = 0; j < upd_per_query; j++) {
        GraphStreamUpdate upd;
        verify_stream.get_update_buffer(&upd, 1);
        verify.edge_update(upd.edge.src, upd.edge.dst);
        ASSERT_NE(upd.type, BREAKPOINT);
      }
      verify.reset_cc_state();
      cc_alg.set_verifier(std::make_unique<MatGraphVerifier>(verify));

      driver.process_stream_until(upd_per_query * (i+1));
      driver.prep_query();
      cc_alg.connected_components();
    }

    num_edges -= 9 * upd_per_query;
    while (num_edges--) {
      GraphStreamUpdate upd;
      verify_stream.get_update_buffer(&upd, 1);
      verify.edge_update(upd.edge.src, upd.edge.dst);
      ASSERT_NE(upd.type, BREAKPOINT);
    }
    verify.reset_cc_state();
    cc_alg.set_verifier(std::make_unique<MatGraphVerifier>(verify));

    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();
    cc_alg.connected_components();
  }
}
