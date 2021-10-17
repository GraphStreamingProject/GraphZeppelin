#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/file_graph_verifier.h"
#include "util/mat_graph_verifier.h"
#include "util/graph_gen.h"

TEST(GraphTestSuite, SmallGraphConnectivity) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
  node_t num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  node_t a, b;
  Graph g{num_nodes};
  while (m--) {
    in >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.set_verifier(std::make_unique<FileGraphVerifier>(curr_dir + "/res/multiples_graph_1024.txt"));
  ASSERT_EQ(78, g.connected_components().size());
}

TEST(GraphTestSuite, IFconnectedComponentsAlgRunTHENupdateLocked) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
  node_t num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  node_t a, b;
  Graph g{num_nodes};
  while (m--) {
    in >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.set_verifier(std::make_unique<FileGraphVerifier>(curr_dir + "/res/multiples_graph_1024.txt"));
  g.connected_components();
  ASSERT_THROW(g.update({{1,2}, INSERT}), UpdateLockedException);
  ASSERT_THROW(g.update({{1,2}, DELETE}), UpdateLockedException);
}

TEST(GraphTestSuite, TestRandomGraphGeneration) {
  generate_stream();
  ifstream in { "./cum_sample.txt" };
  node_t n, m;
  in >> n >> m;
  std::vector<bool> cum_in;
  cum_in.reserve(n*(n-1)/2);
  node_t a, b;
  while (m--) {
    in >> a >> b;
    node_t idx = nondirectional_non_self_edge_pairing_fn(a,b);
    cum_in[idx] = !cum_in[idx];
  }
  MatGraphVerifier graphVerifier {n, cum_in};
}

TEST(GraphTestSuite, TestCorrectnessOnSmallRandomGraphs) {
  int num_trials = 10;
  while (num_trials--) {
    generate_stream();
    ifstream in{"./sample.txt"};
    node_t n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    g.connected_components();
  }
}

TEST(GraphTestSuite, TestCorrectnessOnSmallSparseGraphs) {
  int num_trials = 10;
  while (num_trials--) {
    generate_stream({1024,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
    ifstream in{"./sample.txt"};
    node_t n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    g.connected_components();
  }
}

TEST(GraphTestSuite, TestCorrectnessOfReheating) {
  int num_trials = 1;
  while (num_trials--) {
    generate_stream({1024,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
    ifstream in{"./sample.txt"};
    node_t n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.write_binary("./out_temp.txt");
    g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    auto g_res = g.connected_components();
    printf("number of CC = %lu\n", g_res.size());

    Graph reheated {"./out_temp.txt"};

    reheated.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
    auto reheated_res = reheated.connected_components();
    printf("number of reheated CC = %lu\n", reheated_res.size());
    ASSERT_EQ(g_res.size(), reheated_res.size());
    for (unsigned i = 0; i < g_res.size(); ++i) {
      std::vector<node_t> symdif;
      std::set_symmetric_difference(g_res[i].begin(), g_res[i].end(),
          reheated_res[i].begin(), reheated_res[i].end(),
          std::back_inserter(symdif));
      ASSERT_EQ(0, symdif.size());
    }
  }
}
