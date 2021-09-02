#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/graph_verifier.h"
#include "util/graph_gen.h"

TEST(GraphTestSuite, SmallGraphConnectivity) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  Node a, b;
  Graph g{num_nodes};
  while (m--) {
    in >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.set_cum_in(curr_dir + "/res/multiples_graph_1024.txt");
  ASSERT_EQ(78, g.connected_components().size());
}

TEST(GraphTestSuite, IFconnectedComponentsAlgRunTHENupdateLocked) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/res/multiples_graph_1024.txt"};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  Node a, b;
  Graph g{num_nodes};
  while (m--) {
    in >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.set_cum_in(curr_dir + "/res/multiples_graph_1024.txt");
  g.connected_components();
  ASSERT_THROW(g.update({{1,2}, INSERT}), UpdateLockedException);
  ASSERT_THROW(g.update({{1,2}, DELETE}), UpdateLockedException);
}

TEST(GraphTestSuite, TestRandomGraphGeneration) {
  generate_stream();
  GraphVerifier graphVerifier {};
}

TEST(GraphTestSuite, TestCorrectnessOnSmallRandomGraphs) {
  int num_trials = 10;
  while (num_trials--) {
    generate_stream();
    ifstream in{"./sample.txt"};
    Node n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.set_cum_in("./cum_sample.txt");
    g.connected_components();
  }
}

TEST(GraphTestSuite, TestCorrectnessOnSmallSparseGraphs) {
  int num_trials = 10;
  while (num_trials--) {
    generate_stream({1024,0.002,0.5,0,"./sample.txt","./cum_sample.txt"});
    ifstream in{"./sample.txt"};
    Node n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.set_cum_in("./cum_sample.txt");
    g.connected_components();
  }
}

TEST(GraphTestSuite, TestCorrectnessOfReheating) {
  int num_trials = 10;
  while (num_trials--) {
    generate_stream({1024,0.002,0.5,0,"./sample.txt","./cum_sample.txt"});
    ifstream in{"./sample.txt"};
    Node n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.set_cum_in("./cum_sample.txt");
    g.write_binary("./out_temp.txt");
    auto g_res = g.connected_components();

    Graph reheated {"./out_temp.txt"};
    auto reheated_res = reheated.connected_components();
    ASSERT_EQ(g_res.size(), reheated_res.size());
    for (unsigned i = 0; i < g_res.size(); ++i) {
      std::vector<Node> symdif;
      std::set_symmetric_difference(g_res[i].begin(), g_res[i].end(),
          reheated_res[i].begin(), reheated_res[i].end(),
          std::back_inserter(symdif));
      ASSERT_EQ(0, symdif.size());
    }
  }
}
