#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/deterministic.h"
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
  unsigned long long int num_nodes = 1000;
  Graph g{num_nodes};
  for (unsigned i=1;i<num_nodes;++i) {
    for (unsigned j = i*2;j<num_nodes;j+=i) {
      g.update({{i,j}, INSERT});
    }
  }
  g.connected_components();
  ASSERT_THROW(g.update({{1,2}, INSERT}), UpdateLockedException);
  ASSERT_THROW(g.update({{1,2}, DELETE}), UpdateLockedException);
}

TEST(GraphTestSuite, TestRandomGraphGeneration) {
  generate_stream();
  GraphVerifier graphVerifier {"./cum_sample.txt"};
}

TEST(GraphTestSuite, TestCorrectnessOnSmallRandomGraphs) {
  int num_trials = 10;
  while (num_trials--) {
    generate_stream();
    ifstream in{"./sample.txt"};
    unsigned n, m;
    in >> n >> m >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.connected_components();
  }
}

TEST(GraphTestSuite, TestCorrectnessOnSmallSparseGraphs) {
  int num_trials = 10;
  while (num_trials--) {
    generate_stream({1024,0.0,4096,1,3,"./sample.txt","./cum_sample.txt"});
    ifstream in{"./sample.txt"};
    unsigned n, m;
    in >> n >> m >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.connected_components();
  }
}
