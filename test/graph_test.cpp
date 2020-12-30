#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/deterministic.h"
#include "util/graph_gen.h"

TEST(GraphTestSuite, SmallGraphConnectivity) {
  unsigned long long int num_nodes = 1000;
  Graph g{num_nodes};
  for (unsigned i=1;i<num_nodes;++i) {
    for (unsigned j = i*2;j<num_nodes;j+=i) {
      g.update({{i,j}, INSERT});
    }
  }
  ASSERT_EQ(2, g.connected_components().size());
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
