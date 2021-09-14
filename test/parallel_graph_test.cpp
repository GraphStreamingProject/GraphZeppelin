#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/graph_gen.h"

TEST(ParallelGraphTestSuite, SingleSmallGraphConnectivity) {
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
  g.set_cumul_in(curr_dir + "/res/multiples_graph_1024.txt");
  ASSERT_EQ(78, g.connected_components().size());
}

TEST(ParallelGraphTestSuite, SingleTestCorrectnessOnSmallRandomGraphs) {
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
    g.set_cumul_in("./cumul_sample.txt");
    g.connected_components();
  }
}

TEST(ParallelGraphTestSuite, SingleTestCorrectnessOnSmallSparseGraphs) {
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
    g.set_cumul_in("./cumul_sample.txt");
    g.connected_components();
  }
}

TEST(ParallelGraphTestSuite, ParallelSmallGraphConnectivity) {
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
  g.set_cumul_in(curr_dir + "/res/multiples_graph_1024.txt");
  ASSERT_EQ(78, g.parallel_connected_components().size());
}

TEST(ParallelGraphTestSuite, ParallelTestCorrectnessOnSmallRandomGraphs) {
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
    g.set_cumul_in("./cumul_sample.txt");
    g.parallel_connected_components();
  }
}

TEST(ParallelGraphTestSuite, ParallelTestCorrectnessOnSmallSparseGraphs) {
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
    g.set_cumul_in("./cumul_sample.txt");
    g.parallel_connected_components();
  }
}
