#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/graph_gen.h"

/**
 * For many of these tests (especially for those upon very sparse and small graphs)
 * we allow for a certain number of failures per test.
 * This is because the responsibility of these tests is to quickly alert us 
 * to “this code is very wrong” whereas the statistical testing is responsible 
 * for a more fine grained analysis.
 * In this context a false positive is much worse than a false negative.
 * With 2 failures allowed per test our entire testing suite should fail 1/5000 runs.
 */

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
  int allow_fail = 2; // allow 2 failures
  int fails = 0;
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
    try {
      g.connected_components();
    } catch (OutOfQueriesException& err) {
      fails++;
      if (fails > allow_fail) {
        printf("More than %i failures failing test\n", allow_fail);
        throw;
      }
    }
  }
}

TEST(ParallelGraphTestSuite, SingleTestCorrectnessOnSmallSparseGraphs) {
  int num_trials = 10;
  int allow_fail = 2; // allow 2 failures
  int fails = 0;
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
    try {
      g.connected_components();
    } catch (OutOfQueriesException& err) {
      fails++;
      if (fails > allow_fail) {
        printf("More than %i failures failing test\n", allow_fail);
        throw;
      }
    }
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
  int allow_fail = 2; // allow 2 failures
  int fails = 0;
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
    try {
      g.parallel_connected_components();
    } catch (OutOfQueriesException& err) {
      fails++;
      if (fails > allow_fail) {
        printf("More than %i failures failing test\n", allow_fail);
        throw;
      }
    }
  }
}

TEST(ParallelGraphTestSuite, ParallelTestCorrectnessOnSmallSparseGraphs) {
  int num_trials = 10;
  int allow_fail = 2; // allow 2 failures
  int fails = 0;
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
    try {
      g.parallel_connected_components();
    } catch (OutOfQueriesException& err) {
      fails++;
      if (fails > allow_fail) {
        printf("More than %i failures failing test\n", allow_fail);
        throw;
      }
    }
  }
}
