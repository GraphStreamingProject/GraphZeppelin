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

#include <chrono>

TEST(GraphTestSuite, DISABLED_TestFailureRate) {
  for (int i = 10, n = i; i < 1e4; n = n > i ? (i *= 10) : n * sqrt(10)) {
    int num_trials = 1000;
    int num_failure = 0;
    long total_updates = 0;
    std::chrono::duration<long double> runtime(0);
    while (num_trials--) {
      try {
        generate_stream({n, .03, .5, 0, "./sample.txt", "./cum_sample.txt"});
        ifstream in{"./sample.txt"};
        Node n, m;
        in >> n >> m;
        total_updates += m;
        Graph g{n};
        int type, a, b;
        auto starttime = std::chrono::steady_clock::now();
        while (m--) {
          in >> type >> a >> b;
          if (type == INSERT) {
            g.update({{a, b}, INSERT});
          } else g.update({{a, b}, DELETE});
        }
        runtime += std::chrono::duration<long double>(std::chrono::steady_clock::now() - starttime);
        g.set_cum_in("./cum_sample.txt");
        g.connected_components();
      } catch (const NoGoodBucketException& e) {
        num_failure++;
      } catch (const NotCCException& e) {
        num_failure++;
      } catch (const BadEdgeException& e) {
        num_failure++;
      }
    }
    std::clog << n <<  ',' << total_updates << ':' << num_failure << ',' << runtime.count() << std::endl;
  }
}

void test_continuous(unsigned nodes, unsigned long updates_per_sample, unsigned long samples) {
  srand(time(NULL));
  Graph g(nodes);
  std::vector<std::vector<bool>> adj(nodes, std::vector<bool>(nodes));
  unsigned long num_failure = 0;
  for (unsigned long i = 0; i < samples; i++) {
    for (unsigned long j = 0; j < updates_per_sample; j++) {
      unsigned edgei = rand() % nodes;
      unsigned edgej = rand() % (nodes - 1);
      if (edgei > edgej) {
        std::swap(edgei, edgej);
      } else {
        edgej++;
      }
      g.update({{edgei, edgej}, INSERT});
      adj[edgei][edgej] = !adj[edgei][edgej];
    }
    try {
      Graph g_cc = g;
      vec_t num_edges = 0;
      for (unsigned i = 0; i < nodes; i++) {
        for (unsigned j = i + 1; j < nodes; j++) {
          if (adj[i][j]) {
            num_edges++;
          }
        }
      }
      std::ofstream out("./cum_sample.txt");
      out << nodes << " " << num_edges << std::endl;
      for (unsigned i = 0; i < nodes; i++) {
        for (unsigned j = i + 1; j < nodes; j++) {
          if (adj[i][j]) {
            out << i << " " << j << std::endl;
          }
        }
      }
      out.close();
      g.set_cum_in("./cum_sample.txt");
      g_cc.connected_components();
    } catch (const NoGoodBucketException& e) {
      num_failure++;
    } catch (const NotCCException& e) {
      num_failure++;
    } catch (const BadEdgeException& e) {
      num_failure++;
    }
  }
  std::clog << nodes << ',' << num_failure << std::endl;
}

TEST(GraphTestSuite, DISABLED_TestContinuous) {
  test_continuous(10, 100, 100);
}
