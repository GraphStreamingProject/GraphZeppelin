#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/graph_verifier.h"
#include "util/graph_gen.h"

#include <boost/graph/stoer_wagner_min_cut.hpp>

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

TEST(GraphTestSuite, TestKConnectivityOnRandomGraphs)
{
  using namespace boost;
  typedef adjacency_list<vecS, vecS, undirectedS> UGraph; 

  int num_trials = 10;
  while (num_trials--) {
    generate_stream({100,0.03,0.5,0,"./sample.txt","./cum_sample.txt"});
    ifstream in{"./sample.txt"};
    Node n, m;
    in >> n >> m;
    
    Graph g{n};
    UGraph bg{n};

    int type, a, b;
    while (m--) 
    {
      in >> type >> a >> b;
      if (type == INSERT) 
      {
        g.update({{a, b}, INSERT});
        add_edge(a, b, bg)
      } 
      else 
      {
        g.update({{a, b}, DELETE});
	remove_edge(a, b, bg);
      }
    }
   
    auto min_cut_size = stoer_wagner_min_cut(bg, 
	make_static_property_map<UGraph::edge_descriptor>(1));

    EXPECT_TRUE(is_k_edge_connected(min_cut_size));
    EXPECT_FALSE(is_k_edge_connected(min_cut_size + 1));

    // TODO: Ensure any k \in [min_cut_size] returns true, and
    // any k > min_cut_size returns false. 
  }

}
