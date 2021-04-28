#include <gtest/gtest.h>
#include <fstream>
#include "../include/graph.h"
#include "util/graph_verifier.h"
#include "util/graph_gen.h"

#include <unordered_set>
#include <stack>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/connected_components.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UGraph; 

Graph ingest_stream (std::string filename)
{
  ifstream in{filename};
  Node n, m;
  in >> n >> m;
  
  Graph g{n};

  int type, a, b;
  while (m--) 
  {
    in >> type >> a >> b;
    g.update({{a, b}, (UpdateType)type});
  }

 return g;
}

UGraph ingest_cum_graph (std::string filename)
{
  ifstream in{filename};
  Node n, m;
  in >> n >> m;
  
  UGraph bg{n};

  int a, b;
  while (m--) 
  {
    in >> a >> b;
    boost::add_edge(a, b, bg);
  }
 
  return bg;
}

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

TEST(GraphTestSuite, TestSpanningForestOnRandomGraph)
{
  using namespace boost;
  int num_trials = 10;

  while (num_trials--)
  {
    generate_stream(
	{1024,0.002,0.5,0,"./sample.txt","./cum_sample.txt"});

    UGraph bg = ingest_cum_graph("./cum_sample.txt");
    auto boost_num_vertices = num_vertices(bg);
    vector<int> boost_comp_map(boost_num_vertices);
    int boost_num_comp = 
	    connected_components(bg, &boost_comp_map[0]);
    // Store the cardinality of each boost connected component
    vector<int> boost_comp_sizes(boost_num_comp);
    for (Node i = 0; i < boost_num_vertices; i++)
	    boost_comp_sizes[boost_comp_map[i]]++;

    std::cout << "Verifying spanning forest on graph with " << boost_num_comp << " CC(s)." << std::endl; 

    Graph g = ingest_stream("./sample.txt");
    auto F = g.spanning_forest();

    for (const auto& adj_list : F)
    {
      Node first_node = adj_list.begin()->first;
      
      // Identity of the corresponding boost connected component
      int boost_CC_id = boost_comp_map[first_node];

      // DFS on current component represented by adj_list
      std::unordered_set<Node> visited;
      visited.reserve(adj_list.size()); 
      std::stack<Node> branch_points;    
      branch_points.push(first_node);
      while (!branch_points.empty())
      {
        Node cur = branch_points.top();
	branch_points.pop();

	// Verify node is present in corresponding boost CC
	ASSERT_EQ(boost_CC_id, boost_comp_map[cur])
		<< "Node in wrong component\n";

	visited.insert(cur);

	int visited_neighbor_count = 0;
	for (const Node& neighbor : adj_list.at(cur))
	{
	  if (visited.count(neighbor) > 0)
	  {
	    visited_neighbor_count++;

	    // If we've already visited more than one neighbor of
	    // the current node, then we have found a cycle
	    if (visited_neighbor_count > 1)
	      FAIL() << "Cycle detected\n";

	    continue;  
	  }

	  // Verify that spanning forest is subgraph
	  bool edge_exists = edge(cur, neighbor, bg).second;
	  ASSERT_TRUE(edge_exists) 
		  << "Edge does not exist in supergraph\n";
	  
	  branch_points.push(neighbor);
	}
      }

      // Ensure cardinality of boost_CC and visited vertex set
      // are equal (i.e. that the forest is spanning on this CC)
      ASSERT_EQ(visited.size(), boost_comp_sizes[boost_CC_id]) 
	      << "Incorrect component size \n";
    }  
  }
}

//TEST(GraphTestSuite, TestKConnectivityOnRandomGraphs)
//{
//  using namespace boost;
//
//  int num_trials = 10;
//  while (num_trials--) {
//    generate_stream(
//	{100,0.5,0.5,0,"./sample.txt","./cum_sample.txt"});
//    
//    Graph g = ingest_stream("./sample.txt");
//    UGraph bg = ingest_cum_graph(".cum_sample.txt");
//
//    auto min_cut_size = stoer_wagner_min_cut(bg, 
//	make_static_property_map<UGraph::edge_descriptor>(1));
//
//    std::cout << "Testing " << min_cut_size << "-edge-connectivity on a graph with min edge cut of size " << min_cut_size;
//
//    EXPECT_TRUE(g.is_k_edge_connected(min_cut_size));
////    EXPECT_FALSE(g.is_k_edge_connected(min_cut_size + 1));
//
//    // TODO: Ensure any k \in [min_cut_size] returns true, and
//    // any k > min_cut_size returns false. 
//  }
//}
