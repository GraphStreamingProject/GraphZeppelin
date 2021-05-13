#include <gtest/gtest.h>

#include <fstream>
#include "../include/graph.h"
#include "../include/k_skeleton.h"
#include "util/graph_gen.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/connected_components.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UGraph;

UGraph ingest_boost_graph (std::string filename)
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

TEST(KSkeletonTestSuite, MinCutSizeEdgeConnectedIffHasMinCutSizedSkeleton)
{
  using namespace boost;

  int n = 1000;
  int num_trials = 10;

  while (num_trials--) {
    generate_stream(
        {n,0.04,0.5,0,"./sample.txt","./cum_sample.txt"});

    UGraph bg = ingest_boost_graph("./cum_sample.txt");

    // Vertices that have the same parity after `stoer_wagner_min_cut` 
    // runs are on the same side of the min-cut.
    BOOST_AUTO(bg_parities, make_one_bit_color_map(
                            num_vertices(bg), get(vertex_index, bg)));

    auto G_min_cut_size = stoer_wagner_min_cut(bg,
        make_static_property_map<UGraph::edge_descriptor>(1),
        parity_map(bg_parities));

    std::cout << "Evaluating on graph with " << G_min_cut_size + 1
            << "-edge-connectivity..." << std::endl;

    ifstream * updates_stream = new ifstream("./sample.txt");
    auto U = k_skeleton(updates_stream, G_min_cut_size);
    delete updates_stream;

    UGraph bu(n);
    for (Node i = 0; i < n; i++)
    {
      for (Node neighbor : U[i])
      {
        if (i < neighbor)
        {
          ASSERT_FALSE(edge(i, neighbor, bu).second)
                  << "Forests are not edge-disjoint \n";

          add_edge(i, neighbor, bu);
        }
      }
    }

    // Vertices that have the same parity after `stoer_wagner_min_cut` 
    // runs are on the same side of the min-cut.
    BOOST_AUTO(bu_parities, make_one_bit_color_map(
                            num_vertices(bu), get(vertex_index, bu)));

    auto U_min_cut_size = stoer_wagner_min_cut(bu,
        make_static_property_map<UGraph::edge_descriptor>(1),
        parity_map(bu_parities));

    EXPECT_EQ(G_min_cut_size, U_min_cut_size);
  }
}

