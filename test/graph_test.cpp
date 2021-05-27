#include <gtest/gtest.h>
#include <fstream>
#include <unordered_map>
#include "../include/graph.h"
#include "util/graph_gen.h"

static void assert_correct_cc(const std::vector<std::pair<std::vector<Node>,
    std::vector<Edge>>> ccs, std::string in_file, std::string cum_in_file) {
  std::clog << "Found " << ccs.size() << " connected components" << std::endl;
  // Ensure all returned edges are actually in the stream
  std::ifstream ifs_in(in_file);
  Node in_n, in_m;
  ifs_in >> in_n >> in_m;
  std::unordered_map<vec_t, bool> edge_in;
  edge_in.reserve(ccs.size());
  for (const auto& cc : ccs) {
    for (const Edge& edge : cc.second) {
      edge_in[nondirectional_non_self_edge_pairing_fn(edge.first, edge.second)] = false;
    }
  }
  for (Node i = 0; i < in_m; i++) {
    int type;
    Node a, b;
    ifs_in >> type >> a >> b;
    auto it = edge_in.find(nondirectional_non_self_edge_pairing_fn(a, b));
    if (it != edge_in.end()) {
      if ((type == DELETE) != it->second) {
        std::cerr << "Warning: bad insertion/deletion in stream" << std::endl;
      }
      it->second = !it->second;
    }
  }
  bool all_edges_in = true;
  for (const auto& edge : edge_in) {
    if (!edge.second) {
      all_edges_in = false;
      break;
    }
  }
  ifs_in.close();
  EXPECT_TRUE(all_edges_in);
  // Sanity check, do the same for cumulative input
  // Also run dsu on cumulative input
  std::ifstream ifs_cum_in(cum_in_file);
  Node cum_in_n, cum_in_m;
  ifs_cum_in >> cum_in_n >> cum_in_m;
  std::unordered_map<vec_t, bool> edge_cum_in;
  edge_cum_in.reserve(ccs.size());
  DSU dsu(cum_in_n);
  for (const auto& cc : ccs) {
    for (const Edge& edge : cc.second) {
      edge_cum_in[nondirectional_non_self_edge_pairing_fn(edge.first, edge.second)] = false;
    }
  }
  for (Node i = 0; i < cum_in_m; i++) {
    Node a, b;
    ifs_cum_in >> a >> b;
    dsu.merge(a, b);
    auto it = edge_cum_in.find(nondirectional_non_self_edge_pairing_fn(a, b));
    if (it != edge_cum_in.end()) {
      if (it->second) {
        std::cerr << "Warning: duplicate edge in cumulative stream" << std::endl;
      }
      it->second = true;
    }
  }
  ifs_cum_in.close();
  bool all_edges_cum_in= true;
  for (const auto& edge : edge_cum_in) {
    if (!edge.second) {
      all_edges_cum_in= false;
      break;
    }
  }
  EXPECT_TRUE(all_edges_cum_in);
  // Extract connected components (without edges) from output
  std::vector<std::vector<Node>> ccs_nodes;
  ccs_nodes.reserve(ccs.size());
  for (const auto& cc : ccs) {
    ccs_nodes.push_back(cc.first);
  }
  // Run an actual cc algorithm and compare components
  std::vector<std::vector<Node>> ccs_nodes_ref;
  std::unordered_map<Node, Node> nodemap;
  nodemap.reserve(cum_in_n);
  Node num_comp = 0;
  for (Node i = 0; i < cum_in_n; i++) {
    Node rep = dsu.find(i);
    if (nodemap.find(rep) == nodemap.end()) {
      nodemap[rep] = num_comp++;
      ccs_nodes_ref.emplace_back(std::vector<Node>());
    }
    ccs_nodes_ref[nodemap[rep]].push_back(i);
  }
  // Check equality (sort to establish an order, then element wise compare)
  std::sort(ccs_nodes.begin(), ccs_nodes.end());
  std::sort(ccs_nodes_ref.begin(), ccs_nodes_ref.end());
  EXPECT_EQ(ccs_nodes, ccs_nodes_ref);
}

TEST(GraphTestSuite, SmallGraphConnectivity) {
  srand(time(NULL));
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::string in_file = curr_dir + "/res/multiples_graph_1024.txt";
  ifstream in{in_file};
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
  auto ccs = g.connected_components();
  assert_correct_cc(g.connected_components(), in_file, in_file);
}

TEST(GraphTestSuite, IFconnectedComponentsAlgRunTHENupdateLocked) {
  srand(time(NULL));
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
  g.connected_components();
  ASSERT_THROW(g.update({{1,2}, INSERT}), UpdateLockedException);
  ASSERT_THROW(g.update({{1,2}, DELETE}), UpdateLockedException);
}

TEST(GraphTestSuite, TestRandomGraphGeneration) {
  generate_stream();
}

TEST(GraphTestSuite, TestCorrectnessOnSmallRandomGraphs) {
  srand(time(NULL));
  int num_trials = 10;
  while (num_trials--) {
    generate_stream();
    std::ifstream in{"./sample.txt"};
    Node n, m;
    in >> n >> m;
    std::streampos sp_edge_begin = in.tellg();
    Graph g{n};
    int type, a, b;
    for (Node i = 0; i < m; i++) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else {
        g.update({{a, b}, DELETE});
      }
    }
    in.close();
    assert_correct_cc(g.connected_components(), "./sample.txt", "./cum_sample.txt");
  }
}

TEST(GraphTestSuite, TestCorrectnessOnSmallSparseGraphs) {
  srand(time(NULL));
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
      } else {
        g.update({{a, b}, DELETE});
      }
    }
    in.close();
    assert_correct_cc(g.connected_components(), "./sample.txt", "./cum_sample.txt");
  }
}

TEST(GraphTestSuite, DISABLED_LargeGraphTest) {
  srand(time(NULL));
  int num_trials = 1;
  while (num_trials--) {
    std::cout << "Generating stream" << std::endl;
    generate_stream({30000, 0.000001, 0.5, 0, "./sample.txt", "./cum_sample.txt"});
    std::cout << "Ingesting stream" << std::endl;
    ifstream in{"./sample.txt"};
    Node n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else {
        g.update({{a, b}, DELETE});
      }
    }
    in.close();
    std::cout << "Running ccs" << std::endl;
    auto ccs = g.connected_components();
    std::cout << "Verifying ccs" << std::endl;
    assert_correct_cc(ccs, "./sample.txt", "./cum_sample.txt");
  }
}
