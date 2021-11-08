#include <gtest/gtest.h>
#include <fstream>
#include "../../include/supernode.h"
#include "../../include/graph.h"

TEST(Benchmark, BCHMKpostProcOnPaperclipGraph) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::ifstream in{curr_dir + "/../res/paperclip.stream"};
//  std::ifstream in{"/home/experiment_inputs/streams/kron_13_unique_half_stream.txt"};
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
  std::cout << "Number of CCs:" << g.connected_components().size() << std::endl;
}

TEST(Benchmark, BCHMKpostProcGenOnKron) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  std::ifstream in{curr_dir + "/../res/paperclip.stream"};
//  std::ifstream in{"/home/experiment_inputs/streams/kron_13_unique_half_stream.txt"};
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
  g.write_binary("./kron_13_graph.dmp");
  auto cc = g.connected_components();
  std::ofstream out {"./kron_13_res.txt"};
  out << cc.size() << '\n';
  for (const auto& component : cc) {
    out << component.size() << '\n';
    for (auto node : component) {
      out << node << " ";
    }
    out << '\n';
  }
  out.close();
}

TEST(Benchmark, BCHMKpostProcFromReheat) {
  Graph g { "./kron_13_graph.dmp" };
  auto start_time = std::chrono::steady_clock::now();
  auto cc = g.connected_components();
  std::cout << "Reheated cc took " << std::chrono::duration<long double>(
        std::chrono::steady_clock::now() - start_time).count() << " seconds" <<
        std::endl;

  // assert correctness
  std::ifstream in { "./kron_13_res.txt"};

  unsigned sz; in >> sz;
  ASSERT_EQ(cc.size(), sz);
}
