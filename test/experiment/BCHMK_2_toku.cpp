#include <gtest/gtest.h>
#include <fstream>
#include "../../include/graph.h"
#include "../util/graph_verifier.h"
#include "../util/graph_gen.h"

TEST(Benchmark, BCHMK2Toku) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/../res/1000_0.95_0.5.stream"};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  Node a, b;
  uint8_t u;
  Graph g{num_nodes};
  while (m--) {
    in >> u >> a >> b;
    //printf("a = %lu b = %lu\n", a, b);
    if (u == INSERT)
	g.update({{a, b}, INSERT});
    else
	g.update({{a,b}, DELETE});
  }
  g.set_cum_in(curr_dir + "/../res/1000_0.95_0.5.cum");
  ASSERT_EQ(1, g.connected_components().size());
}
