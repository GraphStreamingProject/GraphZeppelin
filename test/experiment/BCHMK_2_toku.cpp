#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "../../include/graph.h"
#include "../util/graph_verifier.h"
#include "../util/graph_gen.h"

TEST(Benchmark, BCHMK2Toku) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/../res/3000.test.stream"};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  long total = m;
  Node a, b;
  uint8_t u;
  Graph g{num_nodes};
  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  while (m--) {
    if ((total - m) % (int)(total * .05) == 0) {
      int percent = (total - m) / (total * .05);
      printf("Progress:%s%s", std::string(percent, '=').c_str(), std::string(20 - percent, ' ').c_str());
      printf("| %i%%\r", percent * 5); fflush(stdout);
    }
    in >> u >> a >> b;
    //printf("a = %lu b = %lu\n", a, b);
    if (u == INSERT)
      g.update({{a, b}, INSERT});
    else
      g.update({{a,b}, DELETE});
  }
  printf("Progress:====================| Done\n");
  g.set_cum_in(curr_dir + "/../res/3000.test.cum");
  ASSERT_EQ(1, g.connected_components().size());
}
