#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <ctime>
#include "../../include/graph.h"
#include "../util/graph_verifier.h"
#include "../util/graph_gen.h"

TEST(Benchmark, BCHMKGraph) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);
  ifstream in{curr_dir + "/../res/1000_0.95_0.5.stream"};
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
  clock_t start = clock();
  while (m--) {
    if ((total - m) % (int)(total * .05) == 0) {
      clock_t diff = clock() - start;
      float num_seconds = diff / CLOCKS_PER_SEC;
      int percent = (total - m) / (total * .05);
      printf("Progress:%s%s", std::string(percent, '=').c_str(), std::string(20 - percent, ' ').c_str());
      printf("| %i%% -- %.2f per second\r", percent * 5, (total-m)/num_seconds); fflush(stdout);
    }
    in >> u >> a >> b;
    //printf("a = %lu b = %lu\n", a, b);
    if (static_cast<UpdateType>(u) == UpdateType::INSERT)  
    	g.update({{a, b}, UpdateType::INSERT});
    else
      g.update({{a,b}, UpdateType::DELETE});
  }
  printf("Progress:====================| Done\n");
  ASSERT_EQ(1, g.connected_components().size());
}
