#include <gtest/gtest.h>
#include <string>
#include "../../include/graph.h"
#include "../../include/binary_graph_stream.h"
#include "../util/graph_verifier.h"
#include "../util/graph_gen.h"

TEST(Benchmark, BCHMKGraph) {
  BinaryGraphStream stream("/mnt/ssd2/binary_streams/kron_15_stream_binary", 32 * 1024);
  node_t num_nodes = stream.nodes();
  long m         = stream.edges();
  
  Graph g{num_nodes};
  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  auto start = std::chrono::steady_clock::now();
  while (m--) {
    if ((stream.edges() - m) % (int)(stream.edges() * .05) == 0) {
      std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
      double num_seconds = diff.count();
      int percent = (stream.edges() - m) / (stream.edges() * .05);
      printf("Progress:%s%s", std::string(percent, '=').c_str(), std::string(20 - percent, ' ').c_str());
      printf("| %i%% -- %.2f per second\r", percent * 5, (stream.edges()-m)/num_seconds); fflush(stdout);
    }
    g.update(stream.get_edge());
  }
  printf("Progress:====================| Done\n");

  int cc_num = g.connected_components().size();
  std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
  double num_seconds = diff.count();
  printf("Total insertion time was: %lf\n", num_seconds);
  printf("Insertion rate was:       %lf\n", stream.edges() / num_seconds);
  ASSERT_EQ(51, cc_num);

}
