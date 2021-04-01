#include <gtest/gtest.h>
#include <fstream>
#include "../../include/graph.h"

#ifdef VERIFY_SAMPLES_F
#include "../util/graph_verifier.h"
#endif
/**
 * @param stream a filepath, rooted at ../res/
 * @param cum a filepath, rooted at ../res/
 */
void run_expr(const std::string& stream, const std::string& cum) {
  const std::string fname = __FILE__;
  size_t pos = fname.find_last_of("\\/");
  const std::string curr_dir = (std::string::npos == pos) ? "/" : fname.substr
        (0, pos+1) + "../res/";
  ifstream in{curr_dir + stream};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  Node a, b;
  int t;
  Graph g{num_nodes};
  while (m--) {
    in >> t >> a >> b;
    g.update({{a, b}, INSERT});
  }
#ifdef VERIFY_SAMPLES_F
  g.set_cum_in(curr_dir + cum);
#endif
#ifdef EXT_MEM_POST_PROC_F
  std::cout << g.ext_mem_connected_components().size() << std::endl;
#else
  std::cout << g.connected_components().size() << std::endl;
#endif
}

TEST(Experiment, 1K) {
  run_expr("1000_0.95_0.5.stream", "1000_0.95_0.5.cum");
}

TEST(Experiment, 10K) {
  run_expr("10000_0.95_0.5.stream", "10000_0.95_0.5.cum");
}