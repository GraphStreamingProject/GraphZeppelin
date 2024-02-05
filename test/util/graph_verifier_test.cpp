#include <gtest/gtest.h>
#include "../../include/test/file_graph_verifier.h"

const std::string fname = __FILE__;
size_t pos = fname.find_last_of("\\/");
const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0,pos);

TEST(DeterministicToolsTestSuite, TestKruskal) {
  ASSERT_EQ(78,FileGraphVerifier::kruskal(curr_dir+"/../res/multiples_graph_1024.txt").size());
}

TEST(DeterministicToolsTestSuite, TestEdgeVerifier) {
  FileGraphVerifier verifier(1024, curr_dir+"/../res/multiples_graph_1024.txt");
  // add edges of the form {i,2i}
  for (node_id_t i = 2; i < 512; ++i) {
    verifier.verify_edge({i, i*2});
  }
  // throw on nonexistent edge
  ASSERT_THROW(verifier.verify_edge({69,420}), BadEdgeException);
  ASSERT_THROW(verifier.verify_edge({420,69}), BadEdgeException);
}

TEST(DeterministicToolsTestSuite, TestCCVerifier) {
  FileGraphVerifier verifier (1024, curr_dir+"/../res/multiples_graph_1024.txt");
  // {0}, {1}, and primes \in [521,1021] are CCs
  // add edges of the form {i,2i}
  for (node_id_t i = 2; i < 512; ++i) {
    verifier.verify_edge({i, i*2});
  }
}
