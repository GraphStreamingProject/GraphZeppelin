#include <gtest/gtest.h>
#include "mat_graph_verifier.h"

const std::string fname = __FILE__;
size_t pos = fname.find_last_of("\\/");
const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0,pos);

TEST(DeterministicToolsTestSuite, TestKruskal) {
  auto adj = cum_file_to_adj_matrix(curr_dir+"/../res/multiples_graph_1024.txt");
  node_t n = 1024;
  ASSERT_EQ(78,MatGraphVerifier::kruskal(n, adj).size());
}

TEST(DeterministicToolsTestSuite, TestEdgeVerifier) {
  auto adj = cum_file_to_adj_matrix(curr_dir+"/../res/multiples_graph_1024.txt");
  node_t n = 1024;
  MatGraphVerifier verifier {n, adj};
  // add edges of the form {i,2i}
  for (int i = 2; i < 512; ++i) {
    verifier.verify_edge({i,i*2});
  }
  // throw on nonexistent edge
  ASSERT_THROW(verifier.verify_edge({69,420}), BadEdgeException);
  ASSERT_THROW(verifier.verify_edge({420,69}), BadEdgeException);
  // throw on already-included edge
  ASSERT_THROW(verifier.verify_edge({120,240}), BadEdgeException);
  ASSERT_THROW(verifier.verify_edge({240,120}), BadEdgeException);
  // throw on edge within the same set
  ASSERT_THROW(verifier.verify_edge({250,1000}), BadEdgeException);
  ASSERT_THROW(verifier.verify_edge({1000,250}), BadEdgeException);
}

TEST(DeterministicToolsTestSuite, TestCCVerifier) {
  auto adj = cum_file_to_adj_matrix(curr_dir+"/../res/multiples_graph_1024.txt");
  node_t n = 1024;
  MatGraphVerifier verifier {n, adj};
  // {0}, {1}, and primes \in [521,1021] are CCs
  verifier.verify_cc(0);
  verifier.verify_cc(1);
  verifier.verify_cc(911);
  // add edges of the form {i,2i}
  for (int i = 2; i < 512; ++i) {
    verifier.verify_edge({i,i*2});
  }
  // nothing else is currently a CC
  for (int i = 2; i < 512; ++i) {
    ASSERT_THROW(verifier.verify_cc(i), NotCCException);
  }
}
