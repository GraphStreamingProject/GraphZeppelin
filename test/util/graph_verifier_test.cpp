#include <gtest/gtest.h>

#include "../../include/test/file_graph_verifier.h"
#include <cmath>

const std::string fname = __FILE__;
size_t pos = fname.find_last_of("\\/");
const std::string curr_dir = (std::string::npos == pos) ? "" : fname.substr(0, pos);

// just a little primality check.
// based upon https://stackoverflow.com/questions/4424374/determining-if-a-number-is-prime
bool isPrime(size_t number) {
  if (number < 2) return false;
  if (number == 2) return true;
  if (number % 2 == 0) return false;
  size_t max = ceil(sqrt(number));
  for (size_t i = 3; i <= max; i += 2) {
    if (number % i == 0) return false;
  }
  return true;
}

TEST(DeterministicToolsTestSuite, TestKruskal) {
  ASSERT_EQ(78, FileGraphVerifier::kruskal(curr_dir + "/../res/multiples_graph_1024.txt").size());
}

TEST(DeterministicToolsTestSuite, TestEdgeVerifier) {
  FileGraphVerifier verifier(1024, curr_dir + "/../res/multiples_graph_1024.txt");
  // add edges of the form {i,2i}
  for (node_id_t i = 2; i < 512; ++i) {
    verifier.verify_edge({i, i * 2});
  }
  // throw on nonexistent edge
  ASSERT_THROW(verifier.verify_edge({69, 420}), BadEdgeException);
  ASSERT_THROW(verifier.verify_edge({420, 69}), BadEdgeException);
}

TEST(DeterministicToolsTestSuite, TestCCVerifier) {
  FileGraphVerifier verifier(1024, curr_dir + "/../res/multiples_graph_1024.txt");
  // {0}, {1}, and primes \in [521,1021] are CCs
  // add edges of the form {i,2i}
  for (node_id_t i = 2; i < 512; ++i) {
    verifier.verify_edge({i, i * 2});
  }

  std::vector<std::set<node_id_t>> cc;
  cc.emplace_back(std::set<node_id_t>{0});
  cc.emplace_back(std::set<node_id_t>{1});
  std::set<node_id_t> big_cc;
  for (node_id_t i = 2; i < 521; i++) {
    big_cc.insert(i);
  }
  for (node_id_t i = 521; i < 1024; i++) {
    if (isPrime(i)) {
      cc.emplace_back(std::set<node_id_t>{i});
    } else {
      big_cc.insert(i);
    }
  }
  cc.push_back(big_cc);

  verifier.verify_soln(cc);
}
