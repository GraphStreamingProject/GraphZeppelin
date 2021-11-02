#include <gtest/gtest.h>
#include <fstream>
#include <unordered_map>
#include "../include/graph.h"
#include "../include/test/graph_gen.h"
#include "../include/test/mat_graph_verifier.h"

void test_continuous(unsigned nodes, unsigned long updates_per_sample, unsigned long samples) {
  srand(time(NULL));
  Graph g(nodes);
  size_t total_edges = static_cast<size_t>(nodes - 1) * nodes / 2;
  std::vector<bool> adj(total_edges);
  unsigned long num_failure = 0;
  for (unsigned long i = 0; i < samples; i++) {
    std::cout << "Starting updates" << std::endl;
    for (unsigned long j = 0; j < updates_per_sample; j++) {
      unsigned edgei = rand() % nodes;
      unsigned edgej = rand() % (nodes - 1);
      if (edgei > edgej) {
        std::swap(edgei, edgej);
      } else {
        edgej++;
      }
      uint64_t edgeidx = nondirectional_non_self_edge_pairing_fn(edgei, edgej);
      g.update({{edgei, edgej}, INSERT});
      adj[edgeidx] = !adj[edgeidx];
    }
    try {
      g.set_verifier(std::make_unique<MatGraphVerifier>(nodes, adj));
      std::cout << "Running cc" << std::endl;
      g.connected_components(true);
    } catch (const OutOfQueriesException& e) {
      num_failure++;
      std::cout << "CC #" << i << "failed with NoMoreQueries" << std::endl;
    } catch (const NotCCException& e) {
      num_failure++;
      std::cout << "CC #" << i << "failed with NotCC" << std::endl;
    } catch (const BadEdgeException& e) {
      num_failure++;
      std::cout << "CC #" << i << "failed with BadEdge" << std::endl;
    }
  }
  std::clog << nodes << ',' << num_failure << std::endl;
}

void test_continuous(std::ifstream& in, unsigned long samples) {
  node_t n, m;
  in >> n >> m;
  Graph g(n);
  size_t total_edges = static_cast<size_t>(n - 1) * n / 2;
  node_t updates_per_sample = m / samples;
  std::vector<bool> adj(total_edges);
  unsigned long num_failure = 0;

  node_t t, a, b;
  for (unsigned long i = 0; i < samples; i++) {
    std::cout << "Starting updates" << std::endl;
    for (unsigned long j = 0; j < updates_per_sample; j++) {
      in >> t >> a >> b;
      uint64_t edgeidx = nondirectional_non_self_edge_pairing_fn(a, b);
      g.update({{a, b}, INSERT});
      adj[edgeidx] = !adj[edgeidx];
    }
    try {
      g.set_verifier(std::make_unique<MatGraphVerifier>(n, adj));
      std::cout << "Running cc" << std::endl;
      g.connected_components(true);
    } catch (const OutOfQueriesException& e) {
      num_failure++;
      std::cout << "CC #" << i << "failed with NoMoreQueries" << std::endl;
    } catch (const NotCCException& e) {
      num_failure++;
      std::cout << "CC #" << i << "failed with NotCC" << std::endl;
    } catch (const BadEdgeException& e) {
      num_failure++;
      std::cout << "CC #" << i << "failed with BadEdge" << std::endl;
    }
  }
  std::clog << n << ',' << num_failure << std::endl;
}

//TEST(TestContinuous, TestRandom) {
//  test_continuous(1e5, 1e7, 10);
//}

TEST(TestContinuous, DISABLED_StandardKron17) {
  std::ifstream in{ "./kron17" };
  node_t n, m;
  in >> n >> m;
  Graph g{n};
  int type, a, b;
  while (m--) {
    in >> type >> a >> b;
    g.update({{a, b}, INSERT});
  }
  g.connected_components();
}

// Uses ./kron17 in the current working directory
TEST(TestContinuous, TestKron17) {
  std::ifstream input { "./kron17" };
  test_continuous(input, 10);
}
