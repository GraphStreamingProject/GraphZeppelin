#include <gtest/gtest.h>
#include <fstream>
#include <unordered_map>
#include "../include/graph.h"
#include "util/graph_gen.h"

void test_continuous(unsigned nodes, unsigned long updates_per_sample, unsigned long samples) {
  srand(time(NULL));
  Graph g(nodes);
  size_t total_edges = static_cast<size_t>(nodes - 1) * nodes / 2;
  std::vector<bool> adj(total_edges);
  uint64_t num_edges = 0;
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
      if (adj[edgeidx]) {
        num_edges--;
        adj[edgeidx] = false;
      } else {
        num_edges++;
        adj[edgeidx] = true;
      }
    }
    try {
      std::cout << "Writing cumulative" << std::endl;
      std::ofstream out("./cum_sample.txt");
      out << nodes << " " << num_edges << std::endl;
      for (unsigned i = 0; i < total_edges; i++) {
        if (adj[i]) {
          auto edge = inv_nondir_non_self_edge_pairing_fn(i);
          out << edge.first << " " << edge.second << std::endl;
        }
      }
      out.close();
      g.set_cum_in("./cum_sample.txt");
      std::cout << "Running cc" << std::endl;
      g.connected_components();
      g.post_cc_resume();
    } catch (const NoGoodBucketException& e) {
      num_failure++;
    } catch (const NotCCException& e) {
      num_failure++;
    } catch (const BadEdgeException& e) {
      num_failure++;
    }
  }
  std::clog << nodes << ',' << num_failure << std::endl;
}

TEST(TestContinuous, TestContinuous) {
  test_continuous(100'000, 100'000'000, 10);
}
