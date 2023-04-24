#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include "../include/graph.h"

int main() {
  node_id_t num_nodes = (1u<<17);
  Graph g{num_nodes};

  size_t num_updates = 1<<25;

  auto updates = [&]{
    std::vector<Edge> ret;
    ret.reserve(num_updates);

    std::unordered_set<Edge> forest;

    std::vector<node_id_t> parent(num_nodes);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<int> rank(num_nodes);
    const auto rep = [&](node_id_t x){
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    const auto link = [&](node_id_t a, node_id_t b){
      if (rank[a] < rank[b]) std::swap(a, b);
      parent[b] = a;
      if (rank[a] == rank[b]) ++rank[a];
    };

    std::mt19937_64 rng{std::random_device()()};
    std::uniform_int_distribution<node_id_t> rand_node(0, num_nodes - 1);
    while (ret.size() < num_updates) {
      node_id_t a = rand_node(rng);
      node_id_t b = rand_node(rng);
      if (a == b) continue;
      if (a > b) std::swap(a, b);
      Edge e = {a, b};
      if (forest.find(e) != forest.end()) continue;
      ret.push_back(e);
      a = rep(a);
      b = rep(b);
      if (a == b) continue;
      link(a, b);
      forest.insert(e);
    }
    return ret;
  }();
  std::cout << "Doing " << num_updates << " updates" << std::endl;
  auto start_time = std::chrono::steady_clock::now();
  for (const auto& update : updates) {
    g.update({update, INSERT});
  }
  auto end_time = std::chrono::steady_clock::now();
  std::cout << "Total time: " << std::chrono::duration<long double>(end_time - start_time).count() << std::endl;
}
