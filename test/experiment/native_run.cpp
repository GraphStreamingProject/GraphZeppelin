#include <iostream>
#include "../../include/graph.h"

typedef unsigned long ul;

int main() {
  auto start = time(nullptr);
  Node n,m; std::cin >> n >> m;
  Graph g {n};
  bool ins;
  Node a,b;
  while (m--) {
    std::cin >> ins >> a >> b;
    if (ins) {
      g.update({{a, b}, DELETE});
    } else {
      g.update({{a, b}, INSERT});
    }
  }
  for (const auto& v : g.connected_components()) {
    std::cout << v.size() << std::endl;
  }
  std::cout << "Time taken: " << time(nullptr) - start << std::endl;
}