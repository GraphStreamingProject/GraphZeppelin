#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>
#include "include/graph.h"

int main() {
  std::cout << "Input stream file: ";
  std::cout.flush();

  std::string input;
  std::cin >> input;
  std::ifstream in { input };
  Node n, m;
  in >> n >> m;
  Graph g{n};
  int type, a, b;
  while (m--) {
    in >> type >> a >> b;
    g.update({{a, b}, INSERT});
  }
  std::cout << g.connected_components().size() << std::endl;

  return 0;
}
