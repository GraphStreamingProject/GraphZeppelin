//
// Created by victor on 6/7/21.
//
#include <iostream>
#include "../../include/graph.h"

// take a graph file, run connected-components on it, and print the number of
// components and the nodes in each component.
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Incorrect number of arguments. "
                 "Expected one but got " << argc-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream graph_input {argv[1]};
  Node num_nodes; graph_input >> num_nodes;
  Node num_updates; graph_input >> num_updates;

  Graph graph {num_nodes};

  Edge temp;
  int type_throwaway;
  while (num_updates--) {
    graph_input >> type_throwaway;
    graph_input >> temp.first >> temp.second;
    graph.update({temp,(UpdateType)type_throwaway});
  }
  auto result = graph.connected_components();

  std::cout << "Number of connected components: " << result.size() << std::endl;
  for (auto &cc : result) {
    std::cout << cc.size() << ":";
    for (auto node : cc) {
      std::cout << " " << node;
    }
    std::cout << std::endl;
  }

  return 0;
}
