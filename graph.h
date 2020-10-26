#pragma once
#include <cstdlib>

class Graph{
  const int num_nodes;
  vector<map<int,int>> edges;

public:
  Graph(int num_nodes): num_nodes(num_nodes) {
    srand(time(NULL));
    edges = vector<map<int,int>>(num_nodes);
    for (int i = 0; i < num_nodes; i++){
      for (int j = 0; j < num_nodes; j++){
        if (i != j){

        }
      }
    }
  }

}
