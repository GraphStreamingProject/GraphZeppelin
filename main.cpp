#include <iostream>
#include "include/graph.h"

int main() {
  unsigned long long int num_nodes = 1000;
  Graph g{num_nodes};
  for (int i=1;i<num_nodes;++i) {
    for (int j = i*2;j<num_nodes;j+=i) {
      g.update({{i,j}, INSERT});
    }
  }
  vector<set<Node>> res = g.connected_components();
  cout << res.size() << endl;
}
