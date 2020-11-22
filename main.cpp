#include <iostream>
#include <limits>
#include <boost/multiprecision/cpp_int.hpp>

// #include "include/graph.h"
// #include "include/graph_128.h"


using cpp_int = boost::multiprecision::cpp_int;

int main() {
  unsigned long long ULLMAX = std::numeric_limits<unsigned long long>::max();
  cpp_int num_nodes; std::cin >> num_nodes;
  // if (num_nodes > ULLMAX) {
  //   // use special (slower) 128-bit integer types
  //   Graph_w g{num_nodes};
  //   vector<set<Node_w>> res = g.connected_components();
  //   cout << res.size() << endl;
  //   for (auto s : res) {
  //     cout << s.size() << endl;
  //     for (auto n : s) cout << n << " ";
  //     cout << endl;
  //   }
  // } else {
  //   // use normal 64-bit integer types
  //   Graph g{num_nodes};
  //   vector<set<Node>> res = g.connected_components();
  //   cout << res.size() << endl;
  //   for (auto s : res) {
  //     cout << s.size() << endl;
  //     for (auto n : s) cout << n << " ";
  //     cout << endl;
  //   }
  // }
}
