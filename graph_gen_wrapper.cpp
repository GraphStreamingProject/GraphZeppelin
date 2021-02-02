#include <iostream>
#include "test/util/graph_gen.h"

using namespace std;

int main() {
  long n;            // number of nodes
  double p;          // prob of edge between nodes
  long m;           // number of edges
  bool graph_model;     // G(n,p) -- 0, G(n,m) -- 1
  int max_appearances;  // the maximum number of times an edge can show up
  // in the stream
  std::string out_file; // file to write stream
  std::string cum_out_file; // file to write cum graph
  cout << "Number of nodes n: "; cin >> n;
  cout << "Prob of edge between nodes: "; cin >> p;
  cout << "Number of edges: "; cin >> m;
  cout << "Graph model, G(n,p) -- 0, G(n,m) -- 1: "; cin >> graph_model;
  cout << "Max appearances in the stream: "; cin >> max_appearances;
  cout << "Stream output file: "; cin >> out_file;
  cout << "Cumulative output file: "; cin >> cum_out_file;
  cout << "Generating graph..." << endl;
  generate_stream({n,p,m,graph_model,max_appearances,out_file,cum_out_file});
}