#include <fstream>
#include <map>
#include "deterministic.h"

Node* parent;
Node* size;

Node dsu_find(Node i) {
  if (parent[i] == i) return i;
  return parent[i] = dsu_find(parent[i]);
}

void dsu_union(Node i, Node j) {
  i = dsu_find(i); j = dsu_find(j);
  if (size[i] < size[j]) std::swap(i,j);
  parent[j] = i;
  size[i] += size[j];
}

std::vector<std::set<Node>> kruskal(const string& input_file) {
  ifstream in(input_file);
  Node n, m; in >> n >> m;
  parent = (Node*) malloc((n+1)*sizeof(Node));
  size = (Node*) malloc((n+1)*sizeof(Node));

  for (unsigned i = 1; i <= n; ++i) {
    parent[i] = i;
    size[i] = 1;
  }
  int a,b;
  while (m--) {
    in >> a >> b;
    dsu_union(a,b);
  }
  std::map<Node, std::set<Node>> temp;
  for (unsigned i = 1; i <= n; ++i) {
    temp[dsu_find(i)].insert(i);
  }
  free(parent);
  free(size);

  std::vector<std::set<Node>> retval;
  retval.reserve(temp.size());
  for (const auto& entry : temp) {
    retval.push_back(entry.second);
  }
  return retval;
}

void verify_edge(Edge edge) {
  if (!kruskal_run) throw NoPrepException();
}

void verify_cc(Node node);