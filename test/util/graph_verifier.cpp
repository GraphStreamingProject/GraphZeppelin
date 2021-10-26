#include <map>
#include <iostream>
#include "graph_verifier.h"

//node_t* parent;
//node_t* size;

node_t dsu_find(node_t i, node_t* parent) {
  if (parent[i] == i) return i;
  return parent[i] = dsu_find(parent[i], parent);
}

void dsu_union(node_t i, node_t j, node_t* parent, node_t* size) {
  i = dsu_find(i, parent); j = dsu_find(j, parent);
  if (size[i] < size[j]) std::swap(i,j);
  parent[j] = i;
  size[i] += size[j];
}

GraphVerifier::GraphVerifier(const string &input_file) {
  kruskal_ref = kruskal(input_file);
  ifstream in(input_file);
  node_t n,m; in >> n >> m;
  node_t a,b;
  parent = (node_t*) malloc(n*sizeof(node_t));
  size = (node_t*) malloc(n*sizeof(node_t));
  for (unsigned i = 0; i < n; ++i) {
    boruvka_cc.push_back({i});
    det_graph.emplace_back();
    parent[i] = i;
    size[i] = 1;
  }
  while (m--) {
    in >> a >> b;
    det_graph[a].insert(b);
    det_graph[b].insert(a);
  }
  in.close();
}

GraphVerifier::~GraphVerifier() {
  free(parent);
  free(size);
}

std::vector<std::set<node_t>> kruskal(const string& input_file) {
  ifstream in(input_file);
  node_t n, m; in >> n >> m;
  auto* parent = (node_t*) malloc(n*sizeof(node_t));
  auto* size = (node_t*) malloc(n*sizeof(node_t));

  for (unsigned i = 0; i < n; ++i) {
    parent[i] = i;
    size[i] = 1;
  }
  int a,b;
  while (m--) {
    in >> a >> b;
    dsu_union(a,b, parent, size);
  }
  in.close();

  std::map<node_t, std::set<node_t>> temp;
  for (unsigned i = 0; i < n; ++i) {
    temp[dsu_find(i, parent)].insert(i);
  }
  free(parent);
  free(size);

  std::vector<std::set<node_t>> retval;
  retval.reserve(temp.size());
  for (const auto& entry : temp) {
    retval.push_back(entry.second);
  }
  return retval;
}

void GraphVerifier::verify_edge(Edge edge) {
  node_t f = dsu_find(edge.first,parent);
  node_t s = dsu_find(edge.second,parent);
  if (boruvka_cc[f].find(edge.second) != boruvka_cc[f].end()
  || boruvka_cc[s].find(edge.first) != boruvka_cc[s].end()) {
    printf("Got an error of node %u to node (1)%u\n", edge.first, edge.second);
    throw BadEdgeException();
  }
  if (det_graph[edge.first].find(edge.second) == det_graph[edge.first].end()) {
    printf("Got an error of node %u to node (2)%u\n", edge.first, edge.second);
    throw BadEdgeException();
  }

  // if all checks pass, merge supernodes
  if (size[f] < size[s])
    std::swap(f,s);
  for (auto& i : boruvka_cc[s]) boruvka_cc[f].insert(i);
  dsu_union(f, s, parent, size);
}

void GraphVerifier::verify_cc(node_t node) {
  node = dsu_find(node,parent);
  for (const auto& cc : kruskal_ref) {
    if (boruvka_cc[node] == cc) return;
  }
  throw NotCCException();
}

void GraphVerifier::verify_soln(vector<set<node_t>> &retval) {
  vector<set<node_t>> temp {retval};
  sort(temp.begin(),temp.end());
  sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  std::cout << "Solution ok: " << retval.size() << " CCs found." << endl;
}
