#include <fstream>
#include <map>
#include "graph_verifier.h"

//Node* parent;
//Node* size;

Node dsu_find(Node i, Node* parent) {
  if (parent[i] == i) return i;
  return parent[i] = dsu_find(parent[i], parent);
}

void dsu_union(Node i, Node j, Node* parent, Node* size) {
  i = dsu_find(i, parent); j = dsu_find(j, parent);
  if (size[i] < size[j]) std::swap(i,j);
  parent[j] = i;
  size[i] += size[j];
}

GraphVerifier::GraphVerifier(Node n, std::vector<bool>&
      compactified_input) : det_graph(compactified_input){
  kruskal_ref = kruskal(n, compactified_input);
  Node a,b;
  parent = (Node*) malloc(n*sizeof(Node));
  size = (Node*) malloc(n*sizeof(Node));
  for (unsigned i = 0; i < n; ++i) {
    boruvka_cc.push_back({i});
    parent[i] = i;
    size[i] = 1;
  }
}

GraphVerifier::~GraphVerifier() {
  free(parent);
  free(size);
}

std::vector<std::set<Node>> kruskal(Node n, const std::vector<bool>& compactified_input) {
  Node* parent = (Node*) malloc(n*sizeof(Node));
  Node* size = (Node*) malloc(n*sizeof(Node));

  for (unsigned i = 0; i < n; ++i) {
    parent[i] = i;
    size[i] = 1;
  }
  uint64_t num_edges = n * (n - 1) / 2;
  for (uint64_t i = 0; i < num_edges; i++) {
    if (compactified_input[i]) {
      Edge e = inv_nondir_non_self_edge_pairing_fn(i);
      dsu_union(e.first, e.second, parent, size);
    }
  }

  std::map<Node, std::set<Node>> temp;
  for (unsigned i = 0; i < n; ++i) {
    temp[dsu_find(i, parent)].insert(i);
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

void GraphVerifier::verify_edge(Edge edge) {
  Node f = dsu_find(edge.first,parent);
  Node s = dsu_find(edge.second,parent);
  if (boruvka_cc[f].find(edge.second) != boruvka_cc[f].end()
  || boruvka_cc[s].find(edge.first) != boruvka_cc[s].end()) {
    printf("Got an error of node %lu to node (1)%lu\n", edge.first, edge.second);
    throw BadEdgeException();
  }
  if (!det_graph[nondirectional_non_self_edge_pairing_fn(edge.first,edge.second)]) {
    printf("Got an error of node %lu to node (2)%lu\n", edge.first, edge.second);
    throw BadEdgeException();
  }

  // if all checks pass, merge supernodes
  if (size[f] < size[s])
    std::swap(f,s);
  for (auto& i : boruvka_cc[s]) boruvka_cc[f].insert(i);
  dsu_union(f, s, parent, size);
}

void GraphVerifier::verify_cc(Node node) {
  node = dsu_find(node,parent);
  for (const auto& cc : kruskal_ref) {
    if (boruvka_cc[node] == cc) return;
  }
  throw NotCCException();
}

void GraphVerifier::verify_soln(vector<set<Node>> &retval) {
  vector<set<Node>> temp {retval};
  sort(temp.begin(),temp.end());
  sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  cout << "Solution ok: " << retval.size() << " CCs found." << endl;
}
