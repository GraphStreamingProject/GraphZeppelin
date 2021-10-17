#include <map>
#include <iostream>
#include "mat_graph_verifier.h"

//node_t* parent;
//node_t* size;

static node_t dsu_find(node_t i, node_t* parent) {
  if (parent[i] == i) return i;
  return parent[i] = dsu_find(parent[i], parent);
}

static void dsu_union(node_t i, node_t j, node_t* parent, node_t* size) {
  i = dsu_find(i, parent); j = dsu_find(j, parent);
  if (size[i] < size[j]) std::swap(i,j);
  parent[j] = i;
  size[i] += size[j];
}

MatGraphVerifier::MatGraphVerifier(node_t n, std::vector<bool>&
      compactified_input) : det_graph(compactified_input){
  kruskal_ref = kruskal(n, compactified_input);
  parent = (node_t*) malloc(n*sizeof(node_t));
  size = (node_t*) malloc(n*sizeof(node_t));
  for (unsigned i = 0; i < n; ++i) {
    boruvka_cc.push_back({i});
    parent[i] = i;
    size[i] = 1;
  }
}

MatGraphVerifier::~MatGraphVerifier() {
  free(parent);
  free(size);
}

std::vector<std::set<node_t>> MatGraphVerifier::kruskal(node_t n, const std::vector<bool>& compactified_input) {
  node_t* parent = (node_t*) malloc(n*sizeof(node_t));
  node_t* size = (node_t*) malloc(n*sizeof(node_t));

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

void MatGraphVerifier::verify_edge(Edge edge) {
  node_t f = dsu_find(edge.first,parent);
  node_t s = dsu_find(edge.second,parent);
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

void MatGraphVerifier::verify_cc(node_t node) {
  node = dsu_find(node,parent);
  for (const auto& cc : kruskal_ref) {
    if (boruvka_cc[node] == cc) return;
  }
  throw NotCCException();
}

void MatGraphVerifier::verify_soln(vector<set<node_t>> &retval) {
  vector<set<node_t>> temp {retval};
  sort(temp.begin(),temp.end());
  sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  std::cout << "Solution ok: " << retval.size() << " CCs found." << std::endl;
}
