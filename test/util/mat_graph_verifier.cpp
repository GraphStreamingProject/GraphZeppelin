#include <map>
#include <iostream>
#include "mat_graph_verifier.h"

MatGraphVerifier::MatGraphVerifier(node_t n, std::vector<bool>&
      compactified_input) : det_graph(compactified_input), sets(n) {
  kruskal_ref = kruskal(n, compactified_input);
  for (unsigned i = 0; i < n; ++i) {
    boruvka_cc.push_back({i});
  }
}

std::vector<std::set<node_t>> MatGraphVerifier::kruskal(node_t n, const std::vector<bool>& compactified_input) {
  DisjointSetUnion<node_t> sets(n);

  uint64_t num_edges = n * (n - 1) / 2;
  for (uint64_t i = 0; i < num_edges; i++) {
    if (compactified_input[i]) {
      Edge e = inv_nondir_non_self_edge_pairing_fn(i);
      sets.union_set(e.first, e.second);
    }
  }

  std::map<node_t, std::set<node_t>> temp;
  for (unsigned i = 0; i < n; ++i) {
    temp[sets.find_set(i)].insert(i);
  }

  std::vector<std::set<node_t>> retval;
  retval.reserve(temp.size());
  for (const auto& entry : temp) {
    retval.push_back(entry.second);
  }
  return retval;
}

void MatGraphVerifier::verify_edge(Edge edge) {
  node_t f = sets.find_set(edge.first);
  node_t s = sets.find_set(edge.second);
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
  sets.link(f, s);
  if (s == sets.find_set(s))
    std::swap(f,s);
  for (auto& i : boruvka_cc[s]) boruvka_cc[f].insert(i);
}

void MatGraphVerifier::verify_cc(node_t node) {
  node = sets.find_set(node);
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
