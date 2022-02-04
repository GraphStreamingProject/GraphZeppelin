#include "../../include/test/mat_graph_verifier.h"

#include <map>
#include <iostream>
#include <algorithm>
#include <cassert>

MatGraphVerifier::MatGraphVerifier(node_id_t n) : n(n) {
  adj_graph = std::vector<std::vector<bool>>(n);
  for (node_id_t i = 0; i < n; ++i)
    adj_graph[i] = std::vector<bool>(n - i);
}

void MatGraphVerifier::edge_update(node_id_t src, node_id_t dst) {
  if (src > dst) std::swap(src, dst);
  
  dst = dst - src;
  
  // update adj_matrix entry
  adj_graph[src][dst] = !adj_graph[src][dst];
}
  

void MatGraphVerifier::reset_cc_state() {
  kruskal_ref = kruskal();
  sets = DisjointSetUnion<node_id_t>(n);
  boruvka_cc.clear();
  for (node_id_t i = 0; i < n; ++i)
    boruvka_cc.push_back({i});
}

std::vector<std::set<node_id_t>> MatGraphVerifier::kruskal() {
  DisjointSetUnion<node_id_t> sets(n);

  for (node_id_t i = 0; i < n; i++) {
    for (node_id_t j = 0; j < adj_graph[i].size(); j++) {
      if (adj_graph[i][j]) sets.union_set(i, i + j);
    }
  }

  std::map<node_id_t, std::set<node_id_t>> temp;
  for (unsigned i = 0; i < n; ++i) {
    temp[sets.find_set(i)].insert(i);
  }

  std::vector<std::set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto& entry : temp) {
    retval.push_back(entry.second);
  }
  return retval;
}

void MatGraphVerifier::verify_edge(Edge edge) {
  auto f = sets.find_set(edge.first);
  auto s = sets.find_set(edge.second);
  if (boruvka_cc[f].find(edge.second) != boruvka_cc[f].end()
  || boruvka_cc[s].find(edge.first) != boruvka_cc[s].end()) {
    printf("Got an error of node %u to node (1)%u\n", edge.first, edge.second);
    throw BadEdgeException();
  }
  if (edge.first > edge.second) std::swap(edge.first, edge.second);
  if (!adj_graph[edge.first][edge.second - edge.first]) {
    printf("Got an error of node %u to node (2)%u\n", edge.first, edge.second);
    throw BadEdgeException();
  }

  // if all checks pass, merge supernodes
  sets.link(f, s);
  if (s == sets.find_set(s))
    std::swap(f,s);
  for (auto& i : boruvka_cc[s]) boruvka_cc[f].insert(i);
}

void MatGraphVerifier::verify_cc(node_id_t node) {
  node = sets.find_set(node);
  for (const auto& cc : kruskal_ref) {
    if (boruvka_cc[node] == cc) return;
  }
  throw NotCCException();
}

void MatGraphVerifier::verify_soln(std::vector<std::set<node_id_t>> &retval) {
  auto temp {retval};
  std::sort(temp.begin(),temp.end());
  std::sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  std::cout << "Solution ok: " << retval.size() << " CCs found." << std::endl;
}