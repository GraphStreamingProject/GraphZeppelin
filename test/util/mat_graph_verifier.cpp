#include <map>
#include <iostream>
#include "../../include/test/mat_graph_verifier.h"

MatGraphVerifier::MatGraphVerifier(node_id_t n, std::vector<bool>&
      input) : det_graph(input), sets(n) {
  kruskal_ref = kruskal(n, input);
  for (unsigned i = 0; i < n; ++i) {
    boruvka_cc.push_back({i});
  }
}

std::vector<std::set<node_id_t>> MatGraphVerifier::kruskal(node_id_t n, const std::vector<bool>& input) {
  DisjointSetUnion<node_id_t> sets(n);

  uint64_t num_edges = n * (n - 1) / 2;
  for (uint64_t i = 0; i < num_edges; i++) {
    if (input[i]) {
      Edge e = inv_uid(i);
      sets.union_set(e.first, e.second);
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
  if (!det_graph[get_uid(edge.first,edge.second)]) {
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
  sort(temp.begin(),temp.end());
  sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  std::cout << "Solution ok: " << retval.size() << " CCs found." << std::endl;
}
