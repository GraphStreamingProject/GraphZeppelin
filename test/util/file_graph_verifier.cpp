#include <map>
#include <iostream>
#include "../../include/test/file_graph_verifier.h"

FileGraphVerifier::FileGraphVerifier(const std::string &input_file) {
  kruskal_ref = kruskal(input_file);
  std::ifstream in(input_file);
  node_id_t n;
  edge_id_t m;
  node_id_t a, b;
  in >> n >> m;
  sets = DisjointSetUnion<node_id_t>(n);
  for (unsigned i = 0; i < n; ++i) {
    boruvka_cc.push_back({i});
    det_graph.emplace_back();
  }
  while (m--) {
    in >> a >> b;
    det_graph[a].insert(b);
    det_graph[b].insert(a);
  }
  in.close();
}

std::vector<std::set<node_id_t>> FileGraphVerifier::kruskal(const std::string& input_file) {
  std::ifstream in(input_file);
  node_id_t n;
  edge_id_t m;
  in >> n >> m;
  DisjointSetUnion<node_id_t> sets(n);
  int a,b;
  while (m--) {
    in >> a >> b;
    sets.union_set(a,b);
  }
  in.close();

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

void FileGraphVerifier::verify_edge(Edge edge) {
  auto f = sets.find_set(edge.first);
  auto s = sets.find_set(edge.second);
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
  sets.link(f, s);
  if (s == sets.find_set(s))
    std::swap(f,s);
  for (auto& i : boruvka_cc[s]) boruvka_cc[f].insert(i);
}

void FileGraphVerifier::verify_cc(node_id_t node) {
  node = sets.find_set(node);
  for (const auto& cc : kruskal_ref) {
    if (boruvka_cc[node] == cc) return;
  }
  throw NotCCException();
}

void FileGraphVerifier::verify_soln(std::vector<std::set<node_id_t>> &retval) {
  auto temp {retval};
  sort(temp.begin(),temp.end());
  sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  std::cout << "Solution ok: " << retval.size() << " CCs found." << std::endl;
}
