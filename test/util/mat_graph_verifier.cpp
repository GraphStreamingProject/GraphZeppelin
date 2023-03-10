#include "../../include/test/mat_graph_verifier.h"

#include <map>
#include <iostream>
#include <algorithm>
#include <cassert>

MatGraphVerifier::MatGraphVerifier(node_id_t n) : n(n), sets(n) {
  adj_matrix = std::vector<std::vector<bool>>(n);
  for (node_id_t i = 0; i < n; ++i)
    adj_matrix[i] = std::vector<bool>(n - i);
}

void MatGraphVerifier::edge_update(node_id_t src, node_id_t dst) {
  if (src > dst) std::swap(src, dst);
  
  dst = dst - src;
  
  // update adj_matrix entry
  adj_matrix[src][dst] = !adj_matrix[src][dst];
}


void MatGraphVerifier::reset_cc_state() {
  kruskal_ref = kruskal();
  sets.reset();
  boruvka_cc.clear();
  for (node_id_t i = 0; i < n; ++i)
    boruvka_cc.push_back({i});
}

std::vector<std::set<node_id_t>> MatGraphVerifier::kruskal() {
  DisjointSetUnion<node_id_t> kruskal_dsu(n);

  for (node_id_t i = 0; i < n; i++) {
    for (node_id_t j = 0; j < adj_matrix[i].size(); j++) {
      if (adj_matrix[i][j]) kruskal_dsu.merge(i, i + j);
    }
  }

  std::map<node_id_t, std::set<node_id_t>> temp;
  for (unsigned i = 0; i < n; ++i) {
    temp[kruskal_dsu.find_root(i)].insert(i);
  }

  std::vector<std::set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto& entry : temp) {
    retval.push_back(entry.second);
  }
  return retval;
}

void MatGraphVerifier::verify_edge(Edge edge) {
  // verify that the edge in question actually exists
  if (edge.src > edge.dst) std::swap(edge.src, edge.dst);
  if (!adj_matrix[edge.src][edge.dst - edge.src]) {
    printf("Got an error on edge (%u, %u): edge is not in adj_matrix\n", edge.src, edge.dst);
    throw BadEdgeException();
  }

  DSUMergeRet<node_id_t> ret = sets.merge(edge.src, edge.dst); // perform the merge
  if (!ret.merged) {
    printf("Got an error on edge (%u, %u): components already joined!\n", edge.src, edge.dst);
    throw BadEdgeException();
  }

  // if all checks pass, update boruvka_cc by merging in set elements of child
  for (auto& i : boruvka_cc[ret.child]) boruvka_cc[ret.root].insert(i);
}

void MatGraphVerifier::verify_cc(node_id_t node) {
  node = sets.find_root(node);
  for (const auto& cc : kruskal_ref) {
    if (boruvka_cc[node] == cc) return;
  }
  throw NotCCException();
}

void MatGraphVerifier::verify_soln(std::vector<std::set<node_id_t>> &retval) {
  auto temp {retval};
  std::sort(temp.begin(),temp.end());
  std::sort(kruskal_ref.begin(),kruskal_ref.end());
  if (kruskal_ref != temp)
    throw IncorrectCCException();

  std::cout << "Solution ok: " << retval.size() << " CCs found." << std::endl;
}
