#include "../../include/test/file_graph_verifier.h"

#include <map>
#include <iostream>
#include <algorithm>
#include <cassert>

node_id_t num_nodes;
int edge_connectivity;

// pad {x_1, \dots, x_k} to {x_1, \dots, x_k, x_k, \dots, x_k}
// return (x_k, x_k, \dots, x_k, x_{k-1}, \dots, x_1)_{n}
uint128_t concat_tuple_fn(const uint32_t* edge_buf) {
  const auto num_vals = edge_buf[0];
  uint128_t retval = 0;
  for (int i = 0; i < edge_connectivity - num_vals; ++i) {
    retval *= num_nodes;
    retval += edge_buf[num_vals];
  }
  for (int i = num_vals; i > 0; --i) {
    retval *= num_nodes;
    retval += edge_buf[i];
  }
  return retval;
}

void inv_concat_tuple_fn(uint128_t catted, Edge* edge_buf) {
  edge_buf[1] = catted % num_nodes;
  catted /= num_nodes;
  int i = 2;
  while (catted > 0) {
    edge_buf[i] = catted % num_nodes;
    if (edge_buf[i] == edge_buf[i-1]) break;
    catted /= num_nodes;
    ++i;
  }
  edge_buf[0] = i - 1;
}

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
  int r;
  in >> n >> m >> r;
  DisjointSetUnion<node_id_t> sets(n);
  std::vector<node_id_t> buf(r);
  int num;
  while (m--) {
    in >> num;
    for (int i = 0; i < num; ++i) {
      in >> buf[i];
    }
    // TODO: make union set(s) act on hypergraphs
    std::vector<node_id_t> vectorized(buf.begin(), buf.begin() + num);
    sets.union_sets(vectorized);
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

void FileGraphVerifier::verify_edge(Edge* edge) {
  auto n = edge[0];
  decltype(n) reps[n + 1];
  for (int i = 1; i <= n; ++i) {
    reps[i] = sets.find_set(edge[i]);
  }
  for (int i = 1; i <= n; ++i) {
    const auto f = reps[i];
    for (int j = i + 1; j <= n; ++j) {
      const auto s = reps[j];
      if (boruvka_cc[f].find(edge[j]) != boruvka_cc[f].end()
        || boruvka_cc[s].find(edge[i]) != boruvka_cc[s].end()) {
        std::cout << "Got an error(1) of edge { ";
        for (int k = 1; k <= n; ++k) {
          std::cout << edge[k] << ", ";
        }
        std::cout << "}\n";
        throw BadEdgeException();
      }
      if (det_graph[edge[i]].find(edge[j]) == det_graph[edge[i]].end()) {
        std::cout << "Got an error(2) of edge { ";
        for (int k = 1; k <= n; ++k) {
          std::cout << edge[k] << ", ";
        }
        std::cout << "}\n";
        throw BadEdgeException();
      }
    }
  }
  std::vector<Edge> e(edge + 1, edge + n + 1);
  // if all checks pass, merge supernodes
  auto f = sets.link(e);
  for (unsigned i = 1; i <= n; ++i) {
    // merge boruvka_ccs
    if (f == reps[i]) continue;
    for (auto& elem : boruvka_cc[reps[i]]) boruvka_cc[f].insert(elem);
  }
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
  std::sort(temp.begin(),temp.end());
  std::sort(kruskal_ref.begin(),kruskal_ref.end());
  assert(kruskal_ref == temp);
  std::cout << "Solution ok: " << retval.size() << " CCs found." << std::endl;
}
