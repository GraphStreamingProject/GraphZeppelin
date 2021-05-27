#include <unordered_map>
#include <iostream>
#include "include/graph.h"
#include "include/util.h"

Graph::Graph(uint64_t num_nodes): num_nodes(num_nodes), dsu(num_nodes) {
  representatives = new set<Node>();
  supernodes = new Supernode*[num_nodes];
  time_t seed = time(nullptr);
  for (Node i=0;i<num_nodes;++i) {
    representatives->insert(i);
    supernodes[i] = new Supernode(num_nodes,seed);
  }
}

Graph::~Graph() {
  for (unsigned i=0;i<num_nodes;++i)
    delete supernodes[i];
  delete[] supernodes;
  delete representatives;
}

void Graph::update(GraphUpdate upd) {
  if (update_locked) throw UpdateLockedException();
  Edge &edge = upd.first;
  // ensure lhs < rhs
  if (edge.first > edge.second) {
    std::swap(edge.first,edge.second);
  }
  vec_t encoded = nondirectional_non_self_edge_pairing_fn(edge.first, edge.second);
  supernodes[edge.first]->update(encoded);
  supernodes[edge.second]->update(encoded);
}


void Graph::batch_update(uint64_t src, const std::vector<uint64_t>& edges) {
  if (update_locked) throw UpdateLockedException();
  std::vector<vec_t> updates;
  updates.reserve(edges.size());
  for (const auto& edge : edges) {
    if (src < edge) {
      updates.push_back(static_cast<vec_t>(
          nondirectional_non_self_edge_pairing_fn(src, edge)));
    } else {
      updates.push_back(static_cast<vec_t>(
          nondirectional_non_self_edge_pairing_fn(edge, src)));
    }
  }
  supernodes[src]->batch_update(updates);
}

std::vector<std::pair<std::vector<Node>, std::vector<Edge>>>
      Graph::connected_components() {
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
  std::vector<Edge> forest;
  do {
    modified = false;
    std::vector<Node> removed;
    for (Node i: (*representatives)) {
      if (!dsu.is_rep(i)) continue;
      boost::optional<Edge> edge = supernodes[i]->sample();
      if (!edge.is_initialized()) continue;

      forest.push_back(edge.value());

      Node n = dsu.find(edge->first);
      // DSU compression
      if (n == i) {
        n = dsu.find(edge->second);
      } else {
        dsu.find(edge->second);
      }
      removed.push_back(n);
      dsu.link(n, i);
      supernodes[i]->merge(*supernodes[n]);
    }
    if (!removed.empty()) modified = true;
    for (Node i : removed) representatives->erase(i);
  } while (modified);

  // Map parent to returned vector index
  std::unordered_map<Node, Node> nodemap;
  nodemap.reserve(num_nodes);
  Node num_comp = 0;
  std::vector<std::pair<std::vector<Node>, std::vector<Edge>>> retval;
  for (Node i = 0; i < num_nodes; i++) {
    Node parent = dsu.find(i);
    if (nodemap.find(parent) == nodemap.end()) {
      nodemap[parent] = num_comp++;
      retval.emplace_back(std::vector<Node>(), std::vector<Edge>());
    }
    retval[nodemap[parent]].first.push_back(i);
  }
  for (const Edge& edge : forest) {
    if (dsu.find(edge.first) != dsu.find(edge.second)) {
      std::cerr << "Something weird happened with edge DSU" << std::endl;
    }
    retval[nodemap[dsu.find(edge.first)]].second.push_back(edge);
  }

  return retval;
}
