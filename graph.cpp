#include <map>
#include <iostream>
#include "include/graph.h"

Graph::Graph(uint64_t num_nodes): num_nodes(num_nodes), supernodes(num_nodes),
    parent(num_nodes) {
#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  time_t seed = time(nullptr);
  for (Node i=0;i<num_nodes;++i) {
    representatives.insert(i);
    parent[i] = i;
    supernodes[i] = std::unique_ptr<Supernode>(new Supernode(num_nodes, seed));
  }
}

void Graph::update(GraphUpdate upd) {
  if (update_locked) throw UpdateLockedException();
  Edge &edge = upd.first;
  // ensure lhs < rhs
  if (edge.first > edge.second) {
    std::swap(edge.first,edge.second);
  }
  supernodes[edge.first]->update(edge);
  supernodes[edge.second]->update(edge);
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

vector<set<Node>> Graph::connected_components() {
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
#ifdef VERIFY_SAMPLES_F
  GraphVerifier verifier {cum_in};
#endif
  do {
    modified = false;
    vector<Node> removed;
    for (Node i: representatives) {
      if (parent[i] != i) continue;
      boost::optional<Edge> edge = supernodes[i]->sample();
#ifdef VERIFY_SAMPLES_F
      if (edge.is_initialized())
        verifier.verify_edge(edge.value());
      else
        verifier.verify_cc(i);
#endif
      if (!edge.is_initialized()) continue;

      Node n;
      // DSU compression
      if (get_parent(edge->first) == i) {
        n = get_parent(edge->second);
        removed.push_back(n);
        parent[n] = i;
      }
      else {
        get_parent(edge->second);
        n = get_parent(edge->first);
        removed.push_back(n);
        parent[n] = i;
      }
      supernodes[i]->merge(*supernodes[n]);
    }
    if (!removed.empty()) modified = true;
    for (Node i : removed) representatives.erase(i);
  } while (modified);
  map<Node, set<Node>> temp;
  for (Node i=0;i<num_nodes;++i)
    temp[get_parent(i)].insert(i);
  vector<set<Node>> retval;
  retval.reserve(temp.size());
  for (const auto& it : temp) retval.push_back(it.second);
#ifdef VERIFY_SAMPLES_F
  verifier.verify_soln(retval);
#endif
  return retval;
}

Node Graph::get_parent(Node node) {
  if (parent[node] == node) return node;
  return parent[node] = get_parent(parent[node]);
}
