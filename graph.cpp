#include <map>
#include <iostream>
#include "include/graph.h"

Graph::Graph(uint64_t num_nodes): num_nodes(num_nodes) {
#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  representatives = new set<Node>();
  supernodes = new Supernode*[num_nodes];
  parent = new Node[num_nodes];
  time_t seed = time(nullptr);
  for (Node i=0;i<num_nodes;++i) {
    representatives->insert(i);
    supernodes[i] = new Supernode(num_nodes,seed);
    parent[i] = i;
  }
}

Graph::~Graph() {
  for (unsigned i=0;i<num_nodes;++i)
    delete supernodes[i];
  delete supernodes;
  delete representatives;
}

void Graph::update(GraphUpdate upd) {
  if (update_locked) throw UpdateLockedException();
  Edge &edge = upd.first;
  // ensure lhs < rhs
  if (edge.first > edge.second) {
    std::swap(edge.first,edge.second);
  }
  if (upd.second == INSERT) {
    supernodes[edge.first]->update({edge, 1});
    supernodes[edge.second]->update({edge, -1});
  } else {
    supernodes[edge.first]->update({edge, -1});
    supernodes[edge.second]->update({edge, 1});
  }
}

vector<set<Node>> Graph::connected_components() {
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
#ifdef VERIFY_SAMPLES_F
  GraphVerifier verifier {}; // default verifier
#endif
  do {
    modified = false;
    vector<Node> removed;
    for (Node i: (*representatives)) {
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
    for (Node i : removed) representatives->erase(i);
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
