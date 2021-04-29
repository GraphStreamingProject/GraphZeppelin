#include <map>
#include <iostream>
#include "include/graph.h"

Graph::Graph(uint64_t num_nodes): num_nodes(num_nodes) {
#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  time_t seed = time(nullptr);
  supernodes = new Supernode*[num_nodes];
#ifndef EXT_MEM_POST_PROC_F
  representatives = new set<Node>();
  parent = new Node[num_nodes];
  for (Node i = 0; i < num_nodes; ++i) {
    representatives->insert(i);
    supernodes[i] = new Supernode(num_nodes,seed);
    parent[i] = i;
  }
#else
  for (Node i=0;i<num_nodes;++i) {
    supernodes[i] = new Supernode(num_nodes,seed);
    supernodes[i]->ext_mem_parent_ptr = i;
  }
#endif
}

Graph::~Graph() {
  for (unsigned i=0;i<num_nodes;++i)
    delete supernodes[i];
  delete[] supernodes;
  delete[] parent;
#ifndef EXT_MEM_POST_PROC_F
  delete representatives;
#endif
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
#ifdef EXT_MEM_POST_PROC_F
  return Graph::_ext_mem_connected_components();
#else
  return Graph::_connected_components();
#endif
}

#ifndef EXT_MEM_POST_PROC_F
vector<set<Node>> Graph::_connected_components() {
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
#ifdef VERIFY_SAMPLES_F
  GraphVerifier verifier {cum_in};
#endif
  Node size[num_nodes];
  fill(size,size+num_nodes,1);
  bool destroyed[num_nodes];
  fill(destroyed,destroyed+num_nodes,false);
  do {
    modified = false;
    for (unsigned i=0;i<num_nodes;++i) {
      if (destroyed[i]) continue;
      if (parent[i] != i) destroyed[i] = true;
      boost::optional<Edge> edge = supernodes[i]->sample();
      if (!edge.is_initialized()) {
#ifdef VERIFY_SAMPLES_F
        verifier.verify_cc(i);
#endif
        continue;
      }
      Node a = get_parent(edge->first);
      Node b = get_parent(edge->second);
      if (a == b) continue;
#ifdef VERIFY_SAMPLES_F
      verifier.verify_edge(edge.value());
#endif
      if (size[a] < size[b]) std::swap(a,b);
      // DSU compression
      parent[b] = a;
      size[a] += size[b];
      modified = true;
      supernodes[a]->merge(*supernodes[b]);
      if (b <= i) destroyed[b] = true;
    }
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
#endif // ndef EXT_MEM_POST_PROC_F

#ifdef EXT_MEM_POST_PROC_F
/**
 * Runs boruvka and DSU in external memory. Does a final pass through all
 * sketches (using O(n) memory) to collect connected components.
 * TODO: an iterative merging collection scheme in external memory
 */
vector<set<Node>> Graph::_ext_mem_connected_components() {
#ifdef WODS_PROTOTYPE
  db->flush(); // flush everything in toku to make final updates
#endif
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
#ifdef VERIFY_SAMPLES_F
  GraphVerifier verifier {cum_in};
#endif

  do {
    modified = false;
    for (unsigned i = 0; i < num_nodes; ++i) {
      if (supernodes[i]->ext_mem_destroyed) continue;
      if (supernodes[i]->ext_mem_parent_ptr != i)
        supernodes[i]->ext_mem_destroyed = true;
      boost::optional<Edge> edge = supernodes[i]->sample();
      if (!edge.is_initialized()) {
#ifdef VERIFY_SAMPLES_F
        verifier.verify_cc(i);
#endif
        continue;
      }
      Node a = ext_mem_get_parent(edge->first);
      Node b = ext_mem_get_parent(edge->second);
      if (a == b) continue;
#ifdef VERIFY_SAMPLES_F
      verifier.verify_edge(edge.value());
#endif
      if (supernodes[a]->ext_mem_size < supernodes[b]->ext_mem_size)
        std::swap(a,b);
      // DSU compression
      supernodes[b]->ext_mem_parent_ptr = a;
      supernodes[a]->ext_mem_size += supernodes[b]->ext_mem_size;
      modified = true;
      supernodes[a]->merge(*supernodes[b]);
      if (b <= i) supernodes[b]->ext_mem_destroyed = true;
    }
  } while (modified);
  map<Node, set<Node>> temp;
  for (Node i=0;i<num_nodes;++i)
    temp[ext_mem_get_parent(i)].insert(i);
  vector<set<Node>> retval;
  retval.reserve(temp.size());
  for (const auto& it : temp) retval.push_back(it.second);
#ifdef VERIFY_SAMPLES_F
  verifier.verify_soln(retval);
#endif
  return retval;
}

Node Graph::ext_mem_get_parent(Node node) {
  if (supernodes[node]->ext_mem_parent_ptr == node) {
    return node;
  }
  return supernodes[node]->ext_mem_parent_ptr =
               ext_mem_get_parent(supernodes[node]->ext_mem_parent_ptr);
}
#endif // EXT_MEM_POST_PROC_F