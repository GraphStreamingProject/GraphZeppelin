#include <map>
#include <unordered_map>
#include <iostream>
#include "include/graph.h"

Graph::Graph(uint64_t num_nodes): num_nodes(num_nodes) {
#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  // Each representative indexes the number of elements in its 
  // supernode.
  representatives = new map<Node, size_t>(); 
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
  delete[] supernodes;
  delete[] parent;
  delete representatives;
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

//vector<set<Node>> Graph::connected_components() {
//  update_locked = true; // disallow updating the graph after we run the alg
//  bool modified;
//#ifdef VERIFY_SAMPLES_F
//  GraphVerifier verifier {cum_in};
//#endif
//  do {
//    modified = false;
//    vector<Node> removed;
//    for (Node i: (*representatives)) {
//      if (parent[i] != i) continue;
//      boost::optional<Edge> edge = supernodes[i]->sample();
//#ifdef VERIFY_SAMPLES_F
//      if (edge.is_initialized())
//        verifier.verify_edge(edge.value());
//      else
//        verifier.verify_cc(i);
//#endif
//      if (!edge.is_initialized()) continue;
//
//      Node n;
//      // DSU compression
//      if (get_parent(edge->first) == i) {
//        n = get_parent(edge->second);
//        removed.push_back(n);
//        parent[n] = i;
//      }
//      else {
//        get_parent(edge->second);
//        n = get_parent(edge->first);
//        removed.push_back(n);
//        parent[n] = i;
//      }
//      supernodes[i]->merge(*supernodes[n]);
//    }
//    if (!removed.empty()) modified = true;
//    for (Node i : removed) representatives->erase(i);
//  } while (modified);
//  map<Node, set<Node>> temp;
//  for (Node i=0;i<num_nodes;++i)
//    temp[get_parent(i)].insert(i);
//  vector<set<Node>> retval;
//  retval.reserve(temp.size());
//  for (const auto& it : temp) retval.push_back(it.second);
//#ifdef VERIFY_SAMPLES_F
//  verifier.verify_soln(retval);
//#endif
//  return retval;
//}

vector<unordered_map<Node, vector<Node>>> Graph::spanning_forest()
{
  update_locked = true; // disallow updating the graph after we run the alg
  // During Boruvka, if a merge occurs between two supernodes as a
  // consequence of sampling edge (i, j), where i is in the supernode
  // which absorbs the supernode containing j, then append j to the ith vector
  // in this structure.
  vector<vector<Node>> merge_points(num_nodes);

  bool modified;
  do {
    modified = false;
    vector<Node> removed;
    for (Node i: (*representatives)) {
      if (parent[i] != i) continue; //only one edge per cut sampled
      //We sample this node up to potentially
      boost::optional<Edge> oedge = supernodes[i]->sample();
      if (!oedge) continue;
      Edge edge = *oedge;

      Node n;
      // DSU compression
      if (get_parent(edge.first) == i) { // Current node i wins 
        n = get_parent(edge.second);
        removed.push_back(n);
        parent[n] = i;
      }
      else {
        get_parent(edge.second);
        n = get_parent(edge.first);
        removed.push_back(n);
        parent[n] = i; // Shouldnt it be reversed?
      }
      // Ensures sampling occurs along supernode cuts
      supernodes[i]->merge(*supernodes[n]);

      merge_points[edge.first].push_back(edge.second);
    }
    if (!removed.empty()) modified = true;
    for (Node i : removed) representatives->erase(i);
  } while (modified);

  // Maps each represntative (or root) of a connected component to
  // the adjacency list for that connected component 
  unordered_map<Node i, unordered_map<Node, vector<Node>>> 
  	root_adj_list (representatives->size());
  
  // Avoid rehashing of adjacency lists during upcoming insertions
  for (const auto& root_size_pair : *representatives)
    root_adj_list[root_size_pair.first].reserve(
		    root_size_pair.second);

  // Insert edges from merge_points
  for (int i = 0; i < num_nodes; i++)
  {
    for (const Node& span_neighbor : merge_points[i])
    {
      root_adj_list[parents[i]][i].second.push_back(span_neighbor);
      // Include the symmetric edge
      root_adj_list[parents[i]][span_neighbor].second.push_back(i);
    }
  }

  vector<unordered_map<Node, vector<Node>>> 
	  retval(representatives->size());

  for (const auto& root_adj_list_pair : root_adj_list)
	  retval.push_back(root_adj_list_pair.second);

  return retval;
}

bool Graph::is_k_edge_connected (int k)
{
	// TODO: Should we leave the original instance of the sketch
	// unaltered? It consumes (k+1)/k times more
	// memory, but it leaves the original sketch unaltered in
	// case the user wishes to conduct another algorithm.
	vector<Graph> instances(k-1, *this);
	auto F_0 = this->spanning_forest();
	if (F_0.size() > 1) return false;
	for(Graph& g_i : instances)
		for (const Edge& e : F_0[0].second)
			g_i.update({e, DELETE});

	for (int i = 1; i < k - 1; i++)
	{
		auto F_i = instances[i-1].spanning_forest();

		if (F_i.size() > 1) return false;

		for (int j = i; j < k; j++)
			for (const Edge& e : F_i[0].second)
				instances[j].update({e, DELETE});
	}

	return instances[k].spanning_forest().size() < 2;
}

Node Graph::get_parent(Node node) {
  if (parent[node] == node) return node;
  return parent[node] = get_parent(parent[node]);
}
