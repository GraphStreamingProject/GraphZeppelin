#include "include/k_skeleton.h"

vector<vector<Node>> k_skeleton (istream * updates_stream, int k)
{
  int n, m;
  *updates_stream >> n >> m;
  vector<Graph> instances;
  instances.reserve(k);
  for (int i = 0; i < k; i++)
	  instances.push_back(Graph{n});

  vector<vector<Node>> forest_union(n);

  // Ingest stream
  int type, a, b;
  while (m--) 
  {
    *updates_stream >> type >> a >> b;
    
    for (auto& g : instances)
      g.update({{a, b}, (UpdateType)type});
  }

  for (int i = k - 1; i >= 0; i--)
  {
    auto F_i = instances[i].spanning_forest();
    instances.pop_back();

    // Remove current forest edges from remaining instances 
    for (int j = i - 1; j >= 0; j--)
      for (const auto& span_tree: F_i)
        for (const auto& node_list_pair : span_tree)
          for (const Node& neighbor : node_list_pair.second)
            if (node_list_pair.first < neighbor)
              instances[j].update(
                  {{node_list_pair.first, neighbor}, DELETE});

    // Insert current forest edges into union
    for (const auto& span_tree : F_i)
      for (const auto& node_list_pair : span_tree)
        for (const auto& neighbor : node_list_pair.second)
          forest_union[node_list_pair.first].push_back(neighbor);
  }

  return forest_union;
}

