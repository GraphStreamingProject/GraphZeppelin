#pragma once
#include <graph_stream.h>
#include <graph_zeppelin_common.h>
#include <xxhash.h>

#include <functional>
#include <vector>

typedef uint64_t col_hash_t;
static const auto& vec_hash = XXH3_64bits_withSeed;
static const auto& col_hash = XXH3_64bits_withSeed;

// Graph Stream Updates are parsed into the GraphUpdate type for more convinient processing
struct GraphUpdate {
  Edge edge;
  UpdateType type;
};

struct SubgraphTaggedUpdate {
  node_id_t subgraph;  // highest index subgraph the edge maps to (same src, dst -> same subgraph)
  node_id_t dst;       // destination vertex of edge

  bool operator<(const SubgraphTaggedUpdate& oth) const {
    if (subgraph == oth.subgraph) return dst < oth.dst;

    return subgraph < oth.subgraph;
  }

  bool operator>(const SubgraphTaggedUpdate& oth) const {
    if (subgraph == oth.subgraph) return dst > oth.dst;

    return subgraph > oth.subgraph;
  }
};

struct TaggedUpdateBatch {
  node_id_t src;
  node_id_t edge_store_subgraph;
  std::vector<SubgraphTaggedUpdate> dsts_data;
};
