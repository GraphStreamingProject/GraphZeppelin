#include <map>
#include <iostream>
#include <buffer_tree.h>

#include "include/graph.h"
#include "include/util.h"
#include "include/graph_worker.h"

Graph::Graph(uint64_t num_nodes): num_nodes(num_nodes) {
#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples... testing" << endl;
#endif
  representatives = new set<Node>();
  supernodes = new Supernode*[num_nodes];
  parent = new Node[num_nodes];
  seed = time(nullptr);
  //seed = 38382;
  for (Node i=0;i<num_nodes;++i) {
    representatives->insert(i);
    supernodes[i] = new Supernode(num_nodes,seed);
    parent[i] = i;
  }
  std::string buffer_loc_prefix = configure_system(); // read the configuration file to configure the system
#ifdef USE_FBT_F
  // Create buffer tree and start the graphWorkers
  bf = new BufferTree(buffer_loc_prefix, (1<<20), 16, num_nodes, GraphWorker::get_num_groups(), true);
  GraphWorker::start_workers(this, bf);
#else
  unsigned long node_size = 24*pow((log2(num_nodes)), 3);
  node_size /= sizeof(Node);
  wq = new WorkQueue(node_size, num_nodes, 2*GraphWorker::get_num_groups());
  GraphWorker::start_workers(this, wq);
#endif
}

Graph::~Graph() {
  for (unsigned i=0;i<num_nodes;++i)
    delete supernodes[i];
  delete[] supernodes;
  delete[] parent;
  delete representatives;
  GraphWorker::stop_workers(); // join the worker threads
#ifdef USE_FBT_F
  delete bf;
#else
  delete wq;
#endif
}

void Graph::ingest_graph(std::string path)
{
  ifstream in{path};
  Node num_nodes;
  in >> num_nodes;
  long m;
  in >> m;
  Node a, b;
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < m; ++i) {
    in >> a >> b;
    update({{a, b}, UpdateType::INSERT});
    if (i % 1000000 == 0)
      {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = now - start;
        std::cout << i << " of " << m << " processed, running at " << i / (elapsed_seconds.count() * 1e6) << " million edges per second" << std::endl;
      }
  }
  //std::cout << connected_components().size() << std::endl;
}


void Graph::update(GraphUpdate upd) {
  if (update_locked) throw UpdateLockedException();
  Edge &edge = upd.edge;

#ifdef USE_FBT_F
  bf->insert(edge);
  std::swap(edge.first, edge.second);
  bf->insert(edge);
#else
  wq->insert(edge);
  std::swap(edge.first, edge.second);
  wq->insert(edge);
#endif
}

std::vector<vec_t> Graph::make_updates(uint64_t src, const std::vector<uint64_t>& edges) {
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
  return updates;
}

const std::vector<Sketch> &Graph::get_supernode_sketches(uint64_t src) const
{
      	return supernodes[src]->get_sketches();
}

void Graph::apply_supernode_deltas(uint64_t src, const std::vector<Sketch>& deltas)
{
  supernodes[src]->apply_deltas(deltas);
}

void Graph::batch_update(uint64_t src, const std::vector<uint64_t>& edges) {
  if (update_locked) throw UpdateLockedException();
  supernodes[src]->batch_update(make_updates(src, edges));
}

/*void Graph::wait_for_completion()
{
  if (use_external_worker)
  {
    while (!external_workers_completed);
  }
  else
  {
    GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  }
}*/


vector<set<Node>> Graph::connected_components() {
#ifdef USE_FBT_F
  bf->force_flush(); // flush everything in buffertree to make final updates
#else
  wq->force_flush();
#endif
  GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  // after this point all updates have been processed from the buffer tree
  end_time = std::chrono::steady_clock::now();
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
  //std::cout << "Work queue completely dealt with" << std::endl;
  do {
    modified = false;
    vector<Node> removed;
    for (Node i: (*representatives)) {
      if (parent[i] != i) continue;
      boost::optional<Edge> edge = supernodes[i]->sample();
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
  CC_end_time = std::chrono::steady_clock::now();
  //std::cout << "CC completed" << std::endl;
  return retval;
}

Node Graph::get_parent(Node node) {
  if (parent[node] == node) return node;
  return parent[node] = get_parent(parent[node]);
}
