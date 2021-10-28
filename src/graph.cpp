#include <map>
#include <iostream>
#include <chrono>
#include <random>

#include <gutter_tree.h>
#include <standalone_gutters.h>
#include "../include/graph.h"
#include "../include/graph_worker.h"

Graph::Graph(uint64_t num_nodes): num_nodes(num_nodes) {
#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  Supernode::configure(num_nodes);
  representatives = new set<node_t>();
  supernodes = new Supernode*[num_nodes];
  parent = new node_t[num_nodes];
  seed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  std::mt19937_64 r(seed);
  seed = r();

  for (node_t i=0;i<num_nodes;++i) {
    representatives->insert(i);
    supernodes[i] = Supernode::makeSupernode(num_nodes,seed);
    parent[i] = i;
  }
  num_updates = 0; // REMOVE this later
  
  std::pair<bool, std::string> conf = configure_system(); // read the configuration file to configure the system
  std::string buffer_loc_prefix = conf.second;
  // Create the buffering system and start the graphWorkers
  if (conf.first)
    bf = new GutterTree(buffer_loc_prefix, num_nodes, GraphWorker::get_num_groups(), true);
  else
    bf = new StandAloneGutters(num_nodes, GraphWorker::get_num_groups());

  GraphWorker::start_workers(this, bf, Supernode::get_size());
}

Graph::Graph(const std::string& input_file) : num_updates(0) {
  double num_bucket_factor;
  auto binary_in = std::fstream(input_file, std::ios::in | std::ios::binary);
  binary_in.read((char*)&seed, sizeof(long));
  binary_in.read((char*)&num_nodes, sizeof(uint64_t));
  binary_in.read((char*)&num_bucket_factor, sizeof(double));
  Supernode::configure(num_nodes, num_bucket_factor);

#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  representatives = new set<node_t>();
  supernodes = new Supernode*[num_nodes];
  parent = new node_t[num_nodes];
  for (node_t i = 0; i < num_nodes; ++i) {
    representatives->insert(i);
    supernodes[i] = Supernode::makeSupernode(num_nodes, seed, binary_in);
    parent[i] = i;
  }
  binary_in.close();

  std::pair<bool, std::string> conf = configure_system(); // read the configuration file to configure the system
  std::string buffer_loc_prefix = conf.second;
  // Create the buffering system and start the graphWorkers
  if (conf.first)
    bf = new GutterTree(buffer_loc_prefix, num_nodes, GraphWorker::get_num_groups(), true);
  else
    bf = new StandAloneGutters(num_nodes, GraphWorker::get_num_groups());

  GraphWorker::start_workers(this, bf, Supernode::get_size());
}

Graph::~Graph() {
  for (unsigned i=0;i<num_nodes;++i)
    free(supernodes[i]); // free because memory is malloc'd in make_supernode
  delete[] supernodes;
  delete[] parent;
  delete representatives;
  GraphWorker::stop_workers(); // join the worker threads
  delete bf;
}

void Graph::update(GraphUpdate upd) {
  if (update_locked) throw UpdateLockedException();
  Edge &edge = upd.first;

  bf->insert(edge);
  std::swap(edge.first, edge.second);
  bf->insert(edge);
}

void Graph::generate_delta_node(uint64_t node_n, long node_seed, uint64_t
               src, const std::vector<uint64_t>& edges, Supernode *delta_loc) {
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
  Supernode::delta_supernode(node_n, node_seed, updates, delta_loc);
}
void Graph::batch_update(uint64_t src, const std::vector<uint64_t>& edges, Supernode *delta_loc) {
  if (update_locked) throw UpdateLockedException();

  num_updates += edges.size();
  generate_delta_node(supernodes[src]->n, supernodes[src]->seed, src, edges, delta_loc);
  supernodes[src]->apply_delta_update(delta_loc);
}

vector<set<node_t>> Graph::connected_components() {
  bf->force_flush(); // flush everything in buffering system to make final updates
  GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  // after this point all updates have been processed from the buffer tree
  end_time = std::chrono::steady_clock::now();
  printf("Total number of updates to sketches before CC %lu\n", num_updates.load()); // REMOVE this later
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
#ifdef VERIFY_SAMPLES_F
  GraphVerifier verifier {cumul_in };
#endif
  pair<node_t,node_t> query[num_nodes];
  node_t size[num_nodes];
  vector<node_t> reps(num_nodes);
  fill(size, size + num_nodes, 1);
  for (node_t i = 0; i < num_nodes; ++i) {
    reps[i] = i;
  }

  do {
    modified = false;
    bool except = false;
    std::exception_ptr err;
    #pragma omp parallel for default(none) shared(query, reps, except, err)
    for (node_t i = 0; i < reps.size(); ++i) {
      // wrap in a try/catch because exiting through exception is undefined behavior in OMP
      boost::optional<Edge> edge;
      try {
        edge = supernodes[reps[i]]->sample();
      } catch (...) {
        except = true;
        err = std::current_exception();
      }
      if (!edge.is_initialized()) {
        query[reps[i]] = {i,i};
        continue;
      }
      query[reps[i]] = edge.get();
    }

    // Did one of our threads produce an exception?
    if (except) std::rethrow_exception(err);

    vector<node_t> to_remove;
    for (node_t i : reps) {
      node_t a = get_parent(query[i].first);
      node_t b = get_parent(query[i].second);
      if (a == b) continue;
#ifdef VERIFY_SAMPLES_F
      verifier.verify_edge({query[i].first,query[i].second});
#endif

      // make sure a is the one to be merged into
      if (size[a] < size[b]) std::swap(a,b);
      to_remove.push_back(b);
      parent[b] = a;
      size[a] += size[b];
      supernodes[a]->merge(*supernodes[b]);
    }
    if (!to_remove.empty()) modified = true;
    sort(to_remove.begin(), to_remove.end());

    // 2-pointer to find set difference
    vector<node_t> temp_diff;
    node_t ptr1 = 0;
    node_t ptr2 = 0;
    while (ptr1 < reps.size() && ptr2 < to_remove.size()) {
      if (reps[ptr1] == to_remove[ptr2]) {
        ++ ptr1; ++ptr2;
      } else {
        temp_diff.push_back(reps[ptr1]);
        ++ptr1;
      }
    }
    while (ptr1 < reps.size()) {
      temp_diff.push_back(reps[ptr1]);
      ++ptr1;
    }

    swap(reps, temp_diff);
  } while (modified);

  map<node_t, set<node_t>> temp;
  for (node_t i=0;i<num_nodes;++i)
    temp[get_parent(i)].insert(i);
  vector<set<node_t>> retval;
  retval.reserve(temp.size());
  for (const auto& it : temp) retval.push_back(it.second);

  printf("CC done\n");
  return retval;
}

void Graph::post_cc_resume() {
  GraphWorker::unpause_workers();
  update_locked = false;
}

node_t Graph::get_parent(node_t node) {
  if (parent[node] == node) return node;
  return parent[node] = get_parent(parent[node]);
}

void Graph::write_binary(const std::string& filename) {
  bf->force_flush(); // flush everything in buffering system to make final updates
  GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  // after this point all updates have been processed from the buffer tree

  auto binary_out = std::fstream(filename, std::ios::out | std::ios::binary);
  double sketch_factor = Sketch::get_bucket_factor();
  binary_out.write((char*)&seed, sizeof(long));
  binary_out.write((char*)&num_nodes, sizeof(uint64_t));
  binary_out.write((char*)&sketch_factor, sizeof(double));
  for (node_t i = 0; i < num_nodes; ++i) {
    supernodes[i]->write_binary(binary_out);
  }
  binary_out.close();
}
