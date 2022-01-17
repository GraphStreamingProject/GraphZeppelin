#include <map>
#include <iostream>
#include <chrono>
#include <random>

#include <gutter_tree.h>
#include <standalone_gutters.h>
#include "../include/graph.h"
#include "../include/graph_worker.h"

// static variable for enforcing that only one graph is open at a time
bool Graph::open_graph = false;

Graph::Graph(node_id_t num_nodes): num_nodes(num_nodes) {
  if (open_graph) throw MultipleGraphsException();

#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  Supernode::configure(num_nodes);
  representatives = new set<node_id_t>();
  supernodes = new Supernode*[num_nodes];
  parent = new node_id_t[num_nodes];
  seed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  std::mt19937_64 r(seed);
  seed = r();

  for (node_id_t i = 0; i < num_nodes; ++i) {
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
  open_graph = true;
}

Graph::Graph(const std::string& input_file) : num_updates(0) {
  if (open_graph) throw MultipleGraphsException();
  
  int sketch_fail_factor;
  auto binary_in = std::fstream(input_file, std::ios::in | std::ios::binary);
  binary_in.read((char*)&seed, sizeof(long));
  binary_in.read((char*)&num_nodes, sizeof(uint64_t));
  binary_in.read((char*)&sketch_fail_factor, sizeof(int));
  Supernode::configure(num_nodes, sketch_fail_factor);

#ifdef VERIFY_SAMPLES_F
  cout << "Verifying samples..." << endl;
#endif
  representatives = new set<node_id_t>();
  supernodes = new Supernode*[num_nodes];
  parent = new node_id_t[num_nodes];
  for (node_id_t i = 0; i < num_nodes; ++i) {
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
  open_graph = true;
}

Graph::~Graph() {
  for (unsigned i=0;i<num_nodes;++i)
    free(supernodes[i]); // free because memory is malloc'd in make_supernode
  delete[] supernodes;
  delete[] parent;
  delete representatives;
  GraphWorker::stop_workers(); // join the worker threads
  delete bf;
  open_graph = false;
}

void Graph::update(GraphUpdate upd) {
  if (update_locked) throw UpdateLockedException();
  Edge &edge = upd.first;

  bf->insert(edge);
  std::swap(edge.first, edge.second);
  bf->insert(edge);
}

void Graph::generate_delta_node(node_id_t node_n, long node_seed, node_id_t
               src, const vector<node_id_t> &edges, Supernode *delta_loc) {
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
void Graph::batch_update(node_id_t src, const vector<node_id_t> &edges, Supernode *delta_loc) {
  if (update_locked) throw UpdateLockedException();

  num_updates += edges.size();
  generate_delta_node(supernodes[src]->n, supernodes[src]->seed, src, edges, delta_loc);
  supernodes[src]->apply_delta_update(delta_loc);
}

vector<set<node_id_t>> Graph::connected_components() {
  // cc_start_time = std::chrono::steady_clock::now();
  bf->force_flush(); // flush everything in buffering system to make final updates
  GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  // cc_flush_end_time = std::chrono::steady_clock::now();

  // after this point all updates have been processed from the buffer tree
  printf("Total number of updates to sketches before CC %lu\n", num_updates.load()); // REMOVE this later
  update_locked = true; // disallow updating the graph after we run the alg

  cc_start_time = std::chrono::steady_clock::now();
  bool modified;
  std::pair<Edge, SampleSketchRet> query[num_nodes];
  node_id_t size[num_nodes];
  vector<node_id_t> reps(num_nodes);
  fill(size, size + num_nodes, 1);
  for (node_id_t i = 0; i < num_nodes; ++i) {
    reps[i] = i;
  }
  
  int num_rounds = 0;

  do {
    modified = false;
    bool except = false;
    std::exception_ptr err;

    #pragma omp parallel for default(none) shared(query, reps, except, err, modified)
    for (node_id_t i = 0; i < reps.size(); ++i) { // NOLINT(modernize-loop-convert)
      // wrap in a try/catch because exiting through exception is undefined behavior in OMP
      try {
        query[reps[i]] = supernodes[reps[i]]->sample();

      } catch (...) {
        except = true;
        err = std::current_exception();
      }
    }
    // Did one of our threads produce an exception?
    if (except) std::rethrow_exception(err);

    vector<node_id_t> to_remove;
    for (auto i : reps) {
      // unpack query result
      Edge edge = query[i].first;
      SampleSketchRet ret_code = query[i].second;

      // try this query again next round as it failed this round
      if (ret_code == FAIL) {modified = true; continue;} 
      if (ret_code == ZERO) {
#ifdef VERIFY_SAMPLES_F
        verifier->verify_cc(i);
#endif
        continue;
      }

      // query dsu
      node_id_t a = get_parent(edge.first);
      node_id_t b = get_parent(edge.second);
      if (a == b) continue;

#ifdef VERIFY_SAMPLES_F
      verifier->verify_edge(edge);
#endif

      // make sure a is the one to be merged into and merge
      if (size[a] < size[b]) std::swap(a,b);
      to_remove.push_back(b);
      parent[b] = a;
      size[a] += size[b];
      supernodes[a]->merge(*supernodes[b]);
    }
    if (!to_remove.empty()) modified = true;
    sort(to_remove.begin(), to_remove.end());

    // 2-pointer to find set difference
    vector<node_id_t> temp_diff;
    unsigned long ptr1 = 0;
    unsigned long ptr2 = 0;
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
    ++num_rounds;
  } while (modified);

  map<node_id_t, set<node_id_t>> temp;
  for (node_id_t i = 0; i < num_nodes; ++i)
    temp[get_parent(i)].insert(i);
  vector<set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto& it : temp) retval.push_back(it.second);

  cc_end_time = std::chrono::steady_clock::now();
  printf("CC done in %d rounds\n", num_rounds);
  return retval;
}

Supernode** Graph::backup_supernodes() {
  cc_flush_start_time = std::chrono::steady_clock::now();
  bf->force_flush(); // flush everything in buffering system to make final updates
  GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  cc_flush_end_time = std::chrono::steady_clock::now();

  // Copy supernodes
  Supernode** supernodes = new Supernode*[num_nodes];
  for (node_id_t i = 0; i < num_nodes; ++i) {
    supernodes[i] = Supernode::makeSupernode(*this->supernodes[i]);
  }

  return supernodes;
}

void Graph::restore_supernodes(Supernode** supernodes) {
  // Restore supernodes
  for (node_id_t i=0;i<num_nodes;++i) {
    free(this->supernodes[i]);
    this->supernodes[i] = supernodes[i];
    representatives->insert(i);
    parent[i] = i;
  }
  delete[] supernodes;

  GraphWorker::unpause_workers();
  update_locked = false;
}

vector<set<node_id_t>> Graph::connected_components(bool cont) {
  if (!cont)
    return connected_components();

  Supernode** supernodes = backup_supernodes();

  bool except = false;
  std::exception_ptr err;
  vector<set<node_id_t>> ret;

  try {
    ret = connected_components();
  } catch (...) {
    except = true;
    err = std::current_exception();
  }
  restore_supernodes(supernodes);

  // Did one of our threads produce an exception?
  if (except) std::rethrow_exception(err);

  return ret;
}

node_id_t Graph::get_parent(node_id_t node) {
  if (parent[node] == node) return node;
  return parent[node] = get_parent(parent[node]);
}

void Graph::write_binary(const std::string& filename) {
  bf->force_flush(); // flush everything in buffering system to make final updates
  GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  // after this point all updates have been processed from the buffering system

  auto binary_out = std::fstream(filename, std::ios::out | std::ios::binary);
  int fail_factor = Sketch::get_failure_factor();
  binary_out.write((char*)&seed, sizeof(long));
  binary_out.write((char*)&num_nodes, sizeof(uint64_t));
  binary_out.write((char*)&fail_factor, sizeof(int));
  for (node_id_t i = 0; i < num_nodes; ++i) {
    supernodes[i]->write_binary(binary_out);
  }
  binary_out.close();
}
