#include <map>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>

#include <gutter_tree.h>
#include <standalone_gutters.h>
#include "../include/graph.h"
#include "../include/graph_worker.h"

// static variable for enforcing that only one graph is open at a time
bool Graph::open_graph = false;

Graph::Graph(node_id_t num_nodes): num_nodes(num_nodes) {
  if (open_graph) throw MultipleGraphsException();

#ifdef VERIFY_SAMPLES_F
  std::cout << "Verifying samples..." << std::endl;
#endif
  Supernode::configure(num_nodes);
  representatives = new std::set<node_id_t>();
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
  
  std::tuple<bool, bool, std::string> conf = configure_system(); // read the configuration file to configure the system
  copy_in_mem = std::get<1>(conf);
  std::string disk_loc = std::get<2>(conf);
  backup_file = disk_loc + "supernode_backup.data";
  // Create the buffering system and start the graphWorkers
  if (std::get<0>(conf))
    bf = new GutterTree(disk_loc, num_nodes, GraphWorker::get_num_groups(), true);
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
  std::cout << "Verifying samples..." << std::endl;
#endif
  representatives = new std::set<node_id_t>();
  supernodes = new Supernode*[num_nodes];
  parent = new node_id_t[num_nodes];
  for (node_id_t i = 0; i < num_nodes; ++i) {
    representatives->insert(i);
    supernodes[i] = Supernode::makeSupernode(num_nodes, seed, binary_in);
    parent[i] = i;
  }
  binary_in.close();

  std::tuple<bool, bool, std::string> conf = configure_system(); // read the configuration file to configure the system
  copy_in_mem = std::get<1>(conf);
  std::string disk_loc = std::get<2>(conf);
  backup_file = disk_loc + "supernode_backup.data";
  // Create the buffering system and start the graphWorkers
  if (std::get<0>(conf))
    bf = new GutterTree(disk_loc, num_nodes, GraphWorker::get_num_groups(), true);
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

void Graph::generate_delta_node(node_id_t node_n, long node_seed, node_id_t
               src, const std::vector<size_t> &edges, Supernode *delta_loc) {
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
void Graph::batch_update(node_id_t src, const std::vector<size_t> &edges, Supernode *delta_loc) {
  if (update_locked) throw UpdateLockedException();

  num_updates += edges.size();
  generate_delta_node(supernodes[src]->n, supernodes[src]->seed, src, edges, delta_loc);
  supernodes[src]->apply_delta_update(delta_loc);
}

std::vector<std::set<node_id_t>> Graph::boruvka_emulation() {
  cc_alg_start = std::chrono::steady_clock::now();
  printf("Total number of updates to sketches before CC %lu\n", num_updates.load()); // REMOVE this later
  update_locked = true; // disallow updating the graph after we run the alg
  bool modified;
  std::pair<Edge, SampleSketchRet> query[num_nodes];
  node_id_t size[num_nodes];
  std::vector<node_id_t> reps(num_nodes);
  std::fill(size, size + num_nodes, 1);
  for (node_id_t i = 0; i < num_nodes; ++i) {
    reps[i] = i;
  }

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

    // Run the disjoint set union to determine what supernodes
    // Should be merged together. 
    // Map from nodes to a vector of nodes to merge with them
    std::vector<std::vector<node_id_t>> to_merge(num_nodes);
    std::vector<node_id_t> new_reps;
    for (auto i : reps) {
      // unpack query result
      Edge edge = query[i].first;
      SampleSketchRet ret_code = query[i].second;

      // try this query again next round as it failed this round
      if (ret_code == FAIL) {
        modified = true; 
        new_reps.push_back(i); 
        continue;
      } 
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

      // make a the parent of b
      if (size[a] < size[b]) std::swap(a,b);
      parent[b] = a;
      size[a] += size[b];

      // add b and any of the nodes to merge with it to a's vector
      to_merge[a].push_back(b);
      to_merge[a].insert(to_merge[a].end(), to_merge[b].begin(), to_merge[b].end());
      to_merge[b].clear();
      modified = true;
    }

    std::set<node_id_t> found_nodes;

    for (node_id_t a = 0; a < num_nodes; a++)
      if (to_merge[a].size() != 0) new_reps.push_back(a);

    // loop over the to_merge vector and perform supernode merging
    #pragma omp parallel for default(none) shared(new_reps, to_merge, except, err)
    for (node_id_t a = 0; a < num_nodes; a++) {
      try {
        if(to_merge[a].size() == 0) continue;

        // we have stuff to merge into a, so do that
        for (node_id_t b : to_merge[a]) {
          supernodes[a]->merge(*supernodes[b]);
        }
      } catch (...) {
        except = true;
        err = std::current_exception();
      } 
    }

    // Did one of our threads produce an exception?
    if (except) std::rethrow_exception(err);

    swap(reps, new_reps);
  } while (modified);

  std::map<node_id_t, std::set<node_id_t>> temp;
  for (node_id_t i = 0; i < num_nodes; ++i)
    temp[get_parent(i)].insert(i);
  std::vector<std::set<node_id_t>> retval;
  retval.reserve(temp.size());
  for (const auto& it : temp) retval.push_back(it.second);

  cc_alg_end = std::chrono::steady_clock::now();
  printf("CC done\n");
  return retval;
}

Supernode** Graph::backup_supernodes() {
  std::cout << "Backing up supernodes" << std::endl;
  if (copy_in_mem) {
    // Copy supernodes
    Supernode** copy_supernodes = new Supernode*[num_nodes];
    for (node_id_t i = 0; i < num_nodes; ++i)
      copy_supernodes[i] = Supernode::makeSupernode(*supernodes[i]);  
    return copy_supernodes;
  }
  else {
    // Make a copy on disk
    std::fstream binary_out(backup_file, std::ios::out | std::ios::binary);
    if (!binary_out.is_open()) {
      std::cerr << "Failed to open file for writing backup!" << backup_file << std::endl;
      exit(EXIT_FAILURE);
    }
    for (node_id_t i = 0; i < num_nodes; ++i) {
      supernodes[i]->write_binary(binary_out);
    }
    binary_out.close();
    return nullptr; // backup on disk not in memory
  }

}

void Graph::restore_supernodes(Supernode** copy_supernodes) {
  if (copy_in_mem) {
    // Restore supernodes
    for (node_id_t i=0;i<num_nodes;++i) {
      free(this->supernodes[i]);
      this->supernodes[i] = copy_supernodes[i];
      representatives->insert(i);
      parent[i] = i;
    }
    delete[] copy_supernodes;
  } else {
    // restore from disk
    std::fstream binary_in(backup_file, std::ios::in | std::ios::binary);
    if (!binary_in.is_open()) {
      std::cerr << "Failed to open file for reading backup!" << backup_file << std::endl;
      exit(EXIT_FAILURE);
    }
    for (node_id_t i = 0; i < num_nodes; ++i) {
      free(this->supernodes[i]);
      this->supernodes[i] = Supernode::makeSupernode(num_nodes, seed, binary_in);
      representatives->insert(i);
      parent[i] = i;
    }
  }

  GraphWorker::unpause_workers();
  update_locked = false;
}

std::vector<std::set<node_id_t>> Graph::connected_components(bool cont) {
  flush_call = std::chrono::steady_clock::now();
  bf->force_flush(); // flush everything in buffering system to make final updates
  GraphWorker::pause_workers(); // wait for the workers to finish applying the updates
  flush_return = std::chrono::steady_clock::now();
  // after this point all updates have been processed from the buffer tree

  if (!cont)
    return boruvka_emulation();

  create_backup_start = std::chrono::steady_clock::now();
  Supernode** copy_supernodes = backup_supernodes();
  create_backup_end = std::chrono::steady_clock::now();

  std::vector<std::set<node_id_t>> ret = boruvka_emulation();

  restore_backup_start = std::chrono::steady_clock::now();
  restore_supernodes(copy_supernodes);
  restore_backup_end = std::chrono::steady_clock::now();

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
