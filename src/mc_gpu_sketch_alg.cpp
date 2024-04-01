#include "mc_gpu_sketch_alg.h"

#include <iostream>
#include <vector>

void MCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (MCSketchAlg::get_update_locked()) throw UpdateLockedException();

  int stream_id = thr_id * stream_multiplier;
  int stream_offset = 0;
  while(true) {
    if (cudaStreamQuery(streams[stream_id + stream_offset].stream) == cudaSuccess) {
      // Update stream_id
      stream_id += stream_offset;

      // CUDA Stream is available, check if it has any delta sketch
      if(streams[stream_id].delta_applied == 0) {

        if (streams[stream_id].num_graphs < 0 || streams[stream_id].num_graphs > num_sketch_graphs ) {
          std::cout << "Stream #" << stream_id << ": invalid num_graphs! " << streams[stream_id].num_graphs << "\n";
        }

        for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {

          if(trim_enabled) {
            graph_id = trim_graph_id;
          }
          
          Bucket* delta_buckets = new Bucket[num_buckets];
          for (size_t i = 0; i < num_buckets; i++) {
            delta_buckets[i].alpha = cudaUpdateParams[graph_id]->h_bucket_a[(stream_id * num_buckets) + i];
            delta_buckets[i].gamma = cudaUpdateParams[graph_id]->h_bucket_c[(stream_id * num_buckets) + i];
          }

          int prev_src = streams[stream_id].src_vertex;
          
          if(prev_src == -1) {
            std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
          }

          // Apply the delta sketch
          apply_raw_buckets_update((graph_id * num_nodes) + prev_src, delta_buckets);
          free(delta_buckets);
        }
        streams[stream_id].delta_applied = 1;
        streams[stream_id].src_vertex = -1;
        streams[stream_id].num_graphs = -1;
      }
      else {
        if (streams[stream_id].src_vertex != -1) {
          std::cout << "Stream #" << stream_id << ": not applying but has delta sketch: " << streams[stream_id].src_vertex << " deltaApplied: " << streams[stream_id].delta_applied << "\n";
        }
      }
      break;
    }
    stream_offset++;
    if (stream_offset == stream_multiplier) {
        stream_offset = 0;
    }
  }

  int start_index = stream_id * batch_size;
  std::vector<int> sketch_update_size;
  sketch_update_size.assign(num_sketch_graphs, 0);
  int max_depth = 0;

  if(trim_enabled) { // Trimming sketch subgraphs
    if (trim_graph_id < 0 || trim_graph_id > num_sketch_graphs) {
      std::cout << "INVALID trim_graph_id: " << trim_graph_id << "\n";
    }

    int count = 0;
    for (vec_t i = start_index; i < start_index + dst_vertices.size(); i++) {
        cudaUpdateParams[trim_graph_id]->h_edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[count]));
        count++;
    } 

    streams[stream_id].num_graphs = 1; 
    streams[stream_id].src_vertex = src_vertex;
    streams[stream_id].delta_applied = 0;

    cudaMemcpyAsync(&cudaUpdateParams[trim_graph_id]->d_edgeUpdates[start_index], &cudaUpdateParams[trim_graph_id]->h_edgeUpdates[start_index], dst_vertices.size() * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id].stream);
    cudaKernel.k_sketchUpdate(num_device_threads, num_device_blocks, src_vertex, streams[stream_id].stream, start_index, dst_vertices.size(), stream_id * num_buckets, cudaUpdateParams[trim_graph_id], sketchSeed);
    cudaMemcpyAsync(&cudaUpdateParams[trim_graph_id]->h_bucket_a[stream_id * num_buckets], &cudaUpdateParams[trim_graph_id]->d_bucket_a[stream_id * num_buckets], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
    cudaMemcpyAsync(&cudaUpdateParams[trim_graph_id]->h_bucket_c[stream_id * num_buckets], &cudaUpdateParams[trim_graph_id]->d_bucket_c[stream_id * num_buckets], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);

  }
  else {
    for (vec_t i = 0; i < dst_vertices.size(); i++) {
      // Determine the depth of current edge
      vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[i]));
      int depth = Bucket_Boruvka::get_index_depth(edge_id, 0, num_graphs-1);
      max_depth = std::max(depth, max_depth);

      for (int graph_id = 0; graph_id <= depth; graph_id++) {
        if (graph_id >= num_sketch_graphs) { // Add to adj list graphs
          int adjlist_id = graph_id - num_sketch_graphs;

          Adjlist_Edge edge;
          edge.edge = std::make_pair(src_vertex, dst_vertices[i]);
          edge.graph_id = adjlist_id;
          worker_adjlist[thr_id].push_back(edge);
        }
        else {
          cudaUpdateParams[graph_id]->h_edgeUpdates[start_index + sketch_update_size[graph_id]] = edge_id;
          sketch_update_size[graph_id]++;
        }

      } 
    }
    streams[stream_id].num_graphs = std::min(num_sketch_graphs, (max_depth + 1)); 
    streams[stream_id].src_vertex = src_vertex;
    streams[stream_id].delta_applied = 0;

    // Keep tracking how many updates for each sketch subgraph
    for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {
      std::unique_lock<std::mutex> sketch_lk(sketch_mutexes[graph_id]);
      sketch_num_edges[graph_id] += sketch_update_size[graph_id];
      sketch_lk.unlock(); 
    }

    // Go through every sketch subgraphs and apply updates
    for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {
      cudaMemcpyAsync(&cudaUpdateParams[graph_id]->d_edgeUpdates[start_index], &cudaUpdateParams[graph_id]->h_edgeUpdates[start_index], sketch_update_size[graph_id] * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id].stream);
      cudaKernel.k_sketchUpdate(num_device_threads, num_device_blocks, src_vertex, streams[stream_id].stream, start_index, sketch_update_size[graph_id], stream_id * num_buckets, cudaUpdateParams[graph_id], sketchSeed);
      cudaMemcpyAsync(&cudaUpdateParams[graph_id]->h_bucket_a[stream_id * num_buckets], &cudaUpdateParams[graph_id]->d_bucket_a[stream_id * num_buckets], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
      cudaMemcpyAsync(&cudaUpdateParams[graph_id]->h_bucket_c[stream_id * num_buckets], &cudaUpdateParams[graph_id]->d_bucket_c[stream_id * num_buckets], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
    }
  }

};

void MCGPUSketchAlg::apply_flush_updates() {
  for (int stream_id = 0; stream_id < num_host_threads * stream_multiplier; stream_id++) {
    if(streams[stream_id].delta_applied == 0) {

      if (streams[stream_id].num_graphs < 0 || streams[stream_id].num_graphs > num_sketch_graphs ) {
        std::cout << "Stream #" << stream_id << ": invalid num_graphs! " << streams[stream_id].num_graphs << "\n";
      }

      for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {

        if(trim_enabled) {
          graph_id = trim_graph_id;
        }

        Bucket* delta_buckets = new Bucket[num_buckets];
        for (size_t i = 0; i < num_buckets; i++) {
          delta_buckets[i].alpha = cudaUpdateParams[graph_id]->h_bucket_a[(stream_id * num_buckets) + i];
          delta_buckets[i].gamma = cudaUpdateParams[graph_id]->h_bucket_c[(stream_id * num_buckets) + i];
        }

        int prev_src = streams[stream_id].src_vertex;
        
        if(prev_src == -1) {
          std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
        }

        // Apply the delta sketch
        apply_raw_buckets_update((graph_id * num_nodes) + prev_src, delta_buckets);
        free(delta_buckets);
      }
      streams[stream_id].delta_applied = 1;
      streams[stream_id].src_vertex = -1;
      streams[stream_id].num_graphs = -1;
    }
  }
}

void MCGPUSketchAlg::merge_adjlist() {
  for (int i = 0; i < num_host_threads; i++) {
    for (int j = 0; j < worker_adjlist[i].size(); j++) {
      Adjlist_Edge adjlist_edge = worker_adjlist[i][j];
      std::pair<int, int> edge = adjlist_edge.edge;
      int adjlist_id = adjlist_edge.graph_id;

      if (adjlists[adjlist_id].list.find(edge.first) == adjlists[adjlist_id].list.end()) {
        adjlists[adjlist_id].list[edge.first] = std::vector<node_id_t>();
      }
      adjlists[adjlist_id].list[edge.first].push_back(edge.second);
      adjlists[adjlist_id].num_updates++;
    }
  }
}

std::vector<Edge> MCGPUSketchAlg::get_adjlist_spanning_forests(int graph_id, int k) {
  AdjList adjlist = adjlists[graph_id];
  std::vector<Edge> forests;
  for (int k_id = 0; k_id < k; k_id++) {
    for (node_id_t node_id = 0; node_id < num_nodes; node_id++) {
      if (adjlist.list.find(node_id) == adjlist.list.end()) { // Doesn't exist, skip
        continue;
      }

      if (adjlist.list[node_id].size() != 0) {
        forests.push_back({node_id, adjlist.list[node_id].back()});
        adjlist.list[node_id].pop_back();
      }
    }
  }
  return forests;
}