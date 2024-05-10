#include "mc_gpu_sketch_alg.h"

#include <iostream>
#include <vector>

void MCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (MCSketchAlg::get_update_locked()) throw UpdateLockedException();

  // If trim enabled, perform sketch updates in CPU
  if (trim_enabled) {
    if (trim_graph_id < 0 || trim_graph_id >= num_graphs) {
      std::cout << "INVALID trim_graph_id: " << trim_graph_id << "\n";
    }
    
    if (subgraphs[trim_graph_id]->get_type() != SKETCH) {
      std::cout << "Current trim_graph_id isn't SKETCH data structure: " << trim_graph_id << "\n";
    }

    apply_update_batch_single_graph(thr_id, trim_graph_id, src_vertex, dst_vertices);
  }

  else {
    int stream_id = thr_id * stream_multiplier;
    int stream_offset = 0;
    while(true) {
      if (cudaStreamQuery(streams[stream_id + stream_offset].stream) == cudaSuccess) {
        // Update stream_id
        stream_id += stream_offset;

        // CUDA Stream is available, check if it has any delta sketch
        if(streams[stream_id].delta_applied == 0) {

          for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {   
  
            if (subgraphs[graph_id]->get_type() != SKETCH) {
              break;
            }
            
            size_t bucket_offset = thr_id * num_buckets;
            for (size_t i = 0; i < num_buckets; i++) {
              delta_buckets[bucket_offset + i].alpha = subgraphs[graph_id]->get_cudaUpdateParams()->h_bucket_a[(stream_id * num_buckets) + i];
              delta_buckets[bucket_offset + i].gamma = subgraphs[graph_id]->get_cudaUpdateParams()->h_bucket_c[(stream_id * num_buckets) + i];
            }

            int prev_src = streams[stream_id].src_vertex;
            
            if(prev_src == -1) {
              std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
            }

            // Apply the delta sketch
            apply_raw_buckets_update((graph_id * num_nodes) + prev_src, &delta_buckets[bucket_offset]);
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
    sketch_update_size.assign(max_sketch_graphs, 0);
    int max_depth = 0;
    int convert_sketch = -1;

    for (vec_t i = 0; i < dst_vertices.size(); i++) {
      // Determine the depth of current edge
      vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[i]));
      int depth = Bucket_Boruvka::get_index_depth(edge_id, 0, num_graphs-1);
      max_depth = std::max(depth, max_depth);

      for (int graph_id = 0; graph_id <= depth; graph_id++) {
        if (subgraphs[graph_id]->get_type() == FIXED_ADJLIST) { // Graph Type 3: Fixed Adj. List
          subgraphs[graph_id]->insert_fixed_adj_edge(src_vertex, dst_vertices[i]);
        }
        else {
          std::lock_guard<std::mutex> lk(subgraphs[graph_id]->mutex);
          if (subgraphs[graph_id]->get_type() == SKETCH) { // Graph Type 1: Sketch graph
            subgraphs[graph_id]->get_cudaUpdateParams()->h_edgeUpdates[start_index + sketch_update_size[graph_id]] = edge_id;
            sketch_update_size[graph_id]++;
            subgraphs[graph_id]->increment_num_sketch_updates(1);
          }
          else { // Graph Type 2: Adj. list
            subgraphs[graph_id]->insert_adj_edge(src_vertex, dst_vertices[i]);

            // Check the size of adj. list after insertion
            double adjlist_bytes = subgraphs[graph_id]->get_num_updates() * adjlist_edge_bytes;
            if (adjlist_bytes > sketch_bytes) { // With size of current adj. list, it is more space-efficient to convert into sketch graph
              
              // Update num_sketch_updates
              subgraphs[graph_id]->increment_num_sketch_updates(subgraphs[graph_id]->get_num_updates());
              
              // Set subgraph into SKETCH type
              subgraphs[graph_id]->set_type(SKETCH);

              // Init sketches 
              create_sketch_graph(graph_id);

              //convert_sketch = graph_id;
              num_adj_graphs--;
              num_sketch_graphs++;

              convert_sketch = graph_id;
            }
          }
        }

      } 
    }

    streams[stream_id].num_graphs = max_depth + 1; 
    streams[stream_id].src_vertex = src_vertex;
    streams[stream_id].delta_applied = 0;

    // Go every subgraph and apply updates
    for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {
      if (subgraphs[graph_id]->get_type() != SKETCH) {
        break;
      }

      if (graph_id == convert_sketch) {
        convert_sketch = -1; 

        // Start conversion with sketch updates
        std::cout << "Converting graph #" << graph_id << " into sketch\n";
        auto conversion_start = std::chrono::steady_clock::now();

        std::map<node_id_t, std::map<node_id_t, node_id_t>> adjlist = subgraphs[graph_id]->get_adjlist();

        int max_batch_size = 0;
        for (auto it = adjlist.begin(); it != adjlist.end(); it++) { // Go through adj list and get the max batch size
          if (it->second.size() > max_batch_size) {
            max_batch_size = it->second.size();
          }
        }

        // Allocate buffer with the max. batch size
        vec_t *convert_h_edgeUpdates, *convert_d_edgeUpdates;

        gpuErrchk(cudaMallocHost(&convert_h_edgeUpdates, max_batch_size * sizeof(vec_t)));
        gpuErrchk(cudaMalloc(&convert_d_edgeUpdates, max_batch_size * sizeof(vec_t)));

        for (auto it = adjlist.begin(); it != adjlist.end(); it++) {
          node_id_t src = it->first;

          if (adjlist.find(src) == adjlist.end()) { // If current source node not found, skip
            continue;
          }

          // Reset buffer for edge updates
          for (int i = 0; i < max_batch_size; i++) {
            convert_h_edgeUpdates[i] = 0;
          }

          // Go through all neighbor nodes and fill in buffer
          int current_index = 0;
          for (auto dst_it = adjlist[src].begin(); dst_it != adjlist[src].end(); dst_it++) {
            node_id_t dst = dst_it->first;
            convert_h_edgeUpdates[current_index] = static_cast<vec_t>(concat_pairing_fn(src, dst));
            current_index++;
          }

          // Start sketch updates
          CudaUpdateParams* cudaUpdateParams = subgraphs[graph_id]->get_cudaUpdateParams();
          cudaMemcpyAsync(&convert_d_edgeUpdates[0], &convert_h_edgeUpdates[0], adjlist[src].size() * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id].stream);
          cudaKernel.k_sketchUpdate(num_device_threads, num_device_blocks, streams[stream_id].stream, convert_d_edgeUpdates, 0, adjlist[src].size(), 0, cudaUpdateParams, cudaUpdateParams->convert_d_bucket_a, cudaUpdateParams->convert_d_bucket_c, sketchSeed);
          cudaMemcpyAsync(&cudaUpdateParams->convert_h_bucket_a[0], &cudaUpdateParams->convert_d_bucket_a[0], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
          cudaMemcpyAsync(&cudaUpdateParams->convert_h_bucket_c[0], &cudaUpdateParams->convert_d_bucket_c[0], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);

          // Wait until GPU kernel ends and transfer buckets back
          cudaStreamSynchronize(streams[stream_id].stream);

          // Apply delta sketch
          size_t bucket_offset = thr_id * num_buckets;
          for (size_t i = 0; i < num_buckets; i++) {
            delta_buckets[bucket_offset + i].alpha = cudaUpdateParams->convert_h_bucket_a[i];
            delta_buckets[bucket_offset + i].gamma = cudaUpdateParams->convert_h_bucket_c[i];
          }

          // Apply the delta sketch
          apply_raw_buckets_update((graph_id * num_nodes) + src, &delta_buckets[bucket_offset]);

          // Delete from adj list
          subgraphs[graph_id]->adjlist_delete_src(src);
        }

        std::chrono::duration<double> conversion_time = std::chrono::steady_clock::now() - conversion_start;
        std::cout << "Finished Converting graph #" << graph_id << " into sketch. Elpased time: " << conversion_time.count() << "\n";
      }

      // Regular sketch updates
      CudaUpdateParams* cudaUpdateParams = subgraphs[graph_id]->get_cudaUpdateParams();
      cudaMemcpyAsync(&cudaUpdateParams->d_edgeUpdates[start_index], &cudaUpdateParams->h_edgeUpdates[start_index], sketch_update_size[graph_id] * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id].stream);
      cudaKernel.k_sketchUpdate(num_device_threads, num_device_blocks, streams[stream_id].stream, cudaUpdateParams->d_edgeUpdates, start_index, sketch_update_size[graph_id], stream_id * num_buckets, cudaUpdateParams, cudaUpdateParams->d_bucket_a, cudaUpdateParams->d_bucket_c, sketchSeed);
      cudaMemcpyAsync(&cudaUpdateParams->h_bucket_a[stream_id * num_buckets], &cudaUpdateParams->d_bucket_a[stream_id * num_buckets], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
      cudaMemcpyAsync(&cudaUpdateParams->h_bucket_c[stream_id * num_buckets], &cudaUpdateParams->d_bucket_c[stream_id * num_buckets], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
    }    
  }

};

void MCGPUSketchAlg::apply_flush_updates() {
  for (int stream_id = 0; stream_id < num_host_threads * stream_multiplier; stream_id++) {
    if(streams[stream_id].delta_applied == 0) {
      for (int graph_id = 0; graph_id < streams[stream_id].num_graphs; graph_id++) {

        if (subgraphs[graph_id]->get_type() != SKETCH) {
          break;
        }

        for (size_t i = 0; i < num_buckets; i++) {
          delta_buckets[i].alpha = subgraphs[graph_id]->get_cudaUpdateParams()->h_bucket_a[(stream_id * num_buckets) + i];
          delta_buckets[i].gamma = subgraphs[graph_id]->get_cudaUpdateParams()->h_bucket_c[(stream_id * num_buckets) + i];
        }

        int prev_src = streams[stream_id].src_vertex;
        
        if(prev_src == -1) {
          std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
        }

        // Apply the delta sketch
        apply_raw_buckets_update((graph_id * num_nodes) + prev_src, delta_buckets);
      }
      streams[stream_id].delta_applied = 1;
      streams[stream_id].src_vertex = -1;
      streams[stream_id].num_graphs = -1;
    }
  }
}

std::vector<Edge> MCGPUSketchAlg::get_adjlist_spanning_forests(int graph_id, int k) {
  if (subgraphs[graph_id]->get_type() == SKETCH) {
    std::cout << "Subgraph with graph_id: " << graph_id << " is Sketch graph!\n";
  }

  std::vector<Edge> forests;
  for (int k_id = 0; k_id < k; k_id++) {
    for (node_id_t node_id = 0; node_id < num_nodes; node_id++) {
      node_id_t dst = subgraphs[graph_id]->sample_dst_node(node_id);

      if (dst != -1) {
        forests.push_back({node_id, dst});
      }
    }
  }
  return forests;
}