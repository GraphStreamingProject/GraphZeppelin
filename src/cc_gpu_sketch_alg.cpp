#include "cc_gpu_sketch_alg.h"

#include <iostream>
#include <vector>

void CCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();

  int stream_id = thr_id * stream_multiplier;
  int stream_offset = 0;
  while(true) {
    if (cudaStreamQuery(streams[stream_id + stream_offset].stream) == cudaSuccess) {
      // Update stream_id
      stream_id += stream_offset;

      // CUDA Stream is available, check if it has any delta sketch
      if(streams[stream_id].delta_applied == 0) {

        size_t bucket_offset = thr_id * num_buckets;
        for (size_t i = 0; i < num_buckets; i++) {
          delta_buckets[bucket_offset + i].alpha = cudaUpdateParams[0].h_bucket_a[(stream_id * num_buckets) + i];
          delta_buckets[bucket_offset + i].gamma = cudaUpdateParams[0].h_bucket_c[(stream_id * num_buckets) + i];
        }

        int prev_src = streams[stream_id].src_vertex;
        
        if(prev_src == -1) {
          std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
        }

        // Apply the delta sketch
        apply_raw_buckets_update(prev_src, &delta_buckets[bucket_offset]);
        streams[stream_id].delta_applied = 1;
        streams[stream_id].src_vertex = -1;
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
  int count = 0;
  for (vec_t i = start_index; i < start_index + dst_vertices.size(); i++) {
      cudaUpdateParams[0].h_edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[count]));
      count++;
  }        

  streams[stream_id].delta_applied = 0;
  streams[stream_id].src_vertex = src_vertex;
  cudaMemcpyAsync(&cudaUpdateParams[0].d_edgeUpdates[start_index], &cudaUpdateParams[0].h_edgeUpdates[start_index], dst_vertices.size() * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id].stream);
  cudaKernel.sketchUpdate(num_device_threads, num_device_blocks, streams[stream_id].stream, cudaUpdateParams[0].d_edgeUpdates, start_index, dst_vertices.size(), stream_id * num_buckets, cudaUpdateParams, cudaUpdateParams[0].d_bucket_a, cudaUpdateParams[0].d_bucket_c, sketchSeed);
  cudaMemcpyAsync(&cudaUpdateParams[0].h_bucket_a[stream_id * num_buckets], &cudaUpdateParams[0].d_bucket_a[stream_id * num_buckets], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
  cudaMemcpyAsync(&cudaUpdateParams[0].h_bucket_c[stream_id * num_buckets], &cudaUpdateParams[0].d_bucket_c[stream_id * num_buckets], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id].stream);
};

void CCGPUSketchAlg::apply_flush_updates() {
  for (int stream_id = 0; stream_id < num_host_threads * stream_multiplier; stream_id++) {
    if(streams[stream_id].delta_applied == 0) {
      for (size_t i = 0; i < num_buckets; i++) {
        delta_buckets[i].alpha = cudaUpdateParams[0].h_bucket_a[(stream_id * num_buckets) + i];
        delta_buckets[i].gamma = cudaUpdateParams[0].h_bucket_c[(stream_id * num_buckets) + i];
      }

      int prev_src = streams[stream_id].src_vertex;
      
      if(prev_src == -1) {
        std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
      }

      // Apply the delta sketch
      apply_raw_buckets_update(prev_src, delta_buckets);
      streams[stream_id].delta_applied = 1;
      streams[stream_id].src_vertex = -1;
    }
  }
}