#include "cc_gpu_sketch_alg.h"

#include <iostream>
#include <vector>

// Configure CCGPUSketchAlg
void CCGPUSketchAlg::configure(CudaUpdateParams** _cudaUpdateParams, long* _sketchSeeds, int _num_host_threads) {
  cudaUpdateParams = _cudaUpdateParams;
  //sketches = _sketches;
  sketchSeeds = _sketchSeeds;

  num_buckets = cudaUpdateParams[0]->num_buckets;

  num_device_threads = 1024;
  num_device_blocks = 1;
  num_host_threads = _num_host_threads;
  batch_size = cudaUpdateParams[0]->batch_size;
  stream_multiplier = cudaUpdateParams[0]->stream_multiplier;

  // num_host_threads must be even number
  if (num_host_threads % 2 != 0) {
    std::cout << "num_host_threads must be even number!\n";
    exit(EXIT_FAILURE);
  }

  // Initialize CUDA Streams
  for (int i = 0; i < num_host_threads * stream_multiplier; i++) {
    cudaStream_t stream;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    streams.push_back(stream);
    streams_deltaApplied.push_back(1);
    streams_src.push_back(-1);
    streams_num_graphs.push_back(-1);
  }
  
  // Finished configuration, turn on the flag
  isConfigured = true;
}

void CCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (update_locked) throw UpdateLockedException();
  if (!isConfigured) {
    std::cout << "CCGPUSketchAlg has not been configured!\n";
  }
  /*Sketch &delta_sketch = *delta_sketches[thr_id];
  delta_sketch.zero_contents();

  for (const auto &dst : dst_vertices) {
    delta_sketch.update(static_cast<vec_t>(concat_pairing_fn(src_vertex, dst)));
  }

  std::unique_lock<std::mutex>(sketches[src_vertex]->mutex);
  sketches[src_vertex]->merge(delta_sketch);*/

  int stream_id = thr_id * stream_multiplier;
  int stream_offset = 0;
  while(true) {
    if (cudaStreamQuery(streams[stream_id + stream_offset]) == cudaSuccess) {
      // Update stream_id
      stream_id += stream_offset;

      // CUDA Stream is available, check if it has any delta sketch
      if(streams_deltaApplied[stream_id] == 0) {
        streams_deltaApplied[stream_id] = 1;

        // Transfer delta sketch from GPU to CPU
        cudaMemcpyAsync(&cudaUpdateParams[0]->h_bucket_a[stream_id * num_buckets], &cudaUpdateParams[0]->d_bucket_a[stream_id * num_buckets], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id]);
        cudaMemcpyAsync(&cudaUpdateParams[0]->h_bucket_c[stream_id * num_buckets], &cudaUpdateParams[0]->d_bucket_c[stream_id * num_buckets], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id]);

        cudaStreamSynchronize(streams[stream_id]);

        Bucket* delta_buckets = new Bucket[cudaUpdateParams[0]->num_buckets];
        for (size_t i = 0; i < num_buckets; i++) {
          delta_buckets[i].alpha = cudaUpdateParams[0]->h_bucket_a[(stream_id * num_buckets) + i];
          delta_buckets[i].gamma = cudaUpdateParams[0]->h_bucket_c[(stream_id * num_buckets) + i];
        }

        node_id_t prev_src = streams_src[stream_id];
        
        if(prev_src == -1) {
          std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
        }

        // Apply the delta sketch
        std::unique_lock<std::mutex> lk(sketches[prev_src]->mutex);
        sketches[prev_src]->merge_raw_bucket_buffer(delta_buckets);
        lk.unlock();
        streams_src[stream_id] = -1;
      }
      else {
        if (streams_src[stream_id] != -1) {
          std::cout << "Stream #" << stream_id << ": not applying but has delta sketch: " << streams_src[stream_id] << " deltaApplied: " << streams_deltaApplied[stream_id] << "\n";
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
      cudaUpdateParams[0]->h_edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[count]));
      count++;
  }        

  streams_src[stream_id] = src_vertex;
  streams_deltaApplied[stream_id] = 0;
  cudaMemcpyAsync(&cudaUpdateParams[0]->d_edgeUpdates[start_index], &cudaUpdateParams[0]->h_edgeUpdates[start_index], dst_vertices.size() * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id]);
  cudaKernel.gtsStreamUpdate(num_device_threads, num_device_blocks, src_vertex, streams[stream_id], start_index, dst_vertices.size(), stream_id * num_buckets, cudaUpdateParams[0], sketchSeeds[src_vertex]);
};

void CCGPUSketchAlg::apply_flush_updates() {
  for (int stream_id = 0; stream_id < num_host_threads * stream_multiplier; stream_id++) {
    if(streams_deltaApplied[stream_id] == 0) {
      streams_deltaApplied[stream_id] = 1;
        
        cudaMemcpy(&cudaUpdateParams[0]->h_bucket_a[stream_id * num_buckets], &cudaUpdateParams[0]->d_bucket_a[stream_id * num_buckets], num_buckets * sizeof(vec_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cudaUpdateParams[0]->h_bucket_c[stream_id * num_buckets], &cudaUpdateParams[0]->d_bucket_c[stream_id * num_buckets], num_buckets * sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

        Bucket* delta_buckets = new Bucket[cudaUpdateParams[0]->num_buckets];
        for (size_t i = 0; i < num_buckets; i++) {
          delta_buckets[i].alpha = cudaUpdateParams[0]->h_bucket_a[(stream_id * num_buckets) + i];
          delta_buckets[i].gamma = cudaUpdateParams[0]->h_bucket_c[(stream_id * num_buckets) + i];
        }

        node_id_t prev_src = streams_src[stream_id];
        
        if(prev_src == -1) {
          std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
        }

        // Apply the delta sketch
        std::unique_lock<std::mutex> lk(sketches[prev_src]->mutex);
        sketches[prev_src]->merge_raw_bucket_buffer(delta_buckets);
        lk.unlock();
        streams_src[stream_id] = -1;
    }
  }
}