#include "sk_gpu_sketch_alg.h"

#include <iostream>
#include <vector>

void SKGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();
  // Get offset
  size_t offset = edgeUpdate_offset.fetch_add(dst_vertices.size());

  // Fill in buffer
  size_t index = 0;
  for (auto dst : dst_vertices) {
    edgeUpdates[offset + index] = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst));
    index++;
  }

  size_t batch_id = batch_count.fetch_add(1);
  std::lock_guard<std::mutex> lk(batch_mutex);
  batch_sizes.insert({batch_id, dst_vertices.size()});
  batch_src.insert({batch_id, src_vertex});
  batch_start_index.insert({batch_id, offset});
};

void SKGPUSketchAlg::launch_gpu_kernel() {
  // Declare GPU block count and size
  num_device_threads = 1024;
  num_device_blocks = batch_count; // 1 Thread block responsible for one batch of sketch update
  std::cout << "Num GPU threads per block: " << num_device_threads << "\n";
  std::cout << "Num GPU thread blocks: " << num_device_blocks << "\n";

  std::cout << "Preparing update buffers for GPU...\n";
  gpuErrchk(cudaMallocManaged(&update_sizes, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMallocManaged(&update_src, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMallocManaged(&update_start_index, batch_count * sizeof(vec_t)));
  // Fill in update_sizes and update_src
  for (auto it = batch_sizes.begin(); it != batch_sizes.end(); it++) {
    update_sizes[it->first] = it->second;
    update_src[it->first] = batch_src[it->first]; 
    update_start_index[it->first] = batch_start_index[it->first];
  }
  
  // Prefetch buffers to GPU
  gpuErrchk(cudaMemPrefetchAsync(edgeUpdates, num_updates * sizeof(vec_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(bucket_a, num_nodes * num_buckets * sizeof(vec_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(bucket_c, num_nodes * num_buckets * sizeof(vec_hash_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(update_sizes, batch_count * sizeof(vec_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(update_src, batch_count * sizeof(vec_t), device_id));
  gpuErrchk(cudaMemPrefetchAsync(update_start_index, batch_count * sizeof(vec_t), device_id));

  // Launch GPU kernel
  std::cout << "Launching GPU Kernel...\n";
  auto kernel_start = std::chrono::steady_clock::now();
  cudaKernel.single_sketchUpdate(num_device_threads, num_device_blocks, maxBytes, update_src, update_sizes, update_start_index, edgeUpdates, bucket_a, bucket_c, num_buckets, num_columns, bkt_per_col, sketchSeed);

  cudaDeviceSynchronize();
  auto kernel_end = std::chrono::steady_clock::now();
  std::cout << "  GPU Kernel Finished.\n";
  std::chrono::duration<double> kernel_time = kernel_end - kernel_start;
  std::cout << "    Elapsed Time: " << kernel_time.count() << "\n";

  // Prefecth buffers back to CPU
  gpuErrchk(cudaMemPrefetchAsync(bucket_a, num_nodes * num_buckets * sizeof(vec_t), cudaCpuDeviceId));
  gpuErrchk(cudaMemPrefetchAsync(bucket_c, num_nodes * num_buckets * sizeof(vec_hash_t), cudaCpuDeviceId));
}

void SKGPUSketchAlg::apply_delta_sketch() {
  std::cout << "Applying Delta Sketch...\n";
  Bucket* delta_buckets = new Bucket[num_buckets];

  for (node_id_t src = 0; src < num_nodes; src++) {
    for (size_t i = 0; i < num_buckets; i++) {
      delta_buckets[i].alpha = bucket_a[(src * num_buckets) + i];
      delta_buckets[i].gamma = bucket_c[(src * num_buckets) + i];
    }
    apply_raw_buckets_update(src, delta_buckets);
  }

  std::cout << "  Applying Delta Sketch Finished.\n";
}