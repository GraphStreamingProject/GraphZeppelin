#include "cc_gpu_sketch_alg.h"

#include <iostream>
#include <vector>

void CCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();

  int stream_id = (thr_id * stream_multiplier) + streams_offset[thr_id];

  streams_offset[thr_id]++;
  if (streams_offset[thr_id] == stream_multiplier) {
    streams_offset[thr_id] = 0;
  }

  cudaStreamSynchronize(streams[stream_id].stream);
  int start_index = stream_id * batch_size;
  int count = 0;
  for (vec_t i = start_index; i < start_index + dst_vertices.size(); i++) {
      cudaUpdateParams[0].h_edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src_vertex, dst_vertices[count]));
      count++;
  } 

  cudaMemcpyAsync(&cudaUpdateParams[0].d_edgeUpdates[start_index], &cudaUpdateParams[0].h_edgeUpdates[start_index], dst_vertices.size() * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id].stream);
  cudaKernel.sketchUpdate(num_device_threads, num_device_blocks, src_vertex, streams[stream_id].stream, &cudaUpdateParams[0].d_edgeUpdates[start_index], dst_vertices.size(), cudaUpdateParams, sketchSeed);
};