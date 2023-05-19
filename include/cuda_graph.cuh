#pragma once
#include <vector>
#include <map>
#include <util.h>
#include <atomic>
#include <cuda_kernel.cuh>

class CudaGraph {
    public: 
        CudaUpdateParams *cudaUpdateParams;
        CudaSketch* cudaSketches;
        long* sketchSeeds;

        std::vector<std::mutex> mutexes;
        std::atomic<vec_t> offset;
        std::vector<cudaStream_t> streams;

        CudaKernel cudaKernel;

        // Number of threads
        int num_device_threads;
        
        // Number of blocks
        int num_device_blocks;

        int num_host_threads;

        bool isInit = false;

        // Default constructor
        CudaGraph() {}

        void configure(CudaUpdateParams* _cudaUpdateParams, CudaSketch* _cudaSketches, long* _sketchSeeds, int _num_host_threads) {
            cudaUpdateParams = _cudaUpdateParams;
            cudaSketches = _cudaSketches;
            sketchSeeds = _sketchSeeds;
            offset = 0;

            mutexes = std::vector<std::mutex>(cudaUpdateParams[0].num_nodes);

            num_device_threads = 1024;
            num_device_blocks = 1;
            num_host_threads = _num_host_threads;

            for (int i = 0; i < num_host_threads; i++) {
                cudaStream_t stream;
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                streams.push_back(stream);
            }
            isInit = true;
        };

        void batch_update(int id, node_id_t src, const std::vector<node_id_t> &edges) {
            if (!isInit) {
                std::cout << "CudaGraph has not been initialized!\n";
            }
            // Add first to prevent data conflicts
            vec_t prev_offset = std::atomic_fetch_add(&offset, edges.size());
            int count = 0;
            for (vec_t i = prev_offset; i < prev_offset + edges.size(); i++) {
                if (src < edges[count]) {
                    cudaUpdateParams[0].edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src, edges[count]));
                }
                else {
                    cudaUpdateParams[0].edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(edges[count], src));
                }
                count++;
            }

            cudaStreamAttachMemAsync(streams[id], &cudaUpdateParams[0].edgeUpdates[prev_offset]);
            cudaKernel.gtsStreamUpdate(num_device_threads, num_device_blocks, src, streams[id], prev_offset, edges.size(), cudaUpdateParams, cudaSketches, sketchSeeds);
        };
};