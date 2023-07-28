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
        std::vector<std::vector<double>> loop_times;

        CudaKernel cudaKernel;

        // Number of threads
        int num_device_threads;
        
        // Number of blocks
        int num_device_blocks;

        int num_host_threads;
        int batch_size;
        int stream_multiplier;

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
            batch_size = cudaUpdateParams[0].batch_size;
            stream_multiplier = cudaUpdateParams[0].stream_multiplier;

            for (int i = 0; i < num_host_threads; i++) {
                loop_times.push_back(std::vector<double>{});
            }
            
            // Assuming num_host_threads is even number
            for (int i = 0; i < num_host_threads * stream_multiplier; i++) {
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
            // Find which stream is available
            int stream_id = id * stream_multiplier;
            int stream_offset = 0;
            //auto loop_start = std::chrono::steady_clock::now();
            while(true) {
                if (cudaStreamQuery(streams[stream_id + stream_offset]) == cudaSuccess) {
                    //std::chrono::duration<double> loop_time = std::chrono::steady_clock::now() - loop_start;
                    //loop_times[id].push_back(loop_time.count());
                    stream_id += stream_offset;
                    break;
                }
                stream_offset++;
                if (stream_offset == stream_multiplier) {
                    stream_offset = 0;
                }
            }
            int start_index = stream_id * batch_size;
            int count = 0;
            for (vec_t i = start_index; i < start_index + edges.size(); i++) {
                cudaUpdateParams[0].h_edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src, edges[count]));
                count++;
            }
            cudaMemcpyAsync(&cudaUpdateParams[0].d_edgeUpdates[start_index], &cudaUpdateParams[0].h_edgeUpdates[start_index], edges.size() * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id]);
            cudaKernel.gtsStreamUpdate(num_device_threads, num_device_blocks, src, streams[stream_id], start_index, edges.size(), cudaUpdateParams, cudaSketches, sketchSeeds);
        };
};