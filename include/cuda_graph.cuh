#pragma once
#include <vector>
#include <map>
#include <util.h>
#include <cuda_kernel.cuh>
#include <atomic>

class CudaGraph {
    public: 
        CudaUpdateParams *cudaUpdateParams;
        Supernode** supernodes;
        long* sketchSeeds;

        std::vector<std::mutex> mutexes;
        std::atomic<vec_t> offset;
        std::vector<cudaStream_t> streams;
        std::vector<int> streams_deltaApplied;
        std::vector<int> streams_src;
        std::vector<std::vector<double>> loop_times;

        CudaKernel cudaKernel;

        // Number of threads
        int num_device_threads;
        
        // Number of blocks
        int num_device_blocks;

        int num_host_threads;
        int batch_size;
        int stream_multiplier;
        size_t sketch_size;

        bool isInit = false;


        // Default constructor
        CudaGraph() {}

        void configure(CudaUpdateParams* _cudaUpdateParams, Supernode** _supernodes, long* _sketchSeeds, int _num_host_threads) {
            cudaUpdateParams = _cudaUpdateParams;
            supernodes = _supernodes;
            sketchSeeds = _sketchSeeds;
            offset = 0;

            mutexes = std::vector<std::mutex>(cudaUpdateParams[0].num_nodes);

            num_device_threads = 1024;
            num_device_blocks = 1;
            num_host_threads = _num_host_threads;
            batch_size = cudaUpdateParams[0].batch_size;
            stream_multiplier = cudaUpdateParams[0].stream_multiplier;
            sketch_size = cudaUpdateParams[0].num_sketches * cudaUpdateParams[0].num_elems;

            for (int i = 0; i < num_host_threads; i++) {
                loop_times.push_back(std::vector<double>{});
            }
            
            // Assuming num_host_threads is even number
            for (int i = 0; i < num_host_threads * stream_multiplier; i++) {
                cudaStream_t stream;

                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

                streams.push_back(stream);
                streams_deltaApplied.push_back(1);
                streams_src.push_back(-1);
            }

            isInit = true;
        };

        void batch_update(int id, node_id_t src, const std::vector<node_id_t> &edges) {
            if (!isInit) {
                std::cout << "CudaGraph has not been initialized!\n";
            }

            int stream_id = id * stream_multiplier;
            int stream_offset = 0;
            //auto loop_start = std::chrono::steady_clock::now();
            while(true) {
                if (cudaStreamQuery(streams[stream_id + stream_offset]) == cudaSuccess) {
                    //std::chrono::duration<double> loop_time = std::chrono::steady_clock::now() - loop_start;
                    //loop_times[id].push_back(loop_time.count());

                    // Update stream_id
                    stream_id += stream_offset;

                    // CUDA Stream is available, but does not have any delta sketch
                    if(streams_deltaApplied[stream_id] == 0) {
                        streams_deltaApplied[stream_id] = 1;

                        // Bring back delta sketch
                        cudaMemcpyAsync(&cudaUpdateParams[0].h_bucket_a[stream_id * sketch_size], &cudaUpdateParams[0].d_bucket_a[stream_id * sketch_size], sketch_size * sizeof(vec_t), cudaMemcpyDeviceToHost, streams[stream_id]);
                        cudaMemcpyAsync(&cudaUpdateParams[0].h_bucket_c[stream_id * sketch_size], &cudaUpdateParams[0].d_bucket_c[stream_id * sketch_size], sketch_size * sizeof(vec_hash_t), cudaMemcpyDeviceToHost, streams[stream_id]);

                        cudaStreamSynchronize(streams[stream_id]);

                        if(streams_src[stream_id] == -1) {
                            std::cout << "Stream #" << stream_id << ": Shouldn't be here!\n";
                        }

                        // Apply the delta sketch
                        std::unique_lock<std::mutex> lk(mutexes[streams_src[stream_id]]);
                        for (int i = 0; i < cudaUpdateParams[0].num_sketches; i++) {
                            Sketch* sketch = supernodes[streams_src[stream_id]]->get_sketch(i);
                            vec_t* bucket_a = sketch->get_bucket_a();
                            vec_hash_t* bucket_c = sketch->get_bucket_c();

                            for (size_t j = 0; j < cudaUpdateParams[0].num_elems; j++) {
                                bucket_a[j] ^= cudaUpdateParams[0].h_bucket_a[(stream_id * sketch_size) + (i * cudaUpdateParams[0].num_elems) + j];
                                bucket_c[j] ^= cudaUpdateParams[0].h_bucket_c[(stream_id * sketch_size) + (i * cudaUpdateParams[0].num_elems) + j];
                            }
                        }
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
            for (vec_t i = start_index; i < start_index + edges.size(); i++) {
                cudaUpdateParams[0].h_edgeUpdates[i] = static_cast<vec_t>(concat_pairing_fn(src, edges[count]));
                count++;
            }            
            streams_src[stream_id] = src;
            streams_deltaApplied[stream_id] = 0;
            cudaMemcpyAsync(&cudaUpdateParams[0].d_edgeUpdates[start_index], &cudaUpdateParams[0].h_edgeUpdates[start_index], edges.size() * sizeof(vec_t), cudaMemcpyHostToDevice, streams[stream_id]);
            cudaKernel.gtsStreamUpdate(num_device_threads, num_device_blocks, stream_id * sketch_size, src, streams[stream_id], start_index, edges.size(), cudaUpdateParams, sketchSeeds);
        };

        void applyFlushUpdates() {
            //std::cout << "    Applying flush updates\n";

            for (int stream_id = 0; stream_id < num_host_threads * stream_multiplier; stream_id++) {
                if(streams_deltaApplied[stream_id] == 0) {
                    streams_deltaApplied[stream_id] = 1;
                    
                    cudaMemcpy(&cudaUpdateParams[0].h_bucket_a[stream_id * sketch_size], &cudaUpdateParams[0].d_bucket_a[stream_id * sketch_size], sketch_size * sizeof(vec_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&cudaUpdateParams[0].h_bucket_c[stream_id * sketch_size], &cudaUpdateParams[0].d_bucket_c[stream_id * sketch_size], sketch_size * sizeof(vec_hash_t), cudaMemcpyDeviceToHost);

                    // Apply the delta sketch
                    for (int i = 0; i < cudaUpdateParams[0].num_sketches; i++) {
                        Sketch* sketch = supernodes[streams_src[stream_id]]->get_sketch(i);
                        vec_t* bucket_a = sketch->get_bucket_a();
                        vec_hash_t* bucket_c = sketch->get_bucket_c();

                        for (size_t j = 0; j < cudaUpdateParams[0].num_elems; j++) {
                            bucket_a[j] ^= cudaUpdateParams[0].h_bucket_a[(stream_id * sketch_size) + (i * cudaUpdateParams[0].num_elems) + j];
                            bucket_c[j] ^= cudaUpdateParams[0].h_bucket_c[(stream_id * sketch_size) + (i * cudaUpdateParams[0].num_elems) + j];
                        }
                    }
                    streams_src[stream_id] = -1;
                }
            }
        }
};