//#include "cuda_runtime.h"
#include "/usr/local/cuda-11.5/targets/x86_64-linux/include/cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "/usr/local/cuda-11.5/targets/x86_64-linux/include/device_launch_parameters.h"
#include "../bucket.h"

#include <stdio.h>
#include <vector>
#include <graph_zeppelin_common.h>

class CudaSketch {
    private:


    public:
        size_t num_elems;
        size_t num_buckets;
        size_t num_guesses;
        uint64_t seed;

        CudaSketch(size_t numElems, size_t numBuckets, size_t numGuesses, uint64_t currentSeed);
        void update(vec_t* &bucket_a, vec_hash_t* &bucket_c, vec_t* &d_bucket_a, vec_hash_t* &d_bucket_c, col_hash_t* &d_col_index_hashes, const vec_t& update_idx);
        //void update(vec_t* &combined_memory, vec_t* &combined_device_memory, const vec_t& update_idx);
        void query();
};