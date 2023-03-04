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
        size_t num_elems;
        size_t num_buckets;
        size_t num_guesses;
        vec_t *bucket_a;
        vec_hash_t *bucket_c;
        uint64_t seed;

    public:
        CudaSketch(size_t numElems, size_t numBuckets, size_t numGuesses, vec_t* &bucketA, vec_hash_t* &bucketC, uint64_t currentSeed);
        void update(col_hash_t* d_col_index_hash, const vec_t& update_idx, vec_t* &bucket_debug);
};