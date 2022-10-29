#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../bucket.h"

#include <stdio.h>
#include <vector>
#include <graph_zeppelin_common.h>

class CudaSketch {
    private:
        size_t num_elems;
        size_t num_buckets;
        size_t num_guesses;
        vec_t* bucket_a;
        vec_hash_t* bucket_c;
        uint64_t seed;

    public:
        std::vector<vec_t> result = {0};

        CudaSketch(size_t numElems, size_t numBuckets, size_t numGuesses, vec_t* &bucketA, vec_hash_t* &bucketC, uint64_t currentSeed);
        //void update(const vec_t& update_idx, const vec_hash_t update_hash, const col_hash_t* col_index_hashes);
        void update(const vec_t& update_idx, int count);
        void query();
};