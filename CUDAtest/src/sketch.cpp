#include "../include/bucket.h"
#include "../include/cudaSketch.cuh"
#include "../include/sketch.h"

vec_t Sketch::failure_factor = 100;
vec_t Sketch::n;
size_t Sketch::num_elems;
size_t Sketch::num_buckets;
size_t Sketch::num_guesses;

class MultipleQueryException : public std::exception {
    public:
        virtual const char* what() const throw() {
            return "This sketch has already been sampled!";
        }
};

/*
 * Static functions for creating sketches with a provided memory location.
 * We use these in the production system to keep supernodes virtually contiguous.
 */
Sketch* Sketch::makeSketch(void* loc, uint64_t seed) {
    return new (loc) Sketch(seed);
}

Sketch::Sketch(uint64_t seed): seed(seed) {
    // establish the bucket_a and bucket_c locations
    bucket_a = reinterpret_cast<vec_t*>(buckets);
    bucket_c = reinterpret_cast<vec_hash_t*>(buckets + num_elems * sizeof(vec_t));

    // initialize bucket values
    for (size_t i = 0; i < num_elems; ++i) {
        bucket_a[i] = 0;
        bucket_c[i] = 0;
    }
}

std::pair<vec_t, SampleSketchRet> Sketch::query() {
    if (already_queried) {
        throw MultipleQueryException();
    }
    already_queried = true;

    std::cout << "Checking\n"; 
    if (bucket_a[num_elems - 1] == 0 && bucket_c[num_elems - 1] == 0) {
        return {0, ZERO}; // the "first" bucket is deterministic so if it is all zero then there are no edges to return
    }
    if (Bucket_Boruvka::is_good(bucket_a[num_elems - 1], bucket_c[num_elems - 1], seed)) {
        return {bucket_a[num_elems - 1], GOOD};
    }
    std::cout << "Calling CUDA" << "\n";
    CudaSketch::query();
    for (unsigned i = 0; i < num_buckets; ++i) {
        for (unsigned j = 0; j < num_guesses; ++j) {
        unsigned bucket_id = i * num_guesses + j;
        if (Bucket_Boruvka::is_good(bucket_a[bucket_id], bucket_c[bucket_id], i, 1 << j, seed)) {
            return {bucket_a[bucket_id], GOOD};
        }
        }
    }
    return {0, FAIL};
}