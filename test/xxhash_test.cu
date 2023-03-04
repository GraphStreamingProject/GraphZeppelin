#include <vector>
#include <cuda_xxhash64.cuh>
#include <bucket.h>
#include <iostream>

typedef unsigned long long int uint64_cu;
typedef uint64_cu vec_t_cu;

__global__ void xxhash_func(vec_t_cu* d_64, vec_hash_t* d_32, int num_items) {
    vec_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < num_items) {
        d_64[tid] = CUDA_XXH64(&tid, sizeof(tid), tid);
        d_32[tid] = CUDA_XXH32(&tid, sizeof(tid), tid);
    }
}

int main() {

    const int num_items = 1000;

    std::vector<col_hash_t> h_64;
    std::vector<vec_hash_t> h_32;
    vec_t_cu* d_64;
    vec_hash_t* d_32;

    h_64.reserve(num_items);
    h_32.reserve(num_items);

    cudaMallocManaged(&d_64, num_items * sizeof(vec_t_cu));
    cudaMallocManaged(&d_32, num_items * sizeof(vec_hash_t));

    for (vec_t i = 0; i < num_items; i++) {
        h_64.push_back(XXH64(&i, sizeof(i), i));
        h_32.push_back(XXH32(&i, sizeof(i), i));
        d_64[i] = 0;
        d_32[i] = 0;
    }

    int num_threads = 1024;

    int num_blocks = 1;

    xxhash_func<<<num_blocks, num_threads>>>(d_64, d_32, num_items);

    cudaDeviceSynchronize();

    for (int i = 0; i < num_items; i++) {
        if(h_64[i] != d_64[i]) {
            std::cout << "Wrong 64 bit at index: " << i << "\n";
            std::cout << h_64[i] << " != " << d_64[i] << "\n";
        }
        if(h_32[i] != d_32[i]) {
            std::cout << "Wrong 32 bit at index: " << i << "\n";
            std::cout << h_32[i] << " != " << d_32[i] << "\n";
        }
    }
    
    vec_t test = 1020;
    vec_t idx = 1022;
    test ^= idx;

    std::cout << test << "\n";

    return 0;
}