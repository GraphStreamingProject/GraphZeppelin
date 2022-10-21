#include "../include/supernode.h"

size_t Supernode::bytes_size;

Supernode::Supernode(uint64_t n, uint64_t seed): idx(0), num_sketches(log2(n)/(log2(3)-1)),
               n(n), seed(seed), sketch_size(Sketch::sketchSizeof()) {


    std::cout << "num_sketches: " << num_sketches << "\n";

    size_t sketch_width = guess_gen(Sketch::get_failure_factor());

    std::cout << "sketch_width: " << sketch_width << "\n";
    // generate num_sketches sketches for each supernode (read: node)
    for (int i = 0; i < num_sketches; ++i) {
        Sketch::makeSketch(get_sketch(i), seed);
        seed += sketch_width;
    }
}

Supernode* Supernode::makeSupernode(uint64_t n, long seed, void *loc) {
    return new (loc) Supernode(n, seed);
}

std::pair<Edge, SampleSketchRet> Supernode::sample() {
    if (idx == num_sketches) throw OutOfQueriesException();

    std::cout << "Calling query" << "\n";
    std::pair<vec_t, SampleSketchRet> query_ret = get_sketch(idx++)->query();
    vec_t idx = query_ret.first;
    SampleSketchRet ret_code = query_ret.second;
    return {inv_concat_pairing_fn(idx), ret_code};
}