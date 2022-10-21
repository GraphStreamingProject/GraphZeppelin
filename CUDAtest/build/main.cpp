#include <chrono>
#include <random>
#include <gtest/gtest.h>

#include "../src/supernode.cpp"
#include "../src/sketch.cpp"
#include "../include/util.h"

void superNodeTest() {
    uint64_t num_nodes = 8192;
    uint64_t seed;

    seed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    std::mt19937_64 r(seed);
    seed = r();

    std::cout << "Seed: " << seed << "\n";

    Supernode::configure(num_nodes, seed);
    Supernode* supernode = Supernode::makeSupernode(num_nodes, seed);

    SampleSketchRet ret_code = supernode->sample().second;
    ASSERT_EQ(ret_code, ZERO) << "Did not get ZERO when sampling empty vector";
}

int main() {

    superNodeTest();

    return 0;

}
