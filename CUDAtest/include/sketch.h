#pragma once
#include <vector>
#include <stdint.h>
#include <cmath>
#include <iostream>
#include "../src/util.cpp"
#include "./graph_zeppelin_common.h"

// max number of non-zeroes in vector is n/2*n/2=n^2/4
#define guess_gen(x) double_to_ull(log2(x) - 2)
#define bucket_gen(d) double_to_ull((log2(d)+1))

enum SampleSketchRet {
    GOOD,  // querying this sketch returned a single non-zero value
    ZERO,  // querying this sketch returned that there are no non-zero values
    FAIL   // querying this sketch failed to produce a single non-zero value
};

class Sketch {
    private:
        static vec_t failure_factor;     // Failure factor determines number of columns in sketch. Pr(failure) = 1 / factor
        static vec_t n;                  // Length of the vector this is sketching.
        static size_t num_elems;         // length of our actual arrays in number of elements
        static size_t num_buckets;       // Portion of array length, number of buckets
        static size_t num_guesses;       // Portion of array length, number of guesses

        // Seed used for hashing operations in this sketch.
        const uint64_t seed;

        // pointers to buckets
        vec_t*      bucket_a;
        vec_hash_t* bucket_c;

        // Flag to keep track if this sketch has already been queried.
        bool already_queried = false;

        // Buckets of this sketch.
        // Length is bucket_gen(failure_factor) * guess_gen(n).
        // For buckets[i * guess_gen(n) + j], the bucket has a 1/2^j probability
        // of containing an index. The first two are pointers into the buckets array.
        char buckets[1];

        Sketch(uint64_t seed);

    public:
        /*
        * Construct a sketch of a vector of size n
        * The optional parameters are used when building a sketch from a file
        * @param loc        A pointer to a location in memory where the caller would like the sketch constructed
        * @param seed       Seed to use for hashing operations
        * @param binary_in  (Optional) A file which holds an encoding of a sketch
        * @return           A pointer to a newly constructed sketch 
        */
        static Sketch* makeSketch(void* loc, uint64_t seed);

        /* configure the static variables of sketches
        * @param n               Length of the vector to sketch. (static variable)
        * @param failure_factor  The rate at which an individual sketch is allowed to fail (determines column width)
        * @return nothing
        */
        inline static void configure(vec_t _n, vec_t _factor) {
            n = _n;
            failure_factor = _factor;
            num_buckets = bucket_gen(failure_factor);
            num_guesses = guess_gen(n);
            num_elems = num_buckets * num_guesses + 1;

            std::cout << "n: " << n << "\n";
            std::cout << "failure_factor: " << failure_factor << "\n";
            std::cout << "num_buckets: " << num_buckets << "\n";
            std::cout << "num_guesses: " << num_guesses << "\n";
        }

        inline static size_t sketchSizeof()
        { return sizeof(Sketch) + num_elems * (sizeof(vec_t) + sizeof(vec_hash_t)) - sizeof(char); }

        inline static vec_t get_failure_factor() 
        { return failure_factor; }

        /*
         * Function to query a sketch.
         * @return   A pair with the result index and a code indicating if the type of result.
         */
        std::pair<vec_t, SampleSketchRet> query();
};