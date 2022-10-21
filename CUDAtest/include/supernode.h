#pragma once
#include "./sketch.h"

typedef std::pair<node_id_t, node_id_t> Edge;

class Supernode {
    static size_t bytes_size; 

    public:
        const uint64_t n; // for creating a copy
        const uint64_t seed; // for creating a copy
  
    private:
        size_t sketch_size;
        int idx;
        int num_sketches;

        /* collection of logn sketches to qutypedef std::pair<node_id_t, node_id_t> Edge;ery from, since we can't query from one
            sketch more than once */
        // The sketches, off the end.
        alignas(Sketch) char sketch_buffer[1];

        /*
         * @param n     the total number of nodes in the graph.
         * @param seed  the (fixed) seed value passed to each supernode.
         */
        Supernode(uint64_t n, uint64_t seed);


        // get the ith sketch in the sketch array
        inline Sketch* get_sketch(size_t i) {
            return reinterpret_cast<Sketch*>(sketch_buffer + i * sketch_size);
        }

    public:
        /*
         * Supernode construtors
         * @param n       the total number of nodes in the graph.
         * @param seed    the (fixed) seed value passed to each supernode.
         * @param loc     (Optional) the memory location to put the supernode.
         * @return        a pointer to the newly created supernode object
         */
        static Supernode* makeSupernode(uint64_t n, long seed, void *loc = malloc(bytes_size));

        static inline void configure(uint64_t n, vec_t sketch_fail_factor=100) {
            Sketch::configure(n*n, sketch_fail_factor);
            bytes_size = sizeof(Supernode) + log2(n)/(log2(3)-1) * Sketch::sketchSizeof() - sizeof(char);
        }

        /*
         * Function to sample an edge from the cut of a supernode.
         * @return   an edge in the cut, represented as an Edge with LHS <= RHS, 
         *           if one exists. Additionally, returns a code representing the
         *           sample result (good, zero, or fail)
         */
        std::pair<Edge, SampleSketchRet> sample();
};

class OutOfQueriesException : public std::exception {
    virtual const char* what() const throw() {
        return "This supernode cannot be sampled more times!";
    }
};