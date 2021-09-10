#include <fstream>
#include <cstring>
#include "graph.h"

// A class for reading from a binary graph stream
class BinaryGraphStream {
public:
    BinaryGraphStream(std::string file_name, uint32_t _b) {
        bin_file.open(file_name.c_str(), std::ios_base::in | std::ios_base::binary);

        // set the buffer size to be a multiple of an edge size and malloc memory
        buf_size = _b - (_b % edge_size);
        buf = (char *) malloc(buf_size * sizeof(char));
        start_buf = buf;

        // read header from the input file
        bin_file.read(reinterpret_cast<char *>(&num_nodes), 4);
        bin_file.read(reinterpret_cast<char *>(&num_edges), 8);
        
        read_data(); // read in the first block of data
    }
    ~BinaryGraphStream() {
        free(start_buf);
    }
    inline uint32_t nodes() {return num_nodes;}
    inline uint64_t edges() {return num_edges;}

    inline GraphUpdate get_edge() {
        UpdateType u = (UpdateType) *buf;
        uint32_t a;
        uint32_t b;

        std::memcpy(&a, buf + 1, sizeof(uint32_t));
        std::memcpy(&b, buf + 5, sizeof(uint32_t));
        
        buf += edge_size;
        if (buf - start_buf == buf_size) read_data();
    
        return {{a,b}, u};
    }

private:
    inline void read_data() {
        // set buf back to the beginning of the buffer read in data
        buf = start_buf;
        bin_file.read(buf, buf_size);
    }
    const uint32_t edge_size = sizeof(uint8_t) + 2 * sizeof(uint32_t); // size of a binary encoded edge
    std::ifstream bin_file; // file to read from
    char *buf;              // data buffer
    char *start_buf;        // the start of the data buffer
    uint32_t buf_size;      // how big is that data buffer
    uint32_t num_nodes;     // number of nodes in the graph
    uint64_t num_edges;     // number of edges in the graph stream
};
