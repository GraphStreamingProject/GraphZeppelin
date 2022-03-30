#pragma once
#include <fstream>
#include <cstring>
#include <unistd.h> //open and close
#include <fcntl.h>
#include "graph.h"

class BadStreamException : public std::exception {
  virtual const char* what() const throw() {
    return "The stream file was not correctly opened. Does it exist?";
  }
};

// A class for reading from a binary graph stream
class BinaryGraphStream {
public:
  BinaryGraphStream(std::string file_name, uint32_t _b) {
    bin_file.open(file_name.c_str(), std::ios_base::in | std::ios_base::binary);

    if (!bin_file.is_open()) {
      throw BadStreamException();
    }
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
  uint32_t buf_size;      // how big is the data buffer
  uint32_t num_nodes;     // number of nodes in the graph
  uint64_t num_edges;     // number of edges in the graph stream
};

// Class for reading from a binary graph stream using many
// MT_StreamReader threads
class BinaryGraphStream_MT {
public:
  BinaryGraphStream_MT(std::string file_name, uint32_t _b) {
    stream_fd = open(file_name.c_str(), O_RDONLY, S_IRUSR);
    if (stream_fd == -1) {
      throw BadStreamException();
    }

    buf_size = _b - (_b % edge_size); // ensure buffer size is multiple of edge_size

    // read header from the input file
    read(stream_fd, reinterpret_cast<char *>(&num_nodes), 4);
    read(stream_fd, reinterpret_cast<char *>(&num_edges), 8);
    end_of_file = (num_edges * edge_size) + 12;
    stream_off = 12;
  }
  inline uint32_t nodes() {return num_nodes;}
  inline uint64_t edges() {return num_edges;}
  BinaryGraphStream_MT(const BinaryGraphStream_MT &) = delete;
  BinaryGraphStream_MT & operator=(const BinaryGraphStream_MT &) = delete;
  friend class MT_StreamReader;

private:
  int stream_fd;
  std::atomic<uint64_t> stream_off;
  uint32_t num_nodes;     // number of nodes in the graph
  uint64_t num_edges;     // number of edges in the graph stream
  uint32_t buf_size;      // how big is the data buffer
  const uint32_t edge_size = sizeof(uint8_t) + 2 * sizeof(uint32_t); // size of a binary encoded edge
  uint64_t end_of_file;
  inline bool read_data(char *buf) {
    uint64_t read_off = stream_off.fetch_add(buf_size, std::memory_order_relaxed);
    if (read_off >= end_of_file) return false;
    pread(stream_fd, buf, buf_size, read_off); // perform the read
    // TODO: pread may return less data than we asked for because it feels like it
    // we need to ensure that this does not happen unless we've reached the end of the file
    return true;
  }
};

// this class provides an interface for interacting with the
// BinaryGraphStream_MT from a single thread
class MT_StreamReader {
public:
  MT_StreamReader(BinaryGraphStream_MT &stream) : stream(stream) {
    // set the buffer size to be a multiple of an edge size and malloc memory
    buf = (char *) malloc(stream.buf_size * sizeof(char));
    start_buf = buf;

    // initialize buffer by calling read_data
    stream.read_data(start_buf);
  }

  inline GraphUpdate get_edge() {
    // if buffer is empty then read
    if (buf - start_buf == stream.buf_size) {
      if (!stream.read_data(start_buf)) {
        return {{-1, -1}, END_OF_FILE};
      }
      buf = start_buf; // point buf back to beginning of data buffer
    }

    UpdateType u = (UpdateType) *buf;
    uint32_t a;
    uint32_t b;

    std::memcpy(&a, buf + 1, sizeof(uint32_t));
    std::memcpy(&b, buf + 5, sizeof(uint32_t));
    
    buf += stream.edge_size; 

    return {{a,b}, u};
  }

private:
  char *buf;              // data buffer
  char *start_buf;        // the start of the data buffer
  BinaryGraphStream_MT &stream;
};
